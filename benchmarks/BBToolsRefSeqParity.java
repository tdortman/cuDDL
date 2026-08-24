import cardinality.DynamicDemiLog;
import ddl.DDLIndexBase;
import ddl.DDLIndexCSR;
import ddl.DDLRecord;
import ddl.DDLLoader;
import ddl.CSRIndex2;
import fileIO.ByteFile;
import fileIO.FileFormat;
import parse.LineParser1;
import shared.Tools;
import simd.Vector;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeSet;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * BBTools DDLIndex/CSR2 oracle for the cuDDL decoded-row RefSeq parity validation.
 *
 * Loads the official precomputed RefSeq DDL A48 TSV with an order-preserving
 * multithreaded wrapper around the first-party record parser, excludes the
 * rows the parity manifest marks as held out, builds the 32-bit CSR index
 * once as the verification reference, and rebuilds the default 21-bit packed
 * CSR2 index for `WARMUP + RUNS` timed iterations while answering the
 * manifest's selected queries with each backend. The two backends must agree
 * bit-identically in every iteration; their per-query match counts, candidate
 * IDs at the same minimum-match threshold, and exact lower/equal/higher/
 * both-empty summaries for every retained candidate are written to a JSON
 * file that the cuDDL parity command compares against its own decoded-row
 * oracle and GPU index.
 *
 * This harness is benchmark support: it exercises the current BBTools
 * implementation, not a cuDDL reimplementation of it.
 *
 * Usage: java BBToolsRefSeqParity REF_TSV SELECTIONS MIN_HITS THREADS WARMUP RUNS OUT_JSON
 */
public final class BBToolsRefSeqParity {

    private static final int K = 25;
    private static final int VALUES = 65536;

    private BBToolsRefSeqParity() {}

    private static double nowSeconds() {
        return System.nanoTime() * 1e-9;
    }

    /**
     * FNV-1a 64-bit checksum over the absolute 16-bit scores of one decoded row,
     * plus the nonzero count. The cuDDL parity command computes the identical
     * checksum over its independently decoded rows, so a mismatch pinpoints a
     * decode disagreement instead of an index disagreement.
     */
    private static long[] rowChecksum(char[] row) {
        long hash = 0xcbf29ce484222325L;
        long nonzero = 0;
        for (char c : row) {
            hash ^= (c & 0xFFFF);
            hash *= 0x100000001b3L;
            if (c != 0) {
                nonzero++;
            }
        }
        return new long[] {hash, nonzero};
    }

    /** Approximate JVM heap in use, in bytes. */
    private static long heapUsed() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }

    private static final int RECORDS_PER_BUNDLE = 32;
    private static final byte[] PREFIX_TID = "#tid".getBytes();
    private static final byte[] PREFIX_ID = "#id".getBytes();
    private static final byte[] PREFIX_EXPONENT = "#exponent".getBytes();

    /**
     * Order-preserving multithreaded A48 loader.
     *
     * `DDLLoaderMT` parallelises record parsing but merges per-worker batches in worker order,
     * so the manifest's file ordinals would no longer address the same rows. This wrapper keeps
     * the same producer/consumer shape, submits batches to a fixed pool in file order, and
     * collects the futures in submission order, so parsed records remain in file order.
     */
    private static ArrayList<DDLRecord> loadFileOrderedMultithreaded(String path, int k, int threads)
            throws InterruptedException, ExecutionException {
        if (threads < 2) {
            return DDLLoader.loadFile(path, k);
        }

        FileFormat ff = FileFormat.testInput(path, FileFormat.TEXT, null, false, true);
        ByteFile bf = ByteFile.makeByteFile(ff);
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        ArrayList<Future<ArrayList<DDLRecord>>> futures = new ArrayList<>();
        ArrayList<byte[]> currentRecord = new ArrayList<>(12);
        ArrayList<ArrayList<byte[]>> bundle = new ArrayList<>(RECORDS_PER_BUNDLE);

        try {
            for (byte[] line = bf.nextLine(); line != null; line = bf.nextLine()) {
                if (line.length < 1) {
                    continue;
                }
                if (Tools.startsWith(line, PREFIX_EXPONENT, 0)) {
                    LineParser1 headerParser = new LineParser1('\t');
                    headerParser.set(line);
                    if (headerParser.terms() >= 2) {
                        DynamicDemiLog.setExponent((int) headerParser.parseLong(1), true, path);
                    }
                    continue;
                }
                if ((Tools.startsWith(line, PREFIX_TID, 0) || Tools.startsWith(line, PREFIX_ID, 0))
                        && !currentRecord.isEmpty()) {
                    bundle.add(currentRecord);
                    currentRecord = new ArrayList<>(12);
                    currentRecord.add(line);
                    if (bundle.size() >= RECORDS_PER_BUNDLE) {
                        final ArrayList<ArrayList<byte[]>> task = bundle;
                        futures.add(pool.submit(() -> parseRecordBatch(task, k)));
                        bundle = new ArrayList<>(RECORDS_PER_BUNDLE);
                    }
                } else {
                    currentRecord.add(line);
                }
            }

            if (!currentRecord.isEmpty()) {
                bundle.add(currentRecord);
            }
            if (!bundle.isEmpty()) {
                final ArrayList<ArrayList<byte[]>> task = bundle;
                futures.add(pool.submit(() -> parseRecordBatch(task, k)));
            }
        } finally {
            bf.close();
            pool.shutdown();
        }

        ArrayList<DDLRecord> merged = new ArrayList<>();
        for (Future<ArrayList<DDLRecord>> future : futures) {
            merged.addAll(future.get());
        }
        return merged;
    }

    private static ArrayList<DDLRecord> parseRecordBatch(
            ArrayList<ArrayList<byte[]>> batch, int k) {
        ArrayList<DDLRecord> records = new ArrayList<>(batch.size());
        LineParser1 parser = new LineParser1('\t');
        for (ArrayList<byte[]> lines : batch) {
            DDLRecord record = parseRecord(lines, parser, k);
            if (record != null) {
                records.add(record);
            }
        }
        return records;
    }

    private static DDLRecord parseRecord(ArrayList<byte[]> lines, LineParser1 parser, int k) {
        long recordId = -1;
        int taxId = -1;
        String name = null;
        String file = null;
        String origin = null;
        String lineage = null;
        long bases = 0;
        int contigs = 0;
        float gc = -1;
        int offset = -1;
        boolean hasKmers = false;
        byte[] dataLine = null;
        byte[] kmerLine = null;

        for (byte[] line : lines) {
            if (line.length < 1) {
                continue;
            }
            if (line[0] == '#') {
                parser.set(line);
                if (parser.terms() < 2) {
                    continue;
                }
                if (parser.termEquals("#id", 0)) {
                    recordId = parser.parseLong(1);
                } else if (parser.termEquals("#tid", 0)) {
                    taxId = (int) parser.parseLong(1);
                } else if (parser.termEquals("#name", 0)) {
                    name = parser.parseString(1);
                } else if (parser.termEquals("#file", 0)) {
                    file = parser.parseString(1);
                } else if (parser.termEquals("#bases", 0)) {
                    bases = parser.parseLong(1);
                } else if (parser.termEquals("#contigs", 0)) {
                    contigs = (int) parser.parseLong(1);
                } else if (parser.termEquals("#gc", 0)) {
                    gc = parser.parseFloat(1);
                } else if (parser.termEquals("#origin", 0)) {
                    origin = parser.parseString(1);
                } else if (parser.termEquals("#lineage", 0)) {
                    lineage = parser.parseString(1);
                } else if (parser.termEquals("#offset", 0)) {
                    offset = (int) parser.parseLong(1);
                } else if (parser.termEquals("#haskmers", 0)) {
                    hasKmers = parser.parseLong(1) > 0;
                }
            } else if (dataLine == null) {
                dataLine = line;
            } else {
                kmerLine = line;
            }
        }

        if (dataLine == null) {
            return null;
        }

        long[] kmers = null;
        if (hasKmers && kmerLine != null && kmerLine.length > 0) {
            kmers = DDLLoader.parseKmers(kmerLine, parser);
        }
        DynamicDemiLog ddl = DDLLoader.parseDDL(dataLine, parser, k, offset, kmers);
        DDLRecord record = new DDLRecord(ddl, recordId, taxId, name);
        record.filename = file;
        record.bases = bases;
        record.contigs = contigs;
        record.gc = gc;
        record.origin = origin;
        record.lineage = lineage;
        record.cardinality = ddl.cardinality();
        return record;
    }

    public static void main(String[] args) throws IOException, InterruptedException, ExecutionException {
        if (args.length != 7) {
            System.err.println(
                "usage: java BBToolsRefSeqParity REF_TSV SELECTIONS MIN_HITS THREADS WARMUP RUNS OUT_JSON");
            System.exit(2);
        }
        String refPath = args[0];
        String selectionsPath = args[1];
        int minHits = Integer.parseInt(args[2]);
        int threads = Integer.parseInt(args[3]);
        int warmup = Integer.parseInt(args[4]);
        int runs = Integer.parseInt(args[5]);
        String outPath = args[6];
        if (warmup < 0 || runs < 1) {
            throw new IllegalArgumentException("WARMUP must be >= 0 and RUNS >= 1");
        }
        int totalIterations = warmup + runs;

        // Parse the shared query-selection manifest. Every line selects one query:
        // external rows are removed from the database before the index is built, resident rows
        // stay in the database. Legacy holdout/query keys are accepted as aliases.
        TreeSet<Integer> external = new TreeSet<>();
        List<Integer> queries = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(selectionsPath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String trimmed = line.trim();
                if (trimmed.isEmpty() || trimmed.startsWith("#")) {
                    continue;
                }
                if (trimmed.startsWith("external:")) {
                    int ordinal = Integer.parseInt(
                        trimmed.substring(trimmed.indexOf(':') + 1).trim());
                    external.add(ordinal);
                    queries.add(ordinal);
                } else if (trimmed.startsWith("holdout:")) {
                    external.add(Integer.parseInt(
                        trimmed.substring(trimmed.indexOf(':') + 1).trim()));
                } else if (trimmed.startsWith("resident:") || trimmed.startsWith("query:")) {
                    queries.add(Integer.parseInt(
                        trimmed.substring(trimmed.indexOf(':') + 1).trim()));
                } else {
                    throw new IllegalArgumentException("unrecognized selections line: " + trimmed);
                }
            }
        }

        // Load all records with an order-preserving multithreaded wrapper around the
        // first-party parser. File order must be preserved because the shared manifest
        // addresses records by ordinal in both implementations.
        long tLoadStart = System.nanoTime();
        ArrayList<DDLRecord> records = loadFileOrderedMultithreaded(refPath, K, threads);
        double loadSeconds = (System.nanoTime() - tLoadStart) * 1e-9;
        System.err.printf(Locale.ROOT, "Loaded %d records in %.3f s%n", records.size(), loadSeconds);
        if (records.isEmpty()) {
            throw new IllegalStateException("no records loaded from " + refPath);
        }
        int buckets = records.get(0).ddl.maxArray().length;
        if (buckets < 1 || (buckets & (buckets - 1)) != 0) {
            throw new IllegalStateException("non-power-of-two bucket count: " + buckets);
        }

        // Build the database subset: every record except the held-out rows
        ArrayList<DDLRecord> dbRecords = new ArrayList<>(records.size() - external.size());
        for (int ordinal = 0; ordinal < records.size(); ordinal++) {
            if (!external.contains(ordinal)) {
                dbRecords.add(records.get(ordinal));
            }
        }

        // The ordered MT parser leaves transient batch buffers on the heap. Reclaim them so
        // the pre-index rows baseline and the post-CSR2 snapshot measure retained data rather
        // than parser garbage.
        System.gc();
        long heapBeforeIndex = heapUsed();
        double tCsrStart = nowSeconds();
        DDLIndexCSR csr = new DDLIndexCSR(buckets, VALUES);
        csr.addAll(dbRecords, threads);
        double csrBuildSeconds = nowSeconds() - tCsrStart;
        long heapAfterCsr = heapUsed();
        System.err.printf(
            Locale.ROOT, "Built 32-bit CSR index in %.3f s (heap +%.1f MiB)%n",
            csrBuildSeconds, (heapAfterCsr - heapBeforeIndex) / 1048576.0);

        // Answer every selected query with the 32-bit CSR backend once; these counts are the
        // bit-identity reference every rebuild must reproduce, so the reference index itself
        // can be released before the measured loop.
        int[][] csrCounts = new int[queries.size()][];
        for (int qi = 0; qi < queries.size(); qi++) {
            csrCounts[qi] = csr.query(records.get(queries.get(qi)).ddl);
        }
        csr = null;
        System.gc();

        // Measured iterations: warmup runs discarded, measured runs retained
        // Both backends are rebuilt and queried in every iteration, so the 21-bit CSR2 packing
        // can be compared with the native 32-bit CSR at the same work. Queries are answered as
        // a batch on a fixed thread pool, mirroring BBTools' multi-query mode; the per-query
        // CSR2 durations are retained as diagnostics.
        double[] csr2BuildSeconds = new double[totalIterations];
        double[] csrBuildRuns = new double[totalIterations];
        double[] batchSeconds = new double[totalIterations];
        double[] batchCsrSeconds = new double[totalIterations];
        double[][] querySeconds = new double[queries.size()][totalIterations];
        double[][] queryCsrSeconds = new double[queries.size()][totalIterations];
        long heapAfterCsr2 = heapUsed();
        long heapPeak = heapUsed();
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            for (int i = 0; i < totalIterations; i++) {
                final int iteration = i;

                double tStart = nowSeconds();
                CSRIndex2 rebuiltCsr2 = new CSRIndex2(buckets, VALUES);
                rebuiltCsr2.addAll(dbRecords, threads);
                csr2BuildSeconds[i] = nowSeconds() - tStart;
                if (i == 0) {
                    heapAfterCsr2 = heapUsed();
                }

                tStart = nowSeconds();
                DDLIndexCSR rebuiltCsr = new DDLIndexCSR(buckets, VALUES);
                rebuiltCsr.addAll(dbRecords, threads);
                csrBuildRuns[i] = nowSeconds() - tStart;

                // Query batches, verified against the reference counts. The backend order
                // alternates per iteration so neither gains a systematic cache-order advantage.
                boolean csr2First = (i % 2) == 0;
                for (int pass = 0; pass < 2; pass++) {
                    boolean csr2Pass = csr2First == (pass == 0);
                    DDLIndexBase backend = csr2Pass ? rebuiltCsr2 : rebuiltCsr;
                    String backendName = csr2Pass ? "CSR2" : "CSR";
                    double tBatchStart = nowSeconds();
                    List<Future<double[]>> futures = new ArrayList<>(queries.size());
                    for (int qi = 0; qi < queries.size(); qi++) {
                        final int queryIndex = qi;
                        futures.add(pool.submit(() -> {
                            DDLRecord query = records.get(queries.get(queryIndex));
                            long tQueryStart = System.nanoTime();
                            int[] counts = backend.query(query.ddl);
                            double seconds = (System.nanoTime() - tQueryStart) * 1e-9;
                            if (!java.util.Arrays.equals(counts, csrCounts[queryIndex])) {
                                throw new IllegalStateException(
                                    backendName + " disagrees with the reference for query "
                                    + "ordinal " + queries.get(queryIndex)
                                    + " in iteration " + iteration);
                            }
                            return new double[] {seconds, queryIndex};
                        }));
                    }
                    for (Future<double[]> future : futures) {
                        try {
                            double[] result = future.get();
                            if (csr2Pass) {
                                querySeconds[(int) result[1]][i] = result[0];
                            } else {
                                queryCsrSeconds[(int) result[1]][i] = result[0];
                            }
                        } catch (ExecutionException e) {
                            throw new IllegalStateException(
                                backendName + " batch query task failed", e.getCause());
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            throw new IllegalStateException(
                                backendName + " batch query interrupted", e);
                        }
                    }
                    if (csr2Pass) {
                        batchSeconds[i] = nowSeconds() - tBatchStart;
                    } else {
                        batchCsrSeconds[i] = nowSeconds() - tBatchStart;
                    }
                }
                heapPeak = Math.max(heapPeak, heapUsed());
                System.err.printf(
                    Locale.ROOT,
                    "Iteration %d/%d: CSR2 build %.3f s query %.3f s | CSR build %.3f s query %.3f s%n",
                    i + 1, totalIterations, csr2BuildSeconds[i], batchSeconds[i],
                    csrBuildRuns[i], batchCsrSeconds[i]);
            }
        } finally {
            pool.shutdown();
        }

        // Exact index layout sizes
        // The JVM heap snapshots move with garbage collection, so the figure's index-resident
        // comparison uses the exact array arithmetic of each backend instead: the 32-bit CSR
        // keeps (values+1) int offsets per bucket plus one int per posting, while CSR2 packs
        // both fields at 21 bits (three per long, see CSRLine.words).
        long nonemptySlots = 0;
        for (DDLRecord record : dbRecords) {
            char[] row = record.ddl.maxArray();
            for (char score : row) {
                if (score != 0) {
                    nonemptySlots++;
                }
            }
        }
        long csrWords = (long) buckets * ((VALUES + 1 + 2) / 3);
        long csrBytes = (long) buckets * (VALUES + 1) * 4L + nonemptySlots * 4L;
        long csr2Bytes = csrWords * 8L + ((nonemptySlots + 2L * buckets) / 3) * 8L;

        // Parity data (candidates and exact summaries) computed once from the reference
        // counts; the CSR/CSR2 agreement above makes them iteration-invariant
        List<Map<String, Object>> queryReports = new ArrayList<>();
        for (int qi = 0; qi < queries.size(); qi++) {
            int ordinal = queries.get(qi);
            DDLRecord query = records.get(ordinal);

            Map<String, Object> report = new LinkedHashMap<>();
            report.put("ordinal", ordinal);
            report.put("external", external.contains(ordinal));
            long[] checksum = rowChecksum(query.ddl.maxArray());
            report.put("row_checksum", checksum[0]);
            report.put("row_nonzero", checksum[1]);

            List<Map<String, Object>> countRows = new ArrayList<>();
            List<Integer> candidates = new ArrayList<>();
            for (int id = 0; id < csrCounts[qi].length; id++) {
                int count = csrCounts[qi][id];
                if (count >= 1) {
                    Map<String, Object> row = new LinkedHashMap<>();
                    row.put("id", id);
                    row.put("count", count);
                    countRows.add(row);
                }
                if (count >= minHits) {
                    // minHits == 0 is the exhaustive threshold: every reference is a candidate,
                    // including references with zero index matches.
                    candidates.add(id);
                }
            }
            report.put("counts", countRows);
            report.put("candidates", candidates);

            List<Map<String, Object>> summaryRows = new ArrayList<>();
            char[] queryRow = query.ddl.maxArray();
            for (int id : candidates) {
                int[] cmp = Vector.compareDDL(queryRow, dbRecords.get(id).ddl.maxArray());
                Map<String, Object> row = new LinkedHashMap<>();
                row.put("id", id);
                row.put("lower", cmp[0]);
                row.put("equal", cmp[1]);
                row.put("higher", cmp[2]);
                row.put("both_empty", cmp.length > 3 ? cmp[3] : 0);
                summaryRows.add(row);
            }
            report.put("summaries", summaryRows);
            queryReports.add(report);
        }

        // Emit the machine-readable oracle evidence
        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"records\": ").append(records.size()).append(",\n");
        json.append("  \"db_records\": ").append(dbRecords.size()).append(",\n");
        json.append("  \"min_hits\": ").append(minHits).append(",\n");
        json.append("  \"threads\": ").append(threads).append(",\n");
        json.append("  \"warmup\": ").append(warmup).append(",\n");
        json.append("  \"runs\": ").append(runs).append(",\n");
        json.append("  \"buckets\": ").append(buckets).append(",\n");
        json.append("  \"values\": ").append(VALUES).append(",\n");
        json.append("  \"external\": [");
        boolean first = true;
        for (int ordinal : external) {
            json.append(first ? "" : ", ").append(ordinal);
            first = false;
        }
        json.append("],\n");
        json.append("  \"timings_seconds\": {\n");
        json.append("    \"load\": ").append(loadSeconds).append(",\n");
        json.append("    \"index_build_csr\": ").append(csrBuildSeconds).append(",\n");
        json.append("    \"index_build_csr_runs\": [");
        for (int i = warmup; i < totalIterations; i++) {
            json.append(i == warmup ? "" : ", ").append(csrBuildRuns[i]);
        }
        json.append("],\n");
        json.append("    \"index_build_csr2_runs\": [");
        for (int i = warmup; i < totalIterations; i++) {
            json.append(i == warmup ? "" : ", ").append(csr2BuildSeconds[i]);
        }
        json.append("],\n");
        json.append("    \"query_batch_runs\": [");
        for (int i = warmup; i < totalIterations; i++) {
            json.append(i == warmup ? "" : ", ").append(batchSeconds[i]);
        }
        json.append("],\n");
        json.append("    \"query_batch_csr_runs\": [");
        for (int i = warmup; i < totalIterations; i++) {
            json.append(i == warmup ? "" : ", ").append(batchCsrSeconds[i]);
        }
        json.append("],\n");
        json.append("    \"query_runs\": [");
        for (int qi = 0; qi < queries.size(); qi++) {
            if (qi != 0) {
                json.append(", ");
            }
            json.append("[");
            for (int i = warmup; i < totalIterations; i++) {
                json.append(i == warmup ? "" : ", ").append(querySeconds[qi][i]);
            }
            json.append("]");
        }
        json.append("],\n");
        json.append("    \"query_csr_runs\": [");
        for (int qi = 0; qi < queries.size(); qi++) {
            if (qi != 0) {
                json.append(", ");
            }
            json.append("[");
            for (int i = warmup; i < totalIterations; i++) {
                json.append(i == warmup ? "" : ", ").append(queryCsrSeconds[qi][i]);
            }
            json.append("]");
        }
        json.append("]\n");
        json.append("  },\n");
        json.append("  \"memory_bytes\": {\n");
        json.append("    \"heap_used_before_index\": ").append(heapBeforeIndex).append(",\n");
        json.append("    \"heap_used_after_csr\": ").append(heapAfterCsr).append(",\n");
        json.append("    \"heap_used_after_csr2\": ").append(heapAfterCsr2).append(",\n");
        json.append("    \"heap_used_peak\": ").append(heapPeak).append(",\n");
        json.append("    \"nonempty_slots\": ").append(nonemptySlots).append(",\n");
        json.append("    \"index_csr_bytes\": ").append(csrBytes).append(",\n");
        json.append("    \"index_csr2_bytes\": ").append(csr2Bytes).append("\n");
        json.append("  },\n");
        json.append("  \"queries\": [\n");
        for (int qi = 0; qi < queryReports.size(); qi++) {
            Map<String, Object> report = queryReports.get(qi);
            json.append("    {\n");
            json.append("      \"ordinal\": ").append(report.get("ordinal")).append(",\n");
            json.append("      \"external\": ").append(report.get("external")).append(",\n");
            json.append("      \"row_checksum\": ").append(report.get("row_checksum")).append(",\n");
            json.append("      \"row_nonzero\": ").append(report.get("row_nonzero")).append(",\n");
            json.append("      \"candidates\": [");
            @SuppressWarnings("unchecked")
            List<Integer> candidates = (List<Integer>) report.get("candidates");
            for (int ci = 0; ci < candidates.size(); ci++) {
                json.append(ci == 0 ? "" : ", ").append(candidates.get(ci));
            }
            json.append("],\n");
            json.append("      \"summaries\": [");
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> summaryRows = (List<Map<String, Object>>) report.get("summaries");
            for (int si = 0; si < summaryRows.size(); si++) {
                Map<String, Object> row = summaryRows.get(si);
                if (si != 0) {
                    json.append(", ");
                }
                json.append("{\"id\": ").append(row.get("id"))
                    .append(", \"lower\": ").append(row.get("lower"))
                    .append(", \"equal\": ").append(row.get("equal"))
                    .append(", \"higher\": ").append(row.get("higher"))
                    .append(", \"both_empty\": ").append(row.get("both_empty"))
                    .append("}");
            }
            json.append("],\n");
            json.append("      \"counts\": [");
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> countRows = (List<Map<String, Object>>) report.get("counts");
            for (int ci = 0; ci < countRows.size(); ci++) {
                Map<String, Object> row = countRows.get(ci);
                if (ci != 0) {
                    json.append(", ");
                }
                json.append("{\"id\": ").append(row.get("id"))
                    .append(", \"count\": ").append(row.get("count"))
                    .append("}");
            }
            json.append("]\n");
            json.append("    }").append(qi == queryReports.size() - 1 ? "" : ",").append("\n");
        }
        json.append("  ]\n");
        json.append("}\n");

        java.nio.file.Path outParent = Paths.get(outPath).toAbsolutePath().getParent();
        if (outParent != null) {
            Files.createDirectories(outParent);
        }
        try (FileWriter writer = new FileWriter(outPath)) {
            writer.write(json.toString());
        }
        System.err.printf(
            Locale.ROOT, "Wrote %d query reports to %s%n", queryReports.size(), outPath);
    }
}
