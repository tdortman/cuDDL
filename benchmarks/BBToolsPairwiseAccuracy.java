import cardinality.DynamicDemiLog;
import ddl.DDLQueryLoaderSF;
import ddl.DDLRecord;
import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** Replays cuDDL's raw-FASTA pair cases through the BBTools DynamicDemiLog reference. */
public final class BBToolsPairwiseAccuracy {
    private static final int K = 25;
    private static final int BUCKETS = 2048;
    private static final long BASES_PER_THREAD = 1L << 20;
    private static final String[] REQUIRED_COLUMNS = {
        "implementation",
        "reference_path",
        "query_path",
        "reference_bases",
        "query_bases",
        "orientation",
        "lower",
        "equal",
        "higher",
        "both_empty",
        "exact_containment",
        "sketch_containment",
        "containment_signed_error",
        "containment_absolute_error",
        "exact_completeness",
        "sketch_completeness",
        "completeness_signed_error",
        "completeness_absolute_error",
        "exact_wkid",
        "sketch_wkid",
        "wkid_signed_error",
        "wkid_absolute_error",
        "exact_ani",
        "sketch_ani",
        "ani_signed_error",
        "ani_absolute_error",
    };

    private BBToolsPairwiseAccuracy() {}

    private static String[] parseCsvLine(String line) {
        final List<String> values = new ArrayList<>();
        final StringBuilder value = new StringBuilder();
        boolean quoted = false;
        boolean closedQuote = false;
        for (int i = 0; i < line.length(); i++) {
            final char character = line.charAt(i);
            if (quoted) {
                if (character == '"') {
                    if (i + 1 < line.length() && line.charAt(i + 1) == '"') {
                        value.append('"');
                        i++;
                    } else {
                        quoted = false;
                        closedQuote = true;
                    }
                } else {
                    value.append(character);
                }
            } else if (character == ',') {
                values.add(value.toString());
                value.setLength(0);
                closedQuote = false;
            } else if (character == '"') {
                if (value.length() != 0 || closedQuote) {
                    throw new IllegalArgumentException("Malformed cuDDL CSV row");
                }
                quoted = true;
            } else if (closedQuote) {
                throw new IllegalArgumentException("Malformed cuDDL CSV row");
            } else {
                value.append(character);
            }
        }
        if (quoted) {
            throw new IllegalArgumentException("Unterminated quoted cuDDL CSV field");
        }
        values.add(value.toString());
        return values.toArray(String[]::new);
    }

    private static Map<String, String> parse(String[] header, String line) {
        final String[] values = parseCsvLine(line);
        if (values.length != header.length) {
            throw new IllegalArgumentException("Malformed cuDDL CSV row");
        }
        final Map<String, String> row = new HashMap<>();
        for (int i = 0; i < header.length; i++) {
            row.put(header[i], values[i]);
        }
        return row;
    }

    private static void validateHeader(String[] header) {
        final Map<String, Boolean> columns = new HashMap<>();
        for (String column : header) {
            columns.put(column, true);
        }
        for (String required : REQUIRED_COLUMNS) {
            if (!columns.containsKey(required)) {
                throw new IllegalArgumentException("cuDDL CSV is missing column: " + required);
            }
        }
    }

    private static int fileThreads(long bases, int maxThreads) {
        final long usefulThreads = Math.max(1L, (bases + BASES_PER_THREAD - 1L) / BASES_PER_THREAD);
        return (int) Math.min(maxThreads, usefulThreads);
    }

    private static DDLRecord loadRecord(
        Map<Path, DDLRecord> cache,
        String pathText,
        long expectedBases,
        int maxThreads) {
        final Path path = Path.of(pathText).toAbsolutePath().normalize();
        DDLRecord record = cache.get(path);
        if (record == null) {
            record = DDLQueryLoaderSF.loadPerFile(
                path.toString(), BUCKETS, K, false, fileThreads(expectedBases, maxThreads));
            if (record == null) {
                throw new IllegalStateException("BBTools did not load a record from: " + path);
            }
            cache.put(path, record);
        }
        if (record.bases != expectedBases) {
            throw new IllegalStateException(
                "BBTools base count mismatch for " + path + ": expected " + expectedBases
                    + ", got " + record.bases);
        }
        return record;
    }

    private static void setMetric(
        Map<String, String> row,
        String exactColumn,
        String sketchColumn,
        String signedErrorColumn,
        String absoluteErrorColumn,
        double estimate) {
        final double exact = Double.parseDouble(row.get(exactColumn));
        final double signedError = estimate - exact;
        row.put(sketchColumn, Double.toString(estimate));
        row.put(signedErrorColumn, Double.toString(signedError));
        row.put(absoluteErrorColumn, Double.toString(Math.abs(signedError)));
    }

    private static void compare(
        Map<String, String> row,
        DDLRecord reference,
        DDLRecord query) {
        final DynamicDemiLog left;
        final DynamicDemiLog right;
        switch (row.get("orientation")) {
            case "query_to_reference" -> {
                left = query.ddl;
                right = reference.ddl;
            }
            case "reference_to_query" -> {
                left = reference.ddl;
                right = query.ddl;
            }
            default -> throw new IllegalArgumentException(
                "Unknown orientation: " + row.get("orientation"));
        }

        final int[] counts = left.compareToDetailed(right);
        if (counts.length != 4 || counts[0] + counts[1] + counts[2] + counts[3] != BUCKETS) {
            throw new IllegalStateException("BBTools pairwise counts do not sum to " + BUCKETS);
        }
        final int lower = counts[0];
        final int equal = counts[1];
        final int higher = counts[2];

        row.put("implementation", "bbtools");
        row.put("lower", Integer.toString(lower));
        row.put("equal", Integer.toString(equal));
        row.put("higher", Integer.toString(higher));
        row.put("both_empty", Integer.toString(counts[3]));
        setMetric(
            row,
            "exact_containment",
            "sketch_containment",
            "containment_signed_error",
            "containment_absolute_error",
            DynamicDemiLog.containmentAB(lower, equal, higher));
        setMetric(
            row,
            "exact_completeness",
            "sketch_completeness",
            "completeness_signed_error",
            "completeness_absolute_error",
            DynamicDemiLog.completeness(lower, equal, higher));
        setMetric(
            row,
            "exact_wkid",
            "sketch_wkid",
            "wkid_signed_error",
            "wkid_absolute_error",
            DynamicDemiLog.wkid(lower, equal, higher));
        setMetric(
            row,
            "exact_ani",
            "sketch_ani",
            "ani_signed_error",
            "ani_absolute_error",
            DynamicDemiLog.ani(lower, equal, higher, K));
    }

    private static String emit(String[] header, Map<String, String> row) {
        final StringBuilder output = new StringBuilder(1024);
        for (int i = 0; i < header.length; i++) {
            if (i > 0) {
                output.append(',');
            }
            final String value = row.get(header[i]);
            if (value.indexOf(',') < 0 && value.indexOf('"') < 0
                && value.indexOf('\r') < 0 && value.indexOf('\n') < 0) {
                output.append(value);
            } else {
                output.append('"').append(value.replace("\"", "\"\"")).append('"');
            }
        }
        return output.toString();
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        if (args.length != 2) {
            throw new IllegalArgumentException("usage: CUDDL_CSV THREADS");
        }
        final Path csv = Path.of(args[0]);
        final int maxThreads = Integer.parseInt(args[1]);
        if (maxThreads < 1) {
            throw new IllegalArgumentException("THREADS must be positive");
        }

        DynamicDemiLog.setExponent(6);
        final Map<Path, DDLRecord> cache = new HashMap<>();
        try (BufferedReader reader = Files.newBufferedReader(csv)) {
            final String headerLine = reader.readLine();
            if (headerLine == null) {
                throw new IllegalArgumentException("cuDDL CSV is empty");
            }
            final String[] header = parseCsvLine(headerLine);
            validateHeader(header);
            System.out.println(headerLine);
            for (String line = reader.readLine(); line != null; line = reader.readLine()) {
                final Map<String, String> row = parse(header, line);
                final long referenceBases = Long.parseLong(row.get("reference_bases"));
                final long queryBases = Long.parseLong(row.get("query_bases"));
                final DDLRecord reference = loadRecord(
                    cache, row.get("reference_path"), referenceBases, maxThreads);
                final DDLRecord query = loadRecord(
                    cache, row.get("query_path"), queryBases, maxThreads);
                compare(row, reference, query);
                System.out.println(emit(header, row));
            }
        }
    }
}
