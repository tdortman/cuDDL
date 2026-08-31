import cardinality.DynamicDemiLog;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/** Measures comparison-only throughput over batches of independent BBTools DDL pairs. */
public final class BBToolsPairwiseBatchBenchmark {
    private static final int K = 25;
    private static final int BUCKETS = 2048;
    private static final int KMERS_PER_SKETCH = 4096;
    private static final int[] BATCH_SIZES = {1, 8, 32, 128, 512, 2048, 8192};
    private static final long MIDDLE_MASK = (1L << (2 * K - 4)) - 1;
    private static final long CANONICAL_PREFIX = 3L << (2 * K - 2);
    private static volatile long blackhole;

    private record Pair(DynamicDemiLog left, DynamicDemiLog right) {}

    private static long splitmix64(long value) {
        value = (value ^ (value >>> 30)) * 0xbf58476d1ce4e5b9L;
        value = (value ^ (value >>> 27)) * 0x94d049bb133111ebL;
        return value ^ (value >>> 31);
    }

    private static long kmer(long seed, long index) {
        final long multiplier = (splitmix64(seed) & MIDDLE_MASK) | 1;
        final long offset = splitmix64(seed ^ 0x9e3779b97f4a7c15L) & MIDDLE_MASK;
        final long middle = (multiplier * index + offset) & MIDDLE_MASK;
        return CANONICAL_PREFIX | (middle << 2) | 3;
    }

    private static List<Pair> buildPairs(int count) {
        final List<Pair> pairs = new ArrayList<>(count);
        for (int pair = 0; pair < count; pair++) {
            final DynamicDemiLog left = DynamicDemiLog.create(BUCKETS, K, 0, 0, false);
            final DynamicDemiLog right = DynamicDemiLog.create(BUCKETS, K, 0, 0, false);
            final long seed = 0x123456789abcdef0L + pair;
            for (int index = 0; index < KMERS_PER_SKETCH; index++) {
                left.add(kmer(seed, index));
                right.add(kmer(seed, index < KMERS_PER_SKETCH / 2 ? index : index + KMERS_PER_SKETCH));
            }
            pairs.add(new Pair(left, right));
        }
        return pairs;
    }

    private static long compareRange(List<Pair> pairs, int begin, int end) {
        long checksum = 0;
        for (int index = begin; index < end; index++) {
            final int[] counts = pairs.get(index).left().compareToDetailed(pairs.get(index).right());
            checksum += counts[0] + counts[1] + counts[2] + counts[3];
        }
        return checksum;
    }

    private static long compareParallel(
        ExecutorService pool,
        List<Callable<Long>> tasks) {
        long checksum = 0;
        try {
            for (Future<Long> result : pool.invokeAll(tasks)) {
                checksum += result.get();
            }
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("parallel comparison interrupted", error);
        } catch (ExecutionException error) {
            throw new IllegalStateException("parallel comparison failed", error.getCause());
        }
        return checksum;
    }

    private static void emit(
        String implementation,
        int batch,
        int threads,
        int iterations,
        int trial,
        long elapsedNanos,
        long checksum) {
        final double secondsPerBatch = elapsedNanos * 1e-9 / iterations;
        final double comparisonsPerSecond = batch / secondsPerBatch;
        System.out.printf(
            Locale.ROOT,
            "%s,%d,%d,%d,%d,%.12g,%.12g,%.12g,%d%n",
            implementation,
            batch,
            threads,
            iterations,
            trial,
            secondsPerBatch,
            comparisonsPerSecond,
            secondsPerBatch * 1e9 / batch,
            checksum);
        blackhole ^= checksum;
    }

    public static void main(String[] args) {
        Locale.setDefault(Locale.ROOT);
        if (args.length != 3) {
            throw new IllegalArgumentException("usage: THREADS ITERATIONS TRIALS");
        }
        final int threads = Integer.parseInt(args[0]);
        final int iterations = Integer.parseInt(args[1]);
        final int trials = Integer.parseInt(args[2]);
        if (threads < 1 || iterations < 1 || trials < 1) {
            throw new IllegalArgumentException("all arguments must be positive");
        }

        System.out.println(
            "implementation,batch,threads,iterations,trial,seconds_per_batch," +
                "comparisons_per_second,ns_per_comparison,checksum");
        final ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            for (int batch : BATCH_SIZES) {
                final List<Pair> pairs = buildPairs(batch);
                final int workers = Math.min(batch, threads);
                final List<Callable<Long>> tasks = new ArrayList<>(workers);
                for (int worker = 0; worker < workers; worker++) {
                    final int begin = worker * batch / workers;
                    final int end = (worker + 1) * batch / workers;
                    tasks.add(() -> compareRange(pairs, begin, end));
                }

                for (int warmup = 0; warmup < 5; warmup++) {
                    blackhole ^= compareRange(pairs, 0, batch);
                    blackhole ^= compareParallel(pool, tasks);
                }
                for (int trial = 0; trial < trials; trial++) {
                    long checksum = 0;
                    long start = System.nanoTime();
                    for (int iteration = 0; iteration < iterations; iteration++) {
                        checksum += compareRange(pairs, 0, batch);
                    }
                    emit(
                        "bbtools_sequential",
                        batch,
                        1,
                        iterations,
                        trial,
                        System.nanoTime() - start,
                        checksum);

                    checksum = 0;
                    start = System.nanoTime();
                    for (int iteration = 0; iteration < iterations; iteration++) {
                        checksum += compareParallel(pool, tasks);
                    }
                    emit(
                        "bbtools_parallel",
                        batch,
                        threads,
                        iterations,
                        trial,
                        System.nanoTime() - start,
                        checksum);
                }
            }
        } finally {
            pool.shutdown();
        }
        if (blackhole == Long.MIN_VALUE) {
            throw new AssertionError("unreachable");
        }
    }
}
