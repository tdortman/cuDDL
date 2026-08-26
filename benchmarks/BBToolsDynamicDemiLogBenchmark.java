import cardinality.DynamicDemiLog;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/** Times parallel BBTools DynamicDemiLog construction over deterministic packed values. */
public final class BBToolsDynamicDemiLogBenchmark {

    private static long splitmix64(long value) {
        value = (value ^ (value >>> 30)) * 0xbf58476d1ce4e5b9L;
        value = (value ^ (value >>> 27)) * 0x94d049bb133111ebL;
        return value ^ (value >>> 31);
    }

    private static long[] inputs(int count, long trial) {
        final long[] values = new long[count];
        final long mask = (1L << 50) - 1;

        for (int i = 0; i < count; i++) {
            values[i] = splitmix64(42L + i + trial * 0x9e3779b97f4a7c15L) & mask;
        }

        return values;
    }

    private static long run(long[] values, int buckets, int maxThreads) {
        final int threads = Math.min(values.length, maxThreads);
        final ExecutorService pool = Executors.newFixedThreadPool(threads);
        final List<Future<DynamicDemiLog>> partials = new ArrayList<>(threads);

        for (int worker = 0; worker < threads; worker++) {
            final int start = (int) ((long) values.length * worker / threads);
            final int end = (int) ((long) values.length * (worker + 1) / threads);
            partials.add(pool.submit(() -> {
                final DynamicDemiLog partial =
                    DynamicDemiLog.create(buckets, 25, 0, 0, false);
                for (int i = start; i < end; i++) {
                    partial.add(values[i]);
                }
                return partial;
            }));
        }

        final DynamicDemiLog sketch = DynamicDemiLog.create(buckets, 25, 0, 0, false);
        try {
            for (final Future<DynamicDemiLog> partial : partials) {
                sketch.add(partial.get());
            }
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("construction interrupted", error);
        } catch (ExecutionException error) {
            throw new IllegalStateException("construction failed", error.getCause());
        } finally {
            pool.shutdown();
        }

        return sketch.cardinality();
    }

    public static void main(String[] args) {
        if (args.length != 5) {
            throw new IllegalArgumentException("usage: COUNT BUCKETS THREADS WARMUP RUNS");
        }

        final int count = Integer.parseInt(args[0]);
        final int buckets = Integer.parseInt(args[1]);
        final int threads = Integer.parseInt(args[2]);
        final int warmup = Integer.parseInt(args[3]);
        final int runs = Integer.parseInt(args[4]);

        if (count < 1 || buckets < 1 || threads < 1 || warmup < 0 || runs < 1) {
            throw new IllegalArgumentException(
                "COUNT, BUCKETS, THREADS, and RUNS must be positive");
        }

        for (int i = 0; i < warmup; i++) {
            run(inputs(count, -1L - i), buckets, threads);
        }
        for (int i = 0; i < runs; i++) {
            final long[] values = inputs(count, i);
            final long start = System.nanoTime();
            final long estimate = run(values, buckets, threads);
            final double seconds = (System.nanoTime() - start) * 1e-9;
            System.out.printf(
                Locale.ROOT,
                "bbtools,%d,%d,%d,%d,%.9f,%.3f,%d%n",
                count,
                buckets,
                Math.min(count, threads),
                i,
                seconds,
                count / seconds,
                estimate
            );
        }
    }
}
