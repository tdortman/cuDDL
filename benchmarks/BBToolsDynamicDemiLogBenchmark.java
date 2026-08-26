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

    private static final int MAX_SHARD_LENGTH = 1 << 26;

    private static long[][] inputs(long count, int maxThreads, long trial) {
        final int workers = (int) Math.min(count, maxThreads);
        final long requiredShards = (count - 1) / MAX_SHARD_LENGTH + 1;
        final int shardCount = Math.toIntExact(Math.max(workers, requiredShards));
        final long baseLength = count / shardCount;
        final long extra = count % shardCount;
        final long[][] shards = new long[shardCount][];
        final long mask = (1L << 50) - 1;
        final long trialOffset = trial * 0x9e3779b97f4a7c15L;
        long start = 0;

        for (int shard = 0; shard < shardCount; shard++) {
            final int length = Math.toIntExact(baseLength + (shard < extra ? 1 : 0));
            final long[] values = new long[length];
            for (int i = 0; i < length; i++) {
                values[i] = splitmix64(42L + start + i + trialOffset) & mask;
            }
            shards[shard] = values;
            start += length;
        }

        return shards;
    }

    private static long run(
        long[][] values, int buckets, ExecutorService pool, int threads) {
        final List<Future<DynamicDemiLog>> partials = new ArrayList<>(threads);

        for (int worker = 0; worker < threads; worker++) {
            final int firstShard = worker;
            partials.add(pool.submit(() -> {
                final DynamicDemiLog partial =
                    DynamicDemiLog.create(buckets, 25, 0, 0, false);
                for (int shard = firstShard; shard < values.length; shard += threads) {
                    for (final long value : values[shard]) {
                        partial.add(value);
                    }
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
        }

        return sketch.cardinality();
    }

    public static void main(String[] args) {
        if (args.length != 5) {
            throw new IllegalArgumentException("usage: COUNT BUCKETS THREADS WARMUP RUNS");
        }

        final long count = Long.parseLong(args[0]);
        final int buckets = Integer.parseInt(args[1]);
        final int threads = Integer.parseInt(args[2]);
        final int warmup = Integer.parseInt(args[3]);
        final int runs = Integer.parseInt(args[4]);

        if (count < 1 || buckets < 1 || threads < 1 || warmup < 0 || runs < 1) {
            throw new IllegalArgumentException(
                "COUNT, BUCKETS, THREADS, and RUNS must be positive");
        }

        final int workerCount = (int) Math.min(count, threads);
        final ExecutorService pool = Executors.newFixedThreadPool(workerCount);
        try {
            for (int i = 0; i < warmup; i++) {
                run(inputs(count, threads, -1L - i), buckets, pool, workerCount);
            }
            for (int i = 0; i < runs; i++) {
                final long[][] values = inputs(count, threads, i);
                final long start = System.nanoTime();
                final long estimate = run(values, buckets, pool, workerCount);
                final double seconds = (System.nanoTime() - start) * 1e-9;
                System.out.printf(
                    Locale.ROOT,
                    "bbtools,%d,%d,%d,%d,%.9f,%.3f,%d,%d%n",
                    count,
                    buckets,
                    workerCount,
                    i,
                    seconds,
                    count / seconds,
                    estimate,
                    estimate
                );
            }
        } finally {
            pool.shutdown();
        }
    }
}
