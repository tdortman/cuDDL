import cardinality.DynamicDemiLog;
import java.util.Locale;

/** Times the original BBTools DynamicDemiLog over deterministic packed values. */
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

    private static long run(long[] values, int buckets) {
        final DynamicDemiLog sketch = DynamicDemiLog.create(buckets, 25, 0, 0, false);

        for (final long value : values) {
            sketch.add(value);
        }

        return sketch.cardinality();
    }

    public static void main(String[] args) {
        if (args.length != 4) {
            throw new IllegalArgumentException("usage: COUNT BUCKETS WARMUP RUNS");
        }

        final int count = Integer.parseInt(args[0]);
        final int buckets = Integer.parseInt(args[1]);
        final int warmup = Integer.parseInt(args[2]);
        final int runs = Integer.parseInt(args[3]);

        if (count < 1 || buckets < 1 || warmup < 0 || runs < 1) {
            throw new IllegalArgumentException("COUNT, BUCKETS, and RUNS must be positive");
        }

        for (int i = 0; i < warmup; i++) {
            run(inputs(count, -1L - i), buckets);
        }
        for (int i = 0; i < runs; i++) {
            final long[] values = inputs(count, i);
            final long start = System.nanoTime();
            final long estimate = run(values, buckets);
            final double seconds = (System.nanoTime() - start) * 1e-9;
            System.out.printf(
                Locale.ROOT,
                "bbtools,%d,%d,%d,%.9f,%.3f,%d%n",
                count,
                buckets,
                i,
                seconds,
                count / seconds,
                estimate
            );
        }
    }
}
