import cardinality.DynamicDemiLog;
import java.util.Locale;

/**
 * Cardinality estimates of BBTools DynamicDemiLog over the seeded sets of cuddl_hll_accuracy.cu:
 * trial t at 2^p holds (multiplier * i + seed) mod 2^50 for i below 2^p, with seed the SplitMix64
 * finalizer of root ^ (p << 32) ^ t and multiplier the finalizer of seed, made odd. Prints one
 * "items,trial,estimate" line per set.
 *
 * <p>Usage: BBToolsHllAccuracy MIN_POWER MAX_POWER TRIALS ROOT_SEED BUCKETS
 */
public final class BBToolsHllAccuracy {

    /** cuddl::detail::splitmix64: the finalizer alone, without the generator's increment. */
    private static long splitmix64(long value) {
        value = (value ^ (value >>> 30)) * 0xbf58476d1ce4e5b9L;
        value = (value ^ (value >>> 27)) * 0x94d049bb133111ebL;
        return value ^ (value >>> 31);
    }

    public static void main(String[] args) {
        final int minPower = Integer.parseInt(args[0]);
        final int maxPower = Integer.parseInt(args[1]);
        final int trials = Integer.parseInt(args[2]);
        final long rootSeed = Long.parseLong(args[3]);
        final int buckets = Integer.parseInt(args[4]);
        final long mask = (1L << 50) - 1L;
        for (int power = minPower; power <= maxPower; ++power) {
            final long count = 1L << power;
            for (int trial = 0; trial < trials; ++trial) {
                final long seed = splitmix64(rootSeed ^ ((long) power << 32) ^ trial);
                final long multiplier = splitmix64(seed) | 1L;
                final DynamicDemiLog ddl = DynamicDemiLog.create(buckets, 25, 0, 0, false);
                for (long i = 0; i < count; ++i) {
                    ddl.add((multiplier * i + seed) & mask);
                }
                System.out.printf(Locale.ROOT, "%d,%d,%d%n", count, trial, ddl.cardinality());
            }
        }
    }
}
