package ai.nxt.seqpred.util;

/**
 * Created by Jeppe Hallgren on 25/07/15.
 */
public class ProbabilityUtil {
    public static int getWeightedIndex(double[] prediction) {
        double random = Math.random();
        double culm = 0;
        for (int i = 0; i<prediction.length; i++) {
            culm += prediction[i];
            if (culm >= random) return i;
        }
        return prediction.length-1; // last index
    }
}
