package ai.nxt.seqpred.util;

/**
 * Created by jh on 22/06/15.
 */
public class NnUtil {
    public static double sigmoidFunction (double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double derivativeOfSigmoidFunction (double x) {
        // derivative of sigmoid
        double tmp = sigmoidFunction(x);
        return tmp * (1 - tmp);
    }

    public static double tanhFunction (double x) {
        return Math.tanh(x);
    }

    public static double derivativeOfTanhFunction (double x) {
        double tmp = tanhFunction(x);
        return 1 - (tmp*tmp);
    }
}
