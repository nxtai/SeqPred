package ai.nxt.seqpred.util;

import junit.framework.TestCase;
import org.junit.Test;

/**
 * Created by Jeppe Hallgren on 28/06/15.
 */
public class NnUtilTest extends TestCase {

    @Test
    public void testSigmoidFunction() throws Exception {
        assertTrue(NnUtil.sigmoidFunction(5) < 1);
        assertTrue(NnUtil.sigmoidFunction(5) > 0.99);
        assertTrue(NnUtil.sigmoidFunction(0) > 0.49999);
        assertTrue(NnUtil.sigmoidFunction(0) < 5.00001);
        assertTrue(NnUtil.sigmoidFunction(-5) > 0.006);
        assertTrue(NnUtil.sigmoidFunction(-5) < 0.007);
    }

    @Test
    public void testDerivativeOfSigmoidFunction() throws Exception {
        assertTrue(NnUtil.derivativeOfSigmoidFunction(5) > 0.006);
        assertTrue(NnUtil.derivativeOfSigmoidFunction(5) < 0.007);
        assertTrue(NnUtil.derivativeOfSigmoidFunction(0) > 0.24449);
        assertTrue(NnUtil.derivativeOfSigmoidFunction(0) < 0.25001);
        assertTrue(NnUtil.derivativeOfSigmoidFunction(-5) > 0.006);
        assertTrue(NnUtil.derivativeOfSigmoidFunction(-5) < 0.007);
    }

    @Test
    public void testTanhFunction() throws Exception {
        assertTrue(NnUtil.tanhFunction(5) > 0.9999);
        assertTrue(NnUtil.tanhFunction(5) < 1);
        assertTrue(NnUtil.tanhFunction(0) > -0.00001);
        assertTrue(NnUtil.tanhFunction(0) < 0.00001);
        assertTrue(NnUtil.tanhFunction(-5) > -1);
        assertTrue(NnUtil.tanhFunction(-5) < -0.9999);
    }

    @Test
    public void testDerivativeOfTanhFunction() throws Exception {
        assertTrue(NnUtil.derivativeOfTanhFunction(5) > 0.00018);
        assertTrue(NnUtil.derivativeOfTanhFunction(5) < 0.00019);
        assertTrue(NnUtil.derivativeOfTanhFunction(0) > 0.99999);
        assertTrue(NnUtil.derivativeOfTanhFunction(0) < 1.00001);
        assertTrue(NnUtil.derivativeOfTanhFunction(-5) > 0.00018);
        assertTrue(NnUtil.derivativeOfTanhFunction(-5) < 0.00019);
    }
}