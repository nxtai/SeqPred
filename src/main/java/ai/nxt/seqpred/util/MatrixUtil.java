package ai.nxt.seqpred.util;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;

/**
 * Created by Jeppe Hallgren on 28/06/15.
 */
public class MatrixUtil {
    public static ArrayList<RealVector> createListOfVectors(int listSize, int vectorSize) {
        ArrayList<RealVector> list = new ArrayList<RealVector>();
        for(int i = 0; i<listSize; i++) {
            list.add(i, MatrixUtils.createRealVector(new double[vectorSize]));
        }
        return list;
    }
}
