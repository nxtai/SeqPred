package ai.nxt.seqpred.rnn;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Created by Jeppe Hallgren on 21/06/15.
 */
public class RnnParameterPack {
    RealMatrix whx;
    RealMatrix whh;
    RealMatrix wyh;
    RealVector bh;
    RealVector by;
    RealVector binit;

    public static RnnParameterPack createEmptyPack(int vocabSize, int hiddenSize) {
        RnnParameterPack rnnParameterPack = new RnnParameterPack();
        rnnParameterPack.setWhx(MatrixUtils.createRealMatrix(hiddenSize, vocabSize));
        rnnParameterPack.setWhh(MatrixUtils.createRealMatrix(hiddenSize, hiddenSize));
        rnnParameterPack.setWyh(MatrixUtils.createRealMatrix(vocabSize, hiddenSize));
        rnnParameterPack.setBh(MatrixUtils.createRealVector(new double[hiddenSize]));
        rnnParameterPack.setBy(MatrixUtils.createRealVector(new double[vocabSize]));
        rnnParameterPack.setBinit(MatrixUtils.createRealVector(new double[hiddenSize]));
        return rnnParameterPack;
    }

    public static RnnParameterPack createRandomPack(int vocabSize, int hiddenSize) {
        RnnParameterPack rnnParameterPack = createEmptyPack(vocabSize, hiddenSize);
        // TODO: fill with random values
        return rnnParameterPack;
    }

    public RealMatrix getWhx() {
        return whx;
    }

    public void setWhx(RealMatrix whx) {
        this.whx = whx;
    }

    public RealMatrix getWhh() {
        return whh;
    }

    public void setWhh(RealMatrix whh) {
        this.whh = whh;
    }

    public RealMatrix getWyh() {
        return wyh;
    }

    public void setWyh(RealMatrix wyh) {
        this.wyh = wyh;
    }

    public RealVector getBh() {
        return bh;
    }

    public void setBh(RealVector bh) {
        this.bh = bh;
    }

    public RealVector getBy() {
        return by;
    }

    public void setBy(RealVector by) {
        this.by = by;
    }

    public RealVector getBinit() {
        return binit;
    }

    public void setBinit(RealVector binit) {
        this.binit = binit;
    }
}
