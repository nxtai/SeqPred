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

    public int getHiddenSize() {
        return hiddenSize;
    }

    public int getInputSize() {
        return inputSize;
    }

    private int hiddenSize;
    private int inputSize;

    public static RnnParameterPack createEmptyPack(int vocabSize, int hiddenSize) {
        RnnParameterPack rnnParameterPack = new RnnParameterPack();
        rnnParameterPack.hiddenSize = hiddenSize;
        rnnParameterPack.inputSize = vocabSize;
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

        for (int i = 0; i<rnnParameterPack.getWhx().getRowDimension(); i++) {
            for (int j = 0; j<rnnParameterPack.getWhx().getColumnDimension(); j++) {
                rnnParameterPack.getWhx().setEntry(i,j,getRandom());
            }
        }

        for (int i = 0; i<rnnParameterPack.getWhh().getRowDimension(); i++) {
            for (int j = 0; j<rnnParameterPack.getWhh().getColumnDimension(); j++) {
                rnnParameterPack.getWhh().setEntry(i,j,getRandom());
            }
        }

        for (int i = 0; i<rnnParameterPack.getWyh().getRowDimension(); i++) {
            for (int j = 0; j<rnnParameterPack.getWyh().getColumnDimension(); j++) {
                rnnParameterPack.getWyh().setEntry(i,j,getRandom());
            }
        }

        for (int i = 0; i<rnnParameterPack.getBh().getDimension(); i++) {
            rnnParameterPack.getBh().setEntry(i,getRandom());
        }

        for (int i = 0; i<rnnParameterPack.getBy().getDimension(); i++) {
            rnnParameterPack.getBy().setEntry(i,getRandom());
        }

        for (int i = 0; i<rnnParameterPack.getBinit().getDimension(); i++) {
            rnnParameterPack.getBinit().setEntry(i,getRandom());
        }

        return rnnParameterPack;
    }

    private static double getRandom() {
        return Math.random() / 50;
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

    public void weightedAddition(RnnParameterPack pack, double alpha) {
        setWhx(getWhx().add(pack.getWhx().scalarMultiply(alpha)));
        setWhh(getWhh().add(pack.getWhh().scalarMultiply(alpha)));
        setWyh(getWyh().add(pack.getWyh().scalarMultiply(alpha)));
        setBh(getBh().add(pack.getBh().mapMultiply(alpha)));
        setBy(getBy().add(pack.getBy().mapMultiply(alpha)));
        setBinit(getBinit().add(pack.getBinit().mapMultiply(alpha)));
    }

    public JsonRnnParameterPack getJson() {
        JsonRnnParameterPack jsonPack = new JsonRnnParameterPack(whx.getData(),whh.getData(),wyh.getData(),bh.toArray(),by.toArray(),binit.toArray(),getHiddenSize(),getInputSize());
        return jsonPack;
    }

    public static RnnParameterPack createFromJson(JsonRnnParameterPack jsonPack) {
        RnnParameterPack rnnParameterPack = new RnnParameterPack();
        rnnParameterPack.hiddenSize = jsonPack.getHiddenSize();
        rnnParameterPack.inputSize = jsonPack.getInputSize();
        rnnParameterPack.setWhx(MatrixUtils.createRealMatrix(jsonPack.getWhx()));
        rnnParameterPack.setWhh(MatrixUtils.createRealMatrix(jsonPack.getWhh()));
        rnnParameterPack.setWyh(MatrixUtils.createRealMatrix(jsonPack.getWyh()));
        rnnParameterPack.setBh(MatrixUtils.createRealVector(jsonPack.getBh()));
        rnnParameterPack.setBy(MatrixUtils.createRealVector(jsonPack.getBy()));
        rnnParameterPack.setBinit(MatrixUtils.createRealVector(jsonPack.getBinit()));
        return rnnParameterPack;
    }
}
