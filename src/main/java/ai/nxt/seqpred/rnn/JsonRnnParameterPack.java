package ai.nxt.seqpred.rnn;

/**
 * Created by jh on 16/01/16.
 */
public class JsonRnnParameterPack {
    private double[][] whx;
    private double[][] whh;
    private double[][] wyh;
    private double[] bh;
    private double[] by;
    private double[] binit;
    private int hiddenSize;
    private int inputSize;

    public JsonRnnParameterPack(double[][] whx, double[][] whh, double[][] wyh, double[] bh, double[] by, double[] binit, int hiddenSize, int inputSize) {
        this.whx = whx;
        this.whh = whh;
        this.wyh = wyh;
        this.bh = bh;
        this.by = by;
        this.binit = binit;
        this.hiddenSize = hiddenSize;
        this.inputSize = inputSize;
    }

    public double[][] getWhx() {
        return whx;
    }

    public double[][] getWhh() {
        return whh;
    }

    public double[][] getWyh() {
        return wyh;
    }

    public double[] getBh() {
        return bh;
    }

    public double[] getBy() {
        return by;
    }

    public double[] getBinit() {
        return binit;
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    public int getInputSize() {
        return inputSize;
    }
}
