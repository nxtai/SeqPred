package ai.nxt.seqpred.rnn;

import ai.nxt.seqpred.Model;
import ai.nxt.seqpred.Vocab;
import ai.nxt.seqpred.util.FileUtil;
import ai.nxt.seqpred.util.NnUtil;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

/**
 * Created by Jeppe Hallgren on 21/06/15.
 */
public class Rnn extends Model {
    RnnParameterPack currentNetworkParameters;
    private int hiddenLayerSize = 15;
    private int minibatchSize = 10;
    private double lastPerplexity;
    private int testTokensSeen = 0;

    private ArrayList<RealVector> x;
    private ArrayList<RealVector> t;
    private ArrayList<RealVector> h;
    private ArrayList<RealVector> s;
    private ArrayList<RealVector> y;

    public Rnn(Vocab vocab) {
        super(vocab);
    }

    public void train() {
        lastPerplexity = Double.MAX_VALUE;
        int wordCount = FileUtil.countWords(getTrainingFileName());
        int minibatchCount = wordCount / minibatchSize;

        // initialise settings
        currentNetworkParameters = RnnParameterPack.createRandomPack(vocab.getVocabSize(),hiddenLayerSize);

        int gradRound = 0;

        // iterate over minibatches until convergence
        int currentMinibatch = 0;
        do {
            System.out.println("--- Starting grad round " + gradRound + " ---");
            System.out.println("--- Using minibatch " + currentMinibatch + " ---");

            // forward propagate
            double loss = forwardPass(currentNetworkParameters, currentMinibatch, true);
            System.out.println("Loss of forward prop: " + loss);

            RnnParameterPack gradient = computeGrad(currentNetworkParameters);
            currentNetworkParameters.weightedAddition(gradient,1/minibatchSize);

            // switch to next minibatch
            currentMinibatch = (currentMinibatch + 1) % minibatchCount;

            // increment count
            gradRound++;
        } while (! hasConverged(gradRound));
    }

    public boolean hasConverged(int gradRound) {
        if (gradRound < 100) return false;

        double newPerplexity = 80;

        if (newPerplexity < lastPerplexity) {
            lastPerplexity = newPerplexity;
        } else {
            return true;
        }
        return false;
    }

    private double forwardPass(RnnParameterPack networkSettings, int currentMinibatch, boolean updateState) {
        ArrayList<RealVector> x = new ArrayList<RealVector>();
        ArrayList<RealVector> t = new ArrayList<RealVector>();
        ArrayList<RealVector> h = new ArrayList<RealVector>();
        ArrayList<RealVector> s = new ArrayList<RealVector>();
        ArrayList<RealVector> y = new ArrayList<RealVector>();

        for(int i = 0; i<minibatchSize+1; i++){
            x.add(i, MatrixUtils.createRealVector(new double[vocab.getVocabSize()]));
            t.add(i, MatrixUtils.createRealVector(new double[hiddenLayerSize]));
            h.add(i, MatrixUtils.createRealVector(new double[hiddenLayerSize]));
            s.add(i, MatrixUtils.createRealVector(new double[vocab.getVocabSize()]));
            y.add(i, MatrixUtils.createRealVector(new double[vocab.getVocabSize()]));
        }

        double newLogSum = 0.0;
        try {
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(new FileInputStream(getTrainingFileName())));

            // skip words up to start of current minibatch
            for (int i = 0; i < currentMinibatch * minibatchSize; i++){
                FileUtil.readNextWord(reader);
            }

            // iterator over minibatch
            for (int i = 1; i<=minibatchSize; i++) {
                int wordIndex = vocab.getWordIndex(FileUtil.readNextWord(reader));
                x.get(i).setEntry(wordIndex,1);

                if (i == 1) {
                    // Whh*h0 not defined, use binit
                    t.set(i, (networkSettings.getWhx().operate(x.get(i))).add(networkSettings.getBinit()).add(networkSettings.getBh()));
                } else {
                    t.set(i, (networkSettings.getWhx().operate(x.get(i))).add(networkSettings.getWhh().operate(h.get(i - 1))).add(networkSettings.getBh()));
                }
                for (int j = 0; j < t.get(i).getDimension(); j++) {
                    h.get(i).setEntry(j, hiddenActivationFunction(t.get(i).getEntry(j)));
                }
                s.set(i, (networkSettings.getWyh().operate(h.get(i))).add(networkSettings.getBy()));
                for (int j = 0; j < s.get(i).getDimension(); j++) {
                    y.get(i).setEntry(j, outputActivationFunction(s.get(i).getEntry(j)));
                }
                if (i > 1){
                    double ysum = y.get(i-1).getNorm();
                    double base_prob = (ysum / (double)y.get(i-1).getDimension()) * 0.01;
                    double prob_word = (y.get(i-1).getEntry(wordIndex)+base_prob) / (ysum+(base_prob*y.get(i-1).getDimension()));
                    newLogSum += Math.log10(prob_word);
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        // if we should save forward prop result
        if (updateState) {
            this.x = x;
            this.t = t;
            this.h = h;
            this.s = s;
            this.y = y;
        }

        // return perplexity
        return Math.pow(10,-1*newLogSum / minibatchSize);
    }

    private double outputActivationFunction(double a) {
        return NnUtil.sigmoidFunction(a);
    }

    private double derivativeOfOutputActivationFunction(double a) {
        return NnUtil.derivativeOfSigmoidFunction(a);
    }

    private double hiddenActivationFunction(double a) {
        return NnUtil.tanhFunction(a);
    }

    private double derivativeOfHiddenActivationFunction(double a) {
        return NnUtil.derivativeOfTanhFunction(a);
    }

    private RnnParameterPack computeGrad(RnnParameterPack model) {
        ArrayList<RealVector> deltaOutput = new ArrayList<RealVector>();
        for(int i = 0; i<minibatchSize+1; i++) {
            deltaOutput.add(i, MatrixUtils.createRealVector(new double[vocab.getVocabSize()]));
        }

        for (int i = minibatchSize-1; i>0; i--){
            if (i == minibatchSize-1) {
                for (int k = 0; k < s.get(i).getDimension(); k++) {
                    deltaOutput.get(i).setEntry(k, derivativeOfOutputActivationFunction(s.get(i).getEntry(k)) * (x.get(i+1).getEntry(k)-y.get(i).getEntry(k)));
                }
            } else {
                for (int k = 0; k < s.get(i).getDimension(); k++) {
                    deltaOutput.get(i).setEntry(k, derivativeOfOutputActivationFunction(s.get(i).getEntry(k)) * (x.get(i+1).getEntry(k)-y.get(i).getEntry(k)));
                }
            }
        }

        return backwardPass(model, deltaOutput);
    }

    private RnnParameterPack backwardPass(RnnParameterPack model, ArrayList<RealVector> deltaOutput) {
        // no damping
        ArrayList<RealVector> damping = new ArrayList<RealVector>();
        for(int i = 0; i<minibatchSize+1; i++) {
            damping.add(i, MatrixUtils.createRealVector(new double[hiddenLayerSize]));
        }
        return backwardsPassWithDamping(model, deltaOutput, damping);
    }

    private RnnParameterPack backwardsPassWithDamping(RnnParameterPack model, ArrayList<RealVector> deltaOutput, ArrayList<RealVector> damping){
        ArrayList<RealVector> deltaHidden = new ArrayList<RealVector>();
        ArrayList<RealVector> deltaZ = new ArrayList<RealVector>();
        for(int i = 0; i<minibatchSize+1; i++) {
            deltaHidden.add(i, MatrixUtils.createRealVector(new double[hiddenLayerSize]));
            deltaZ.add(i, MatrixUtils.createRealVector(new double[hiddenLayerSize]));
        }

        for (int i = minibatchSize; i>0; i--){
            if (i == minibatchSize) {
                deltaHidden.set(i, currentNetworkParameters.getWyh().transpose().operate(deltaOutput.get(i)));
                RealVector tmp2 = t.get(i);
                for (int k = 0; k < tmp2.getDimension(); k++) {
                    tmp2.setEntry(k, derivativeOfHiddenActivationFunction(t.get(i).getEntry(k)));
                }
                deltaZ.set(i, deltaHidden.get(i).ebeMultiply(tmp2));
            } else {
                deltaHidden.set(i, currentNetworkParameters.getWyh().transpose().operate(deltaOutput.get(i)));
                deltaHidden.get(i).add(currentNetworkParameters.getWhh().transpose().operate(deltaZ.get(i+1)));
                RealVector tmp2 = t.get(i);
                for (int k = 0; k < tmp2.getDimension(); k++) {
                    tmp2.setEntry(k, derivativeOfHiddenActivationFunction(t.get(i).getEntry(k)));
                }
                deltaZ.set(i, deltaHidden.get(i).ebeMultiply(tmp2));
            }

            deltaZ.set(i, deltaZ.get(i).add(damping.get(i)));
        }

        RnnParameterPack gradPack = RnnParameterPack.createEmptyPack(vocab.getVocabSize(), hiddenLayerSize);

        for (int i = minibatchSize; i>0; i--){
            gradPack.setWyh(gradPack.getWyh().add(deltaOutput.get(i).outerProduct(h.get(i))));
            gradPack.setWhh(gradPack.getWhh().add(deltaZ.get(i).outerProduct(h.get(i-1))));
            gradPack.setWhx(gradPack.getWhx().add(deltaZ.get(i).outerProduct(x.get(i))));

            gradPack.setBh(gradPack.getBh().add(deltaZ.get(i)));
            gradPack.setBy(gradPack.getBy().add(deltaOutput.get(i)));
        }

        gradPack.setBinit(currentNetworkParameters.getWhh().transpose().operate(deltaZ.get(1)));

        return gradPack;
    }

    public void feedNextToken(int tokenId) {
        // TODO: Implement this
        testTokensSeen++;
    }
    public double[] predictNextToken() {
        // TODO: Implement this
        return null;
    }
}
