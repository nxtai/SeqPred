package ai.nxt.seqpred.rnn;

import ai.nxt.seqpred.Model;
import ai.nxt.seqpred.Vocab;
import ai.nxt.seqpred.util.FileUtil;
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
    private int minibatchSize = 5;
    private double lastPerplexity;

    private ArrayList<RealVector> x;
    private ArrayList<RealVector> t;
    private ArrayList<RealVector> h;
    private ArrayList<RealVector> s;
    private ArrayList<RealVector> y;

    public Rnn(Vocab vocab) {
        super(vocab);
    }

    public void train() {
        // TODO: Implement this
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

            RnnParameterPack gradient = computeGrad();
            currentNetworkParameters.weightedAddition(gradient,1);

            // switch to next minibatch
            currentMinibatch = (currentMinibatch + 1) % minibatchCount;

            // increment count
            gradRound++;
        } while (! hasConverged());
    }

    public boolean hasConverged() {
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

    private double hiddenActivationFunction(double a) {
        // TODO: Implement this
        return a;
    }

    private double outputActivationFunction(double a) {
        // TODO: Implement this
        return a;
    }

    private RnnParameterPack computeGrad() {
        // TODO: Implement this
        return RnnParameterPack.createRandomPack(vocab.getVocabSize(), hiddenLayerSize);
    }

    public void feedNextToken(int tokenId) {
        // TODO: Implement this
    }
    public double[] predictNextToken() {
        // TODO: Implement this
        return null;
    }
}
