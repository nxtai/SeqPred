package ai.nxt.seqpred.rnn;

import ai.nxt.seqpred.Model;
import ai.nxt.seqpred.Vocab;
import ai.nxt.seqpred.util.FileUtil;

/**
 * Created by Jeppe Hallgren on 21/06/15.
 */
public class Rnn extends Model {
    RnnParameterPack currentNetworkParameters;
    private int hiddenLayerSize = 15;
    private int minibatchSize = 5;
    private double lastPerplexity;

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
            double loss = forwardPass(currentNetworkParameters, currentMinibatch);

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

    private double forwardPass(RnnParameterPack networkParameters, int currentMinibatch) {
        // TODO: Implement this
        return 0;
    }

    private RnnParameterPack computeGrad() {
        // TODO: Implement this
        return RnnParameterPack.createRandomPack(vocab.getVocabSize(),hiddenLayerSize);
    }

    public void feedNextToken(int tokenId) {
        // TODO: Implement this
    }
    public double[] predictNextToken() {
        // TODO: Implement this
        return null;
    }
}
