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

    public Rnn(Vocab vocab) {
        super(vocab);
    }

    public void train() {
        // TODO: Implement this
        int lineCount = FileUtil.countLines(getTrainingFileName());

        // initialise settings
        currentNetworkParameters = RnnParameterPack.createEmptyPack(vocab.getVocabSize(),hiddenLayerSize);
    }
    public void feedNextToken(int tokenId) {
        // TODO: Implement this
    }
    public double[] predictNextToken() {
        // TODO: Implement this
        return null;
    }
}
