package ai.nxt.seqpred;

/**
 * Created by Jeppe Hallgren on 21/06/15.
 * This is a very simple model that just predicts the last word in the vocab every time
 */
public class BaselineModel extends Model {

    public BaselineModel(Vocab vocab) {
        super(vocab);
    }

    public void train() {
        // no training needed
    }
    public void feedNextToken(int tokenId) {
        // don't need to keep track of token history
    }
    public double[] predictNextToken() {
        double[] prediction = new double[vocab.getVocabSize()];
        for (int i = 0; i<prediction.length; i++) {
            prediction[i] = 0.0;
        }
        // predict last word in vocab
        prediction[vocab.getVocabSize()-1] = 1.0;
        return prediction;
    }
}
