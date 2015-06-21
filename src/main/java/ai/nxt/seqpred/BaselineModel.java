package ai.nxt.seqpred;

import java.util.ArrayList;

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
    public ArrayList<Integer> predictNextToken() {
        ArrayList<Integer> prediction = new ArrayList<Integer>(vocab.getVocabSize());
        for (Integer i : prediction) {
            i = 0;
        }
        // predict last word in vocab
        prediction.set(vocab.getVocabSize()-1,1);
        return prediction;
    }
}
