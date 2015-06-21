package ai.nxt.seqpred;

import java.util.ArrayList;

/**
 * Created by Jeppe Hallgren on 21/06/15.
 */
public abstract class Model {
    private String trainingFileName;

    public void init() {
        // models can override this, but don't have to
    }

    public abstract void train();

    public void setTrainingFile(String trainingFileName) {
        this.trainingFileName = trainingFileName;
    }

    // these method are used for testing model performance
    public abstract void feedNextToken(int tokenId);
    public abstract ArrayList<Integer> predictNextToken();
}
