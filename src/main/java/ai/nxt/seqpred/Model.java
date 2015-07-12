package ai.nxt.seqpred;

/**
 * Created by Jeppe Hallgren on 21/06/15.
 */
public abstract class Model {
    private String trainingFileName;
    protected Vocab vocab;

    public Model(Vocab vocab) {
        this.vocab = vocab;
    }

    public void init() {
        // models can override this, but don't have to
    }

    public void prepareForTesting() {
        // models can override this, but don't have to
    }

    public abstract void train();

    public void setTrainingFile(String trainingFileName) {
        this.trainingFileName = trainingFileName;
    }

    protected String getTrainingFileName() {
        return trainingFileName;
    }

    // these method are used for testing model performance
    public abstract void feedNextToken(int tokenId);
    public abstract double[] predictNextToken();
}
