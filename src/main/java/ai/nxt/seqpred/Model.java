package ai.nxt.seqpred;

/**
 * Created by Jeppe Hallgren on 21/06/15.
 */
public abstract class Model {
    private String trainingFileName;

    public void init() {
        // models can override this, but don't have to
    }

    public abstract void train();
    public abstract void test();

    public void setTrainingFile(String trainingFileName) {
        this.trainingFileName = trainingFileName;
    }
}
