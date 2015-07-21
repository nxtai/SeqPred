package ai.nxt.seqpred.api.request;

/**
 * Created by Jeppe Hallgren on 17/07/15.
 */
public class TrainModelRequest {
    private String modelId;
    private String trainingData;

    public TrainModelRequest() {

    }

    public String getModelId() {
        return modelId;
    }

    public String getTrainingData() {
        return trainingData;
    }
}
