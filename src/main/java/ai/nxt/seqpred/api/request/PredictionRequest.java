package ai.nxt.seqpred.api.request;

import ai.nxt.seqpred.JsonModel;

/**
 * Created by Jeppe Hallgren on 17/07/15.
 */
public class PredictionRequest {
    private String modelId;
    private JsonModel jsonModel;
    private String sequence;

    public PredictionRequest() {

    }

    public String getModelId() {
        return modelId;
    }

    public JsonModel getJsonModel() {
        return jsonModel;
    }

    public String getSequence() {
        return sequence;
    }
}
