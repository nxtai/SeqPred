package ai.nxt.seqpred.api.request;

import ai.nxt.seqpred.rnn.JsonRnn;

/**
 * Created by Jeppe Hallgren on 17/07/15.
 */
public class PredictionRequest {
    private String modelId;
    private JsonRnn jsonModel;
    private String sequence;

    public PredictionRequest() {

    }

    public String getModelId() {
        return modelId;
    }

    public JsonRnn getJsonModel() {
        return jsonModel;
    }

    public String getSequence() {
        return sequence;
    }
}
