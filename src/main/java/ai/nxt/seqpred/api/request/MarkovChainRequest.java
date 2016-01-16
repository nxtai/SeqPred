package ai.nxt.seqpred.api.request;

import ai.nxt.seqpred.JsonModel;

/**
 * Created by Jeppe Hallgren on 17/07/15.
 */
public class MarkovChainRequest {
    private String modelId;
    private JsonModel jsonModel;
    private int length;

    public MarkovChainRequest() {

    }

    public String getModelId() {
        return modelId;
    }

    public JsonModel getJsonModel() {
        return jsonModel;
    }

    public int getLength() {
        return length;
    }
}
