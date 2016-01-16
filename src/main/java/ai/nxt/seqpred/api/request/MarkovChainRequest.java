package ai.nxt.seqpred.api.request;

import ai.nxt.seqpred.rnn.JsonRnn;

/**
 * Created by Jeppe Hallgren on 17/07/15.
 */
public class MarkovChainRequest {
    private String modelId;
    private JsonRnn jsonModel;
    private int length;

    public MarkovChainRequest() {

    }

    public String getModelId() {
        return modelId;
    }

    public JsonRnn getJsonModel() {
        return jsonModel;
    }

    public int getLength() {
        return length;
    }
}
