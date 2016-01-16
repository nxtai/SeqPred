package ai.nxt.seqpred.rnn;

import ai.nxt.seqpred.JsonModel;
import ai.nxt.seqpred.Vocab;

/**
 * Created by Jeppe Hallgren on 25/07/15.
 */
public class JsonRnn extends JsonModel {
    private Rnn rnn;
    private JsonRnnParameterPack jsonRnnParameterPack;
    private Vocab vocab;
    private String modelId;

    public JsonRnn(){}

    public JsonRnn(JsonRnnParameterPack jsonRnnParameterPack, Vocab vocab) {
        this.jsonRnnParameterPack = jsonRnnParameterPack;
        this.vocab = vocab;
    }

    public String getModelId() {
        return "0";
    }

    public Vocab getVocab() {
        return vocab;
    }

    public JsonRnnParameterPack getJsonRnnParameterPack() {return jsonRnnParameterPack; }
}

