package ai.nxt.seqpred.rnn;

import ai.nxt.seqpred.JsonModel;
import ai.nxt.seqpred.Vocab;

/**
 * Created by Jeppe Hallgren on 25/07/15.
 */
public class JsonRnn extends JsonModel {
    private Rnn rnn;

    public JsonRnn(Rnn rnn) {
        this.rnn = rnn;
    }

    public String getModelId() {
        return "0";
    }

    public Vocab getVocab() {
        return rnn.getVocab();
    }

    public JsonRnnParameterPack getJsonRnnParameterPack() {return rnn.getCurrentNetworkParameters().getJson(); }
}

