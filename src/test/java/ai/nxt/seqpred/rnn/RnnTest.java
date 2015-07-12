package ai.nxt.seqpred.rnn;

import ai.nxt.seqpred.Vocab;
import junit.framework.TestCase;
import org.junit.Test;

/**
 * Created by Jeppe Hallgren on 28/06/15.
 */
public class RnnTest extends TestCase {

    @Test
    public void testTrain() throws Exception {
        int hiddenSize = 2;
        String testFileName = "src/test/resources/data/rnn-test-set.txt";
        Vocab vocab = new Vocab(testFileName);

        Rnn model = new Rnn(vocab);
        model.setCurrentNetworkParameters(RnnParameterPack.createRandomPack(vocab.getVocabSize(), hiddenSize));

        model.setTrainingFile(testFileName);

        double loss1 = model.forwardPass(model.getCurrentNetworkParameters(), 0, true);

        for (int i = 1; i<10; i++) {
            int minibatch = i % (vocab.getTrainingFileSize() / model.getMinibatchSize());
            model.forwardPass(model.getCurrentNetworkParameters(), minibatch, true);
            RnnParameterPack grad = model.computeGrad(model.getCurrentNetworkParameters());
            model.getCurrentNetworkParameters().weightedAddition(grad, 1);
        }
        double loss2 = model.forwardPass(model.getCurrentNetworkParameters(), 0, true);
        assertTrue(loss1 > loss2);

    }
}