package ai.nxt.seqpred;

import junit.framework.TestCase;
import org.junit.Test;

/**
 * Created by Jeppe Hallgren on 28/06/15.
 */
public class VocabTest extends TestCase {

    @Test
    public void testGetVocabSize() throws Exception {
        Vocab vocab = new Vocab("src/test/resources/data/vocab-test-set.txt");
        assertEquals(vocab.getVocabSize(),16);
    }

    @Test
    public void testGetTrainingFileSize() throws Exception {
        Vocab vocab = new Vocab("src/test/resources/data/vocab-test-set.txt");
        assertEquals(vocab.getTrainingFileSize(),19);
    }
}