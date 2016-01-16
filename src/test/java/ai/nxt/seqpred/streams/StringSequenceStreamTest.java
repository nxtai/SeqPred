package ai.nxt.seqpred.streams;

import junit.framework.TestCase;
import org.junit.Test;

/**
 * Created by jh on 16/01/16.
 */
public class StringSequenceStreamTest extends TestCase {

    @Test
    public void testGetNextWord() throws Exception {
        StringSequenceStream stream = new StringSequenceStream("a b c",null);
        assertTrue("a".equals(stream.getNextWord()));
        assertTrue("b".equals(stream.getNextWord()));
        assertTrue("c".equals(stream.getNextWord()));
    }
}