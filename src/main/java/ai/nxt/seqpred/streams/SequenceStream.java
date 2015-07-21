package ai.nxt.seqpred.streams;

import java.io.IOException;

/**
 * Created by Jeppe Hallgren on 20/07/15.
 */
public interface SequenceStream {
    public String getNextLine()  throws IOException;
    public int getLineCount();
    public String getStreamId();
}
