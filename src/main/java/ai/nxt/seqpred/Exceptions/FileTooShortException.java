package ai.nxt.seqpred.Exceptions;

/**
 * Created by jh on 21/06/15.
 */
public class FileTooShortException extends Exception {
    private int lineCount;

    public FileTooShortException(int count){
        super();
        lineCount = count;
    }

    public int getLineCount() {
        return lineCount;
    }
}
