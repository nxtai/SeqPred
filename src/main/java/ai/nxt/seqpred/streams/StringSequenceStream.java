package ai.nxt.seqpred.streams;

import java.util.Scanner;

/**
 * Created by Jeppe Hallgren on 20/07/15.
 */
public class StringSequenceStream implements SequenceStream {
    private final String dataString;
    private Scanner scanner;
    private String sequenceId;
    private int curLineCount = 0;

    public StringSequenceStream(String data, String sequenceId) {
        dataString = data;
        scanner = new Scanner(dataString);
        this.sequenceId = sequenceId;
    }

    public String getNextLine() {
        if (curLineCount >= getLineCount()) return null;
        curLineCount++;
        String line = scanner.nextLine();
        return line;
    }

    public int getLineCount() {
        return dataString.split(System.getProperty("line.separator")).length;
    }

    public String getStreamId() {
        return sequenceId;
    }

    public String getNextWord() {
        return scanner.next("%s");
    }
}
