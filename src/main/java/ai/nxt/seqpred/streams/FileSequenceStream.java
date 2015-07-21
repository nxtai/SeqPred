package ai.nxt.seqpred.streams;

import ai.nxt.seqpred.util.FileUtil;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 * Created by Jeppe Hallgren on 20/07/15.
 */
public class FileSequenceStream implements SequenceStream {
    private final String fileName;
    private BufferedReader br;

    public FileSequenceStream(String fileName) throws FileNotFoundException{
        this.fileName = fileName;
        br = new BufferedReader(new FileReader(fileName));
    }

    public String getNextLine() throws IOException {
        String line = br.readLine();
        return line;
    }

    public int getLineCount() {
        return FileUtil.countLines(fileName);
    }

    public String getStreamId() {
        // find file name
        int lastSlashPos = fileName.lastIndexOf('/');
        String trainingFileName;
        if (lastSlashPos != -1) {
            trainingFileName = fileName.substring(fileName.lastIndexOf('/')+1, fileName.length());
        } else {
            trainingFileName = fileName;
        }
        return trainingFileName;
    }
}
