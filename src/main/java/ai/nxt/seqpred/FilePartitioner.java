package ai.nxt.seqpred;

import ai.nxt.seqpred.Exceptions.FileTooShortException;
import ai.nxt.seqpred.streams.SequenceStream;

import java.io.IOException;
import java.io.PrintWriter;

/**
 * Created by Jeppe Hallgren on 21/06/15.
 */
public class FilePartitioner {

    private String trainingFile;
    private String validationFile;
    private String testFile;

    public FilePartitioner(SequenceStream sequenceStream) throws FileTooShortException {
        // count lines in file
        int lineCount = sequenceStream.getLineCount();

        if (lineCount < 5) {
            throw new FileTooShortException();
        }

        // define file names
        trainingFile = "data/processed/" + sequenceStream.getStreamId() + ".train";
        validationFile = "data/processed/" + sequenceStream.getStreamId() + ".valid";
        testFile = "data/processed/" + sequenceStream.getStreamId() + ".test";

        try {
            PrintWriter trainingFileWriter = new PrintWriter(trainingFile);
            PrintWriter validationFileWriter = new PrintWriter(validationFile);
            PrintWriter testFileWriter = new PrintWriter(testFile);

            int currentLine = 0;

            String line;
            while ((line = sequenceStream.getNextLine()) != null) {
                if (lineCount * 0.6 > currentLine) {
                    // add to training file
                    trainingFileWriter.println(line);
                } else if (lineCount * 0.8 > currentLine){
                    // add to validation file
                    validationFileWriter.println(line);
                } else {
                    // add to test file
                    testFileWriter.println(line);
                }
                currentLine++;
            }
            trainingFileWriter.close();
            validationFileWriter.close();
            testFileWriter.close();
            System.out.println("Wrote data to files: ");
            System.out.println(trainingFile);
            System.out.println(validationFile);
            System.out.println(testFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public String getTrainingFile() {
        return trainingFile;
    }

    public String getValidationFile() {
        return validationFile;
    }

    public String getTestFile() {
        return testFile;
    }
}
