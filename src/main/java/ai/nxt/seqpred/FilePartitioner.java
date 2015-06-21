package ai.nxt.seqpred;

import ai.nxt.seqpred.Exceptions.FileTooShortException;
import ai.nxt.seqpred.util.FileUtil;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * Created by Jeppe Hallgren on 21/06/15.
 */
public class FilePartitioner {

    private String trainingFile;
    private String validationFile;
    private String testFile;

    public FilePartitioner(String trainingFileName) throws FileTooShortException {
        // count lines in file
        int lineCount = FileUtil.countLines(trainingFileName);

        if (lineCount < 5) {
            throw new FileTooShortException();
        }

        // find file name
        int lastSlashPos = trainingFileName.lastIndexOf('/');
        String fileName;
        if (lastSlashPos != -1) {
            fileName = trainingFileName.substring(trainingFileName.lastIndexOf('/')+1, trainingFileName.length());
        } else {
            fileName = trainingFileName;
        }

        // define file names
        trainingFile = "data/processed/" + fileName + ".train";
        validationFile = "data/processed/" + fileName + ".valid";
        testFile = "data/processed/" + fileName + ".test";

        try {
            PrintWriter trainingFileWriter = new PrintWriter(trainingFile);
            PrintWriter validationFileWriter = new PrintWriter(validationFile);
            PrintWriter testFileWriter = new PrintWriter(testFile);

            int currentLine = 0;


            BufferedReader br = new BufferedReader(new FileReader(trainingFileName));
            String line;
            while ((line = br.readLine()) != null) {
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
