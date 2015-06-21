package ai.nxt.seqpred;

import ai.nxt.seqpred.Exceptions.InvalidPredictionException;
import ai.nxt.seqpred.util.FileUtil;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 * Created by Jeppe Hallgren on 21/06/15.
 */
public class ModelEvaluator {
    private String testFileName;
    private Vocab vocab;

    public ModelEvaluator(String testFileName, Vocab vocab) {
        this.testFileName = testFileName;
        this.vocab = vocab;
    }

    public void testModel(Model model) throws InvalidPredictionException {
        System.out.println("Testing model");
        try {
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(new FileInputStream(testFileName)));
            String nextWord = FileUtil.readNextWord(reader);
            int wordsRead = 0;
            double logSum = 0.0;
            while (!(nextWord == null || nextWord.equals(""))) {
                wordsRead++;
                double[] prediction = model.predictNextToken();
                assertPrediction(prediction);
                logSum += Math.log10(prediction[vocab.getWordIndex(nextWord)]);
                nextWord = FileUtil.readNextWord(reader);
            }
            double perplexity = Math.pow(10.0,-1*logSum/wordsRead);
            System.out.println("Model had perplexity: " + perplexity);
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Tested model");
    }

    public void assertPrediction(double[] prediction) throws InvalidPredictionException {
        // prediction must have same length as vocab size
        if (vocab.getVocabSize() != prediction.length) {
            throw new InvalidPredictionException();
        }

        // prediction must sum to 1
        double sum = 0.0;
        for (int i = 0; i<prediction.length; i++) {
            sum += prediction[i];
        }
        if (sum > 1.0 || sum < 0.99) {
            throw new InvalidPredictionException();
        }
    }
}
