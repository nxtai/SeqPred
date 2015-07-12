package ai.nxt.seqpred;

import ai.nxt.seqpred.Exceptions.InvalidPredictionException;
import ai.nxt.seqpred.util.FileUtil;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.DecimalFormat;

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
            model.prepareForTesting();
            int wordsRead = 0;
            double logSum = 0.0;
            int wordCorrectlyGuessed = 0;
            while (!(nextWord == null || nextWord.equals(""))) {
                wordsRead++;
                double[] prediction = model.predictNextToken();
                assertPrediction(prediction);
                logSum += Math.log10(prediction[vocab.getWordIndex(nextWord)]);
                if (getMostLikelyWord(prediction) == vocab.getWordIndex(nextWord))
                    wordCorrectlyGuessed++;

                // feed the word we just asked for a prediction about
                model.feedNextToken(vocab.getWordIndex(nextWord));

                // read next word from test file
                nextWord = FileUtil.readNextWord(reader);
            }
            double perplexity = Math.pow(10.0,-1*logSum/wordsRead);
            System.out.println("The test used " + wordsRead + " words.");
            System.out.println("Model had perplexity: " + perplexity);
            DecimalFormat df = new DecimalFormat("#.##");
            System.out.println("Model had top-1 accuracy: " + df.format((double) wordCorrectlyGuessed / (double) wordsRead * 100) + "%");
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Tested model");
    }

    public void assertPrediction(double[] prediction) throws InvalidPredictionException {
        // prediction cannot be null
        if (prediction == null) {
            throw new InvalidPredictionException("No array was given (prediction == null).");
        }

        // prediction must have same length as vocab size
        if (vocab.getVocabSize() != prediction.length) {
            throw new InvalidPredictionException("Prediction array has wrong length. Expected " +
                vocab.getVocabSize() + ", got " + prediction.length + ".");
        }

        // prediction must sum to 1
        double sum = 0.0;
        for (int i = 0; i<prediction.length; i++) {
            sum += prediction[i];
        }
        if (sum > 1.0 || sum < 0.99) {
            throw new InvalidPredictionException("Prediction didn't sum to 1, sum was " + sum + ".");
        }
    }

    public void printPrediction(double[] prediction, int actualWord) {
        System.out.print(vocab.getWordString(actualWord).toUpperCase() + ": ");
        DecimalFormat df = new DecimalFormat("#.####");
        for (int i = 0; i<prediction.length; i++) {
            System.out.print(vocab.getWordString(i) + " " + df.format(prediction[i]) + ", ");
        }
        System.out.println();
    }

    private static int getMostLikelyWord(double[] prediction) {
        double highestPredictionSeen = -1;
        int bestIndex = -1;
        for (int i = 0; i<prediction.length; i++) {
            if (prediction[i] > highestPredictionSeen) {
                highestPredictionSeen = prediction[i];
                bestIndex = i;
            }
        }
        return bestIndex;
    }
}
