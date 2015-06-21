package ai.nxt.seqpred;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 * Created by Jeppe Hallgren on 21/06/15.
 */
public class Vocab {
    private String trainingFileName;
    private BiMap<String, Integer> vocabBiMap;
    private int totalWordCount;

    public Vocab (String trainingFileName) {
        this.trainingFileName = trainingFileName;
        processTrainingFile();
    }

    private void processTrainingFile() {
        totalWordCount = 0;
        vocabBiMap = HashBiMap.create();
        vocabBiMap.put("</s>",0); //end of sentence token

        try {
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(new FileInputStream(trainingFileName)));
            String nextWord = readNextWord(reader);
            while (! (nextWord == null || nextWord.equals(""))) {
                totalWordCount++;
                if (! vocabBiMap.containsKey(nextWord)) {
                    vocabBiMap.put(nextWord, vocabBiMap.size());
                }
                nextWord = readNextWord(reader);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private String readNextWord(BufferedReader reader) {
        String currentWord = "";
        try {
            int currentChar;
            while ((currentChar = reader.read()) != -1) {
                if ((currentChar == 13) || (currentChar == ' ') || (currentChar == '\t') || (currentChar == '\n')) { // if white space
                    if (currentWord.length() > 0) {
                        break;
                    } else {
                        if (currentChar == '\n') {
                            return "</s>";
                        }
                    }
                } else {
                    currentWord += (char) currentChar;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return currentWord;
    }

    public int getVocabSize() {
        return vocabBiMap.size();
    }

    public int getTrainingFileSize() {
        return totalWordCount;
    }
}
