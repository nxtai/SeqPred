package ai.nxt.seqpred;

import ai.nxt.seqpred.util.FileUtil;
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
    public static final int START_TOKEN = 0;

    public Vocab (String trainingFileName) {
        this.trainingFileName = trainingFileName;
        processTrainingFile();
    }

    private void processTrainingFile() {
        totalWordCount = 0;
        vocabBiMap = HashBiMap.create();
        vocabBiMap.put("<start>", START_TOKEN);
        vocabBiMap.put("</s>",1); //end of sentence token

        try {
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(new FileInputStream(trainingFileName)));
            String nextWord = FileUtil.readNextWord(reader);
            while (! (nextWord == null || nextWord.equals(""))) {
                totalWordCount++;
                if (! vocabBiMap.containsKey(nextWord)) {
                    vocabBiMap.put(nextWord, vocabBiMap.size());
                }
                nextWord = FileUtil.readNextWord(reader);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public int getVocabSize() {
        return vocabBiMap.size();
    }

    public int getTrainingFileSize() {
        return totalWordCount;
    }

    public int getWordIndex(String word) {
        return vocabBiMap.get(word);
    }

    public String getWordString(int wordIndex) {
        return vocabBiMap.inverse().get(wordIndex);
    }
}
