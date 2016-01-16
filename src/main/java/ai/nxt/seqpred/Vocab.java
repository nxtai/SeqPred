package ai.nxt.seqpred;

import ai.nxt.seqpred.Exceptions.TokenNotInVocabException;
import ai.nxt.seqpred.util.FileUtil;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;

/**
 * Created by Jeppe Hallgren on 21/06/15.
 */
public class Vocab {
    private String trainingFileName;
    private HashMap<String, Integer> vocabMap;
    private int trainingFileSize;
    private int vocabSize;
    public static final int START_TOKEN = 0;

    public Vocab(){
        // empty constructor for json decode
    }

    public Vocab (String trainingFileName) {
        this.trainingFileName = trainingFileName;
        processTrainingFile();
    }

    private void processTrainingFile() {
        trainingFileSize = 0;
        vocabMap = new HashMap<String, Integer>();
        vocabMap.put("<start>", START_TOKEN);
        vocabMap.put("</s>",1); //end of sentence token

        try {
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(new FileInputStream(trainingFileName)));
            String nextWord = FileUtil.readNextWord(reader);
            while (! (nextWord == null || nextWord.equals(""))) {
                trainingFileSize++;
                if (! vocabMap.containsKey(nextWord)) {
                    vocabMap.put(nextWord, vocabMap.size());
                }
                nextWord = FileUtil.readNextWord(reader);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        vocabSize = vocabMap.size();
    }

    public int getVocabSize() {
        return vocabSize;
    }

    public int getTrainingFileSize() {
        return trainingFileSize;
    }

    public int getWordIndex(String word) throws TokenNotInVocabException{
        if (! vocabMap.containsKey(word))
            throw new TokenNotInVocabException(word);
        return vocabMap.get(word);
    }

    public HashMap<String, Integer> getVocabMap(){
        return vocabMap;
    }

    public String getWordString(int wordIndex) {
        BiMap<String,Integer> biMap = HashBiMap.create(vocabMap);
        return biMap.inverse().get(wordIndex);
    }
}
