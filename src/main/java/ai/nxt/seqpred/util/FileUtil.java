package ai.nxt.seqpred.util;

import java.io.BufferedReader;
import java.io.IOException;

/**
 * Created by jh on 21/06/15.
 */
public class FileUtil {
    public static String readNextWord(BufferedReader reader) {
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
}
