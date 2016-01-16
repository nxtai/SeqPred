package ai.nxt.seqpred.Exceptions;

/**
 * Created by jh on 16/01/16.
 */
public class TokenNotInVocabException extends Exception {
    String token;
    public TokenNotInVocabException(String token) {
        super();
        this.token = token;
    }

    public String getToken() {
        return token;
    }
}
