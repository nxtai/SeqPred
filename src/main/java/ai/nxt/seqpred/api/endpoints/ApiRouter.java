package ai.nxt.seqpred.api.endpoints;

import ai.nxt.seqpred.Exceptions.FileTooShortException;
import ai.nxt.seqpred.Exceptions.InvalidPredictionException;
import ai.nxt.seqpred.Exceptions.TokenNotInVocabException;
import ai.nxt.seqpred.FilePartitioner;
import ai.nxt.seqpred.Model;
import ai.nxt.seqpred.ModelEvaluator;
import ai.nxt.seqpred.Vocab;
import ai.nxt.seqpred.api.request.MarkovChainRequest;
import ai.nxt.seqpred.api.request.PredictionRequest;
import ai.nxt.seqpred.api.request.TrainModelRequest;
import ai.nxt.seqpred.rnn.JsonRnn;
import ai.nxt.seqpred.rnn.Rnn;
import ai.nxt.seqpred.streams.StringSequenceStream;
import ai.nxt.seqpred.util.ProbabilityUtil;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;

/**
 * Created by Jeppe Hallgren on 12/07/15.
 */
@Path("/api/v1")
public class ApiRouter {
    @GET
    @Path("/")
    @Produces(MediaType.APPLICATION_JSON)
    public String getIndex() {
        return "Welcome to the prediction API";
    }

    @POST
    @Path("/rnn/trainModel")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public JsonRnn trainModel(TrainModelRequest req) {
        if (req == null) {
            System.err.println("Failed to decode TrainModelRequest.");
            return null;
        }
        System.out.println("Post request: " + req.getModelId());
        // partition
        FilePartitioner filePartitioner;
        try {
            filePartitioner = new FilePartitioner(new StringSequenceStream(req.getTrainingData(), "api01"));
        } catch (FileTooShortException e) {
            System.err.println("The training file must be at least 5 lines long, was: " + e.getLineCount());
            return null;
        }

        // process training file
        Vocab vocab = new Vocab(filePartitioner.getTrainingFile());

        // print out statistics
        System.out.println("Processed " + vocab.getTrainingFileSize() + " words in training file");
        System.out.println("Vocab size is " + vocab.getVocabSize());

        // initialise model
        Model model = new Rnn(vocab);
        model.setTrainingFile(filePartitioner.getTrainingFile());
        model.init();

        // train model
        model.train();

        // test model
        double perplexity = -1;
        try {
            ModelEvaluator evaluator = new ModelEvaluator(filePartitioner.getTestFile(), vocab);
            perplexity = evaluator.testModel(model);
        } catch (InvalidPredictionException e) {
            System.err.println("The model returned an invalid prediction: " + e.getMessage());
            return null;
        }
        return new JsonRnn( ((Rnn) model).getCurrentNetworkParameters().getJson(), model.getVocab() );
    }

    @POST
    @Path("/rnn/predict")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public double[] predict(PredictionRequest req) {
        if (req == null) {
            System.err.println("Failed to decode PredictRequest.");
            return null;
        }
        Rnn rnn = new Rnn((JsonRnn) req.getJsonModel());
        if (req.getSequence() == null) {
            System.err.println("Err: Failed to decode input sequence in PredictRequest.");
            return null;
        }
        System.out.println("Running model");
        StringSequenceStream sequenceStream = new StringSequenceStream(req.getSequence(), null);
        rnn.prepareForTesting();
        String nextToken = sequenceStream.getNextWord();
        while (nextToken != null) {
            try {
                rnn.feedNextToken(rnn.getVocab().getWordIndex(nextToken));
            } catch (TokenNotInVocabException e) {
                System.err.println("Word in not in vocab: " + e.getToken());
                return null;
            }
            nextToken = sequenceStream.getNextWord();
        }
        double[] prediction = rnn.predictNextToken();
        return prediction;
    }


    @POST
    @Path("/rnn/markovChain")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public String markovChain(MarkovChainRequest req) {
        if (req.getLength() < 1) return "";
        Rnn rnn = new Rnn((JsonRnn) req.getJsonModel());
        rnn.prepareForTesting();
        StringBuilder chain = new StringBuilder();
        int length = 0;

        while (length <= req.getLength()) {
            double[] prediction = rnn.predictNextToken();
            int nextToken = ProbabilityUtil.getWeightedIndex(prediction);
            chain.append(rnn.getVocab().getWordString(nextToken) + " ");
            length++;
            rnn.feedNextToken(nextToken);
        }

        return chain.toString();
    }
}
