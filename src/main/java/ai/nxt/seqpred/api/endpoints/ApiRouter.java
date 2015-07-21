package ai.nxt.seqpred.api.endpoints;

import ai.nxt.seqpred.Exceptions.FileTooShortException;
import ai.nxt.seqpred.Exceptions.InvalidPredictionException;
import ai.nxt.seqpred.FilePartitioner;
import ai.nxt.seqpred.Model;
import ai.nxt.seqpred.ModelEvaluator;
import ai.nxt.seqpred.Vocab;
import ai.nxt.seqpred.api.request.TrainModelRequest;
import ai.nxt.seqpred.rnn.Rnn;
import ai.nxt.seqpred.streams.StringSequenceStream;

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
    public Double trainModel(TrainModelRequest req) {
        System.out.println("Post request: " + req.getModelId());
        // partition
        FilePartitioner filePartitioner;
        try {
            filePartitioner = new FilePartitioner(new StringSequenceStream(req.getTrainingData(), "api01"));
        } catch (FileTooShortException e) {
            System.err.println("The training file must be at least 5 lines long");
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
        return perplexity;
    }
}
