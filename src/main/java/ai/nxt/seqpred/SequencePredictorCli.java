package ai.nxt.seqpred;

import ai.nxt.seqpred.Exceptions.FileTooShortException;
import ai.nxt.seqpred.Exceptions.InvalidPredictionException;
import ai.nxt.seqpred.rnn.Rnn;
import org.apache.commons.cli.*;

/**
 * Created by Jeppe Hallgren on 21/06/15.
 */
public class SequencePredictorCli {
    public static String versionString = "v0.0.1";
    public static String yearOfRelease = "2015";

    public static void main(String args[]) {
        Options cliOptions = getCliOptions();

        String trainingFileName = null;

        try {
            // parse arguments
            CommandLineParser parser = new DefaultParser();
            CommandLine line = parser.parse(cliOptions, args);

            // validate that training-file is present
            if(line.hasOption("training-file")) {
                trainingFileName = line.getOptionValue("training-file");
            }
        } catch (ParseException e) {
            System.out.println("Cli parsing failed:" + e.getMessage());
        }

        System.out.println("* Sequence Predictor " + versionString + " © NXT.AI " + yearOfRelease + " *");

        // print out parsed CLI arguments
        System.out.println("Training file: " + trainingFileName);

        // process training file
        Vocab vocab = new Vocab(trainingFileName);

        // print out statistics
        System.out.println("Processed " + vocab.getTrainingFileSize() + " words in training file");
        System.out.println("Vocab size is " + vocab.getVocabSize());

        // partition
        FilePartitioner filePartitioner;
        try {
            filePartitioner = new FilePartitioner(trainingFileName);
        } catch (FileTooShortException e) {
            System.err.println("The training file must be at least 5 lines long");
            return;
        }

        // initialise model
        Model model = new Rnn(vocab);
        model.setTrainingFile(trainingFileName);
        model.init();

        // train model
        model.train();

        // test model
        try {
            ModelEvaluator evaluator = new ModelEvaluator(filePartitioner.getTestFile(), vocab);
            evaluator.testModel(model);
        } catch (InvalidPredictionException e) {
            System.err.println("The model returned an invalid prediction: " + e.getMessage());
            return;
        }

        System.out.println("Cli terminated");
    }

    public static Options getCliOptions() {
        Options options = new Options();
        Option trainingFileOption = Option.builder("tf")
                .required(true)
                .hasArg(true)
                .longOpt("training-file")
                .build();
        options.addOption(trainingFileOption);
        return options;
    }
}
