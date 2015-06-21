package ai.nxt.seqpred;

import org.apache.commons.cli.*;

/**
 * Created by Jeppe Hallgren on 21/06/15.
 */
public class SequencePredictorCli {
    public static String versionString = "v0.0.1";
    public static String yearOfRelease = "2015";

    public static void main(String args[]) {
        Options cliOptions = getCliOptions(args);

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

        System.out.println("Cli terminated");
    }

    public static Options getCliOptions(String args[]) {
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
