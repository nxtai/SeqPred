package ai.nxt.seqpred;

import org.apache.commons.cli.*;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;

/**
 * Created by Jeppe Hallgren on 21/06/15.
 */
public class PredictionApi {
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

        System.out.println("* Prediction API " + versionString + " © NXT.AI " + yearOfRelease + " *");

        Server server = new Server(3000);
        ServletContextHandler context = new ServletContextHandler(server, "/", ServletContextHandler.SESSIONS);
        ServletHolder jerseyServlet = context.addServlet(org.glassfish.jersey.servlet.ServletContainer.class, "/*");
        jerseyServlet.setInitOrder(1);
        jerseyServlet.setInitParameter("jersey.config.server.provider.packages","ai.nxt.seqpred.api");

        try {
            server.start();
            server.join();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            server.destroy();
        }

        System.out.println("Api terminated");
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
