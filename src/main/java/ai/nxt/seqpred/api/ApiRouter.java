package ai.nxt.seqpred.api;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
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
}
