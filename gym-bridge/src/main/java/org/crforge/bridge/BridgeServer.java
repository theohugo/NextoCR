// Modified by NextoCR contributors; see NOTICE for attribution.
package org.crforge.bridge;

import org.crforge.bridge.protocol.ZmqTransport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Entry point for the gym-bridge server. Binds a ZMQ PAIR socket and runs a session loop.
 *
 * <p>Usage: java -jar gym-bridge.jar [port] [--allow-remote] Default: 127.0.0.1:9876
 */
public class BridgeServer {

  private static final Logger log = LoggerFactory.getLogger(BridgeServer.class);
  private static final int DEFAULT_PORT = 9876;
  private static final String LOOPBACK_HOST = "127.0.0.1";
  private static final String REMOTE_HOST = "*";

  public static void main(String[] args) {
    int port = DEFAULT_PORT;
    if (args.length > 0) {
      try {
        port = Integer.parseInt(args[0]);
      } catch (NumberFormatException e) {
        log.error("Invalid port number: {}", args[0]);
        System.exit(1);
      }
    }

    if (port < 1 || port > 65535) {
      log.error("Port must be between 1 and 65535: {}", port);
      System.exit(1);
      return;
    }

    boolean allowRemote = false;
    if (args.length > 1) {
      if ("--allow-remote".equals(args[1]) && args.length == 2) {
        allowRemote = true;
      } else {
        log.error("Usage: gym-bridge [port] [--allow-remote]");
        System.exit(1);
        return;
      }
    }

    String endpoint = endpoint(port, allowRemote);
    log.info("Starting NextoCR Bridge Server on {}", endpoint);
    if (allowRemote) {
      log.warn("Remote bridge access is enabled without protocol authentication");
    }

    // Force CardRegistry static initialization before accepting connections
    log.info(
        "Loading card registry ({} cards)...",
        org.crforge.data.card.CardRegistry.getAllIds().size());

    try (ZmqTransport transport = new ZmqTransport(endpoint)) {
      transport.bind();

      // Run session loop -- reconnects automatically after client disconnects
      while (true) {
        log.info("Waiting for client connection on port {}...", port);
        BridgeSession session = new BridgeSession(transport);
        session.run();
        log.info("Session ended, ready for next connection");
      }
    } catch (Exception e) {
      log.error("Bridge server error", e);
      System.exit(1);
    }
  }

  static String endpoint(int port, boolean allowRemote) {
    return "tcp://" + (allowRemote ? REMOTE_HOST : LOOPBACK_HOST) + ":" + port;
  }
}
