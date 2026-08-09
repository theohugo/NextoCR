// Modified by NextoCR contributors; see NOTICE for attribution.
package org.crforge.core.engine;

import org.crforge.core.player.Team;

/** Canonical terminal outcome of a game, independent of how the match ended. */
public enum GameOutcome {
  ONGOING(0, null),
  BLUE_WIN(1, Team.BLUE),
  RED_WIN(2, Team.RED),
  DRAW(3, null);

  private final int binaryCode;
  private final Team winner;

  GameOutcome(int binaryCode, Team winner) {
    this.binaryCode = binaryCode;
    this.winner = winner;
  }

  /** Stable integer used by the binary bridge trailer. */
  public int binaryCode() {
    return binaryCode;
  }

  /** Winning team, or {@code null} for an ongoing game or draw. */
  public Team winner() {
    return winner;
  }

  public boolean isTerminal() {
    return this != ONGOING;
  }

  public static GameOutcome fromWinner(Team winner) {
    if (winner == Team.BLUE) {
      return BLUE_WIN;
    }
    if (winner == Team.RED) {
      return RED_WIN;
    }
    return DRAW;
  }
}
