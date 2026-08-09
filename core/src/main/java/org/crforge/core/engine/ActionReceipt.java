package org.crforge.core.engine;

import java.util.Objects;
import org.crforge.core.player.Player;
import org.crforge.core.player.dto.PlayerActionDTO;

/**
 * Tracks whether a queued action was actually accepted by the deployment system.
 *
 * <p>Placement validation can reject a receipt immediately. Otherwise it remains {@link
 * Status#PENDING} until the next deployment update performs the authoritative resource check and,
 * on success, spends elixir and cycles the hand.
 */
public final class ActionReceipt {

  public enum Status {
    PENDING,
    ACCEPTED,
    REJECTED
  }

  private final Player player;
  private final PlayerActionDTO action;
  private volatile Status status;

  ActionReceipt(Player player, PlayerActionDTO action) {
    this.player = Objects.requireNonNull(player, "player");
    this.action = Objects.requireNonNull(action, "action");
    this.status = Status.PENDING;
  }

  static ActionReceipt rejected(Player player, PlayerActionDTO action) {
    ActionReceipt receipt = new ActionReceipt(player, action);
    receipt.reject();
    return receipt;
  }

  Player player() {
    return player;
  }

  PlayerActionDTO action() {
    return action;
  }

  void accept() {
    complete(Status.ACCEPTED);
  }

  void reject() {
    complete(Status.REJECTED);
  }

  private void complete(Status completedStatus) {
    if (status != Status.PENDING) {
      throw new IllegalStateException("Action receipt already resolved as " + status);
    }
    status = completedStatus;
  }

  public Status status() {
    return status;
  }

  public boolean isResolved() {
    return status != Status.PENDING;
  }

  public boolean isAccepted() {
    return status == Status.ACCEPTED;
  }

  public boolean isRejected() {
    return status == Status.REJECTED;
  }
}
