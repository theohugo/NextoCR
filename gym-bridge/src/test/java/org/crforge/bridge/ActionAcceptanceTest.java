package org.crforge.bridge;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.List;
import org.crforge.bridge.dto.InitConfig;
import org.crforge.bridge.dto.StepAction;
import org.crforge.bridge.dto.StepResultDTO;
import org.crforge.core.card.Card;
import org.junit.jupiter.api.Test;

class ActionAcceptanceTest {

  private static final float ONE_TICK_REGEN = (1f / 2.8f) / 20f;

  private static final List<String> TEST_DECK =
      List.of(
          "knight", "archer", "fireball", "arrows", "giant", "musketeer", "minions", "valkyrie");

  @Test
  void jsonStepReportsActualActionAcceptance() {
    assertAcceptanceContract(false);
  }

  @Test
  void binaryStepReportsActualActionAcceptance() {
    assertAcceptanceContract(true);
  }

  private void assertAcceptanceContract(boolean binary) {
    GameSession mixedSession = newSession();
    Card blueCardBefore = mixedSession.getBluePlayer().getHand().getCard(0);
    Card redCardBefore = mixedSession.getRedPlayer().getHand().getCard(0);
    assertThat(mixedSession.getRedPlayer().getElixir().spend(5)).isTrue();

    StepResultDTO mixedResult =
        execute(mixedSession, binary, new StepAction(0, 9f, 5f), new StepAction(0, 9f, 22f));

    assertThat(mixedResult.blueActionFailed()).as("affordable valid action").isFalse();
    assertThat(mixedResult.redActionFailed()).as("unaffordable action").isTrue();
    assertThat(mixedSession.getBluePlayer().getHand().getCard(0)).isNotSameAs(blueCardBefore);
    assertThat(mixedSession.getRedPlayer().getHand().getCard(0)).isSameAs(redCardBefore);
    assertThat(mixedSession.getBluePlayer().getElixir().getCurrent())
        .isCloseTo(5f + ONE_TICK_REGEN - blueCardBefore.getCost(), within(0.0001f));
    assertThat(mixedSession.getRedPlayer().getElixir().getCurrent())
        .isCloseTo(ONE_TICK_REGEN, within(0.0001f));

    GameSession invalidSession = newSession();
    Card invalidCardBefore = invalidSession.getBluePlayer().getHand().getCard(0);
    StepResultDTO invalidResult =
        execute(invalidSession, binary, new StepAction(0, -1f, -1f), null);

    assertThat(invalidResult.blueActionFailed()).as("invalid placement").isTrue();
    assertThat(invalidSession.getBluePlayer().getHand().getCard(0)).isSameAs(invalidCardBefore);
    assertThat(invalidSession.getBluePlayer().getElixir().getCurrent())
        .isCloseTo(5f + ONE_TICK_REGEN, within(0.0001f));

    GameSession noopSession = newSession();
    Card noopCardBefore = noopSession.getBluePlayer().getHand().getCard(0);
    StepResultDTO noopResult = execute(noopSession, binary, new StepAction(-1, 0f, 0f), null);

    assertThat(noopResult.blueActionFailed()).as("explicit no-op").isFalse();
    assertThat(noopResult.redActionFailed()).as("null no-op").isFalse();
    assertThat(noopSession.getBluePlayer().getHand().getCard(0)).isSameAs(noopCardBefore);
    assertThat(noopSession.getBluePlayer().getElixir().getCurrent())
        .isCloseTo(5f + ONE_TICK_REGEN, within(0.0001f));
    if (binary) {
      assertThat(noopResult.observation()).isNull();
    } else {
      assertThat(noopResult.observation()).isNotNull();
    }
  }

  private GameSession newSession() {
    GameSession session = new GameSession();
    session.init(new InitConfig(TEST_DECK, TEST_DECK, 11, 1, 42L));
    return session;
  }

  private StepResultDTO execute(
      GameSession session, boolean binary, StepAction blueAction, StepAction redAction) {
    return binary ? session.stepBinary(blueAction, redAction) : session.step(blueAction, redAction);
  }
}
