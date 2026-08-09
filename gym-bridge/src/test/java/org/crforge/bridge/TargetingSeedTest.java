package org.crforge.bridge;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;
import org.crforge.bridge.dto.InitConfig;
import org.crforge.core.combat.TargetSelectAlgorithm;
import org.crforge.core.combat.TargetingSystem;
import org.crforge.core.component.Combat;
import org.crforge.core.component.Movement;
import org.crforge.core.component.Position;
import org.crforge.core.entity.base.Entity;
import org.crforge.core.entity.base.MovementType;
import org.crforge.core.entity.unit.Troop;
import org.crforge.core.player.Team;
import org.junit.jupiter.api.Test;

class TargetingSeedTest {

  private static final List<String> TEST_DECK =
      List.of(
          "knight", "archer", "fireball", "arrows", "giant", "musketeer", "minions", "valkyrie");

  @Test
  void sameSeedMatchesAcrossEnginesAndReusedEpisodes() {
    GameSession first = sessionWithSeed(1234L);
    GameSession second = sessionWithSeed(1234L);

    List<String> expected = sampleRandomTargets(first.getEngine().getTargetingSystem());

    assertThat(sampleRandomTargets(second.getEngine().getTargetingSystem()))
        .as("independent engines with the same seed")
        .isEqualTo(expected);

    first.reset(1234L);

    assertThat(sampleRandomTargets(first.getEngine().getTargetingSystem()))
        .as("a reused engine reset to the same episode seed")
        .isEqualTo(expected);
  }

  @Test
  void differentSeedsCanProduceDifferentTargetSequences() {
    GameSession first = sessionWithSeed(1234L);
    GameSession second = sessionWithSeed(9876L);

    assertThat(sampleRandomTargets(second.getEngine().getTargetingSystem()))
        .as("different episode seeds should select a distinct random-target sequence")
        .isNotEqualTo(sampleRandomTargets(first.getEngine().getTargetingSystem()));
  }

  private GameSession sessionWithSeed(long seed) {
    GameSession session = new GameSession();
    session.init(new InitConfig(TEST_DECK, TEST_DECK, 11, 1, seed));
    return session;
  }

  private List<String> sampleRandomTargets(TargetingSystem targetingSystem) {
    List<String> targets = new ArrayList<>();
    for (int sample = 0; sample < 24; sample++) {
      Troop attacker = randomTargetingTroop("attacker-" + sample, Team.BLUE, 9, 16);
      List<Entity> entities = new ArrayList<>();
      entities.add(attacker);

      for (int candidate = 0; candidate < 5; candidate++) {
        entities.add(
            deployedTroop(
                "candidate-" + candidate, Team.RED, 5 + candidate * 2, 18 + sample * 0.001f));
      }

      targetingSystem.updateTargets(entities);
      targets.add(attacker.getCombat().getCurrentTarget().getName());
    }
    return targets;
  }

  private Troop randomTargetingTroop(String name, Team team, float x, float y) {
    Troop troop =
        Troop.builder()
            .name(name)
            .team(team)
            .position(new Position(x, y))
            .deployTime(0)
            .movement(new Movement(0f, 0f, 0.5f, 0.5f, MovementType.GROUND))
            .combat(
                Combat.builder()
                    .sightRange(20f)
                    .targetSelectAlgorithm(TargetSelectAlgorithm.RANDOM)
                    .build())
            .build();
    troop.onSpawn();
    return troop;
  }

  private Troop deployedTroop(String name, Team team, float x, float y) {
    Troop troop =
        Troop.builder()
            .name(name)
            .team(team)
            .position(new Position(x, y))
            .deployTime(0)
            .movement(new Movement(0f, 0f, 0.5f, 0.5f, MovementType.GROUND))
            .build();
    troop.onSpawn();
    return troop;
  }
}
