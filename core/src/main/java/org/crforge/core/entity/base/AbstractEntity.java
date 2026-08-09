// Modified by NextoCR contributors; see NOTICE for attribution.
package org.crforge.core.entity.base;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import lombok.experimental.SuperBuilder;
import org.crforge.core.component.Health;
import org.crforge.core.component.Movement;
import org.crforge.core.component.Position;
import org.crforge.core.component.SpawnerComponent;
import org.crforge.core.effect.AppliedEffect;
import org.crforge.core.effect.BuffDefinition;
import org.crforge.core.effect.BuffRegistry;
import org.crforge.core.effect.StatusEffectType;
import org.crforge.core.player.Team;

@Getter
@SuperBuilder
@ToString(of = {"id", "name", "team"})
public abstract class AbstractEntity implements Entity {

  // Kept for entities built outside a GameState. The owning GameState replaces this temporary ID
  // with an ID from its world-local sequence when the entity is spawned.
  private static final AtomicLong DETACHED_ID_SEQUENCE = new AtomicLong(1);

  @Builder.Default protected long id = DETACHED_ID_SEQUENCE.getAndIncrement();

  @Getter(AccessLevel.NONE)
  private boolean gameIdAssigned;

  protected final String name;
  protected final Team team;
  protected final Position position;

  @Builder.Default protected final Health health = new Health(100);

  @Builder.Default
  protected final Movement movement = new Movement(0f, 0f, 0.5f, 0.5f, MovementType.GROUND);

  @Builder.Default protected final SpawnerComponent spawner = null;

  @Builder.Default protected final List<AppliedEffect> appliedEffects = new ArrayList<>();

  // Level used when this entity was created (1 = base stats, no scaling)
  @Builder.Default protected final int level = 1;

  @Builder.Default protected boolean spawned = false;

  @Builder.Default protected boolean dead = false;

  @Setter @Builder.Default protected boolean invulnerable = false;

  /** Assigns the deterministic ID allocated by the {@code GameState} owning this entity. */
  public final synchronized void assignGameId(long id) {
    if (id <= 0) {
      throw new IllegalArgumentException("Entity ID must be positive");
    }
    if (gameIdAssigned) {
      throw new IllegalStateException("Entity already has a game ID: " + this.id);
    }
    this.id = id;
    this.gameIdAssigned = true;
  }

  /** Resets IDs for detached entities only. Running games use their own world-local allocator. */
  public static void resetIdCounter() {
    DETACHED_ID_SEQUENCE.set(1);
  }

  @Override
  public float getCollisionRadius() {
    return movement.getCollisionRadius();
  }

  @Override
  public float getVisualRadius() {
    return movement.getVisualRadius();
  }

  @Override
  public MovementType getMovementType() {
    return movement.getType();
  }

  @Override
  public boolean isAlive() {
    return !dead && health.isAlive();
  }

  @Override
  public boolean isTargetable() {
    return isAlive() && spawned && !invulnerable;
  }

  public void markDead() {
    this.dead = true;
  }

  @Override
  public void onSpawn() {
    this.spawned = true;
  }

  @Override
  public void onDeath() {
    this.dead = true;
  }

  @Override
  public List<AppliedEffect> getAppliedEffects() {
    return appliedEffects;
  }

  @Override
  public void addEffect(AppliedEffect effect) {
    // Check if this buff allows stacking (e.g. Earthquake)
    BuffDefinition buffDef = BuffRegistry.get(effect.getBuffName());
    boolean stackable = buffDef != null && buffDef.isEnableStacking();

    if (!stackable) {
      // Prevent stacking of effects of the same type.
      // Instead, refresh the duration of the existing effect.
      // Exception: different CURSE buffs can coexist (e.g. GoblinCurse + VoodooCurse).
      for (AppliedEffect existing : appliedEffects) {
        if (existing.getType() == effect.getType()) {
          if (existing.getType() == StatusEffectType.CURSE
              && !Objects.equals(existing.getBuffName(), effect.getBuffName())) {
            continue;
          }
          existing.refresh(effect.getRemainingDuration());
          return;
        }
      }
    }
    appliedEffects.add(effect);
  }
}
