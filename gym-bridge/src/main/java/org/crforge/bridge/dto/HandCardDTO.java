package org.crforge.bridge.dto;

import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * A card in a player's hand, as seen by the RL agent.
 *
 * @param cardIndex legacy 0-based index into the current CardRegistry order
 * @param identityId stable namespaced card identity; zero is reserved for unknown/collision
 * @param allowsEnemyPlacement whether this card may be placed on the opponent's half
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record HandCardDTO(
    String id,
    String name,
    String type,
    int cost,
    int cardIndex,
    int identityId,
    boolean allowsEnemyPlacement) {}
