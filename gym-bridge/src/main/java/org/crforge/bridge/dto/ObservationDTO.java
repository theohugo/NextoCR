package org.crforge.bridge.dto;

import java.util.List;

/** Full game state observation sent to the RL agent. */
public record ObservationDTO(
    int schemaVersion,
    int frame,
    float gameTimeSeconds,
    boolean isOvertime,
    int elixirMultiplier,
    PlayerObsDTO bluePlayer,
    PlayerObsDTO redPlayer,
    List<EntityDTO> entities) {}
