// Modified by NextoCR contributors; see NOTICE for attribution.
package org.crforge.bridge.observation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.zip.CRC32;
import org.crforge.data.card.CardRegistry;

/** Stable, bounded identities for cards and runtime entity archetypes in RL observations. */
final class ObservationIdentity {

  static final int MAX_ID = 1 << 24;

  private static final String CARD_NAMESPACE = "card";
  private static final String ENTITY_NAMESPACE = "entity";
  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final IdentityVocabulary VOCABULARY = buildVocabulary();
  private static final Map<String, Integer> CARD_IDENTITIES =
      VOCABULARY.identitiesFor(CARD_NAMESPACE);
  private static final Map<String, Integer> ENTITY_IDENTITIES =
      VOCABULARY.identitiesFor(ENTITY_NAMESPACE);

  private ObservationIdentity() {}

  static int cardId(String cardId) {
    return lookup(CARD_IDENTITIES, cardId);
  }

  static int entityId(String entityName) {
    return lookup(ENTITY_IDENTITIES, entityName);
  }

  static float normalize(int identityId) {
    return identityId / (float) MAX_ID;
  }

  static Set<String> collisionKeys() {
    return VOCABULARY.collisionKeys();
  }

  private static int lookup(Map<String, Integer> identities, String key) {
    return key == null || key.isBlank() ? 0 : identities.getOrDefault(key, 0);
  }

  /** Returns 1..2^24 for a non-empty key, with zero reserved for padding/unknown/collision. */
  static int hash(String namespace, String key) {
    if (namespace == null || namespace.isBlank() || key == null || key.isBlank()) {
      return 0;
    }
    CRC32 crc = new CRC32();
    crc.update((namespace + ":" + key).getBytes(StandardCharsets.UTF_8));
    return 1 + (int) (crc.getValue() & 0x00FF_FFFFL);
  }

  private static IdentityVocabulary buildVocabulary() {
    IdentityVocabulary vocabulary = new IdentityVocabulary();
    for (String cardId : CardRegistry.getAllIds()) {
      vocabulary.resolve(CARD_NAMESPACE, cardId);
    }

    registerEntityNames(vocabulary, "/cards/units.json");
    registerEntityNames(vocabulary, "/cards/cards.json");
    registerEntityNames(vocabulary, "/cards/projectiles.json");
    vocabulary.resolve(ENTITY_NAMESPACE, "Crown Tower");
    vocabulary.resolve(ENTITY_NAMESPACE, "Princess Tower");
    vocabulary.resolve(ENTITY_NAMESPACE, "AttackAreaEffect");
    vocabulary.resolve(ENTITY_NAMESPACE, "SpawnedAreaEffect");
    vocabulary.freeze();
    return vocabulary;
  }

  private static void registerEntityNames(IdentityVocabulary vocabulary, String resourcePath) {
    try (InputStream stream = ObservationIdentity.class.getResourceAsStream(resourcePath)) {
      if (stream == null) {
        throw new IllegalStateException("Missing observation identity resource: " + resourcePath);
      }
      collectNames(MAPPER.readTree(stream), vocabulary);
    } catch (IOException e) {
      throw new IllegalStateException(
          "Cannot load observation identity resource: " + resourcePath, e);
    }
  }

  private static void collectNames(JsonNode node, IdentityVocabulary vocabulary) {
    if (node.isObject()) {
      node.properties()
          .forEach(
              entry -> {
                if ("name".equals(entry.getKey()) && entry.getValue().isTextual()) {
                  vocabulary.resolve(ENTITY_NAMESPACE, entry.getValue().asText());
                }
                collectNames(entry.getValue(), vocabulary);
              });
    } else if (node.isArray()) {
      node.forEach(child -> collectNames(child, vocabulary));
    }
  }

  static final class IdentityVocabulary {

    private final Map<String, Integer> idByKey = new HashMap<>();
    private final Map<Integer, String> keyById = new HashMap<>();
    private final Set<Integer> collidingIds = new HashSet<>();
    private final Set<String> collidingKeys = new HashSet<>();
    private boolean frozen;

    synchronized int resolve(String namespace, String key) {
      if (key == null || key.isBlank()) {
        return 0;
      }

      String namespacedKey = namespace + ":" + key;
      Integer existingId = idByKey.get(namespacedKey);
      if (existingId != null) {
        return existingId;
      }
      if (frozen) {
        return 0;
      }

      int candidate = hash(namespace, key);
      if (collidingIds.contains(candidate)) {
        idByKey.put(namespacedKey, 0);
        collidingKeys.add(namespacedKey);
        return 0;
      }

      String existingKey = keyById.get(candidate);
      if (existingKey != null && !existingKey.equals(namespacedKey)) {
        collidingIds.add(candidate);
        collidingKeys.add(existingKey);
        collidingKeys.add(namespacedKey);
        idByKey.put(existingKey, 0);
        idByKey.put(namespacedKey, 0);
        keyById.remove(candidate);
        return 0;
      }

      keyById.put(candidate, namespacedKey);
      idByKey.put(namespacedKey, candidate);
      return candidate;
    }

    synchronized void freeze() {
      frozen = true;
    }

    synchronized Map<String, Integer> identitiesFor(String namespace) {
      String prefix = namespace + ":";
      Map<String, Integer> identities = new HashMap<>();
      idByKey.forEach(
          (key, id) -> {
            if (key.startsWith(prefix)) {
              identities.put(key.substring(prefix.length()), id);
            }
          });
      return Collections.unmodifiableMap(identities);
    }

    synchronized Set<String> collisionKeys() {
      return Collections.unmodifiableSet(new HashSet<>(collidingKeys));
    }
  }
}
