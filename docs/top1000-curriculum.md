# Top-1000 opponent curriculum

NextoCR can build a local opponent catalog from the current decks of the players in an
official Clash Royale Ranked leaderboard snapshot. This is an offline RL input, not a live-game
integration.

The precise claim matters: the official ranking response identifies ranked players, while the
player profile exposes each player's `currentDeck`. A profile can change after the player earned
their rank. The result is therefore **the current decks of Top-1000 players at retrieval time**,
not proof of the decks used in their ranked matches and not a global Top-1000 deck-usage table.

## Official endpoints and credentials

The pipeline uses only the authenticated [official Clash Royale developer
API](https://developer.clashroyale.com/#/documentation):

```text
GET /v1/locations/global/pathoflegend/{seasonId}/rankings/players
    ?limit=50&after={cursor}

GET /v1/players/{urlEncodedPlayerTag}
```

The first endpoint is cursor-paginated and supplies ranks and player tags. The second supplies
`currentDeck` and, when present, `currentDeckSupportCards`. With pages of 50, a complete
1,000-player snapshot normally requires 20 leaderboard requests plus 1,000 profile requests.

Create a key at the official developer portal and allow-list the machine's public egress IP. The
portal states that the JWT is bound to IP and rate limitations, but it does not publish a numeric
quota on the public getting-started page. The script's default of five requests per second is a
conservative client setting, **not an official quota**. HTTP 429 and transient 5xx responses use
bounded retry/backoff.

Never put the token in a command line, Git, logs, fixtures, or a client application. Provide it
through `CLASH_ROYALE_API_TOKEN` or a local token file:

```powershell
$env:CLASH_ROYALE_API_TOKEN = '<developer JWT>'
```

The key itself must never be pasted into an issue or task. A 403 generally means that the token is
invalid/expired or the caller's current public IP is not in the key's allow-list.

## Reproducible two-stage workflow

Use an explicit season ID. Do not silently substitute the wall-clock month: a training manifest
must retain the exact leaderboard season requested.

```powershell
python scripts/build_top1000_curriculum.py fetch `
  --season 2026-08 `
  --target 1000 `
  --output .local/top1000-2026-08-source.json

python scripts/build_top1000_curriculum.py compile `
  --source .local/top1000-2026-08-source.json `
  --tracker docs/card_tracker.md `
  --fidelity-policy base-fallback `
  --output .local/top1000-2026-08-curriculum.json
```

Both outputs are intended for `.local/`, which is ignored by Git. `fetch` allow-lists only rank,
card identity, equipped form hints, and Tower Troop identity. It drops player names, player tags,
levels, icons, asset URLs, and unknown profile fields before writing. The bearer token is never
serialized. A source hash, retrieval interval, endpoints, season, mapping-policy version, tracker
hash, and compiler version make the derived snapshot auditable.

The fetch and compile phases are separate so an expensive API acquisition can be reviewed and
recompiled under a newer support map without querying players again. Do not combine snapshots
silently during a training run. Pin one `snapshotId` in that run's manifest.

For a CI quality gate, add either or both flags:

```powershell
python scripts/build_top1000_curriculum.py compile `
  --source .local/top1000-2026-08-source.json `
  --output .local/top1000-2026-08-curriculum.json `
  --require-target-complete `
  --require-exact-coverage
```

The file is still written for diagnosis, but the command returns exit code 3 when the requested
gate is not met. Validation/transport failures return exit code 2.

## Mapping and fidelity

The compiler joins official display names to the explicit `CR Name`, `Internal ID`, and status in
`docs/card_tracker.md`. It then applies the reviewed, append-only
`scripts/top1000_mapping_overrides.json` entries for playable cards that exist in the engine but
are absent from that Markdown table. The initial exceptions are Cannon Cart (`movingcannon`) and
Three Musketeers (`threemusketeers`). An override cannot replace an existing tracker row. It does
not guess a simulator card from edit distance. Both sources are hashed into the snapshot so a
changed mapping cannot masquerade as the same curriculum.

Two fidelity policies are available:

- `strict` admits only `DONE` base cards and the default Tower Princess. Evolution, Hero,
  `PARTIAL`, missing/unmapped cards, unknown Tower Troops, and non-default Tower Troops exclude
  the deck.
- `base-fallback` admits mapped `PARTIAL` cards and replaces Evolution/Hero forms and non-default
  or unknown Tower Troops with their base/default approximation. Missing or unmapped playable
  cards still exclude the deck.

`base-fallback` is appropriate only for the first mechanics curriculum. It is not exact Clash
Royale coverage. A later run should use `strict` after the relevant forms and Tower Troops have
golden interaction tests.

No observation is silently discarded. The compiled file contains:

- every unique observed deck, including excluded decks and their reasons;
- every observed card/form, its official ID, simulator mapping, tracker status, occurrence count,
  and rank mass;
- explicit `unmappedCards` and profile-fetch failures;
- counts for exact, approximated, partial, and excluded deck observations;
- raw and rank-weighted trainable/exact coverage;
- `completeTargetPopulation`, `exactSimulatorCoverage`, and `top1000Exact` booleans.

The `trainingClaim` remains `official-ranked-current-decks-partial-coverage` unless all 1,000
requested profiles are present and exact. A trainer or README must not call the population
"Top-1000 complete" merely because some eligible decks were emitted.

## Deduplication and sampling

Deck identity is the unordered set of official card IDs plus observed forms and Tower Troop
identity. Thus the same eight base cards with different Evolution assignments remain distinct in
the audit catalog, even when `base-fallback` maps both to the same simulator deck.

Only eligible decks receive a nonzero sampling weight. The default mixture is:

- 70% observed player frequency;
- 20% rank mass, where an observation contributes `1 / sqrt(rank)`;
- 10% uniform mass across unique eligible decks.

Frequency preserves the meta distribution, the moderate rank term gives the strongest ranks some
extra influence without letting rank 1 dominate, and the uniform floor keeps rare archetypes in
the curriculum. All three component weights and the combined normalized weight are retained in
the snapshot. Change the mixture with `--population-weight`, `--rank-weight`, and
`--uniform-weight`; the values must sum to one.

For self-play, use this catalog as the **opponent-deck distribution**, not as a replacement for
the checkpoint league. A useful schedule is:

1. fixed Mortar mirror self-play for basic mechanics and action validity;
2. 50% checkpoint-league mirror / 50% catalog opponents;
3. increase catalog sampling while retaining at least 20% mirror games for regression detection;
4. prioritize decks producing high loss rates, but retain the frozen catalog weight as a minimum
   exploration probability.

Snapshot IDs must be logged per episode or evaluation batch. Evaluation should report results by
deck quality (`exact-base`, `approximated`, `partial`) and never merge them into one unlabeled win
rate.

## Known limitations

- `currentDeck` is a point-in-time profile field. It can be unrelated to the battle that produced
  the listed rank.
- The leaderboard changes during the roughly 1,020-call acquisition, so the snapshot is not an
  atomic server-side transaction. Start/end timestamps expose that window.
- A player lookup can fail after the leaderboard page succeeds. Failures remain in the report by
  rank and reduce target coverage.
- The API schema can add new form markers. The pipeline currently retains `evolutionLevel`,
  `heroLevel`, and Hero/Evolution name hints; unknown forms will require a reviewed schema update.
- Card levels are intentionally ignored because the simulator uses its configured match level.
- Tower Troops are part of matchup behavior. Replacing Cannoneer, Dagger Duchess, Royal Chef, or
  another troop with Tower Princess is a material approximation.
- Champions marked `PARTIAL` lack live ability cycling. Hero and Evolution fallbacks likewise do
  not simulate their special mechanics.
- The official portal exposes rate limitations without a public numeric allowance. Lower
  `--requests-per-second` after 429 responses or when required by updated developer terms.
- No open-data redistribution license was identified for API responses. Keep generated snapshots
  private/local unless the current API agreement and publication plan have been reviewed.

Battle logs could support a separate "recent decks actually observed in Ranked battles" dataset,
but that has different biases: a short history window, duplicate matches, non-ranked modes, and
more opponent/player records. It must use a separate schema and claim rather than silently being
mixed with this current-profile snapshot.

## Policy boundary

This collector uses the documented read-only API for an offline research curriculum. It does not
read the game client, scrape rendered sites, download card art, automate gameplay, or control an
account. Review the [Supercell Fan Content Policy](https://supercell.com/en/fan-content-policy/)
and current developer/API terms before publishing any generated data or branded application.

> This material is unofficial and is not endorsed by Supercell. For more information see
> Supercell's Fan Content Policy: www.supercell.com/fan-content-policy.
