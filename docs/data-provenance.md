# Card Data Provenance

This document defines which card data NextoCR may ingest, how every value must be
attributed, and which sources must not be copied into the repository. It is a data
engineering policy, not legal advice and not a statement that Supercell has approved
this project.

## Snapshot status

| Item | Value |
| --- | --- |
| Source audit date | 2026-08-09 |
| Inherited engine commit | [`voonhous/crforge@c41c16a`](https://github.com/voonhous/crforge/tree/c41c16a85e6474e8a59b088ace8358a76b9a9d1a) |
| Inherited card-data claim | Season 80 (`202602`, February 2026) |
| Inherited upstream tracker | Claims 113 done, 7 partial, 11 missing, and 128 playable total; those category counts do not reconcile |
| Corrected NextoCR tracker | 112 done, 6 partial, 10 missing; 128 playable total |
| Live-data status | Not yet synchronized and not certified as August 2026 balance data |

The upstream tracker category values add up to 131 rather than its stated 128 total.
NextoCR has corrected the summary against the detailed tables, but **128 is still not a
current live-card count**. The tracker has a different scope from the official catalog
and contains special, missing, and possibly internal or unreleased entries. Its known
gaps include six champion abilities, Goblinstein, Boss Bandit, Cannoneer, Dagger
Duchess, Royal Chef, every Evolution and Hero mechanic, and Ronin. Its Tower Troop list
also contains `Apprentice`; an inherited internal name is not proof that an item is
live. Never decide whether a card is live from an inherited file, wiki category, or
search result. A fresh official API response, including its `supportItems` collection,
is the catalog authority.

The last upstream data baseline predates Ronin and later Evolution/Hero releases. The
simulator must therefore report the baseline above, rather than presenting the data as
current merely because the repository was audited in August 2026.

## Source hierarchy

No single public source provides a complete, current, openly licensed simulation
dataset. NextoCR uses sources for distinct purposes:

1. **Official catalog:** the Supercell API determines active card identity, stable ID,
   rarity, elixir cost, and whether an item is a normal card or Tower Troop.
2. **Official changes:** Supercell release notes and balance posts determine dated
   changes and their stated scope.
3. **Licensed detailed values:** revision-pinned community wiki text may seed fields
   that the official API does not expose, subject to its license and API terms.
4. **Independent observation:** non-invasive, reproducible measurements may resolve
   hidden timing or physics fields when their method and uncertainty are recorded.
5. **Cross-check only:** proprietary, stale, or unlicensed datasets may reveal that a
   value needs investigation, but their content must not be copied.

When sources disagree, do not silently choose one. Preserve the conflicting
observations, mark the canonical field `unresolved`, and add a focused interaction test
or an allowed independent measurement.

## Audited sources

The machine-readable companion to this table is
[`data-sources.json`](data-sources.json).

| Source | Coverage and format | Currency | Rights and decision |
| --- | --- | --- | --- |
| [Official Clash Royale API](https://developer.clashroyale.com/#/documentation), `GET https://api.clashroyale.com/v1/cards` | Authenticated JSON. Current responses contain `items` and `supportItems`, with fields such as `name`, `id`, `maxLevel`, `maxEvolutionLevel`, `elixirCost`, `rarity`, and `iconUrls`. It does **not** provide hitpoints, damage, hit speed, first hit/load time, range, movement, projectile behavior, ability logic, or complete per-level combat tables. | Live, queried on demand. No historical snapshots or balance-version identifier are supplied. | Authoritative for catalog metadata only. Access requires a bearer token and IP allow-listing. The API response and linked art are not published under an open-data license. Fetch server-side; never commit credentials or redistribute an unreviewed raw dump or downloaded icons. |
| [Official Clash Royale release notes](https://supercell.com/en/games/clashroyale/blog/release-notes/) and [support news](https://support.supercell.com/clash-royale/en/index.html) | Human-readable release descriptions and balance deltas, normally at Level 11. Examples include the [July 2026 balance changes](https://supercell.com/en/games/clashroyale/blog/release-notes/july-balance-changes-2026/) and the [Ronin release](https://supercell.com/en/games/clashroyale/blog/release-notes/new-season-honor-and-exile/). | Official but event-driven; older posts can be corrected. | Cite and encode small factual deltas with their effective date and URL. Do not copy article prose or media. A patch note is not a complete snapshot, and phrases such as “most of the time” must not be converted into a universal inheritance rule. |
| [Clash Royale Wiki on Fandom](https://clashroyale.fandom.com/wiki/Clash_Royale_Wiki), via its [MediaWiki API](https://clashroyale.fandom.com/api.php) | Wikitext, templates, tables, revision ID, and timestamp. Broad base-card, Tower Troop, Evolution, Hero, and per-level coverage, but schemas vary and new pages may be placeholders. | Frequently updated, but subject to lag, vandalism, formula mistakes, and incomplete newly announced forms. | Fandom states that community text is generally [CC BY-SA 3.0](https://www.fandom.com/licensing) unless otherwise noted. Import text/table-derived data only with article URL, page/revision ID, retrieval time, attribution, and ShareAlike handling. Images and other media have separate licenses and must not be copied. Use the API, not brittle HTML scraping. |
| [Liquipedia Clash Royale](https://liquipedia.net/clashroyale/Main_Page), via its MediaWiki API | Often structured card infoboxes and explicit level tables; useful as a second licensed source for base stats and hidden attributes. | Page-dependent. Some cards/forms lag releases. | Text/code reuse is governed by CC BY-SA terms and the [Liquipedia API Terms of Use](https://liquipedia.net/api-terms-of-use). Automated HTML access is prohibited. Use a descriptive User-Agent, cache responses, and respect the published limits (including at most one ordinary API request per two seconds and one `action=parse` request per 30 seconds). Media licensing varies. |
| [`voonhous/crforge`](https://github.com/voonhous/crforge) | Deterministic Java simulator, typed JSON schemas, units, projectiles, buffs, tests, and more than 100 implemented cards. The checked-in tracker says Season 80 (`202602`); Evolutions, Hero mechanics, champion abilities, and several Tower Troops are missing or partial. | Upstream code commit audited 2026-07-13; gameplay dataset claims February 2026. | Code is Apache-2.0 and is the permitted engineering foundation when `LICENSE` and `NOTICE` are retained. Its notice describes card statistics as “community-decoded from publicly available game data”; that is not field-level provenance and does not establish that Supercell-derived data is Apache-licensed. Treat inherited values as `inherited_unverified` until re-sourced or measured. |
| [`RoyaleAPI/cr-api-data`](https://github.com/RoyaleAPI/cr-api-data) | Convenient JSON identity files and hidden-stat tables (`cards*.json`, characters, buildings, spells, projectiles, buffs, and early Evolutions). | Latest audited commit [`d5461b0`](https://github.com/RoyaleAPI/cr-api-data/tree/d5461b0a59bff33c4da2fc845b07275b66b2d6ff), 2023-10-18: too old for Tower Troops, most Evolutions/Heroes, Ronin, and current level/balance data. | No explicit repository license was present at the audited commit. Do not copy or relicense its JSON. It is historical research context only. |
| [RoyaleAPI website](https://royaleapi.com/) | Current card pages, decks, usage/win statistics, and editorial release coverage. Its former developer API is an archive: RoyaleAPI says it was [sunset on 2020-03-01](https://docs.royaleapi.com/getting_started.html) in favor of the official API. | Current website; dead public API. | The [RoyaleAPI Terms](https://royaleapi.com/tos) reserve site rights and restrict copying/redistribution. No automated ingestion or republishing without written permission. Manual links may be used to flag a discrepancy for independent verification. |
| [Kaggle “Clash Royale Cards Dataset”](https://www.kaggle.com/datasets/emirdm/clash-royale-cards-dataset) | Small CSV with a limited set of aggregate columns, advertised as 2025 data. It lacks full level tables, hidden physics, newer variants, and a complete current Tower Troop catalog. | 2025 snapshot; incomplete for simulation. | Dataset page declares CC0. It can be used as a low-confidence seed or validation fixture after downloading the exact version and preserving its Kaggle metadata, but it cannot make the simulator current or mechanically complete. |
| [Noff card directory](https://www.noff.gg/clash-royale/cards) | Human-facing active-card directory with level selectors and detailed card pages; useful for sanity checks. | The directory reported 126 active cards at audit time. This number is informational and must be rechecked against the official API. | No open-data license or supported public API was found. Do not scrape or redistribute its tables. Use only as a manual discrepancy signal unless written permission is obtained. |
| Public APK/game-file dumps and asset repositories | Extracted CSVs, internal names, client binaries, textures, sounds, and unpacked assets can appear complete and machine-readable. | Often current but provenance and legality are poor. | **Rejected.** A repository being public or labeled MIT/Apache does not let its uploader sublicense Supercell assets or data obtained contrary to Supercell terms. Do not download, derive from, import, or link an automated pipeline to extracted game files. |

Other public simulators without an explicit license may be studied through their public
documentation, but their code and data must not be copied. In particular, a GitHub
repository is “publicly visible,” not automatically open source.

## Supercell policy boundary

Before publishing a branded simulator, maintainers must review both the
[Supercell Fan Content Policy](https://supercell.com/en/fan-content-policy/) and the
[Supercell Terms of Service](https://supercell.com/en/terms-of-service/). At the audit
date, these create material risk that cannot be solved by choosing Apache-2.0 for this
repository:

- the Fan Content Policy limits use of Supercell assets and says fan creators may not
  create new games or products based on Supercell game characters, even when free;
- the Terms prohibit unauthorized bots, automation, mods, reverse engineering, and
  obtaining game information through methods not expressly permitted;
- Supercell names, characters, artwork, audio, client code, and other assets remain
  Supercell property.

Accordingly, NextoCR must remain an offline research simulator and must not connect to,
control, instrument, or impersonate the live client or service. It must not include a
bot for playing accounts, packet capture, memory inspection, decompilation, private
server protocol, anti-cheat bypass, extracted client files, or credentials.

Even an offline implementation may still require permission under the Fan Content
Policy. The safest publication path is written permission from Supercell. Without it,
use a generic headless mechanics engine, original neutral identifiers and visuals, and
a user-supplied import format; obtain qualified legal review before distributing a
branded, stat-complete recreation.

Every public project surface must retain this disclaimer:

> This material is unofficial and is not endorsed by Supercell. For more information
> see Supercell's Fan Content Policy: www.supercell.com/fan-content-policy.

## Repository rules

### Permitted with provenance

- Apache-2.0 code inherited from crforge, with the upstream `LICENSE`, `NOTICE`, and
  modification history retained.
- Small factual official-API metadata required to identify a catalog item, when API
  terms and Fan Content Policy permit it.
- Append-only factual balance deltas transcribed from official posts, with URL,
  publication date, effective date, stated level, and explicit affected form(s).
- CC BY-SA wiki-derived text/table data in a separately identified ShareAlike data
  package with page-level revision attribution.
- Independently measured facts with a reproducible protocol, game version/date,
  repeated trials, uncertainty, and contributor attestation that no prohibited
  extraction or automation was used.
- Original placeholder geometry and original UI assets created for NextoCR.

### Not permitted

- Supercell card art, icons, logos, sounds, fonts, animations, screenshots used as a
  reusable asset pack, or downloaded API asset files. Storing a source URL for audit is
  not permission to mirror the file.
- APK/IPA contents, decompiled code, extracted `*.csv` tables, internal resource packs,
  memory dumps, private protocol captures, leaked or unreleased content.
- Raw data copied from RoyaleAPI, Noff, an unlicensed GitHub repository, or another
  proprietary site.
- A scraper that depends on rendered HTML when a permitted API exists, bypasses rate
  limits, evades access controls, or ignores robots/terms.
- An API key, bearer token, player token, account identifier, or any other credential
  in source control, logs, fixtures, CI artifacts, or client-side applications.
- A guessed value presented as exact. Use `unknown`, a range, or an uncertainty field.

## Recommended ingestion pipeline

### 1. Synchronize the official catalog

NextoCR provides `scripts/sync_official_catalog.py` for this narrow task. It reads the token from
`CLASH_ROYALE_API_TOKEN` or a local `--token-file`, emits only allow-listed metadata, keeps
`items`/`supportItems` distinct, and excludes icon URLs. For example:

```bash
mkdir -p .local
python scripts/sync_official_catalog.py \
  --output .local/official-catalog.json
```

The `.local/` directory is ignored by Git. Review the official API terms and do not commit the raw
or normalized response without a separate redistribution decision.

Run a server-side job after releases and at least weekly:

1. Read the API token from a secret store and call `GET /v1/cards` from an allow-listed
   IP.
2. Preserve `items` and `supportItems` as distinct source collections. `supportItems`
   contains Tower Troops and must not be dropped.
3. Normalize only documented or observed metadata: official ID, display name, rarity,
   elixir cost, maximum API level fields, and the presence of variant icon URL keys.
4. Store the raw response only in a restricted build cache if permitted. Record a
   SHA-256 hash, retrieval timestamp, HTTP validators, and normalized diff in the
   source ledger. Do not commit the token or mirror images.
5. Fail the update if the official catalog and the simulation catalog differ. New
   official IDs must begin as `active` plus `stats_status: missing`, not as guessed
   clones of a similar card.

Do not treat `maxLevel` as the displayed level cap without conversion. Official API
level fields have historically been rarity-relative. Current clients align display
levels, while API values can require rarity offsets. Keep both `api_level` and
`display_level`, document the conversion table for the observed API version, and test
it against real player-card responses. `maxEvolutionLevel` is not a complete mechanics
description; variant availability must not be inferred beyond what a validated schema
supports.

### 2. Build a licensed detailed-stat seed

Use the Fandom or Liquipedia MediaWiki API under their terms. For each imported page,
record:

- canonical URL, wiki name, page ID, revision ID, revision timestamp, and retrieval
  timestamp;
- content license and attribution text or author-list URL;
- parser version and SHA-256 of the exact source wikitext;
- whether the page is complete, a formula, an explicit table, or a “coming soon”
  placeholder;
- source references per field rather than one source label for the entire card.

Wiki images are out of scope. A page's text license does not automatically cover its
media. Cache API responses and obey the provider's User-Agent and rate limits.

### 3. Apply official changes as events

Store each official balance change as an immutable event before materializing a new
snapshot. An event needs `published_at`, `effective_at`, `source_url`, `stated_level`,
`subject_id`, `form`, `field`, `old_value`, `new_value`, and a note about stated scope.

Base, Evolution, Hero, champion ability, spawned entity, and Tower Troop values are
separate subjects. Never assume a base-form change applies to every form. If an
official post says that it applies “most of the time,” encode the named exceptions or
leave dependent forms pending verification.

### 4. Normalize without losing evidence

The simulator may use canonical units, but the provenance record retains source units
and source precision. Examples include milliseconds versus seconds, game subtiles
versus tiles, and integer percentages versus ratios.

Level tables require special care. Do not extrapolate by one floating-point multiplier
and round only at the end. Clash Royale values may use iterative per-level rounding,
rarity-relative API levels, distinct early Tower scaling, or bespoke ability scaling.
Prefer explicit official/UI tables; otherwise store the formula, rounding mode, anchor
level, and every generated result, then validate representative levels.

### 5. Validate and publish

CI should reject a data snapshot when any of these checks fail:

- every active official ID maps to exactly one base card or Tower Troop;
- no deprecated, special-mode, internal spawn, or unreleased entity is exported as an
  active playable card;
- every simulation field has a source reference, a documented measurement, or an
  explicit `unknown` status;
- source revisions, licenses, retrieval dates, and hashes are present;
- IDs are unique, level tables are monotonic where the mechanic requires it, units are
  normalized, and derived DPS agrees with damage/cooldown within documented rounding;
- each Evolution, Hero, champion ability, projectile, spawn, buff, and Tower Troop has
  independent coverage rather than inheriting undocumented behavior;
- golden interaction tests cover timing, targeting, projectile travel, splash,
  knockback, death spawns, and level-boundary rounding for changed cards.

Publish the generated snapshot date, effective balance date, official catalog hash,
source-revision manifest, missing-field count, and simulator coverage count. Never
label a build “current” merely because all JSON files parse.

## Minimal provenance record

Each canonical value should be traceable without reading a commit message. A record can
use this shape (illustrative, not yet a stable schema):

```json
{
  "subject": { "officialId": 26000000, "form": "base" },
  "field": "hitpoints",
  "value": 1766,
  "unit": "hp",
  "atDisplayLevel": 11,
  "status": "verified",
  "confidence": "licensed-source-plus-interaction-test",
  "sources": [
    {
      "kind": "cc-wiki-revision",
      "url": "https://example.invalid/wiki/Card",
      "revisionId": "123456",
      "retrievedAt": "2026-08-09T00:00:00Z",
      "license": "CC-BY-SA-3.0"
    }
  ],
  "effectiveFrom": "2026-07-06",
  "notes": "Example only; URL and values are not an import."
}
```

Recommended statuses are `verified`, `conflicting`, `measured`, `derived`,
`inherited_unverified`, and `unknown`. “Community decoded” by itself is never
`verified`.

## Update cadence and ownership

- Query the official catalog weekly and immediately after a release announcement.
- Review official release and balance posts at each effective-date boundary.
- Refresh only affected wiki revisions; do not re-scrape every page on every build.
- Cut immutable, versioned data snapshots rather than mutating an undated `latest`.
- Re-run catalog, schema, attribution, and interaction gates before each public release.
- Assign a maintainer to provenance review separately from mechanics review.

The intended first clean snapshot is **not** “August 2026 by inheritance.” It is the
first snapshot for which the official catalog was fetched, every retained field has an
allowed source or measurement, all inherited unverified values were resolved or marked
as such, and the effective official balance date is recorded.
