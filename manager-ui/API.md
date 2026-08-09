# Contrat HTTP du tableau de bord

L'interface appelle uniquement des routes relatives de même origine. Toutes les
réponses JSON utilisent l'enveloppe `{ "ok": true, "data": ... }`; une erreur
utilise `{ "ok": false, "error": { "code": "...", "message": "..." } }`.

## Session et lectures

- `GET /api/session` → `{ token }`.
- `GET /api/state` → état global et `selectedRunId`.
- `GET /api/runs` → liste des runs.
- `GET /api/runs/{runId}` → détail, checkpoints, ligue et actions disponibles.
- `GET /api/runs/{runId}/metrics` → agrégats, `latest` et `recentEpisodes`.
- `GET /api/runs/{runId}/logs?tail=180` → `{ lines: string[] }`.
- `GET /api/runs/{runId}/replays/{replayId}` → job puis document replay v1.

`runId` et `replayId` sont encodés avec `encodeURIComponent`.

## Mutations

Chaque `POST` envoie le token de session dans `X-NextoCR-Token` et un corps
JSON, même vide.

- `POST /api/runs` avec `{ label, targetTotalTimesteps? }`.
- `POST /api/runs/{runId}/checkpoint`.
- `POST /api/runs/{runId}/pause` : écrit un checkpoint sûr avant l'arrêt.
- `POST /api/runs/{runId}/resume`.
- `POST /api/runs/{runId}/versions` avec
  `{ label, targetTotalTimesteps?, checkpoint? }`.
- `POST /api/runs/{runId}/replays` avec
  `{ checkpoint?: "latest" | string, seed?: number, opponent?: "rule_based" | "league" }`.

La création d'un replay renvoie `{ replayId, status }`. L'interface interroge
ensuite la route de lecture jusqu'à l'état `ready` ou `failed`.

## Replay v1

Le document prêt contient notamment `trainingLevel`, `evaluation`, `arena` et
`frames`. Les coordonnées de l'arène sont `x: 0..18`, `y: 0..32`, avec BLUE en
bas et RED en haut. Chaque frame fournit `towers` et `entities` avec position,
équipe, PV et PV maximum. L'interface inverse l'axe Y pour le repère CSS et
interpole les déplacements entre deux frames.

Le serveur statique peut appliquer `script-src 'self'`. Les positions et barres
dynamiques utilisent des attributs de style, donc la CSP doit autoriser
`style-src 'self' 'unsafe-inline'`.
