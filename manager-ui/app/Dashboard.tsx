"use client";

import {
  type CSSProperties,
  type FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import "./dashboard.css";

type JsonRecord = Record<string, any>;
type ViewName = "overview" | "replay";

type RunView = {
  id: string;
  label: string;
  status: string;
  current: number;
  target: number;
  rate: number;
  etaSeconds: number | null;
  createdAt?: string;
  updatedAt?: string;
  parentRunId?: string;
  raw: JsonRecord;
};

type Confirmation = {
  title: string;
  description: string;
  confirmLabel: string;
  tone?: "default" | "warning";
  action: () => Promise<void>;
};

type ReplayEntity = {
  id: string;
  team: "blue" | "red";
  kind: string;
  x: number;
  y: number;
  hp: number;
  maxHp: number;
};

type ReplayTower = ReplayEntity & { tower: true };

type ReplayFrame = {
  t: number;
  entities: ReplayEntity[];
  towers: ReplayTower[];
};

type ReplayData = {
  id: string;
  status: string;
  progress: number;
  error?: string;
  metadata: JsonRecord;
  frames: ReplayFrame[];
};

const POLL_INTERVAL_MS = 2_500;
const REPLAY_POLL_INTERVAL_MS = 1_300;

const apiErrorMessages: Record<string, string> = {
  invalid_tail: "La taille demandée pour le journal n'est pas valide.",
  not_found: "La ressource demandée n'existe plus.",
  invalid_run_id: "L'identifiant du run n'est pas valide.",
  invalid_host: "Cette interface doit être ouverte depuis le manager local.",
  invalid_session: "La session locale a expiré. Rechargez la page.",
  invalid_origin: "Cette action est autorisée uniquement depuis l'interface locale.",
  invalid_content_type: "Le manager n'a pas reconnu le format de la requête.",
  invalid_body: "La requête envoyée au manager est invalide.",
  body_too_large: "La requête dépasse la taille autorisée.",
  invalid_json: "La requête envoyée au manager est invalide.",
  target_reached: "Ce run a déjà atteint son objectif total.",
  invalid_target: "L'objectif doit être supérieur au checkpoint et compris dans les limites autorisées.",
  invalid_seed: "La seed de simulation n'est pas valide.",
  invalid_opponent: "Cet adversaire n'est pas disponible pour la simulation.",
  replay_busy: "Une autre partie est déjà en cours de génération.",
  invalid_replay_id: "L'identifiant de la partie n'est pas valide.",
  unsupported_deck: "Le manager local est configuré uniquement pour le deck Mortier prévu.",
  unsupported_opponent: "Ce run doit utiliser l'auto-jeu avec l'évaluation prévue.",
  invalid_config: "La configuration d'entraînement n'est pas valide.",
  command_failed: "Le trainer n'a pas pu exécuter cette commande.",
  trainer_stopped: "Le trainer s'est arrêté avant de confirmer la commande.",
  command_timeout: "Le trainer met trop de temps à confirmer la commande.",
  training_active: "Mettez le run actif en pause avant de lancer cette action.",
  trainer_not_running: "Ce run n'est pas en cours d'entraînement.",
  no_bridge_port: "Aucun port local n'est disponible pour le simulateur.",
  trainer_exit_timeout: "Le checkpoint est écrit, mais le trainer termine encore son arrêt.",
  bridge_build_failed: "Le simulateur local n'a pas pu être préparé.",
};

const statusLabels: Record<string, string> = {
  running: "En entraînement",
  starting: "Démarrage",
  pausing: "Sauvegarde en cours",
  stopping: "Arrêt en cours",
  paused: "En pause",
  interrupted: "En pause",
  stopped: "En pause",
  checkpointed: "Checkpoint prêt",
  idle: "Prêt",
  ready: "Prêt",
  completed: "Terminé",
  failed: "À vérifier",
  failed_to_start: "Échec au démarrage",
  error: "À vérifier",
  shutdown_error: "Arrêt incomplet",
  external: "Processus externe",
  unknown: "État inconnu",
};

const outcomeLabels: Record<string, string> = {
  blue_win: "Victoire bleue",
  red_win: "Victoire rouge",
  win: "Victoire",
  loss: "Défaite",
  draw: "Égalité",
  ongoing: "En cours",
  unknown: "Non disponible",
};

function record(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonRecord)
    : {};
}

function array(value: unknown): any[] {
  return Array.isArray(value) ? value : [];
}

function numberFrom(...values: unknown[]): number {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string" && value.trim() !== "") {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) return parsed;
    }
  }
  return 0;
}

function stringFrom(...values: unknown[]): string {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value;
    if (typeof value === "number") return String(value);
  }
  return "";
}

function userError(error: unknown, fallback: string): string {
  if (error instanceof TypeError) {
    return "Impossible de joindre le manager local. Vérifiez qu'il est toujours ouvert.";
  }
  if (error instanceof Error && error.message.trim()) return error.message;
  return fallback;
}

function unwrap(payload: any): any {
  if (payload && typeof payload === "object" && "ok" in payload) {
    if (payload.ok === false) {
      const error = record(payload.error);
      const code = stringFrom(error.code, payload.code);
      throw new Error(
        apiErrorMessages[code] ??
          (code ? `Le manager local a refusé l'action (${code}).` : "Le manager local a signalé une erreur."),
      );
    }
    return "data" in payload ? payload.data : payload;
  }
  return payload;
}

async function parseResponse(response: Response): Promise<any> {
  const contentType = response.headers.get("content-type") ?? "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const body = record(payload);
    const error = record(body.error);
    const code = stringFrom(error.code, body.code);
    throw new Error(
      apiErrorMessages[code] ??
        (code
          ? `Le manager local a refusé l'action (${code}).`
          : `Le manager local ne répond pas comme prévu (erreur ${response.status}).`),
    );
  }
  return unwrap(payload);
}

function normaliseRun(rawValue: any): RunView {
  const wrapper = record(rawValue);
  const raw = Object.keys(record(wrapper.run)).length ? record(wrapper.run) : wrapper;
  const progress = record(raw.progress);
  const metrics = record(raw.metrics);
  const current = numberFrom(
    raw.currentTimesteps,
    raw.timesteps,
    raw.totalTimesteps,
    progress.current,
    progress.timesteps,
    metrics.timesteps,
  );
  const target = numberFrom(
    raw.targetTotalTimesteps,
    raw.targetTimesteps,
    raw.target,
    progress.target,
    10_000_000,
  );
  const rate = numberFrom(
    raw.throughputStepsPerSecond,
    raw.stepsPerSecond,
    raw.rate,
    progress.stepsPerSecond,
    metrics.stepsPerSecond,
  );
  const eta = numberFrom(raw.etaSeconds, progress.etaSeconds);
  const id = stringFrom(raw.id, raw.runId, raw.slug, raw.name, "run-local");

  return {
    id,
    label: stringFrom(raw.label, raw.displayName, raw.name, id),
    status: stringFrom(raw.status, wrapper.status, "unknown").toLowerCase(),
    current,
    target,
    rate,
    etaSeconds: eta > 0 ? eta : rate > 0 && target > current ? (target - current) / rate : null,
    createdAt: stringFrom(raw.createdAtUtc, raw.createdAt, raw.created_at) || undefined,
    updatedAt: stringFrom(raw.updatedAtUtc, raw.updatedAt, raw.updated_at, progress.updatedAt) || undefined,
    parentRunId: stringFrom(raw.parentRunId, raw.parent_run_id, raw.parent) || undefined,
    raw,
  };
}

function runsFromPayload(payload: any): RunView[] {
  const value = unwrap(payload);
  const values = Array.isArray(value)
    ? value
    : array(record(value).runs).length
      ? array(record(value).runs)
      : array(record(value).items);
  return values.map(normaliseRun);
}

function formatInteger(value: number): string {
  return new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 }).format(value || 0);
}

function formatRate(value: number): string {
  return `${new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 1 }).format(value || 0)} pas/s`;
}

function formatDuration(seconds: number | null | undefined): string {
  if (!seconds || !Number.isFinite(seconds) || seconds <= 0) return "—";
  if (seconds < 60) return `${Math.ceil(seconds)} s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  if (hours < 48) return `${hours} h ${remainingMinutes.toString().padStart(2, "0")}`;
  const days = Math.floor(hours / 24);
  return `${days} j ${hours % 24} h`;
}

function formatClock(seconds: number | null | undefined): string {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const remaining = Math.floor(safe % 60);
  return `${minutes}:${remaining.toString().padStart(2, "0")}`;
}

function formatDate(value: unknown, includeTime = true): string {
  if (!value) return "—";
  const date = new Date(String(value));
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("fr-FR", {
    day: "2-digit",
    month: "short",
    ...(includeTime ? { hour: "2-digit", minute: "2-digit" } : {}),
  }).format(date);
}

function formatReward(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

function compactCheckpointName(value: unknown): string {
  const name = stringFrom(value, "checkpoint");
  const pieces = name.replaceAll("\\", "/").split("/");
  return pieces.at(-1) || name;
}

function checkpointItems(detail: JsonRecord): JsonRecord[] {
  const artifacts = record(detail.artifacts);
  const values = array(detail.checkpoints).length
    ? array(detail.checkpoints)
    : array(artifacts.checkpoints);
  return values
    .map((item) => (typeof item === "string" ? { path: item } : record(item)))
    .sort(
      (a, b) =>
        numberFrom(b.timesteps, b.step, b.createdAt) -
        numberFrom(a.timesteps, a.step, a.createdAt),
    );
}

function metricEpisodes(metrics: JsonRecord): JsonRecord[] {
  const candidates = [metrics.recentEpisodes, metrics.episodes, metrics.history, metrics.items];
  for (const candidate of candidates) {
    if (Array.isArray(candidate)) return candidate.map(record);
  }
  const latest = record(metrics.latest);
  return Object.keys(latest).length ? [latest] : [];
}

function latestMetric(metrics: JsonRecord, detail: JsonRecord): JsonRecord {
  const episodes = metricEpisodes(metrics);
  return {
    ...record(detail.lastEpisode),
    ...record(metrics.lastEpisode),
    ...record(metrics.latest),
    ...(episodes.at(-1) ?? {}),
  };
}

function humanOutcome(value: unknown): string {
  const key = stringFrom(value, "unknown").toLowerCase().replaceAll("-", "_");
  return outcomeLabels[key] ?? String(value || "Non disponible");
}

// The simulator emits internal unit names ("SkeletonContainerNew", "Goblin_Stab").
// Abbreviating those to a single initial made Mortier, Gargouille and Chariot all
// render as "M", so every unit carries a readable label and a distinct badge.
const UNIT_LABELS: Record<string, { label: string; badge: string }> = {
  barbarian: { label: "Barbare", badge: "BA" },
  brokencannon: { label: "Canon démonté (chariot détruit)", badge: "CN" },
  goblin_stab: { label: "Gobelin", badge: "GO" },
  minion: { label: "Gargouille", badge: "GA" },
  mortar: { label: "Mortier", badge: "MO" },
  movingcannon: { label: "Chariot à canon", badge: "CC" },
  rascalboy: { label: "Canaille (garçon)", badge: "CG" },
  rascalgirl: { label: "Canaille (fille)", badge: "CF" },
  skeleton: { label: "Squelette", badge: "SQ" },
  skeletonballoon: { label: "Fût à squelettes", badge: "FS" },
  skeletoncontainernew: { label: "Fût à squelettes (projectile)", badge: "FP" },
  fireball: { label: "Boule de feu", badge: "BF" },
  crown: { label: "Tour du roi", badge: "K" },
  princess: { label: "Tour de princesse", badge: "T" },
};

/** Split an internal name into words so unknown units stay readable, not a single letter. */
function prettifyUnitName(kind: string): string {
  const words = kind
    .replace(/[_-]+/g, " ")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function describeUnit(kind: string): { label: string; badge: string } {
  const known = UNIT_LABELS[kind.trim().toLowerCase()];
  if (known) {
    return known;
  }
  const label = prettifyUnitName(kind);
  // Initials of each word keep unrelated units visually distinct.
  const initials = label
    .split(/\s+/)
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
  return { label, badge: initials || "U" };
}

function normaliseCoordinate(value: unknown, max: number): number {
  const raw = numberFrom(value);
  const normalised = Math.abs(raw) <= 1 ? raw : raw / max;
  return Math.max(0.025, Math.min(0.975, normalised));
}

function normaliseEntity(rawValue: any, index: number, fallbackTeam?: string): ReplayEntity {
  const raw = record(rawValue);
  const position = record(raw.position);
  const teamValue = stringFrom(raw.team, raw.side, fallbackTeam, "blue").toLowerCase();
  const hp = numberFrom(raw.hp, raw.health, raw.hitpoints);
  const maxHp = numberFrom(raw.maxHp, raw.maxHealth, raw.max_hitpoints, hp, 1);
  return {
    id: stringFrom(raw.id, raw.entityId, `${teamValue}-${index}`),
    team: teamValue.includes("red") || teamValue.includes("enemy") ? "red" : "blue",
    kind: stringFrom(raw.kind, raw.card, raw.name, raw.type, "unité"),
    x: normaliseCoordinate(raw.x ?? position.x, 18),
    // Simulator Y grows from BLUE (bottom) to RED (top), while CSS top grows downward.
    y: 1 - normaliseCoordinate(raw.y ?? position.y, 32),
    hp,
    maxHp: Math.max(maxHp, 1),
  };
}

function defaultTowers(): ReplayTower[] {
  return [
    { id: "blue-left", team: "blue", kind: "tour", x: 0.22, y: 0.84, hp: 1, maxHp: 1, tower: true },
    { id: "blue-king", team: "blue", kind: "roi", x: 0.5, y: 0.92, hp: 1, maxHp: 1, tower: true },
    { id: "blue-right", team: "blue", kind: "tour", x: 0.78, y: 0.84, hp: 1, maxHp: 1, tower: true },
    { id: "red-left", team: "red", kind: "tour", x: 0.22, y: 0.16, hp: 1, maxHp: 1, tower: true },
    { id: "red-king", team: "red", kind: "roi", x: 0.5, y: 0.08, hp: 1, maxHp: 1, tower: true },
    { id: "red-right", team: "red", kind: "tour", x: 0.78, y: 0.16, hp: 1, maxHp: 1, tower: true },
  ];
}

function normaliseFrame(rawValue: any, index: number): ReplayFrame {
  const raw = record(rawValue);
  const entities = array(raw.entities).map((item, entityIndex) =>
    normaliseEntity(item, entityIndex),
  );
  const explicitTowers = array(raw.towers).map((item, towerIndex) => ({
    ...normaliseEntity(item, towerIndex),
    tower: true as const,
  }));
  const blueTowers = array(raw.blueTowers).map((item, towerIndex) => ({
    ...normaliseEntity(item, towerIndex, "blue"),
    tower: true as const,
  }));
  const redTowers = array(raw.redTowers).map((item, towerIndex) => ({
    ...normaliseEntity(item, towerIndex, "red"),
    tower: true as const,
  }));
  const towers = [...explicitTowers, ...blueTowers, ...redTowers];
  return {
    t: numberFrom(raw.t, raw.time, raw.elapsed, index / 10),
    entities,
    towers: towers.length ? towers : defaultTowers(),
  };
}

function normaliseReplay(payload: any, fallbackId: string): ReplayData {
  const root = record(unwrap(payload));
  const replay = Object.keys(record(root.replay)).length ? record(root.replay) : root;
  const frameValues = array(replay.frames).length
    ? array(replay.frames)
    : array(record(replay.data).frames);
  const status = stringFrom(replay.status, frameValues.length ? "ready" : "queued").toLowerCase();
  const trainingLevel = record(replay.trainingLevel);
  const checkpoint = record(trainingLevel.checkpoint);
  const evaluation = record(replay.evaluation);
  const opponent = record(evaluation.opponent);
  const opponentLabel = opponent.type
    ? `${stringFrom(opponent.type)}${stringFrom(opponent.opponentId) ? ` · ${stringFrom(opponent.opponentId)}` : ""}`
    : "";
  return {
    id: stringFrom(replay.id, replay.replayId, fallbackId),
    status,
    progress: Math.max(0, Math.min(1, numberFrom(replay.progress, status === "ready" ? 1 : 0))),
    error: stringFrom(record(replay.error).message, replay.error, replay.message) || undefined,
    metadata: {
      checkpoint: stringFrom(checkpoint.path, checkpoint.id),
      checkpointTimesteps: numberFrom(
        trainingLevel.checkpointTimesteps,
        checkpoint.numTimesteps,
        checkpoint.step,
      ),
      currentTimesteps: numberFrom(trainingLevel.currentTimesteps),
      checkpointLagTimesteps: numberFrom(trainingLevel.checkpointLagTimesteps),
      runId: stringFrom(trainingLevel.runId),
      runStatus: stringFrom(trainingLevel.runStatus),
      seed: evaluation.seed,
      opponent: opponentLabel,
      opponentType: stringFrom(opponent.type),
      outcome: evaluation.outcome,
      reward: evaluation.totalReward,
      steps: evaluation.steps,
      ticksPerStep: evaluation.ticksPerStep,
      ...record(replay.metadata),
      ...record(record(replay.data).metadata),
    },
    frames: frameValues.map(normaliseFrame),
  };
}

function statusClass(status: string): string {
  if (["running", "starting"].includes(status)) return "is-live";
  if (["paused", "pausing", "stopping", "stopped", "interrupted", "checkpointed"].includes(status)) return "is-paused";
  if (status.includes("failed") || status.includes("error")) return "is-error";
  if (status === "completed") return "is-complete";
  return "is-idle";
}

function StatusPill({ status }: { status: string }) {
  return (
    <span className={`status-pill ${statusClass(status)}`}>
      <span className="status-dot" aria-hidden="true" />
      {statusLabels[status] ?? statusLabels.unknown}
    </span>
  );
}

function EmptyState({ title, copy }: { title: string; copy: string }) {
  return (
    <div className="empty-state">
      <span className="empty-orbit" aria-hidden="true" />
      <strong>{title}</strong>
      <p>{copy}</p>
    </div>
  );
}

function ConfirmationDialog({
  confirmation,
  busy,
  onClose,
}: {
  confirmation: Confirmation;
  busy: boolean;
  onClose: () => void;
}) {
  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [busy, onClose]);

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={() => !busy && onClose()}>
      <section
        className="modal-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className={`modal-symbol ${confirmation.tone === "warning" ? "warning" : ""}`}>
          {confirmation.tone === "warning" ? "!" : "✓"}
        </div>
        <p className="eyebrow">Confirmation</p>
        <h2 id="confirm-title">{confirmation.title}</h2>
        <p>{confirmation.description}</p>
        <div className="modal-actions">
          <button className="button secondary" type="button" onClick={onClose} disabled={busy}>
            Annuler
          </button>
          <button
            className={`button ${confirmation.tone === "warning" ? "warning-button" : "primary"}`}
            type="button"
            disabled={busy}
            onClick={async () => {
              try {
                await confirmation.action();
                onClose();
              } catch {
                // The dashboard error banner already exposes the manager message.
              }
            }}
          >
            {busy ? "Traitement…" : confirmation.confirmLabel}
          </button>
        </div>
      </section>
    </div>
  );
}

function VersionDialog({
  run,
  checkpoint,
  busy,
  onClose,
  onCreate,
}: {
  run: RunView;
  checkpoint?: JsonRecord;
  busy: boolean;
  onClose: () => void;
  onCreate: (body: JsonRecord) => Promise<void>;
}) {
  const [label, setLabel] = useState(`${run.label} · v2`);
  const [target, setTarget] = useState(String(run.target || 10_000_000));

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    try {
      await onCreate({
        label: label.trim(),
        targetTotalTimesteps: Number(target),
        ...(checkpoint
          ? { checkpoint: stringFrom(checkpoint.path, checkpoint.id, checkpoint.name) }
          : {}),
      });
    } catch {
      // Keep the form open so the user can correct it after reading the banner.
    }
  };

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={() => !busy && onClose()}>
      <form
        className="modal-card version-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="version-title"
        onSubmit={submit}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <p className="eyebrow">Nouvelle branche d'entraînement</p>
        <h2 id="version-title">Créer une version</h2>
        <p>
          La version repartira du {checkpoint ? "checkpoint choisi" : "dernier checkpoint valide"} sans modifier le run actuel.
        </p>
        <label className="field">
          <span>Nom de la version</span>
          <input value={label} onChange={(event) => setLabel(event.target.value)} required maxLength={80} />
        </label>
        <label className="field">
          <span>Objectif total de pas</span>
          <input
            type="number"
            min="1000"
            step="1000"
            value={target}
            onChange={(event) => setTarget(event.target.value)}
            required
          />
        </label>
        <div className="version-source">
          <span>Source</span>
          <strong>{checkpoint ? compactCheckpointName(checkpoint.path ?? checkpoint.name) : "Dernier checkpoint"}</strong>
        </div>
        <div className="modal-actions">
          <button className="button secondary" type="button" onClick={onClose} disabled={busy}>
            Annuler
          </button>
          <button className="button primary" type="submit" disabled={busy || !label.trim()}>
            {busy ? "Création…" : "Créer la version"}
          </button>
        </div>
      </form>
    </div>
  );
}

export function Dashboard() {
  const [token, setToken] = useState("");
  const [view, setView] = useState<ViewName>("overview");
  const [runs, setRuns] = useState<RunView[]>([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const selectedRunIdRef = useRef("");
  const [detail, setDetail] = useState<JsonRecord>({});
  const [metrics, setMetrics] = useState<JsonRecord>({});
  const [logs, setLogs] = useState<string[]>([]);
  const [initialLoading, setInitialLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [actionBusy, setActionBusy] = useState(false);
  const [apiError, setApiError] = useState("");
  const [toast, setToast] = useState("");
  const [confirmation, setConfirmation] = useState<Confirmation | null>(null);
  const [versionSource, setVersionSource] = useState<JsonRecord | null | undefined>(undefined);
  const [replay, setReplay] = useState<ReplayData | null>(null);
  const [replayBusy, setReplayBusy] = useState(false);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const replayPollRef = useRef<number | null>(null);

  const api = useCallback(
    async (path: string, init?: RequestInit) => {
      const headers = new Headers(init?.headers);
      headers.set("Accept", "application/json");
      if (init?.body) headers.set("Content-Type", "application/json");
      if (init?.method && init.method !== "GET" && token) {
        headers.set("X-NextoCR-Token", token);
      }
      const response = await fetch(path, { ...init, headers, cache: "no-store" });
      return parseResponse(response);
    },
    [token],
  );

  const showToast = useCallback((message: string) => {
    setToast(message);
    window.setTimeout(() => setToast(""), 4_000);
  }, []);

  const refreshGlobal = useCallback(
    async (silent = false) => {
      if (!silent) setRefreshing(true);
      try {
        const [stateResult, runsResult] = await Promise.all([
          api("/api/state"),
          api("/api/runs"),
        ]);
        const nextRuns = runsFromPayload(runsResult);
        setRuns(nextRuns);
        const state = record(stateResult);
        const preferredId = stringFrom(
          state.selectedRunId,
          state.activeRunId,
          record(state.selectedRun).id,
          record(state.activeRun).id,
        );
        const currentSelection = selectedRunIdRef.current;
        const validCurrent = nextRuns.some((run) => run.id === currentSelection);
        const nextSelection = validCurrent
          ? currentSelection
          : nextRuns.some((run) => run.id === preferredId)
            ? preferredId
            : nextRuns[0]?.id ?? "";
        if (nextSelection !== currentSelection) {
          selectedRunIdRef.current = nextSelection;
          setSelectedRunId(nextSelection);
        }
        setApiError("");
      } catch (error) {
        setApiError(userError(error, "Le manager local est indisponible."));
      } finally {
        setRefreshing(false);
        setInitialLoading(false);
      }
    },
    [api],
  );

  const refreshSelected = useCallback(
    async (runId: string, silent = false) => {
      if (!runId) return;
      if (!silent) setRefreshing(true);
      const encoded = encodeURIComponent(runId);
      const results = await Promise.allSettled([
        api(`/api/runs/${encoded}`),
        api(`/api/runs/${encoded}/metrics`),
        api(`/api/runs/${encoded}/logs?tail=180`),
      ]);
      const [detailResult, metricsResult, logsResult] = results;
      if (detailResult.status === "fulfilled") setDetail(record(detailResult.value));
      if (metricsResult.status === "fulfilled") setMetrics(record(metricsResult.value));
      if (logsResult.status === "fulfilled") {
        const payload = logsResult.value;
        const groupedLogs = record(record(payload).logs);
        const flattenedLogs = Object.entries(groupedLogs).flatMap(([filename, fileLines]) =>
          array(fileLines).map((line) => `[${filename}] ${String(line)}`),
        );
        const values = Array.isArray(payload)
          ? payload
          : array(record(payload).lines).length
            ? array(record(payload).lines)
            : flattenedLogs.length
              ? flattenedLogs
            : typeof payload === "string"
              ? payload.split(/\r?\n/)
              : [];
        setLogs(values.map((line) => (typeof line === "string" ? line : JSON.stringify(line))));
      }
      const rejected = results.find((result) => result.status === "rejected") as PromiseRejectedResult | undefined;
      if (detailResult.status === "rejected") {
        setApiError(userError(detailResult.reason, "Ce run est indisponible."));
      } else if (rejected && !silent) {
        setApiError(userError(rejected.reason, "Certaines données du run sont indisponibles."));
      } else {
        setApiError("");
      }
      setRefreshing(false);
    },
    [api],
  );

  useEffect(() => {
    let active = true;
    fetch("/api/session", { headers: { Accept: "application/json" }, cache: "no-store" })
      .then(parseResponse)
      .then((payload) => {
        if (!active) return;
        setToken(stringFrom(record(payload).token, payload));
      })
      .catch((error) => {
        if (active) setApiError(userError(error, "La session locale est indisponible."));
      });
    const initialHash = window.location.hash.replace("#", "");
    if (initialHash === "replay") setView("replay");
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    refreshGlobal();
    const interval = window.setInterval(() => refreshGlobal(true), POLL_INTERVAL_MS * 2);
    return () => window.clearInterval(interval);
  }, [refreshGlobal]);

  useEffect(() => {
    if (!selectedRunId) {
      setDetail({});
      setMetrics({});
      setLogs([]);
      return;
    }
    refreshSelected(selectedRunId);
    const interval = window.setInterval(
      () => refreshSelected(selectedRunId, true),
      POLL_INTERVAL_MS,
    );
    return () => window.clearInterval(interval);
  }, [refreshSelected, selectedRunId]);

  useEffect(() => {
    if (!playing || !replay?.frames.length) return;
    let lastAdvance = performance.now();
    let animationFrame = 0;
    const tick = (now: number) => {
      const current = replay.frames[frameIndex];
      const next = replay.frames[frameIndex + 1];
      const frameDelay = next
        ? Math.max(40, ((next.t - current.t) * 1_000) / playbackSpeed)
        : 100;
      if (now - lastAdvance >= frameDelay) {
        lastAdvance = now;
        if (frameIndex >= replay.frames.length - 1) {
          setPlaying(false);
          return;
        }
        setFrameIndex((value) => Math.min(value + 1, replay.frames.length - 1));
      }
      animationFrame = requestAnimationFrame(tick);
    };
    animationFrame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animationFrame);
  }, [frameIndex, playbackSpeed, playing, replay]);

  useEffect(
    () => () => {
      if (replayPollRef.current) window.clearTimeout(replayPollRef.current);
    },
    [],
  );

  const selectedSummary = useMemo(
    () => runs.find((run) => run.id === selectedRunId) ?? null,
    [runs, selectedRunId],
  );
  const run = useMemo(
    () => (selectedSummary ? normaliseRun({ ...selectedSummary.raw, ...detail }) : null),
    [detail, selectedSummary],
  );
  const latest = useMemo(() => latestMetric(metrics, detail), [metrics, detail]);
  const episodes = useMemo(() => metricEpisodes(metrics).slice(-12), [metrics]);
  const checkpoints = useMemo(() => checkpointItems(detail), [detail]);
  const league = useMemo<JsonRecord>(
    () => ({
      ...record(detail.league),
      ...record(metrics.league),
      selections: record(metrics.leagueSelections),
    }),
    [detail, metrics],
  );
  const currentFrame = replay?.frames[Math.min(frameIndex, Math.max(0, replay.frames.length - 1))];
  const nextFrame = replay?.frames[Math.min(frameIndex + 1, Math.max(0, replay.frames.length - 1))];
  const frameTransitionMs = currentFrame && nextFrame
    ? Math.max(60, Math.min(900, ((nextFrame.t - currentFrame.t) * 1_000) / playbackSpeed))
    : 95;
  const progress = run?.target ? Math.max(0, Math.min(1, run.current / run.target)) : 0;
  const actions = record(detail.actions);
  const isRunning = run
    ? Boolean(actions.canPause) || ["running", "starting", "pausing"].includes(run.status)
    : false;
  const isPaused = run
    ? Boolean(actions.canResume) || ["paused", "stopped", "interrupted", "idle", "ready"].includes(run.status)
    : false;

  const perform = useCallback(
    async (action: () => Promise<any>, successMessage: string) => {
      setActionBusy(true);
      setApiError("");
      try {
        const result = await action();
        showToast(successMessage);
        await refreshGlobal(true);
        const nextId = stringFrom(record(result).runId, record(result).id, selectedRunIdRef.current);
        if (nextId) {
          selectedRunIdRef.current = nextId;
          setSelectedRunId(nextId);
          await refreshSelected(nextId, true);
        }
      } catch (error) {
        setApiError(userError(error, "L'action n'a pas pu être effectuée."));
        throw error;
      } finally {
        setActionBusy(false);
      }
    },
    [refreshGlobal, refreshSelected, showToast],
  );

  const requestPrimaryAction = () => {
    if (isRunning && run) {
      setConfirmation({
        title: "Mettre l'entraînement en pause ?",
        description:
          "NextoCR termine l'étape en cours, écrit un checkpoint sûr puis arrête les processus. Vous pourrez éteindre le PC après confirmation.",
        confirmLabel: "Pause + checkpoint",
        tone: "warning",
        action: () =>
          perform(
            () => api(`/api/runs/${encodeURIComponent(run.id)}/pause`, { method: "POST", body: "{}" }),
            "Checkpoint créé, entraînement en pause.",
          ),
      });
      return;
    }

    const canResume = run && isPaused;
    setConfirmation({
      title: canResume ? "Reprendre l'entraînement ?" : "Démarrer l'entraînement ?",
      description: canResume
        ? "Le dernier checkpoint valide sera chargé et le compteur global continuera sans repartir de zéro."
        : "Un nouveau run Mortier en auto-jeu sera créé avec la configuration locale actuelle.",
      confirmLabel: canResume ? "Reprendre" : "Démarrer",
      action: () =>
        perform(
          () =>
            canResume
              ? api(`/api/runs/${encodeURIComponent(run.id)}/resume`, { method: "POST", body: "{}" })
              : api("/api/runs", {
                  method: "POST",
                  body: JSON.stringify({ label: `Mortier self-play · ${formatDate(new Date().toISOString(), false)}` }),
                }),
          canResume ? "Entraînement repris." : "Nouveau run démarré.",
        ),
    });
  };

  const requestCheckpoint = () => {
    if (!run) return;
    setConfirmation({
      title: "Créer un checkpoint maintenant ?",
      description:
        "L'entraînement continue pendant l'écriture. Le modèle, l'optimiseur, la ligue et la configuration seront versionnés ensemble.",
      confirmLabel: "Créer le checkpoint",
      action: () =>
        perform(
          () => api(`/api/runs/${encodeURIComponent(run.id)}/checkpoint`, { method: "POST", body: "{}" }),
          "Checkpoint manuel demandé.",
        ),
    });
  };

  const createVersion = async (body: JsonRecord) => {
    if (!run) return;
    await perform(
      () =>
        api(`/api/runs/${encodeURIComponent(run.id)}/versions`, {
          method: "POST",
          body: JSON.stringify(body),
        }),
      "Nouvelle version créée.",
    );
    setVersionSource(undefined);
  };

  const pollReplay = useCallback(
    async (runId: string, replayId: string) => {
      try {
        const payload = await api(
          `/api/runs/${encodeURIComponent(runId)}/replays/${encodeURIComponent(replayId)}`,
        );
        const next = normaliseReplay(payload, replayId);
        setReplay(next);
        if (["queued", "running", "generating"].includes(next.status)) {
          replayPollRef.current = window.setTimeout(
            () => pollReplay(runId, replayId),
            REPLAY_POLL_INTERVAL_MS,
          );
        } else {
          setReplayBusy(false);
          if (next.status === "ready") {
            setFrameIndex(0);
            showToast("Partie prête à être observée.");
          }
        }
      } catch (error) {
        setReplayBusy(false);
        setReplay((current) => ({
          ...(current ?? { id: replayId, progress: 0, metadata: {}, frames: [] }),
          status: "failed",
          error: userError(error, "La partie est indisponible."),
        }));
      }
    },
    [api, showToast],
  );

  const generateReplay = async () => {
    if (!run) return;
    setView("replay");
    window.location.hash = "replay";
    setPlaying(false);
    setReplayBusy(true);
    setReplay({ id: "pending", status: "queued", progress: 0, metadata: {}, frames: [] });
    try {
      const payload = await api(`/api/runs/${encodeURIComponent(run.id)}/replays`, {
        method: "POST",
        body: JSON.stringify({ opponent: "league", checkpoint: "latest" }),
      });
      const next = normaliseReplay(payload, stringFrom(record(payload).replayId, "pending"));
      setReplay(next);
      if (next.status === "ready") {
        setReplayBusy(false);
        setFrameIndex(0);
      } else {
        await pollReplay(run.id, next.id);
      }
    } catch (error) {
      setReplayBusy(false);
      setReplay({
        id: "failed",
        status: "failed",
        progress: 0,
        metadata: {},
        frames: [],
        error: userError(error, "Impossible de générer la partie."),
      });
    }
  };

  const changeView = (next: ViewName) => {
    setView(next);
    window.location.hash = next === "replay" ? "replay" : "";
  };

  const status = run?.status ?? "idle";
  const outcome = humanOutcome(latest.outcome ?? latest.result ?? detail.lastOutcome);
  const reward = numberFrom(latest.reward, latest.episodeReward, detail.lastReward);
  const elapsed = numberFrom(latest.durationSeconds, latest.elapsedSeconds);
  const episodeLength = numberFrom(latest.length, latest.steps);
  const poolEntries = array(league.entries).length
    ? array(league.entries)
    : array(league.opponents).length
      ? array(league.opponents)
      : array(league.snapshots).length
        ? array(league.snapshots)
        : [
            ...(Object.keys(record(league.initial)).length ? [record(league.initial)] : []),
            ...array(league.recent),
            ...array(league.historical),
          ];
  const poolSize = numberFrom(league.poolSize, league.size, poolEntries.length);
  const leagueSelections = record(league.selections);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div>
            <strong>NextoCR</strong>
            <small>learning lab</small>
          </div>
        </div>

        <nav className="main-nav" aria-label="Navigation principale">
          <button className={view === "overview" ? "active" : ""} onClick={() => changeView("overview")}>
            <span className="nav-glyph grid-glyph" aria-hidden="true" />
            Vue d'ensemble
          </button>
          <button className={view === "replay" ? "active" : ""} onClick={() => changeView("replay")}>
            <span className="nav-glyph replay-glyph" aria-hidden="true" />
            Voir une partie
          </button>
        </nav>

        <div className="sidebar-spacer" />
        <section className="local-card" aria-label="Sécurité locale">
          <span className="local-icon" aria-hidden="true">⌁</span>
          <div>
            <strong>Session locale</strong>
            <p>Accessible uniquement sur ce PC</p>
          </div>
        </section>
        <div className="sidebar-foot">
          <span>Deck</span>
          <strong>Mortier · 8 cartes</strong>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div className="mobile-brand">
            <span className="brand-mark mini" aria-hidden="true"><span /><span /><span /></span>
            <strong>NextoCR</strong>
          </div>
          <div className="run-picker">
            <label htmlFor="run-select">Run actif</label>
            <select
              id="run-select"
              value={selectedRunId}
              onChange={(event) => {
                const next = event.target.value;
                selectedRunIdRef.current = next;
                setSelectedRunId(next);
              }}
              disabled={!runs.length}
            >
              {!runs.length && <option value="">Aucun run</option>}
              {runs.map((item) => (
                <option key={item.id} value={item.id}>{item.label}</option>
              ))}
            </select>
          </div>
          <div className="topbar-status">
            <StatusPill status={status} />
            <button
              className="icon-button"
              type="button"
              aria-label="Actualiser les données"
              title="Actualiser"
              onClick={async () => {
                await refreshGlobal();
                if (selectedRunIdRef.current) await refreshSelected(selectedRunIdRef.current);
              }}
              disabled={refreshing}
            >
              <span className={refreshing ? "spin" : ""} aria-hidden="true">↻</span>
            </button>
          </div>
        </header>

        {apiError && (
          <div className="error-banner" role="alert">
            <span aria-hidden="true">!</span>
            <div><strong>Le manager répond avec une erreur</strong><p>{apiError}</p></div>
            <button aria-label="Fermer l'erreur" onClick={() => setApiError("")}>×</button>
          </div>
        )}

        {initialLoading ? (
          <div className="loading-screen" role="status">
            <div className="loading-mark"><span /><span /><span /></div>
            <strong>Connexion au laboratoire local</strong>
            <p>Lecture des runs et du dernier checkpoint…</p>
          </div>
        ) : view === "overview" ? (
          <div className="content overview-view">
            <section className="page-heading">
              <div>
                <p className="eyebrow">Centre d'entraînement local</p>
                <h1>Le modèle apprend.<br /><span>Vous gardez la main.</span></h1>
                <p className="heading-copy">
                  Suivez l'auto-jeu Mortier, sécurisez chaque avancée et reprenez exactement là où vous vous êtes arrêté.
                </p>
                <p className="continuity-note"><span aria-hidden="true">●</span> Fermer cet onglet n'arrête pas l'entraînement.</p>
              </div>
              <div className="heading-actions">
                <button className="button watch-button" type="button" onClick={generateReplay} disabled={!run || replayBusy}>
                  <span className="play-symbol" aria-hidden="true">▶</span>
                  Voir une partie
                </button>
                <button
                  className={`button ${isRunning ? "pause-button" : "primary"}`}
                  type="button"
                  onClick={requestPrimaryAction}
                  disabled={actionBusy || ["pausing", "stopping", "starting"].includes(status)}
                >
                  <span aria-hidden="true">{isRunning ? "Ⅱ" : "▶"}</span>
                  {isRunning ? "Pause + checkpoint" : isPaused && run ? "Reprendre" : "Démarrer"}
                </button>
              </div>
            </section>

            {!run ? (
              <section className="panel no-run-panel">
                <EmptyState
                  title="Aucun entraînement pour l'instant"
                  copy="Démarrez un premier run pour créer le modèle Mortier et suivre sa progression ici."
                />
                <button className="button primary" onClick={requestPrimaryAction}>Démarrer le premier run</button>
              </section>
            ) : (
              <>
                <section className="progress-panel panel">
                  <div className="progress-topline">
                    <div>
                      <span className="section-label">Progression globale</span>
                      <strong>{formatInteger(run.current)} <small>/ {formatInteger(run.target)} pas</small></strong>
                    </div>
                    <div className="progress-percent">{(progress * 100).toFixed(progress < 0.01 ? 2 : 1)}%</div>
                  </div>
                  <div className="progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={run.target} aria-valuenow={run.current}>
                    <span style={{ width: `${Math.max(progress * 100, progress > 0 ? 0.35 : 0)}%` }} />
                    {[25, 50, 75].map((marker) => <i key={marker} style={{ left: `${marker}%` }} />)}
                  </div>
                  <div className="progress-footer">
                    <div><span>Cadence</span><strong>{formatRate(run.rate)}</strong></div>
                    <div><span>Fin estimée</span><strong>{formatDuration(run.etaSeconds)}</strong></div>
                    <div><span>Dernière synchro</span><strong>{formatDate(run.updatedAt)}</strong></div>
                  </div>
                </section>

                <div className="metrics-grid">
                  <section className="metric-card panel outcome-card">
                    <div className="metric-icon outcome-icon" aria-hidden="true">↗</div>
                    <div>
                      <span>Dernier résultat</span>
                      <strong>{outcome}</strong>
                      <small>{elapsed ? `Partie de ${formatDuration(elapsed)}` : episodeLength ? `${formatInteger(episodeLength)} décisions` : "Dernier épisode terminé"}</small>
                    </div>
                  </section>
                  <section className="metric-card panel reward-card">
                    <div className="metric-icon reward-icon" aria-hidden="true">±</div>
                    <div>
                      <span>Récompense</span>
                      <strong className={reward < 0 ? "negative" : "positive"}>{formatReward(reward)}</strong>
                      <small>{stringFrom(latest.opponent, latest.opponentType, "contre la ligue")}</small>
                    </div>
                  </section>
                  <section className="metric-card panel league-card">
                    <div className="metric-icon league-icon" aria-hidden="true">◎</div>
                    <div>
                      <span>Ligue d'auto-jeu</span>
                      <strong>{formatInteger(poolSize)} adversaires</strong>
                      <small>{formatInteger(numberFrom(league.matches, league.gamesPlayed, leagueSelections.episodes))} matchs échantillonnés</small>
                    </div>
                  </section>
                </div>

                <div className="main-grid">
                  <section className="panel learning-panel">
                    <div className="panel-heading">
                      <div><span className="section-label">Signal d'apprentissage</span><h2>12 derniers épisodes</h2></div>
                      <span className="legend"><i /> récompense</span>
                    </div>
                    {episodes.length ? (
                      <div className="reward-chart" aria-label="Récompenses des derniers épisodes">
                        {episodes.map((episode, index) => {
                          const value = numberFrom(episode.reward, episode.episodeReward);
                          const maxAbs = Math.max(1, ...episodes.map((item) => Math.abs(numberFrom(item.reward, item.episodeReward))));
                          const height = 14 + (Math.abs(value) / maxAbs) * 72;
                          return (
                            <div className="reward-column" key={`${index}-${stringFrom(episode.episode, episode.id)}`} title={`Épisode ${stringFrom(episode.episode, index + 1)} · ${formatReward(value)}`}>
                              <span className={value < 0 ? "negative" : "positive"} style={{ height: `${height}%` }} />
                              <small>{index + 1}</small>
                            </div>
                          );
                        })}
                        <div className="chart-zero" />
                      </div>
                    ) : (
                      <EmptyState title="La courbe arrive bientôt" copy="Elle apparaîtra dès la fin du premier épisode." />
                    )}
                  </section>

                  <section className="panel control-panel">
                    <div className="panel-heading"><div><span className="section-label">Actions sûres</span><h2>Piloter ce run</h2></div></div>
                    <button className="control-row" onClick={requestCheckpoint} disabled={actions.canCheckpoint === false || !isRunning || actionBusy}>
                      <span className="control-symbol save-symbol" aria-hidden="true">↓</span>
                      <span><strong>Checkpoint manuel</strong><small>Sans interrompre l'apprentissage</small></span>
                      <i aria-hidden="true">→</i>
                    </button>
                    <button className="control-row" onClick={() => setVersionSource(null)} disabled={actions.canVersion === false || !checkpoints.length || actionBusy}>
                      <span className="control-symbol branch-symbol" aria-hidden="true">⑂</span>
                      <span><strong>Nouvelle version</strong><small>Repartir du dernier checkpoint</small></span>
                      <i aria-hidden="true">→</i>
                    </button>
                    <button className="control-row" onClick={generateReplay} disabled={actions.canReplay === false || replayBusy}>
                      <span className="control-symbol view-symbol" aria-hidden="true">▶</span>
                      <span><strong>Observer le niveau actuel</strong><small>Simuler depuis le dernier checkpoint</small></span>
                      <i aria-hidden="true">→</i>
                    </button>
                  </section>
                </div>

                <div className="secondary-grid">
                  <section className="panel checkpoints-panel">
                    <div className="panel-heading">
                      <div><span className="section-label">Restaurations</span><h2>Checkpoints</h2></div>
                      <button className="text-button" onClick={requestCheckpoint} disabled={!isRunning}>+ Créer maintenant</button>
                    </div>
                    <div className="table-shell">
                      <table>
                        <thead><tr><th>Fichier</th><th>Pas</th><th>Créé</th><th><span className="sr-only">Action</span></th></tr></thead>
                        <tbody>
                          {checkpoints.slice(0, 6).map((checkpoint, index) => (
                            <tr key={stringFrom(checkpoint.id, checkpoint.path, checkpoint.name, index)}>
                              <td><span className="checkpoint-glyph" aria-hidden="true">◇</span><strong>{compactCheckpointName(checkpoint.path ?? checkpoint.name ?? checkpoint.id)}</strong></td>
                              <td>{formatInteger(numberFrom(checkpoint.timesteps, checkpoint.step))}</td>
                              <td>{formatDate(checkpoint.timestampUtc ?? checkpoint.createdAt ?? checkpoint.created_at ?? checkpoint.mtime)}</td>
                              <td><button className="mini-action" onClick={() => setVersionSource(checkpoint)}>Créer une version</button></td>
                            </tr>
                          ))}
                          {!checkpoints.length && (
                            <tr><td colSpan={4}><div className="inline-empty">Aucun checkpoint disponible pour ce run.</div></td></tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </section>

                  <section className="panel versions-panel">
                    <div className="panel-heading"><div><span className="section-label">Historique</span><h2>Versions & runs</h2></div><span className="count-badge">{runs.length}</span></div>
                    <div className="run-list">
                      {runs.slice(0, 6).map((item) => (
                        <button
                          key={item.id}
                          className={`run-list-item ${item.id === run.id ? "selected" : ""}`}
                          onClick={() => {
                            selectedRunIdRef.current = item.id;
                            setSelectedRunId(item.id);
                          }}
                        >
                          <span className="version-rail"><i /></span>
                          <span className="run-list-copy">
                            <strong>{item.label}</strong>
                            <small>{formatInteger(item.current)} pas · {formatDate(item.updatedAt)}</small>
                          </span>
                          <span className={`micro-status ${statusClass(item.status)}`}>{statusLabels[item.status] ?? item.status}</span>
                        </button>
                      ))}
                    </div>
                  </section>
                </div>

                <section className="panel logs-panel">
                  <div className="panel-heading">
                    <div><span className="section-label">Sortie temps réel</span><h2>Journal du trainer</h2></div>
                    <span className="terminal-live"><i /> suivi automatique</span>
                  </div>
                  <div className="terminal" role="log" aria-live="polite">
                    {logs.length ? logs.slice(-90).map((line, index) => {
                      const level = /error|exception|failed/i.test(line) ? "log-error" : /warn/i.test(line) ? "log-warn" : "";
                      return <div key={`${index}-${line.slice(0, 20)}`} className={level}><span>{String(index + Math.max(1, logs.length - 89)).padStart(3, "0")}</span><code>{line}</code></div>;
                    }) : <div className="terminal-empty">En attente des premières lignes du trainer…</div>}
                  </div>
                </section>
              </>
            )}
          </div>
        ) : (
          <div className="content replay-view">
            <section className="page-heading replay-heading">
              <div>
                <p className="eyebrow">Observation du modèle</p>
                <h1>Voir une partie.<br /><span>Comprendre le niveau.</span></h1>
                <p className="heading-copy">Une simulation locale depuis le dernier checkpoint, sans interaction avec le client du jeu.</p>
              </div>
              <button className="button primary" onClick={generateReplay} disabled={!run || replayBusy}>
                <span aria-hidden="true">↻</span>{replayBusy ? "Génération…" : replay?.frames.length ? "Nouvelle partie" : "Générer une partie"}
              </button>
            </section>

            <div className="replay-layout">
              <section className="panel arena-panel">
                <div className="arena-toolbar">
                  <div><span className="section-label">Simulation abstraite</span><strong>{run?.label ?? "Aucun run"}</strong></div>
                  {replay && <span className={`replay-state ${replay.status}`}>{replay.status === "ready" ? "Prête" : replay.status === "failed" ? "Échec" : "Calcul en cours"}</span>}
                </div>

                {replayBusy || (replay && ["queued", "running", "generating"].includes(replay.status)) ? (
                  <div className="replay-generating" role="status">
                    <div className="generation-orbit"><span /><i /></div>
                    <strong>Le modèle joue sa partie</strong>
                    <p>Chargement du checkpoint et simulation contre la ligue…</p>
                    <div className="generation-track"><span style={{ width: `${Math.max(8, replay?.progress ? replay.progress * 100 : 18)}%` }} /></div>
                  </div>
                ) : replay?.status === "failed" ? (
                  <div className="replay-generating error-replay">
                    <div className="modal-symbol warning">!</div>
                    <strong>Partie indisponible</strong>
                    <p>{replay.error ?? "La génération du replay a échoué."}</p>
                    <button className="button secondary" onClick={generateReplay}>Réessayer</button>
                  </div>
                ) : (
                  <>
                    <div
                      className={`arena ${!currentFrame ? "arena-empty" : ""}`}
                      style={{ "--frame-duration": `${frameTransitionMs}ms` } as CSSProperties}
                    >
                      <div className="arena-grid" aria-hidden="true" />
                      <div className="river" aria-hidden="true"><i /><i /></div>
                      {(currentFrame?.towers ?? defaultTowers()).map((tower) => {
                        const health = Math.max(0, Math.min(1, tower.hp / tower.maxHp));
                        const { label, badge } = describeUnit(tower.kind);
                        const caption = `${label} · ${formatInteger(tower.hp)} / ${formatInteger(tower.maxHp)} PV`;
                        return (
                          <div
                            className={`arena-tower ${tower.team}`}
                            key={tower.id}
                            style={{ left: `${tower.x * 100}%`, top: `${tower.y * 100}%` }}
                            data-label={caption}
                            role="img"
                            aria-label={caption}
                          >
                            <span>{badge}</span>
                            <i><b style={{ width: `${health * 100}%` }} /></i>
                          </div>
                        );
                      })}
                      {currentFrame?.entities.map((entity) => {
                        const health = Math.max(0, Math.min(1, entity.hp / entity.maxHp));
                        const { label, badge } = describeUnit(entity.kind);
                        const side = entity.team === "red" ? "Adversaire" : "NextoCR";
                        const caption = `${label} · ${side} · ${formatInteger(entity.hp)} / ${formatInteger(entity.maxHp)} PV`;
                        return (
                          <div
                            className={`arena-entity ${entity.team}`}
                            key={entity.id}
                            style={{
                              left: `${entity.x * 100}%`,
                              top: `${entity.y * 100}%`,
                              "--health": `${health * 100}%`,
                            } as CSSProperties}
                            data-label={caption}
                            role="img"
                            aria-label={caption}
                          >
                            <span>{badge}</span>
                          </div>
                        );
                      })}
                      {!currentFrame && (
                        <div className="arena-empty-copy">
                          <span className="play-orbit" aria-hidden="true">▶</span>
                          <strong>Aucune partie chargée</strong>
                          <p>Générez une simulation pour observer les décisions du checkpoint actuel.</p>
                          <button className="button primary" onClick={generateReplay} disabled={!run}>Générer maintenant</button>
                        </div>
                      )}
                    </div>

                    <div className="playback-controls">
                      <button
                        className="play-button"
                        type="button"
                        disabled={!replay?.frames.length}
                        aria-label={playing ? "Mettre le replay en pause" : "Lire le replay"}
                        onClick={() => {
                          if (frameIndex >= (replay?.frames.length ?? 1) - 1) setFrameIndex(0);
                          setPlaying((value) => !value);
                        }}
                      >{playing ? "Ⅱ" : "▶"}</button>
                      <span className="play-time">{formatClock(currentFrame?.t)}</span>
                      <input
                        className="timeline"
                        aria-label="Position dans la partie"
                        type="range"
                        min={0}
                        max={Math.max(0, (replay?.frames.length ?? 1) - 1)}
                        value={Math.min(frameIndex, Math.max(0, (replay?.frames.length ?? 1) - 1))}
                        disabled={!replay?.frames.length}
                        onChange={(event) => {
                          setPlaying(false);
                          setFrameIndex(Number(event.target.value));
                        }}
                        style={{ "--timeline-progress": `${replay?.frames.length ? (frameIndex / Math.max(1, replay.frames.length - 1)) * 100 : 0}%` } as CSSProperties}
                      />
                      <span className="play-time">{formatClock(replay?.frames.at(-1)?.t)}</span>
                      <div className="speed-control" aria-label="Vitesse de lecture">
                        {[0.5, 1, 2, 4].map((speed) => (
                          <button className={playbackSpeed === speed ? "active" : ""} key={speed} onClick={() => setPlaybackSpeed(speed)}>{speed}×</button>
                        ))}
                      </div>
                    </div>
                  </>
                )}
              </section>

              <aside className="replay-aside">
                <section className="panel replay-meta">
                  <div className="panel-heading"><div><span className="section-label">Contexte</span><h2>Fiche de partie</h2></div></div>
                  <dl>
                    <div><dt>Checkpoint</dt><dd>{compactCheckpointName(replay?.metadata.checkpoint ?? checkpoints[0]?.path ?? "—")}</dd></div>
                    <div><dt>Niveau du checkpoint</dt><dd>{formatInteger(numberFrom(replay?.metadata.checkpointTimesteps, replay?.metadata.timesteps, run?.current))} pas</dd></div>
                    <div><dt>Run courant</dt><dd>{formatInteger(numberFrom(replay?.metadata.currentTimesteps, run?.current))} pas</dd></div>
                    <div>
                      <dt>Retard sur le run</dt>
                      <dd className={numberFrom(replay?.metadata.checkpointLagTimesteps) > 0 ? "lag-value" : ""}>
                        {formatInteger(numberFrom(replay?.metadata.checkpointLagTimesteps))} pas
                      </dd>
                    </div>
                    <div><dt>Seed</dt><dd className="mono">{stringFrom(replay?.metadata.seed, "—")}</dd></div>
                    <div><dt>Adversaire</dt><dd>{stringFrom(replay?.metadata.opponent, "Ligue auto-jeu")}</dd></div>
                    <div><dt>Issue</dt><dd>{humanOutcome(replay?.metadata.outcome)}</dd></div>
                    <div><dt>Récompense</dt><dd className={numberFrom(replay?.metadata.reward) < 0 ? "negative" : "positive"}>{replay?.metadata.reward == null ? "—" : formatReward(numberFrom(replay.metadata.reward))}</dd></div>
                  </dl>
                </section>
                <section className="panel replay-legend">
                  <div className="panel-heading"><div><span className="section-label">Lecture</span><h2>Légende</h2></div></div>
                  <div className="legend-row"><i className="blue-unit" /><span><strong>NextoCR</strong><small>Deck Mortier entraîné</small></span></div>
                  <div className="legend-row"><i className="red-unit" /><span><strong>Adversaire</strong><small>Échantillon de la ligue</small></span></div>
                  <p>Les formes représentent les entités du simulateur. Aucun visuel du jeu original n'est utilisé.</p>
                </section>
              </aside>
            </div>
          </div>
        )}

        <nav className="mobile-nav" aria-label="Navigation mobile">
          <button className={view === "overview" ? "active" : ""} onClick={() => changeView("overview")}><span className="nav-glyph grid-glyph" />Vue d'ensemble</button>
          <button className={view === "replay" ? "active" : ""} onClick={() => changeView("replay")}><span className="nav-glyph replay-glyph" />Voir une partie</button>
        </nav>
      </main>

      {toast && <div className="toast" role="status"><span>✓</span>{toast}</div>}
      {confirmation && (
        <ConfirmationDialog confirmation={confirmation} busy={actionBusy} onClose={() => setConfirmation(null)} />
      )}
      {versionSource !== undefined && run && (
        <VersionDialog
          run={run}
          checkpoint={versionSource ?? undefined}
          busy={actionBusy}
          onClose={() => setVersionSource(undefined)}
          onCreate={createVersion}
        />
      )}
    </div>
  );
}
