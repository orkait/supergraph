// Categorized examples covering every cluster of the supergraph DSL.
// Each example is a small, runnable script that doubles as a teaching
// moment: the description explains WHY, the script shows HOW.
//
// To add a category, push a new object into `categories`. Each category
// holds 2-5 examples; we deliberately keep it small so the picker stays
// scannable.

export type Example = {
  id: string
  name: string
  description: string
  script: string
  // True when the example seeds graph state and benefits from a reset
  // before loading. Toolbar honors this hint.
  resetsGraph?: boolean
}

export type ExampleCategory = {
  id: string
  name: string
  description: string
  examples: Example[]
}

import { functionCallGraph } from './function-call-graph'
import { classHierarchy } from './class-hierarchy'
import { codeGraph } from './code-graph'
import { microservicesMap } from './microservices-map'

// ------------------------------------------------------------
// 1. Foundations
// ------------------------------------------------------------

const foundationsCreate: Example = {
  id: 'foundations.create',
  name: 'Create + retrieve a node',
  description: 'CREATE NODE with a DOCUMENT clause; the smallest write that shows up in the graph.',
  resetsGraph: true,
  script: `// CREATE NODE writes typed columns; DOCUMENT also fills the vector
// + FTS5 index in one shot. Without DOCUMENT a node is structured
// data only and won't show up in REMEMBER / LEXICAL.
CREATE NODE "mem:paris" kind = "memory" topic = "geography" DOCUMENT "Paris is the capital of France, famous for the Eiffel Tower."

NODE "mem:paris"`,
}

const foundationsCreateBatch: Example = {
  id: 'foundations.batch',
  name: 'Batched create with rollback',
  description: 'BEGIN ... COMMIT runs as one atomic batch. Rollback fires automatically on any error.',
  resetsGraph: true,
  script: `BEGIN
CREATE NODE "mem:paris" kind = "memory" DOCUMENT "Paris is the capital of France."
CREATE NODE "mem:rome"  kind = "memory" DOCUMENT "Rome is the capital of Italy."
CREATE NODE "mem:berlin" kind = "memory" DOCUMENT "Berlin is the capital of Germany."
CREATE EDGE "mem:paris"  -> "mem:rome"   kind = "both_european_capitals"
CREATE EDGE "mem:paris"  -> "mem:berlin" kind = "both_european_capitals"
CREATE EDGE "mem:rome"   -> "mem:berlin" kind = "both_european_capitals"
COMMIT

NODES WHERE kind = "memory" LIMIT 5`,
}

const foundationsUpdate: Example = {
  id: 'foundations.update',
  name: 'Update + delete',
  description: 'UPDATE NODE patches typed fields; DELETE NODES with WHERE removes by predicate.',
  script: `CREATE NODE "draft:1" kind = "draft" status = "open" DOCUMENT "First draft."

UPDATE NODE "draft:1" SET status = "shipped"

NODE "draft:1"`,
}

// ------------------------------------------------------------
// 2. Retrieval (REMEMBER fusion + single-leg verbs)
// ------------------------------------------------------------

const retrievalRemember: Example = {
  id: 'retrieval.remember',
  name: 'REMEMBER hybrid fusion',
  description: 'REMEMBER fuses vector cosine + BM25 + recency + graph signal. Each result carries per-signal scores.',
  resetsGraph: true,
  script: `BEGIN
CREATE NODE "mem:croissant" kind = "memory" DOCUMENT "I had a buttery croissant in Paris last Tuesday."
CREATE NODE "mem:colosseum" kind = "memory" DOCUMENT "The Colosseum in Rome is older than I remembered."
CREATE NODE "mem:pasta"     kind = "memory" DOCUMENT "Carbonara with guanciale is the canonical Roman pasta."
CREATE NODE "mem:metro"     kind = "memory" DOCUMENT "Paris Metro line 1 runs east-west across the city."
COMMIT

REMEMBER "Italian food" LIMIT 3`,
}

const retrievalSimilar: Example = {
  id: 'retrieval.similar',
  name: 'SIMILAR TO (vector only)',
  description: 'SIMILAR TO is the pure vector leg. Returns nearest neighbors by cosine, ignoring text + recency + graph.',
  script: `SIMILAR TO "European architecture" LIMIT 3`,
}

const retrievalLexical: Example = {
  id: 'retrieval.lexical',
  name: 'LEXICAL SEARCH (BM25 only)',
  description: 'BM25 over the FTS5 index. Catches keyword matches that vector cosine smooths over.',
  script: `LEXICAL SEARCH "Paris" LIMIT 5`,
}

const retrievalRecall: Example = {
  id: 'retrieval.recall',
  name: 'RECALL FROM (graph walk)',
  description: 'RECALL spreads activation outward from an anchor node through edges, weighting by edge degree.',
  script: `RECALL FROM "mem:paris" DEPTH 2 LIMIT 5`,
}

const retrievalAnswer: Example = {
  id: 'retrieval.answer',
  name: 'ANSWER (retrieval + reader)',
  description: 'ANSWER pulls top-K via REMEMBER then synthesizes a short answer through the configured reader.',
  script: `// Requires SuperGraph(reader=...) wired in the host. Without it,
// the verb errors with a clear "no reader configured" message.
ANSWER "What is the capital of France?" LIMIT 3`,
}

// ------------------------------------------------------------
// 3. Beliefs & Facts
// ------------------------------------------------------------

const beliefsAssertRetract: Example = {
  id: 'beliefs.assert',
  name: 'ASSERT belief, then RETRACT',
  description: 'ASSERT writes a fact node with confidence + source. RETRACT supersedes it with a reason.',
  resetsGraph: true,
  script: `ASSERT "fact:lunch_spot" kind = "belief" value = "Cafe Paloma" CONFIDENCE 0.9 SOURCE "msg:42"

RETRACT "fact:lunch_spot" REASON "user said 'I go to Cafe Centro now'"

ASSERT "fact:lunch_spot" kind = "belief" value = "Cafe Centro" CONFIDENCE 0.95 SOURCE "msg:43"

NODE "fact:lunch_spot"`,
}

const beliefsContradict: Example = {
  id: 'beliefs.contradictions',
  name: 'Find contradictions',
  description: 'SYS CONTRADICTIONS surfaces beliefs whose value flipped without a RETRACT.',
  script: `SYS CONTRADICTIONS WHERE kind = "belief" FIELD value GROUP BY topic`,
}

// ------------------------------------------------------------
// 4. Graph Algorithms
// ------------------------------------------------------------

const algoTraverse: Example = {
  id: 'algo.traverse',
  name: 'TRAVERSE depth-bounded',
  description: 'BFS outward from an anchor, respecting depth + WHERE filters. Useful for "everything reachable in 2 hops."',
  script: `TRAVERSE FROM "mem:paris" DEPTH 2 LIMIT 20`,
}

const algoPath: Example = {
  id: 'algo.path',
  name: 'PATH FROM ... TO',
  description: 'Single shortest hop sequence. SHORTEST PATH and PATHS (plural) cover weighted + multi-path variants.',
  script: `PATH FROM "mem:paris" TO "mem:berlin" MAX_DEPTH 5`,
}

const algoCommonNeighbors: Example = {
  id: 'algo.common',
  name: 'COMMON NEIGHBORS',
  description: 'Set intersection of two nodes\' adjacency. Surfaces shared associations in agent memory.',
  script: `COMMON NEIGHBORS OF "mem:paris" AND "mem:berlin"`,
}

const algoAncestors: Example = {
  id: 'algo.ancestors',
  name: 'ANCESTORS / DESCENDANTS',
  description: 'Directed traversal up or down the edge DAG. Honors edge direction unlike TRAVERSE.',
  script: `ANCESTORS OF "fn_login" DEPTH 3`,
}

// ------------------------------------------------------------
// 5. Aggregation & Match
// ------------------------------------------------------------

const aggCount: Example = {
  id: 'agg.count',
  name: 'COUNT NODES / EDGES',
  description: 'Cheapest read for cardinality. WHERE filters apply.',
  script: `COUNT NODES WHERE kind = "memory"

COUNT EDGES`,
}

const aggGroupBy: Example = {
  id: 'agg.groupby',
  name: 'AGGREGATE GROUP BY',
  description: 'AGGREGATE NODES grouped by a typed column with COUNT() / SUM() / AVG().',
  script: `AGGREGATE NODES GROUP BY kind SELECT COUNT() ORDER BY COUNT() DESC LIMIT 10`,
}

const aggMatch: Example = {
  id: 'agg.match',
  name: 'MATCH pattern',
  description: 'Cypher-style multi-hop pattern matching. Each step is bound to a node with optional WHERE.',
  script: `MATCH ("mem:paris") -[]-> ("mem:berlin") LIMIT 5`,
}

// ------------------------------------------------------------
// 6. Time & Lifecycle
// ------------------------------------------------------------

const timeEventAt: Example = {
  id: 'time.event_at',
  name: 'EVENT_AT + recency',
  description: 'EVENT_AT pins the canonical timestamp; REMEMBER recency decays exponentially from it.',
  resetsGraph: true,
  script: `// EVENT_AT seconds-since-epoch. Use NOW() in WHERE for relative time.
CREATE NODE "mem:fresh"   kind = "memory" DOCUMENT "Just happened."   EVENT_AT 1730000000
CREATE NODE "mem:stale"   kind = "memory" DOCUMENT "Six months ago."  EVENT_AT 1714000000
CREATE NODE "mem:ancient" kind = "memory" DOCUMENT "Two years ago."   EVENT_AT 1666000000

REMEMBER "anything" LIMIT 5`,
}

const timeExpires: Example = {
  id: 'time.expires',
  name: 'EXPIRES IN',
  description: 'Node TTL. SYS EXPIRE garbage-collects expired nodes.',
  script: `CREATE NODE "session:tmp" kind = "session" DOCUMENT "Ephemeral context." EXPIRES IN 5 MINUTES`,
}

// ------------------------------------------------------------
// 7. Schema
// ------------------------------------------------------------

const schemaRegisterNode: Example = {
  id: 'schema.register_node',
  name: 'Register a typed node kind',
  description: 'Strict schemas: declared required + optional fields, optional EMBED hint for the auto-embed column.',
  script: `SYS REGISTER NODE KIND "task" REQUIRED title : string, status : string OPTIONAL due_date : string EMBED title

SYS DESCRIBE NODE "task"`,
}

const schemaRegisterEdge: Example = {
  id: 'schema.register_edge',
  name: 'Register a typed edge kind',
  description: 'Constrains edges to declared (from-kind, to-kind) pairs.',
  script: `SYS REGISTER EDGE KIND "assigned_to" FROM "task" TO "person"`,
}

// ------------------------------------------------------------
// 8. Vault (markdown notes)
// ------------------------------------------------------------

const vaultBasic: Example = {
  id: 'vault.basic',
  name: 'VAULT NEW + READ',
  description: 'Create a markdown note in the vault, then read it back. Notes auto-embed for VAULT SEARCH.',
  script: `VAULT NEW "meeting-2026-05-02" KIND "meeting" TAGS "team,planning"

VAULT WRITE "meeting-2026-05-02" SECTION "agenda" CONTENT "1. Standup\\n2. Roadmap review"

VAULT READ "meeting-2026-05-02"`,
}

const vaultSearch: Example = {
  id: 'vault.search',
  name: 'VAULT SEARCH + BACKLINKS',
  description: 'Hybrid vector+text search across vault notes. BACKLINKS lists notes that reference a target.',
  script: `VAULT SEARCH "roadmap" LIMIT 5

VAULT BACKLINKS "meeting-2026-05-02"`,
}

const vaultSync: Example = {
  id: 'vault.sync',
  name: 'VAULT SYNC + DAILY',
  description: 'SYNC reconciles the on-disk markdown directory with the graph; DAILY creates today\'s note.',
  script: `VAULT SYNC

VAULT DAILY`,
}

// ------------------------------------------------------------
// 9. Bulk & Ingest
// ------------------------------------------------------------

const ingestText: Example = {
  id: 'ingest.text',
  name: 'INGEST a text file',
  description: 'INGEST routes through the tiered pipeline. Plain text and markdown are direct; PDFs use pymupdf4llm.',
  script: `// File path is interpreted server-side. Use a path the server has access to.
INGEST "./docs/example.md" AS "doc:example" KIND "doc"`,
}

const ingestVision: Example = {
  id: 'ingest.vision',
  name: 'INGEST USING VISION',
  description: 'Image / scanned-PDF caption via the local llama.cpp VLM sidecar (supergraph vision serve).',
  script: `INGEST "./screenshots/dashboard.png" USING VISION "smolvlm2-2.2b" AS "img:dashboard"`,
}

const ingestMerge: Example = {
  id: 'ingest.merge',
  name: 'MERGE duplicate nodes',
  description: 'MERGE folds one node\'s edges + fields into another. Self-merge is rejected.',
  script: `MERGE NODE "mem:paris-fr" INTO "mem:paris"`,
}

// ------------------------------------------------------------
// 10. System
// ------------------------------------------------------------

const sysStats: Example = {
  id: 'sys.stats',
  name: 'SYS STATS / KINDS',
  description: 'High-level counters. STATS gives totals; KINDS lists all node + edge kinds in use.',
  script: `SYS STATS

SYS KINDS`,
}

const sysExplain: Example = {
  id: 'sys.explain',
  name: 'SYS EXPLAIN REMEMBER',
  description: 'Dry-runs a retrieval pipeline without mutating recall counters. Returns per-signal scores + meta.',
  script: `SYS EXPLAIN REMEMBER "Italian food" LIMIT 3`,
}

const sysSnapshot: Example = {
  id: 'sys.snapshot',
  name: 'SYS SNAPSHOT + ROLLBACK',
  description: 'Named graph snapshots. ROLLBACK TO restores by name. File-based, lightweight - not a full backup.',
  script: `SYS SNAPSHOT "before-experiment"

// ... experiments here ...

SYS ROLLBACK TO "before-experiment"`,
}

const sysHealth: Example = {
  id: 'sys.health',
  name: 'SYS HEALTH / STATUS',
  description: 'Operational health: WAL depth, vector index size, embedder + reranker status.',
  script: `SYS HEALTH

SYS STATUS`,
}

const sysOptimize: Example = {
  id: 'sys.optimize',
  name: 'SYS OPTIMIZE / REBUILD INDICES',
  description: 'Maintenance: vacuum the SQLite blob store + rebuild the FTS5 index in place.',
  script: `SYS OPTIMIZE

SYS REBUILD INDICES`,
}

// ------------------------------------------------------------
// 11. Observability
// ------------------------------------------------------------

const obsSlowQueries: Example = {
  id: 'obs.slow',
  name: 'SYS SLOW / FREQUENT / FAILED',
  description: 'Per-query telemetry: slowest by p99, most frequent by count, recently failed by traceback.',
  script: `SYS SLOW QUERIES SINCE "1h" LIMIT 10

SYS FREQUENT QUERIES LIMIT 10

SYS FAILED QUERIES LIMIT 5`,
}

const obsLog: Example = {
  id: 'obs.log',
  name: 'SYS LOG (filtered)',
  description: 'Tail the structured event log. WHERE filters by node id or kind.',
  script: `SYS LOG WHERE kind = "memory" LIMIT 20`,
}

// ------------------------------------------------------------
// 12. Cron
// ------------------------------------------------------------

const cronAdd: Example = {
  id: 'cron.add',
  name: 'SYS CRON ADD',
  description: 'Schedule a DSL query on a cron expression. Useful for periodic VAULT SYNC or SYS CONNECT.',
  script: `SYS CRON ADD "nightly-sync" SCHEDULE "0 2 * * *" QUERY "VAULT SYNC"

SYS CRON LIST`,
}

const cronRun: Example = {
  id: 'cron.run',
  name: 'SYS CRON RUN (manual)',
  description: 'Fire a scheduled job once, immediately. Verifies the query before letting the scheduler own it.',
  script: `SYS CRON RUN "nightly-sync"`,
}

// ------------------------------------------------------------
// 13. Evolve
// ------------------------------------------------------------

const evolveRule: Example = {
  id: 'evolve.rule',
  name: 'SYS EVOLVE RULE',
  description: 'Self-tuning fusion weights: WHEN a metric crosses a threshold, THEN adjust the weight vector.',
  script: `SYS EVOLVE RULE "promote-recency-when-stale"
WHEN avg_age_days > 30
THEN SET remember_weights = [0.45, 0.20, 0.30, 0.05]
COOLDOWN 3600
PRIORITY 5

SYS EVOLVE LIST`,
}

// ------------------------------------------------------------
// Categories
// ------------------------------------------------------------

export const categories: ExampleCategory[] = [
  {
    id: 'foundations',
    name: 'Foundations',
    description: 'CREATE / UPDATE / DELETE - the smallest write that shows up in the graph.',
    examples: [foundationsCreate, foundationsCreateBatch, foundationsUpdate],
  },
  {
    id: 'retrieval',
    name: 'Retrieval',
    description: 'REMEMBER hybrid fusion plus the single-leg verbs (SIMILAR, LEXICAL, RECALL, ANSWER).',
    examples: [retrievalRemember, retrievalSimilar, retrievalLexical, retrievalRecall, retrievalAnswer],
  },
  {
    id: 'beliefs',
    name: 'Beliefs & Facts',
    description: 'ASSERT / RETRACT with confidence + source. The agent-memory backbone.',
    examples: [beliefsAssertRetract, beliefsContradict],
  },
  {
    id: 'algo',
    name: 'Graph Algorithms',
    description: 'TRAVERSE, PATH, ANCESTORS, COMMON NEIGHBORS - graph reasoning over edges.',
    examples: [algoTraverse, algoPath, algoCommonNeighbors, algoAncestors],
  },
  {
    id: 'agg',
    name: 'Aggregation & Match',
    description: 'COUNT, AGGREGATE GROUP BY, MATCH patterns. Cypher-flavored reads.',
    examples: [aggCount, aggGroupBy, aggMatch],
  },
  {
    id: 'time',
    name: 'Time & Lifecycle',
    description: 'EVENT_AT, EXPIRES, recency decay - first-class time semantics.',
    examples: [timeEventAt, timeExpires],
  },
  {
    id: 'schema',
    name: 'Schema',
    description: 'SYS REGISTER NODE / EDGE KIND - typed columns, validation, embed hints.',
    examples: [schemaRegisterNode, schemaRegisterEdge],
  },
  {
    id: 'vault',
    name: 'Vault (markdown)',
    description: 'VAULT NEW / READ / WRITE / SEARCH / BACKLINKS - markdown notes integrated with the graph.',
    examples: [vaultBasic, vaultSearch, vaultSync],
  },
  {
    id: 'ingest',
    name: 'Bulk & Ingest',
    description: 'INGEST text / PDF / image / audio. MERGE duplicate folds.',
    examples: [ingestText, ingestVision, ingestMerge],
  },
  {
    id: 'sys',
    name: 'System',
    description: 'SYS STATS / EXPLAIN / SNAPSHOT / HEALTH / OPTIMIZE - operational verbs.',
    examples: [sysStats, sysExplain, sysSnapshot, sysHealth, sysOptimize],
  },
  {
    id: 'obs',
    name: 'Observability',
    description: 'SYS SLOW / FREQUENT / FAILED queries + structured log tail.',
    examples: [obsSlowQueries, obsLog],
  },
  {
    id: 'cron',
    name: 'Cron',
    description: 'SYS CRON ADD / LIST / RUN - schedule DSL on a cron expression.',
    examples: [cronAdd, cronRun],
  },
  {
    id: 'evolve',
    name: 'Evolve',
    description: 'SYS EVOLVE RULE - self-tuning fusion weights via measured metrics.',
    examples: [evolveRule],
  },
  {
    id: 'demos',
    name: 'Code analysis demos',
    description: 'Pre-baked graphs that exercise non-agent use cases (call graphs, class hierarchies, microservices).',
    examples: [
      { id: 'demo.call', name: functionCallGraph.name, description: functionCallGraph.description, script: functionCallGraph.script, resetsGraph: true },
      { id: 'demo.class', name: classHierarchy.name, description: classHierarchy.description, script: classHierarchy.script, resetsGraph: true },
      { id: 'demo.code', name: codeGraph.name, description: codeGraph.description, script: codeGraph.script, resetsGraph: true },
      { id: 'demo.microservices', name: microservicesMap.name, description: microservicesMap.description, script: microservicesMap.script, resetsGraph: true },
    ],
  },
]

// Backward-compat: flat list still exported. Any consumer that imported
// `examples` keeps working until it migrates to `categories`.
export const examples = categories.flatMap((c) => c.examples)
