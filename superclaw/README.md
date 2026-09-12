# superclaw

The harness layer. Everything around the model that is not the model.

supergraph has exactly two parts:

| Part | Path | Owns |
|---|---|---|
| **supergraph** | `src/supergraph/` | the cognitive substrate - storage, DSL, retrieval, memory, ingestion |
| **superclaw** | `superclaw/` | the control plane - identity, permission, modes, planning, compaction, delegation, completion |

Nothing else. The earlier nine-component design (`power`, `forge`, `regate`,
`temporal`, `cloak`, `pulse`, `whitebox`) is gone; whatever those layers were
going to do is now one of these two parts' problem.

## Status

Scaffold. Nothing here is built yet.

## What superclaw owns

A harness is twelve layers. supergraph already answers five of them, so
superclaw's scope is the other seven:

| Layer | Owner |
|---|---|
| Identity | superclaw |
| Output channel | superclaw |
| Tool protocol | superclaw |
| Edit primitive | superclaw |
| Shell model | superclaw |
| **Permission** | **superclaw** - nothing else can own this |
| **Modes** (plan vs act) | **superclaw** |
| **Planning state** | **superclaw**, stored in supergraph |
| Memory | **supergraph** - `ASSERT`/`RETRACT`, TTL, `SYS CONSOLIDATE` |
| **Compaction** | **superclaw** summarises, supergraph stores |
| **Sub-agents** | **superclaw** |
| **Completion** | **superclaw** |

## Design constraints

Three of these are settled and should not be relitigated without evidence.

**A harness is a swarm of narrow prompts, not one system prompt.** The 2026
flagships ship a dozen or more separate model calls, each with a rigid output
shape and a hard scope: a goal driver, an independent verifier, a compaction
summariser, a memory extractor, a memory selector, a title generator. The main
context never carries side work. superclaw should be built as
`superclaw/prompts/*`, one file per call, not as a growing preamble.

**Completion is adversarial.** The verifier is a separate read-only call that
may use no tools and must be able to fail. Passing tests, a finished plan, a
complete manifest and visible effort are not evidence unless they cover every
named requirement. The working turn may not mark itself done.

**Sub-agents default to off.** Delegation is the last thing to build, not the
first, and the shape that works is narrow: read-only, returns text rather than
files, spawned only when explicitly asked. Fan-out by default is the
best-documented way to lose the ability to see and correct the work.

## What supergraph gives superclaw for free

The prompt-layer harnesses cannot date a stored fact or expire one; every one
of them carries a staleness warning and none carries a mechanism. supergraph
already has `__created_at__`, `__updated_at__`, `__event_at__`, `__expires_at__`
with `SYS EXPIRE`, soft `RETRACT` that preserves history, `SYS CONTRADICTIONS`
for detection, and `SYS CONSOLIDATE` for episodic-to-semantic promotion that is
additive rather than lossy.

That is the reason the harness lives next to the substrate instead of on top of
a vector store.

## Known gaps to close first

| Gap | Why it blocks superclaw |
|---|---|
| No spend ceiling in `IngestConfig` | superclaw cannot trade money against certainty if the substrate has no budget to report |
| No `__origin__` class on facts | superclaw cannot tell a user-stated fact from a model-inferred one, so it cannot apply an origin test |
| No `__invalid_at__` | beliefs are overwritten in place by `ASSERT`, so there is no invalidation window to reason over |
