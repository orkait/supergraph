---
name: supergraph-bonsai-dsl
description: Minimal ingest-only DSL cheat sheet for tiny local LLMs (Ternary-Bonsai 1.7B / 4B / 8B GGUF). Uses few-shot examples instead of reference text. Pair with GBNF grammar-constrained decoding for guaranteed-parseable output.
compatibility: supergraph >= 0.4.0
metadata:
  author: orkait
  version: "2.0"
  target_tokens: 400
---

Convert one user message into SuperGraph DSL. Output **only** DSL lines. No prose, no markdown, no `<think>` tags.

Valid verbs:
```
CREATE NODE "id" kind = "K" f = v ... [DOCUMENT "text"]
UPSERT NODE "id" kind = "K" name = "N"
CREATE EDGE "a" -> "b" kind = "K"
ASSERT "fact:X" kind = "belief" value = "V" CONFIDENCE 0.9 SOURCE "msg:id"
RETRACT "fact:X" REASON "why"
```

Escape `"` as `\"` inside strings.

Emit each entity once per message. Entity id = `ent:<lower_slug>`. Belief id = `fact:<topic>`.

Required per message (copy this pattern):
1. One `CREATE NODE "m:<session>:<idx>" kind = "message" session = "..." role = "..." DOCUMENT "..."`
2. For every person / org / named thing in the text: one `UPSERT NODE "ent:<slug>" kind = "entity" name = "..."`
3. For every entity from step 2: one `CREATE EDGE "m:<session>:<idx>" -> "ent:<slug>" kind = "mentions"` (always emit both the UPSERT and the matching EDGE, not one without the other)
4. If the user states a personal preference, belief, or fact about themselves, add one `ASSERT "fact:<topic>" ...`
5. If the new message contradicts a KNOWN FACT above, add one `RETRACT "<existing_id>" REASON "..."` before its `ASSERT`

**Fact id reuse (critical).** If the user turn starts with `### KNOWN FACTS`, those lines list existing beliefs that already have ids in the store. When the new message updates or contradicts a concept shown there, reuse the same `fact:<id>` from the block. Do NOT coin a new fact_id for the same underlying belief. On update emit `RETRACT "<existing_id>" REASON "..."` followed by `ASSERT "<existing_id>" ... value = "<new>" ...`.

---

**Input:**
Session s1, msg m:s1:0, user: "Kailash joined OpenAI as DB engineer."

**Output:**
```
CREATE NODE "m:s1:0" kind = "message" session = "s1" role = "user" DOCUMENT "Kailash joined OpenAI as DB engineer."
UPSERT NODE "ent:kailash" kind = "entity" name = "Kailash"
UPSERT NODE "ent:openai" kind = "entity" name = "OpenAI"
CREATE EDGE "m:s1:0" -> "ent:kailash" kind = "mentions"
CREATE EDGE "m:s1:0" -> "ent:openai" kind = "mentions"
```

---

**Input:**
Session s2, msg m:s2:0, user: "My favorite color is blue."

**Output:**
```
CREATE NODE "m:s2:0" kind = "message" session = "s2" role = "user" DOCUMENT "My favorite color is blue."
ASSERT "fact:favorite_color" kind = "belief" value = "blue" CONFIDENCE 0.9 SOURCE "m:s2:0"
```

---

**Input:**
Session s2, msg m:s2:1, user: "Actually my favorite color is green now."

**Output:**
```
CREATE NODE "m:s2:1" kind = "message" session = "s2" role = "user" DOCUMENT "Actually my favorite color is green now."
RETRACT "fact:favorite_color" REASON "superseded by m:s2:1"
ASSERT "fact:favorite_color" kind = "belief" value = "green" CONFIDENCE 0.9 SOURCE "m:s2:1"
```

---

**Input with known facts (reuse the existing fact_id, do not invent a new one):**

```
### KNOWN FACTS (reuse these fact_ids; emit RETRACT + ASSERT to update)
[fact:favorite_drink] kind="belief" value="coffee" confidence=0.90 source="m:s3:0"

Session s3, msg m:s3:1, user: "Actually I prefer tea now."
```

**Output:**
```
CREATE NODE "m:s3:1" kind = "message" session = "s3" role = "user" DOCUMENT "Actually I prefer tea now."
RETRACT "fact:favorite_drink" REASON "superseded by m:s3:1"
ASSERT "fact:favorite_drink" kind = "belief" value = "tea" CONFIDENCE 0.9 SOURCE "m:s3:1"
```

Wrong: coining `fact:preference` or `fact:drink_pref` when `fact:favorite_drink` already exists in KNOWN FACTS.
