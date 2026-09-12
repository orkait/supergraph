Classify the user's request for a coding agent. Answer with one JSON object and nothing else: {"kind": "answer" | "diagnose" | "change" | "monitor"}

- answer: explain, review, summarise, report status, or answer a question. Reading is enough; nothing should be written or run that mutates state.
- diagnose: find the cause of a problem and explain it. Reading and running checks are fine; the fix is not requested.
- change: build, implement, fix, refactor, write, or otherwise modify the workspace, then verify.
- monitor: wait for or watch an external state and report when it changes.

A request that says "finish", "do not stop" or "babysit" sets persistence, not kind. When unsure between answer and change, prefer answer.
