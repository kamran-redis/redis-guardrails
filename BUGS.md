# Bugs

Running log of known bugs. Add new ones at the top with status `OPEN`/`FIXED`;
when fixed, note the commit that fixed it instead of deleting the entry.

## OPEN

### Web app doesn't see guardrails added by another process until restarted

**Symptom:** Start the web app with no guardrails loaded. Add guardrails via
the CLI (`redis-guardrails add ...`) while the web app keeps running. The web
app's guardrail list / evaluate results don't reflect the new guardrails —
restarting the web app is required to see them.

**Likely cause:** `GuardrailStore.__init__` (`src/redis_guardrails/store.py`)
builds one long-lived `SemanticRouter` per scope and holds it in
`self._routers`. `create_app()`'s FastAPI lifespan builds this store once at
process startup. The CLI is a separate short-lived process — it builds its
own `GuardrailStore`/`SemanticRouter`, mutates Redis, then exits. The running
web process's in-memory `SemanticRouter` instances never get told about
routes added by that other process, so matching/listing against them is
stale until the web process rebuilds its routers from Redis (i.e. restarts).

**Not yet investigated:** whether this also affects two CLI invocations
racing each other, or is specific to a long-lived process (web) missing
writes from a short-lived one (CLI).
