# Backlog

Feature/content ideas not yet scheduled. Add new ones at the top; when
picked up, move to a written spec (see `docs/superpowers/specs/`) and
remove from here.

## Home page: explain the app

Home page (`src/redis_guardrails/web/templates/home.html`) currently jumps
straight to the three nav cards (Run a prompt / Run a benchmark / Manage
guardrails) with no context for a first-time visitor.

Add a brief description above the cards, in this order:

1. **Problem** — what this solves (checking chatbot input/output against
   guardrail rules before it reaches a user, for a banking chatbot use
   case), in plain language.
2. **Technical information** — how it works at a glance (semantic
   similarity via RedisVL's `SemanticRouter`, ALLOW/FLAG/BLOCK decisions),
   pointing to `docs/vector-guardrail-algorithm.md` for the full algorithm.
3. **Diagrams, if useful** — e.g. a simple input/output → chunk → match →
   decision flow diagram. Only add one if it genuinely clarifies the flow
   better than prose; don't force a diagram in.
