from __future__ import annotations

from redis import Redis
from redisvl.extensions.router import Route, RoutingConfig, SemanticRouter
from redisvl.extensions.router.schema import DistanceAggregationMethod
from redisvl.utils.vectorize.base import BaseVectorizer

from redis_guardrails.errors import (
    DuplicateGuardrailError,
    EmbeddingError,
    GuardrailNotFoundError,
    IncompleteCoverageError,
    SearchError,
)
from redis_guardrails.models import Chunk, Guardrail, Match, Stage

_MAX_K = 100
_STAGES_REGISTRY_KEY = "guardrails:stages"
_DEFAULT_STAGES = {"input", "output"}


def _router_name(stage: str) -> str:
    return f"guardrails-{stage}"


class GuardrailStore:
    def __init__(self, redis_url: str, vectorizer: BaseVectorizer, overwrite: bool = False):
        self._redis_url = redis_url
        self._vectorizer = vectorizer
        stages = self._read_known_stages(redis_url) | _DEFAULT_STAGES
        self._routers: dict[str, SemanticRouter] = {
            stage: self._attach_or_create(_router_name(stage), redis_url, vectorizer, overwrite)
            for stage in stages
        }

    @staticmethod
    def _read_known_stages(redis_url: str) -> set[str]:
        """Which stages have ever had a guardrail added, per the registry SET.

        Empty on a fresh Redis (or one that predates this registry). Callers
        union this with _DEFAULT_STAGES (not "or" -- a non-empty registry
        must never suppress "input"/"output", since their Redis indices can
        still hold real data even if those two stages were never explicitly
        written into the registry themselves) so existing input/output-only
        deployments keep working identically regardless of what custom
        stages have been registered elsewhere.
        """
        client = Redis.from_url(redis_url)
        try:
            raw = client.smembers(_STAGES_REGISTRY_KEY)
        finally:
            client.close()
        return {s.decode() if isinstance(s, bytes) else s for s in raw}

    @staticmethod
    def _register_stage(redis_url: str, stage: str) -> None:
        client = Redis.from_url(redis_url)
        try:
            client.sadd(_STAGES_REGISTRY_KEY, stage)
        finally:
            client.close()

    @staticmethod
    def _attach_or_create(
        name: str, redis_url: str, vectorizer: BaseVectorizer, overwrite: bool
    ) -> SemanticRouter:
        # Ruling (controller, after Task 1's contract test): do NOT use
        # from_existing() here. Confirmed against redisvl 0.27.2 —
        # from_existing() does not accept an explicit vectorizer= override
        # (the kwarg is misrouted into Redis connection kwargs and raises
        # TypeError), and it can only reconstruct RedisVL's built-in
        # vectorizer types anyway (raises ValueError for a custom vectorizer
        # like HashVectorizer, whose .type is the inherited "base"). See
        # Task 1's tests/test_redisvl_contract.py for the reproduction.
        #
        # CORRECTED (empirically, via this task's own
        # test_second_store_instance_sees_guardrails_added_by_first): simply
        # constructing SemanticRouter(routes=[], overwrite=False) against an
        # index that already exists is NOT enough to "attach without
        # wiping". Reading redisvl 0.27.2's SemanticRouter.__init__ /
        # _initialize_index:
        #   - when the index already exists and overwrite=False, the
        #     constructor does NOT call _add_routes(self.routes) to
        #     repopulate self.routes from what's already indexed — it just
        #     leaves self.routes as whatever was passed in (here: []).
        #     That alone breaks .get()/.list(), which read self.routes.
        #   - worse, __init__ unconditionally re-persists
        #     f"{name}:route_config" from self.to_dict() at the very end,
        #     regardless of existed/overwrite. So constructing with
        #     routes=[] against an existing index silently CLOBBERS the
        #     previously-stored route config (metadata, thresholds, route
        #     names) with an empty one, even though the underlying indexed
        #     hash documents survive untouched.
        #
        # Fix: read that same f"{name}:route_config" JSON key ourselves
        # first and pass the real routes back into the constructor. Route
        # objects need no vectorizer to reconstruct (just
        # name/references/metadata/distance_threshold), so this sidesteps
        # from_existing()'s custom-vectorizer limitation entirely while
        # avoiding both the empty-.routes bug and the config-clobber bug.
        routes = []
        if overwrite:
            GuardrailStore._drop_index(name, redis_url)
        else:
            routes = GuardrailStore._load_existing_routes(name, redis_url)
        return SemanticRouter(
            name=name,
            routes=routes,
            vectorizer=vectorizer,
            routing_config=RoutingConfig(
                max_k=_MAX_K, aggregation_method=DistanceAggregationMethod.min
            ),
            redis_url=redis_url,
            overwrite=overwrite,
        )

    @staticmethod
    def _drop_index(name: str, redis_url: str) -> None:
        """Explicitly drop the index and its documents before a real fresh start.

        RedisVL's SemanticRouter.__init__ calls self._index.create(overwrite=
        overwrite, drop=False) internally. With drop=False, overwrite=True only
        recreates the index DEFINITION — any reference hash documents already
        indexed under the same key prefix are left in Redis untouched, and get
        silently re-indexed alongside whatever routes the new construction
        adds. That means a caller who asks for overwrite=True (expecting a
        clean slate) can end up with BOTH old and new data matchable.

        Fix: drop the index and its documents ourselves, directly via
        FT.DROPINDEX <name> DD, before SemanticRouter.__init__ ever runs — so
        that when it calls .create(overwrite=True, drop=False), the index
        genuinely does not exist yet and creation is a true fresh start.

        Dropping an index that doesn't exist yet raises a ResponseError from
        Redis (e.g. "Unknown index name"); that's the expected first-time-ever
        case, not a real failure, so it's swallowed as a no-op. Same narrow,
        direct-connection pattern as _load_existing_routes.
        """
        client = Redis.from_url(redis_url)
        try:
            client.execute_command("FT.DROPINDEX", name, "DD")
        except Exception:
            pass
        finally:
            client.close()

    @staticmethod
    def _load_existing_routes(name: str, redis_url: str) -> list[Route]:
        """Read a router's persisted route config directly from Redis.

        Used instead of SemanticRouter.from_existing(), which cannot
        reconstruct a custom (non-builtin) vectorizer. Returns [] only
        when the router has never been created yet — RedisJSON returns
        None for a missing key rather than raising, so that case is
        handled by the isinstance check below, not by catching an
        exception. A genuine connection/protocol failure here is left to
        propagate rather than being swallowed into an empty list: silently
        treating "we couldn't reach Redis" the same as "nothing exists
        yet" would risk reconstructing with routes=[] against an index
        that DOES have data — reintroducing the exact route_config-
        clobbering bug this method exists to prevent.
        """
        client = Redis.from_url(redis_url)
        try:
            stored = client.json().get(f"{name}:route_config")
        finally:
            client.close()
        if not isinstance(stored, dict):
            return []
        return [Route(**route) for route in stored.get("routes", [])]

    def _ensure_router(self, stage: str) -> None:
        """Lazily create (and register) a router for a stage never seen before.

        Shared by add() and update(): both can be handed a stage that has
        never had a guardrail in it yet -- add() via a brand-new guardrail,
        update() via moving an existing guardrail onto a new stage name --
        and in either case self._routers must gain a live entry for it
        before anything tries to look it up with [] rather than .get().
        """
        if stage not in self._routers:
            self._routers[stage] = self._attach_or_create(
                _router_name(stage), self._redis_url, self._vectorizer, overwrite=False
            )
            self._register_stage(self._redis_url, stage)

    def add(self, guardrail: Guardrail) -> None:
        if self.get(guardrail.id) is not None:
            raise DuplicateGuardrailError(guardrail.id)
        self._ensure_router(guardrail.stage)
        try:
            self._routers[guardrail.stage].add_route(self._to_route(guardrail))
        except Exception as exc:
            raise SearchError(f"failed to add guardrail {guardrail.id!r}: {exc}") from exc

    def update(self, guardrail: Guardrail) -> None:
        existing = self.get(guardrail.id)
        if existing is None:
            raise GuardrailNotFoundError(guardrail.id)

        old_router = self._routers[existing.stage]
        old_route = self._to_route(existing)
        old_router.remove_route(guardrail.id)
        self._ensure_router(guardrail.stage)
        try:
            self._routers[guardrail.stage].add_route(self._to_route(guardrail))
        except Exception as exc:
            old_router.add_route(old_route)  # best-effort rollback
            raise SearchError(f"failed to update guardrail {guardrail.id!r}: {exc}") from exc

    def delete(self, guardrail_id: str) -> None:
        existing = self.get(guardrail_id)
        if existing is None:
            raise GuardrailNotFoundError(guardrail_id)
        self._routers[existing.stage].remove_route(guardrail_id)

    def get(self, guardrail_id: str) -> Guardrail | None:
        for stage, router in self._routers.items():
            route = router.get(guardrail_id)
            if route is None:
                continue
            return self._to_guardrail(stage, router, route)
        return None

    def list(self, stage: Stage | None = None) -> list[Guardrail]:
        stages = [stage] if stage is not None else list(self._routers)
        result: list[Guardrail] = []
        for s in stages:
            router = self._routers.get(s)
            if router is None:
                continue
            for route in router.routes:
                result.append(self._to_guardrail(s, router, route))
        return result

    def _to_guardrail(self, stage: Stage, router: SemanticRouter, route: Route) -> Guardrail:
        references = router.get_route_references(route_name=route.name)
        return Guardrail(
            id=route.name,
            stage=stage,
            category=route.metadata["category"],
            description=route.metadata["description"],
            examples=[ref["reference"] for ref in references],
            action=route.metadata["action"],
            match_threshold=route.distance_threshold,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._check_token_limits(texts)
        try:
            # skip_cache=True: bypass any embedding cache so embedding_ms
            # (measured by the caller around this call) always reflects a
            # genuine embedding computation, never a cache hit. No
            # vectorizer in this codebase has caching configured today, so
            # this is currently a no-op -- but it keeps the code honest
            # about the timing guarantee if that ever changes.
            return self._vectorizer.embed_many(texts, skip_cache=True)
        except Exception as exc:
            raise EmbeddingError(f"failed to embed {len(texts)} chunk(s): {exc}") from exc

    def _check_token_limits(self, texts: list[str]) -> None:
        """Best-effort guard against embedding-model truncation being silent.

        DEFAULT_MAX_CHARS in chunking.py is a character budget with no fixed
        relationship to an embedding model's actual token limit. Text that
        fits the character budget can still exceed the model's token limit
        (verified for token-dense content like CJK text or URLs) --
        sentence-transformers then truncates internally and silently, and
        the resulting embedding covers only part of the text even though the
        evaluation still reports status="COMPLETED". That violates the rule
        that un-inspected text must never be treated as allowed.

        This is intentionally scoped to the HFTextVectorizer-backed path:
        HFTextVectorizer wraps a sentence_transformers.SentenceTransformer at
        `._client`, which exposes `.max_seq_length` (the model's true token
        budget) and `.tokenizer` (to count tokens without embedding). Any
        other vectorizer -- including a custom one like HashVectorizer, or a
        future API-backed one such as OpenAI's -- won't have both of these
        attributes, and the check is a silent no-op for it: the embedding
        provider/model choice itself is out of scope here, so this must not
        become a hard requirement for every vectorizer type.
        """
        client = getattr(self._vectorizer, "_client", None)
        max_seq_length = getattr(client, "max_seq_length", None)
        tokenizer = getattr(client, "tokenizer", None)
        if max_seq_length is None or tokenizer is None:
            return

        for text in texts:
            try:
                token_count = len(tokenizer(text, add_special_tokens=True)["input_ids"])
            except Exception:
                # Best-effort only: if the tokenizer itself can't process
                # this text for some unrelated reason, don't block on it --
                # any real problem will surface (more informatively) from
                # the embed_many() call that follows.
                continue
            if token_count > max_seq_length:
                raise IncompleteCoverageError(
                    f"text requires {token_count} tokens, which exceeds the "
                    f"embedding model's max_seq_length={max_seq_length}; "
                    "embedding it would silently truncate the text and "
                    "produce an embedding covering only part of it"
                )

    def search(self, vector: list[float], chunk: Chunk, stage: Stage) -> list[Match]:
        router = self._routers.get(stage)
        if router is None or not router.routes:
            raise SearchError(f"no guardrails configured for stage {stage!r}")

        try:
            # No distance_threshold kwarg here: route_many's distance_threshold
            # override is deprecated and, per RedisVL's implementation, isn't
            # actually read for filtering anyway -- passing it only produced a
            # DeprecationWarning on every call. The real filtering contract is
            # each route's own distance_threshold (set at Route construction
            # in _to_route) combined with this router's routing_config; that's
            # what determines which candidates route_many returns here.
            # evaluator.py's own `distance <= threshold` re-check downstream
            # is intentional defense-in-depth against that contract, not
            # redundant dead logic -- keep both.
            route_matches = router.route_many(
                vector=vector,
                max_k=_MAX_K,
                aggregation_method=DistanceAggregationMethod.min,
            )
        except Exception as exc:
            raise SearchError(f"search failed for stage {stage!r}: {exc}") from exc

        matches: list[Match] = []
        for route_match in route_matches:
            if route_match.name is None or route_match.distance is None:
                continue
            route = router.get(route_match.name)
            matches.append(
                Match(
                    rule_id=route_match.name,
                    category=route.metadata["category"],
                    action=route.metadata["action"],
                    distance=route_match.distance,
                    threshold=route.distance_threshold,
                    chunk_id=chunk.id,
                    evaluated_text=chunk.evaluated_text,
                )
            )
        return matches

    def _to_route(self, guardrail: Guardrail) -> Route:
        return Route(
            name=guardrail.id,
            references=guardrail.examples,
            metadata={
                "category": guardrail.category,
                "description": guardrail.description,
                "action": guardrail.action,
            },
            distance_threshold=guardrail.match_threshold,
        )
