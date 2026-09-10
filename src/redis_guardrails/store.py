from __future__ import annotations

from redis import Redis
from redisvl.extensions.router import Route, RoutingConfig, SemanticRouter
from redisvl.extensions.router.schema import DistanceAggregationMethod
from redisvl.utils.vectorize.base import BaseVectorizer

from redis_guardrails.errors import (
    DuplicateGuardrailError,
    EmbeddingError,
    GuardrailNotFoundError,
    SearchError,
)
from redis_guardrails.models import Chunk, Guardrail, Match, Stage

_MAX_K = 100
_ROUTER_NAMES: dict[str, str] = {"input": "guardrails-input", "output": "guardrails-output"}


class GuardrailStore:
    def __init__(self, redis_url: str, vectorizer: BaseVectorizer, overwrite: bool = False):
        self._vectorizer = vectorizer
        self._routers: dict[Stage, SemanticRouter] = {
            stage: self._attach_or_create(name, redis_url, vectorizer, overwrite)
            for stage, name in _ROUTER_NAMES.items()
        }

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
        routes = [] if overwrite else GuardrailStore._load_existing_routes(name, redis_url)
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
    def _load_existing_routes(name: str, redis_url: str) -> list[Route]:
        """Read a router's persisted route config directly from Redis.

        Used instead of SemanticRouter.from_existing(), which cannot
        reconstruct a custom (non-builtin) vectorizer. Returns [] if the
        router has never been created yet (no stored config).
        """
        client = Redis.from_url(redis_url)
        try:
            stored = client.json().get(f"{name}:route_config")
        except Exception:
            return []
        finally:
            client.close()
        if not isinstance(stored, dict):
            return []
        return [Route(**route) for route in stored.get("routes", [])]

    def add(self, guardrail: Guardrail) -> None:
        if self.get(guardrail.id) is not None:
            raise DuplicateGuardrailError(guardrail.id)
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
            router = self._routers[s]
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
        try:
            return self._vectorizer.embed_many(texts)
        except Exception as exc:
            raise EmbeddingError(f"failed to embed {len(texts)} chunk(s): {exc}") from exc

    def search(self, vector: list[float], chunk: Chunk, stage: Stage) -> list[Match]:
        router = self._routers[stage]
        if not router.routes:
            raise SearchError(f"no guardrails configured for stage {stage!r}")

        try:
            route_matches = router.route_many(
                vector=vector,
                max_k=_MAX_K,
                aggregation_method=DistanceAggregationMethod.min,
                distance_threshold=None,
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
