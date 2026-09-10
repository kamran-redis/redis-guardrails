import os

import pytest
import redis

_ALLOW_OVERWRITE_ENV_VAR = "REDIS_GUARDRAILS_ALLOW_TEST_OVERWRITE"


def _redis_stack_available(redis_url: str) -> bool:
    try:
        client = redis.Redis.from_url(redis_url)
        client.execute_command("FT._LIST")
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def redis_url() -> str:
    url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    if not _redis_stack_available(url):
        pytest.skip(
            f"no Redis Stack (RediSearch) reachable at {url!r}; "
            "set REDIS_URL or run `docker run -p 6379:6379 redis/redis-stack-server`"
        )
    return url


@pytest.fixture(scope="session")
def allow_test_overwrite() -> None:
    """Explicit opt-in gate for tests that construct a store/router with
    overwrite=True against a real, reachable Redis.

    `redis_url` above only confirms Redis Stack is *reachable* -- that says
    nothing about whether it's safe to destructively overwrite. Reachable
    at redis://localhost:6379 (the default) could just as easily mean a
    real staging/dev Redis someone has port-forwarded to their machine for
    an unrelated reason, not the disposable `redis-stack-guardrails`
    container this suite expects.

    Any test that ends up calling GuardrailStore(overwrite=True) (directly,
    via a fixture, or via a raw SemanticRouter(overwrite=True)) must depend
    on this fixture. Without REDIS_GUARDRAILS_ALLOW_TEST_OVERWRITE=1 set,
    such tests are skipped rather than silently run against whatever Redis
    happens to be reachable.
    """
    if os.environ.get(_ALLOW_OVERWRITE_ENV_VAR) != "1":
        pytest.skip(
            "skipping test that overwrites Redis data: this test calls "
            "GuardrailStore/SemanticRouter with overwrite=True, which "
            "destructively clears index data at the target Redis. Set "
            f"{_ALLOW_OVERWRITE_ENV_VAR}=1 to confirm it's safe to do this "
            "against the Redis at REDIS_URL (e.g. it's the disposable "
            "redis-stack-guardrails container, not a real staging/dev Redis)."
        )
