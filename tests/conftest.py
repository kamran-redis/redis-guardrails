import os

import pytest
import redis


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
