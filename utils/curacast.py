# curatarr
# Copyright (C) 2026 OrchestratedChaos
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Curacast API client for Curatarr.

Optional integration: consumes curacast's graded watch-credit feed
(`GET /api/analytics/watch-credits`) as a supplemental history source for
live-TV viewing. curacast (a sibling product) plays a Plex library back
through simulated TV channels and marks watched items via Plex's
`/:/scrobble` endpoint - that bumps viewCount/lastViewedAt but creates NO
row in `/status/sessions/history/all` (verified against a live server:
14/14 real scrobbles bumped viewCount, 0/14 produced a history row), so
live-TV viewing is otherwise invisible to Curatarr's default profile
path (see utils/plex.py's fetch_plex_watch_history_movies/shows).

Disabled by default. Never raises out of the high-level get_watch_credits()/
apply_watch_credits() helpers below - if curacast is disabled, unreachable,
or misconfigured, callers must transparently fall back to Plex-only
behavior, exactly like the tautulli/trakt integrations this mirrors.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from .api_client import BaseAPIClient
from .cache import load_json_cache, save_json_cache
from .counters import process_counters_from_cache
from .display import log_warning
from .scoring import calculate_recency_multiplier

logger = logging.getLogger("curatarr")

CURACAST_RATE_LIMIT_DELAY = 0.1
CURACAST_REQUEST_TIMEOUT = 30

# Default/hard-cap for the `limit` query param, matching curacast's own
# documented API contract (default 500, server hard-caps at 1000).
CURACAST_DEFAULT_PAGE_LIMIT = 500
CURACAST_MAX_PAGE_LIMIT = 1000

# Bounds the since/next_since pagination loop in get_watch_credits() - a
# safety net against a server bug that never advances next_since (or
# advances it by less than one full page), which would otherwise loop
# forever. Hitting this is logged loudly, never silently truncated -
# whatever was fetched by then is still returned and the cursor still
# advances, so the rest is simply picked up on the next run instead of
# lost.
CURACAST_MAX_PAGES = 50

# Weight below which a curacast credit is dropped (config
# `curacast.min_weight`, sent as the `min_weight` query param). Matches
# curacast's own tier -> weight table (sampled=-0.3, tasted=0.15,
# partial=0.4, substantial=0.8, complete=1.0): the default keeps
# "partial" and up.
DEFAULT_CURACAST_MIN_WEIGHT = 0.4

# Weight at/above which a credit also counts as "already watched" for
# recommendation exclusion (config `curacast.exclude_at_weight`), not just
# a scoring signal. Default 0.8 = curacast's "substantial" tier
# (>=70% of the program's runtime seen) and up. 70% is the completion bar
# Netflix used for its own "viewer" definition before 2019 - a defensible
# "they've seen this" line. Below it (partial/tasted/sampled) they bailed
# early, and the item should stay recommendable.
DEFAULT_CURACAST_EXCLUDE_AT_WEIGHT = 0.8


class CuracastAPIError(Exception):
    """Raised when a curacast API request fails."""

    pass


class CuracastClient(BaseAPIClient):
    """
    curacast analytics API client.

    Authenticated via the `x-api-key` header - curacast applies its
    ApiAuth middleware to every `/api/*` route
    (~/dev/curacast/src/boot/routes.js:157-161's `app.use('/api', ...
    auth.middleware())`), which checks `req.headers['x-api-key']`
    (~/dev/curacast/src/middleware/api-auth.js:214).
    """

    api_name = "Curacast"
    exception_class = CuracastAPIError
    rate_limit_delay = CURACAST_RATE_LIMIT_DELAY
    request_timeout = CURACAST_REQUEST_TIMEOUT

    def __init__(self, url: str, api_key: str):
        """
        Initialize curacast client.

        Args:
            url: Base curacast URL, e.g. http://localhost:8000
            api_key: curacast API key (Settings -> API)
        """
        super().__init__()
        self.base_url = url.rstrip("/")
        self.api_key = api_key

    def _get_headers(self) -> Dict[str, str]:
        return {"x-api-key": self.api_key}

    def get_watch_credits_page(
        self,
        since: int = 0,
        since_id: int = 0,
        username: Optional[str] = None,
        min_weight: Optional[float] = None,
        limit: int = CURACAST_DEFAULT_PAGE_LIMIT,
    ) -> Dict[str, Any]:
        """
        Fetch a single page of `GET {base_url}/api/analytics/watch-credits`.

        Compound cursor (`since`, `since_id`): the server's WHERE clause is
        `(ended_at > since) OR (ended_at = since AND id > since_id)`, ordered
        `ended_at ASC, id ASC`. Plain `ended_at > since` alone can silently
        skip rows forever when two credits share the exact same millisecond
        and a page boundary lands between them; `since_id` breaks that tie.
        Both are always sent, together - see get_watch_credits() for how
        they're advanced together across pages.

        Args:
            since: Epoch ms cursor (exclusive on its own; see since_id)
            since_id: Tiebreaker row id for credits sharing `since`'s exact
                millisecond - always sent alongside `since`
            username: Optional curacast viewer username to restrict to
            min_weight: Optional minimum credit weight to include
            limit: Max rows to return (server hard-caps at 1000)

        Returns:
            The raw {"credits": [...], "next_since": int, "next_since_id": int,
            "count": int} envelope

        Raises:
            CuracastAPIError: On HTTP errors, timeouts, connection failures,
                or a malformed/unexpected response shape
        """
        params: Dict[str, Any] = {"since": since, "since_id": since_id, "limit": limit}
        if username:
            params["username"] = username
        if min_weight is not None:
            params["min_weight"] = min_weight

        url = f"{self.base_url}/api/analytics/watch-credits"
        result = self._make_request_to_url("GET", url, params=params)

        if not isinstance(result, dict) or "credits" not in result:
            raise CuracastAPIError("Unexpected response shape from curacast watch-credits endpoint")

        return result


def create_curacast_client(config: Dict) -> Optional[CuracastClient]:
    """
    Create a CuracastClient from config, if configured and enabled.

    Args:
        config: Full config dict containing an optional 'curacast' section

    Returns:
        CuracastClient if configured and enabled, None otherwise
    """
    curacast_config = config.get("curacast", {}) or {}

    if not curacast_config.get("enabled", False):
        return None

    url = curacast_config.get("url")
    api_key = curacast_config.get("api_key")

    if not url or not api_key:
        log_warning("Curacast enabled but 'url'/'api_key' not configured - skipping curacast watch credits")
        return None

    return CuracastClient(url, api_key)


def _cursor_path(cache_dir: str, media_type: str) -> str:
    """
    Path to the persisted compound (`since`, `since_id`) pagination cursor
    for `media_type`.

    Deliberately one cursor file per media_type (curacast_cursor_movie.json
    / curacast_cursor_tv.json), not one shared cursor, even though both are
    read from the exact same underlying feed. Pagination is exclusive (see
    CuracastClient.get_watch_credits_page's compound-cursor docstring) and
    the movie and TV recommenders consume this feed independently, in
    separate process invocations, each filtering out the other's credits
    client-side (see apply_watch_credits). A single shared cursor advanced
    by (say) the movie recommender's run would permanently skip any
    episode credits in that same time range before the TV recommender ever
    got a chance to see them.
    """
    return os.path.join(cache_dir, f"curacast_cursor_{media_type}.json")


def _load_cursor(cache_dir: str, media_type: str) -> Tuple[int, int]:
    """Load the persisted (since, since_id) compound cursor for
    `media_type`, defaulting to (0, 0) (start of time) on first run or any
    read failure.

    Upgrade path: a cursor file written by the pre-compound-cursor format
    has `since` but no `since_id` key at all - defaults to 0 rather than
    raising, exactly like a missing/corrupt cursor file does.
    """
    data = load_json_cache(_cursor_path(cache_dir, media_type))
    if not data:
        return 0, 0
    since = data.get("since", 0)
    since = int(since) if isinstance(since, (int, float)) else 0
    since_id = data.get("since_id", 0)
    since_id = int(since_id) if isinstance(since_id, (int, float)) else 0
    return since, since_id


def _save_cursor(cache_dir: str, media_type: str, since: int, since_id: int) -> None:
    """Persist (since, since_id) as the next run's starting cursor for `media_type`."""
    save_json_cache(_cursor_path(cache_dir, media_type), {"since": since, "since_id": since_id})


def get_watch_credits(
    client: CuracastClient,
    since: int = 0,
    since_id: int = 0,
    username: Optional[str] = None,
    min_weight: Optional[float] = None,
    max_pages: int = CURACAST_MAX_PAGES,
) -> Tuple[List[Dict], Optional[int], Optional[int]]:
    """
    Paginate `GET /api/analytics/watch-credits` from the compound
    (`since`, `since_id`) cursor, following (`next_since`, `next_since_id`)
    until the server returns `count: 0` or `max_pages` is hit.

    The compound cursor exists because plain `ended_at > since` can
    silently skip credits forever: if two rows share the exact same
    millisecond `ended_at` and a page boundary lands between them, the
    second is never `> since` of the first once `since` advances past that
    millisecond. `since_id` (ordered `ended_at ASC, id ASC` server-side)
    breaks the tie - see CuracastClient.get_watch_credits_page.

    Never raises: a connection error, non-200, or malformed JSON on ANY
    page logs a warning and returns ([], None, None) - discarding whatever
    partial pages already succeeded this call, so a flaky/partial fetch
    can never advance the persisted cursor past credits the caller never
    actually saw.

    Args:
        client: CuracastClient instance
        since: Epoch ms to start from (paired with since_id - see above)
        since_id: Tiebreaker row id to start from, paired with `since`
        username: Optional curacast viewer username to restrict to
        min_weight: Optional minimum credit weight to include
        max_pages: Safety cap on pagination iterations (default 50)

    Returns:
        (credits, next_since, next_since_id) - the two cursor values the
        caller should persist TOGETHER for its next call. Both None means
        "don't move the cursor, retry from the same point next time" (i.e.
        the fetch failed outright); they are never None on a successful
        fetch, even one whose very first page already returned zero rows.
        If a page's response omits `next_since_id` entirely (an older
        curacast that doesn't yet speak the compound cursor), this falls
        back to 0 rather than crashing - logged at debug level each time,
        so a stuck-on-0 since_id against an old server stays visible
        rather than silently persisting forever.
    """
    all_credits: List[Dict] = []
    cursor = since
    cursor_id = since_id

    for page_num in range(1, max_pages + 1):
        try:
            page_result = client.get_watch_credits_page(
                since=cursor, since_id=cursor_id, username=username, min_weight=min_weight
            )
        except CuracastAPIError as e:
            log_warning(f"Curacast: watch-credits fetch failed, discarding partial results this run: {e}")
            return [], None, None

        page_credits = page_result.get("credits") or []
        count = page_result.get("count", len(page_credits))
        next_since = page_result.get("next_since")
        next_since_id = page_result.get("next_since_id")
        if next_since_id is None:
            logger.debug(
                "Curacast: watch-credits response missing 'next_since_id' (older curacast server?) - falling back to 0"
            )
            next_since_id = 0

        all_credits.extend(page_credits)

        if not count:
            return all_credits, (next_since if next_since is not None else cursor), next_since_id

        if next_since is None:
            log_warning(
                "Curacast: watch-credits response had a nonzero count but no 'next_since' - stopping pagination early"
            )
            return all_credits, cursor, cursor_id

        cursor = next_since
        cursor_id = next_since_id

        if page_num == max_pages:
            log_warning(
                f"Curacast: watch-credits pagination hit the {max_pages}-page safety cap "
                f"({len(all_credits)} credits fetched this run) - the rest will be picked up "
                "on a future run, not lost."
            )

    return all_credits, cursor, cursor_id


def apply_watch_credits(
    config: Dict,
    counters: Dict,
    media_type: str,
    watched_ids: Set[int],
    media_info_cache: Dict[str, Dict],
    plex: Any,
    cache_dir: str,
    plex_username: Optional[str] = None,
    client: Optional[CuracastClient] = None,
) -> int:
    """
    Fold curacast watch credits into `counters` as weighted signals.

    Weighting: each credit contributes `weight = credit['weight'] *
    recency_decay(ended_at)`, via the exact same
    process_counters_from_cache()/_apply_capped_weight() path (and the
    same `negative_signals.bad_ratings.cap_penalty` clamp) every other
    watched item goes through - see utils/counters.py. Deliberately NOT
    the rating/rewatch multipliers real Plex/Tautulli history gets: a
    live-TV credit carries no user star rating, and rewatch is already
    implicit in getting a second (or third...) credit for the same item,
    each contributing its own weight on top.

    Dedup rule: a credit is skipped if its resolved top-level library id
    (a movie's own ratingKey, or an episode's grandparentRatingKey -
    i.e. its show - for TV) is already in `watched_ids`, the set this
    run's normal Plex/Tautulli history builder already populated BEFORE
    this function is called. That item is already counted at full
    (real) weight; a curacast credit for the same item would double-
    count it.

    Exclusion: a credit whose OWN weight (before recency decay - recency
    is a scoring-only concept, irrelevant to "did they finish it") is >=
    `curacast.exclude_at_weight` (default 0.8, curacast's "substantial"
    tier and up) has its resolved id added to `watched_ids` itself, IN
    PLACE - the exact same set this function reads for the dedup check
    above, and (critically) the exact same set object the caller's
    exclusion/library-filtering logic consults downstream. Callers MUST
    pass their real, live watched-id set here (not a disposable local
    copy already snapshotted elsewhere) or this mutation is silently
    inert - see recommenders/movie.py's/tv.py's call sites, which pass
    `self.watched_ids` for exactly this reason (traced: their own local
    `watched_ids` variable is copied into `self.watched_ids` via
    `.update()` BEFORE this function ever runs, so passing the local
    variable here would mutate a copy nothing downstream ever reads
    again). A credit below the threshold (partial/tasted/sampled) is
    never added - the user bailed early, and the item stays
    recommendable. Exclusion is applied even if `media_info_cache` has
    no entry for the id (i.e. even when there's nothing to score) -
    "did they watch it" doesn't depend on whether metadata happened to
    be cached.

    Never raises: any failure (curacast disabled/unreachable, an
    unresolvable ratingKey, a cache miss) degrades to skipping that
    credit or the whole fetch, never to an exception reaching the
    profile build.

    Args:
        config: Full config dict containing an optional 'curacast' section
        counters: Counter dict to update (see utils.counters.create_empty_counters)
        media_type: 'movie' or 'tv'
        watched_ids: This run's REAL, live Plex/Tautulli-derived watched-id
            set - read for dedup AND mutated in place for credits at/above
            `curacast.exclude_at_weight` (see "Exclusion" above)
        media_info_cache: The movie/show metadata cache dict, keyed by str(id)
            (e.g. self.movie_cache.cache["movies"] / self.show_cache.cache["shows"])
        plex: Connected plexapi PlexServer, used to resolve each credit's
            program_key (a Plex ratingKey) to a real library item
        cache_dir: This run's cache directory, for the persisted since-cursor
        plex_username: The Plex user this profile is being built for. Used
            as the curacast `username` filter when curacast.username is
            blank ("the configured Plex user" - see config.example.yml).
            May be None (e.g. a managed-users run with no single configured
            user), in which case credits from every curacast viewer are
            pulled unfiltered.
        client: Optional pre-built CuracastClient (mainly for testing)

    Returns:
        Number of credits actually applied to `counters` (exclusion-only
        additions to `watched_ids` for a cache-miss credit are not counted
        here - see "Exclusion" above)
    """
    client = client or create_curacast_client(config)
    if not client:
        return 0

    curacast_cfg = config.get("curacast", {}) or {}
    min_weight = curacast_cfg.get("min_weight", DEFAULT_CURACAST_MIN_WEIGHT)
    exclude_at_weight = curacast_cfg.get("exclude_at_weight", DEFAULT_CURACAST_EXCLUDE_AT_WEIGHT)
    username = curacast_cfg.get("username") or plex_username or None

    since, since_id = _load_cursor(cache_dir, media_type)
    credits, next_since, next_since_id = get_watch_credits(
        client, since=since, since_id=since_id, username=username, min_weight=min_weight
    )

    if next_since is not None:
        # Paired by construction (get_watch_credits() never returns one
        # without the other) - asserted, not just assumed, so a future
        # change that breaks that pairing fails loudly here instead of
        # silently persisting a bogus since_id.
        assert next_since_id is not None
        _save_cursor(cache_dir, media_type, next_since, next_since_id)

    if not credits:
        return 0

    ns_config = config.get("negative_signals", {})
    cap_penalty = ns_config.get("bad_ratings", {}).get("cap_penalty", 0.5)
    recency_config = config.get("recency_decay", {})

    applied = 0
    for credit in credits:
        program_key = credit.get("program_key")
        weight = credit.get("weight")
        if not program_key or weight is None:
            continue

        try:
            item = plex.fetchItem(int(program_key))
        except Exception as e:
            # Broad except deliberately: plexapi surfaces a NotFound/
            # BadRequest/Unauthorized/etc zoo of its own exception types
            # here (on top of the plain ValueError a non-numeric
            # program_key would raise), and this must never crash the
            # profile build over one unresolvable credit (#see module
            # docstring).
            logger.debug(f"Curacast credit: could not resolve ratingKey {program_key!r}: {e}")
            continue

        item_type = getattr(item, "type", None)
        if media_type == "movie":
            if item_type != "movie":
                continue
            item_id = int(item.ratingKey)
        else:
            if item_type != "episode":
                continue
            grandparent = getattr(item, "grandparentRatingKey", None)
            if not grandparent:
                logger.debug(f"Curacast credit: episode ratingKey {program_key} has no parent show, skipping")
                continue
            item_id = int(grandparent)

        if item_id in watched_ids:
            continue

        # Exclusion check FIRST, on the credit's own (pre-recency-decay)
        # weight, independent of whether metadata for this id is cached -
        # "did they watch it" isn't a scoring question. Mutates the
        # caller's real watched_ids set in place (see docstring).
        if float(weight) >= exclude_at_weight:
            watched_ids.add(item_id)
            logger.debug(
                f"Curacast credit: ratingKey {item_id} weight {weight} >= {exclude_at_weight}, marking watched"
            )

        media_info = media_info_cache.get(str(item_id))
        if not media_info:
            logger.debug(f"Curacast credit: ratingKey {item_id} not in {media_type} cache, skipping")
            continue

        ended_at = credit.get("ended_at")
        recency_multiplier = calculate_recency_multiplier(ended_at / 1000, recency_config) if ended_at else 1.0
        total_weight = float(weight) * recency_multiplier

        process_counters_from_cache(
            media_info, counters, media_type=media_type, weight=total_weight, cap_penalty=cap_penalty
        )
        if tmdb_id := media_info.get("tmdb_id"):
            counters["tmdb_ids"].add(tmdb_id)
        applied += 1

    return applied
