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

"""Tests for utils/curacast.py - curacast watch-credit API client and profile integration."""

from collections import Counter
from unittest.mock import Mock, patch

import pytest

from utils.counters import create_empty_counters
from utils.curacast import (
    CuracastAPIError,
    CuracastClient,
    _load_cursor,
    _save_cursor,
    apply_watch_credits,
    create_curacast_client,
    get_watch_credits,
)

# ---------------------------------------------------------------------------
# CuracastClient
# ---------------------------------------------------------------------------


class TestCuracastClientInit:
    def test_init_strips_trailing_slash(self):
        client = CuracastClient(url="http://localhost:8000/", api_key="key123")
        assert client.base_url == "http://localhost:8000"
        assert client.api_key == "key123"

    def test_headers_use_x_api_key(self):
        client = CuracastClient(url="http://localhost:8000", api_key="secret-key")
        assert client._get_headers() == {"x-api-key": "secret-key"}


class TestCuracastClientGetWatchCreditsPage:
    @patch("utils.api_client.requests.request")
    def test_success_returns_envelope(self, mock_request):
        mock_response = Mock()
        # BaseAPIClient streams+caps the response body (see
        # utils.helpers.read_response_capped) - a plain Mock() needs
        # .headers/.iter_content spelled out explicitly.
        mock_response.headers = {}
        mock_response.iter_content = Mock(return_value=[])
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "credits": [{"id": 1}],
            "next_since": 1000,
            "next_since_id": 5,
            "count": 1,
        }
        mock_request.return_value = mock_response

        client = CuracastClient("http://localhost:8000", "key123")
        result = client.get_watch_credits_page(since=0)

        assert result == {"credits": [{"id": 1}], "next_since": 1000, "next_since_id": 5, "count": 1}
        called_url = mock_request.call_args.kwargs["url"]
        called_headers = mock_request.call_args.kwargs["headers"]
        called_params = mock_request.call_args.kwargs["params"]
        assert called_url == "http://localhost:8000/api/analytics/watch-credits"
        assert called_headers["x-api-key"] == "key123"
        # since_id is always sent, even at its default (0) - compound cursor
        # contract, not an optional extra like username/min_weight.
        assert called_params == {"since": 0, "since_id": 0, "limit": 500}

    @patch("utils.api_client.requests.request")
    def test_since_id_always_sent_alongside_since(self, mock_request):
        mock_response = Mock()
        mock_response.headers = {}
        mock_response.iter_content = Mock(return_value=[])
        mock_response.status_code = 200
        mock_response.json.return_value = {"credits": [], "next_since": 0, "next_since_id": 0, "count": 0}
        mock_request.return_value = mock_response

        client = CuracastClient("http://localhost:8000", "key123")
        client.get_watch_credits_page(since=500, since_id=7, username="testuser", min_weight=0.4, limit=250)

        called_params = mock_request.call_args.kwargs["params"]
        assert called_params == {
            "since": 500,
            "since_id": 7,
            "limit": 250,
            "username": "testuser",
            "min_weight": 0.4,
        }

    @patch("utils.api_client.requests.request")
    def test_unexpected_shape_raises(self, mock_request):
        mock_response = Mock()
        mock_response.headers = {}
        mock_response.iter_content = Mock(return_value=[])
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected": True}
        mock_request.return_value = mock_response

        client = CuracastClient("http://localhost:8000", "key123")
        with pytest.raises(CuracastAPIError, match="Unexpected response shape"):
            client.get_watch_credits_page()

    @patch("utils.api_client.requests.request")
    def test_malformed_json_raises_curacast_error(self, mock_request):
        import requests

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.iter_content = Mock(return_value=[])
        mock_response.status_code = 200
        # requests.Response.json() raises requests.exceptions.JSONDecodeError
        # on malformed JSON - a real RequestException subclass (unlike a bare
        # ValueError/json.JSONDecodeError), which is what makes
        # BaseAPIClient._make_request_to_url's own except clause translate it
        # into CuracastAPIError.
        mock_response.json.side_effect = requests.exceptions.JSONDecodeError("not json", "not json", 0)
        mock_request.return_value = mock_response

        client = CuracastClient("http://localhost:8000", "key123")
        with pytest.raises(CuracastAPIError):
            client.get_watch_credits_page()

    @patch("utils.api_client.requests.request")
    def test_connection_error_raises_curacast_error(self, mock_request):
        import requests

        mock_request.side_effect = requests.exceptions.ConnectionError()

        client = CuracastClient("http://localhost:8000", "key123")
        with pytest.raises(CuracastAPIError, match="Could not connect"):
            client.get_watch_credits_page()

    @patch("utils.api_client.requests.request")
    def test_401_raises_curacast_error(self, mock_request):
        mock_response = Mock()
        mock_response.headers = {}
        mock_response.iter_content = Mock(return_value=[])
        mock_response.status_code = 401
        mock_request.return_value = mock_response

        client = CuracastClient("http://localhost:8000", "bad-key")
        with pytest.raises(CuracastAPIError):
            client.get_watch_credits_page()


# ---------------------------------------------------------------------------
# create_curacast_client
# ---------------------------------------------------------------------------


class TestCreateCuracastClient:
    def test_disabled_by_default_returns_none(self):
        assert create_curacast_client({}) is None
        assert create_curacast_client({"curacast": {}}) is None

    def test_enabled_but_missing_url_returns_none(self, caplog):
        import logging

        config = {"curacast": {"enabled": True, "url": "", "api_key": "key123"}}
        with caplog.at_level(logging.WARNING, logger="curatarr"):
            assert create_curacast_client(config) is None
        assert "not configured" in caplog.text

    def test_enabled_but_missing_api_key_returns_none(self, caplog):
        import logging

        config = {"curacast": {"enabled": True, "url": "http://localhost:8000", "api_key": ""}}
        with caplog.at_level(logging.WARNING, logger="curatarr"):
            assert create_curacast_client(config) is None
        assert "not configured" in caplog.text
        # Never logs the actual key value, only that one is missing.
        assert "key123" not in caplog.text

    def test_enabled_and_configured_returns_client(self):
        config = {"curacast": {"enabled": True, "url": "http://localhost:8000", "api_key": "key123"}}
        client = create_curacast_client(config)
        assert isinstance(client, CuracastClient)
        assert client.base_url == "http://localhost:8000"
        assert client.api_key == "key123"


# ---------------------------------------------------------------------------
# Cursor persistence (compound: since + since_id)
# ---------------------------------------------------------------------------


class TestCursorPersistence:
    def test_load_cursor_defaults_to_zero(self, tmp_path):
        assert _load_cursor(str(tmp_path), "movie") == (0, 0)

    def test_save_then_load_round_trips(self, tmp_path):
        _save_cursor(str(tmp_path), "movie", 1787584815000, 42)
        assert _load_cursor(str(tmp_path), "movie") == (1787584815000, 42)

    def test_movie_and_tv_cursors_are_independent(self, tmp_path):
        _save_cursor(str(tmp_path), "movie", 100, 1)
        _save_cursor(str(tmp_path), "tv", 200, 2)
        assert _load_cursor(str(tmp_path), "movie") == (100, 1)
        assert _load_cursor(str(tmp_path), "tv") == (200, 2)

    def test_reads_old_format_cursor_file_missing_since_id(self, tmp_path):
        """Upgrade path: a cursor file written by the pre-compound-cursor
        format has 'since' but no 'since_id' key at all - must default to
        0, not crash."""
        import json

        cursor_path = tmp_path / "curacast_cursor_movie.json"
        cursor_path.write_text(json.dumps({"since": 999}), encoding="utf-8")

        assert _load_cursor(str(tmp_path), "movie") == (999, 0)


# ---------------------------------------------------------------------------
# get_watch_credits (pagination)
# ---------------------------------------------------------------------------


def _credit(id_, weight=1.0, ended_at=1000):
    return {"id": id_, "weight": weight, "ended_at": ended_at, "program_key": str(id_)}


class TestGetWatchCredits:
    def test_single_page_terminates_on_zero_count(self):
        client = Mock()
        client.get_watch_credits_page.return_value = {
            "credits": [],
            "next_since": 0,
            "next_since_id": 0,
            "count": 0,
        }

        credits, cursor, cursor_id = get_watch_credits(client, since=0)

        assert credits == []
        assert cursor == 0
        assert cursor_id == 0
        assert client.get_watch_credits_page.call_count == 1

    def test_paginates_across_multiple_pages(self):
        client = Mock()
        client.get_watch_credits_page.side_effect = [
            {"credits": [_credit(1), _credit(2)], "next_since": 100, "next_since_id": 2, "count": 2},
            {"credits": [_credit(3)], "next_since": 200, "next_since_id": 3, "count": 1},
            {"credits": [], "next_since": 200, "next_since_id": 3, "count": 0},
        ]

        credits, cursor, cursor_id = get_watch_credits(client, since=0)

        assert [c["id"] for c in credits] == [1, 2, 3]
        assert cursor == 200
        assert cursor_id == 3
        assert client.get_watch_credits_page.call_count == 3
        # (since, since_id) is passed through from the previous page's
        # (next_since, next_since_id) - both advance together.
        calls = client.get_watch_credits_page.call_args_list
        assert calls[0].kwargs["since"] == 0
        assert calls[0].kwargs["since_id"] == 0
        assert calls[1].kwargs["since"] == 100
        assert calls[1].kwargs["since_id"] == 2
        assert calls[2].kwargs["since"] == 200
        assert calls[2].kwargs["since_id"] == 3

    def test_hits_max_pages_safety_cap_without_raising(self):
        client = Mock()
        # Never terminates on its own (count always 1, cursor always advances by 1)
        call_count = {"n": 0}

        def _never_ending(since, since_id=0, username=None, min_weight=None):
            call_count["n"] += 1
            return {
                "credits": [_credit(call_count["n"])],
                "next_since": since + 1,
                "next_since_id": since_id + 1,
                "count": 1,
            }

        client.get_watch_credits_page.side_effect = _never_ending

        with patch("utils.curacast.log_warning") as mock_warn:
            credits, cursor, cursor_id = get_watch_credits(client, since=0, max_pages=5)

        assert client.get_watch_credits_page.call_count == 5
        assert len(credits) == 5
        assert cursor == 5
        assert cursor_id == 5
        assert any("safety cap" in str(call.args[0]) for call in mock_warn.call_args_list)

    def test_connection_error_returns_empty_list_no_raise(self):
        client = Mock()
        client.get_watch_credits_page.side_effect = CuracastAPIError("Could not connect to Curacast")

        credits, cursor, cursor_id = get_watch_credits(client, since=0)

        assert credits == []
        assert cursor is None
        assert cursor_id is None

    def test_malformed_json_returns_empty_list_no_raise(self):
        client = Mock()
        client.get_watch_credits_page.side_effect = CuracastAPIError("Curacast response rejected: bad json")

        credits, cursor, cursor_id = get_watch_credits(client, since=0)

        assert credits == []
        assert cursor is None
        assert cursor_id is None

    def test_failure_on_later_page_discards_partial_results(self):
        client = Mock()
        client.get_watch_credits_page.side_effect = [
            {"credits": [_credit(1)], "next_since": 100, "next_since_id": 1, "count": 1},
            CuracastAPIError("timeout"),
        ]

        credits, cursor, cursor_id = get_watch_credits(client, since=0)

        assert credits == []
        assert cursor is None
        assert cursor_id is None

    def test_nonzero_count_missing_next_since_stops_early(self):
        """A server bug (nonzero count with no next_since to advance by)
        must stop pagination rather than loop - the credits fetched so far
        are still usable, and BOTH cursor halves stay put for a retry."""
        client = Mock()
        client.get_watch_credits_page.return_value = {"credits": [_credit(1)], "count": 1}  # no next_since key

        with patch("utils.curacast.log_warning") as mock_warn:
            credits, cursor, cursor_id = get_watch_credits(client, since=0, since_id=5)

        assert len(credits) == 1
        assert cursor == 0
        assert cursor_id == 5  # unchanged - next_since never advanced, so neither does since_id
        assert client.get_watch_credits_page.call_count == 1
        assert mock_warn.called

    def test_missing_next_since_id_falls_back_to_zero_and_logs_debug(self, caplog):
        """Older curacast server that doesn't speak the compound cursor yet:
        next_since is present but next_since_id simply isn't in the
        response at all. Must fall back to 0, keep working (not abort the
        fetch), and say so at debug level every time - not silently
        degrade forever with no signal at all."""
        import logging

        client = Mock()
        client.get_watch_credits_page.side_effect = [
            {"credits": [_credit(1)], "next_since": 100, "count": 1},  # no next_since_id key
            {"credits": [], "next_since": 100, "next_since_id": 0, "count": 0},
        ]

        with caplog.at_level(logging.DEBUG, logger="curatarr"):
            credits, cursor, cursor_id = get_watch_credits(client, since=0)

        assert len(credits) == 1
        assert cursor == 100
        assert cursor_id == 0
        assert "next_since_id" in caplog.text

        # "Keep working" means the fallback is actually SENT on the next
        # request, not just silently substituted into the return value.
        second_call_kwargs = client.get_watch_credits_page.call_args_list[1].kwargs
        assert second_call_kwargs["since_id"] == 0


# ---------------------------------------------------------------------------
# apply_watch_credits (profile integration)
# ---------------------------------------------------------------------------


def _plex_movie_item(rating_key, item_type="movie"):
    item = Mock()
    item.type = item_type
    item.ratingKey = rating_key
    return item


def _plex_episode_item(rating_key, grandparent_rating_key):
    item = Mock()
    item.type = "episode"
    item.ratingKey = rating_key
    item.grandparentRatingKey = grandparent_rating_key
    return item


class TestApplyWatchCreditsDisabled:
    def test_disabled_config_is_a_full_noop(self):
        """With curacast.enabled: false, apply_watch_credits must never touch
        Plex, never fetch credits, and never modify counters - the code path
        is completely inert."""
        counters = create_empty_counters("movie")
        plex = Mock()

        applied = apply_watch_credits(
            config={"curacast": {"enabled": False}},
            counters=counters,
            media_type="movie",
            watched_ids=set(),
            media_info_cache={},
            plex=plex,
            cache_dir="/does/not/matter",
        )

        assert applied == 0
        assert counters["genres"] == Counter()
        plex.fetchItem.assert_not_called()

    def test_missing_curacast_section_is_a_full_noop(self):
        counters = create_empty_counters("movie")
        plex = Mock()

        applied = apply_watch_credits(
            config={},
            counters=counters,
            media_type="movie",
            watched_ids=set(),
            media_info_cache={},
            plex=plex,
            cache_dir="/does/not/matter",
        )

        assert applied == 0
        plex.fetchItem.assert_not_called()


class TestApplyWatchCreditsWeighting:
    def _config(self, min_weight=-1.0):
        return {
            "curacast": {
                "enabled": True,
                "url": "http://localhost:8000",
                "api_key": "key123",
                "min_weight": min_weight,
            },
            "recency_decay": {"enabled": False},
            "negative_signals": {"bad_ratings": {"cap_penalty": 0.5}},
        }

    def test_complete_tier_weight_applied_with_recency_disabled(self, tmp_path):
        counters = create_empty_counters("movie")
        plex = Mock()
        plex.fetchItem.return_value = _plex_movie_item(rating_key=555)
        media_info_cache = {"555": {"genres": ["drama"], "tmdb_id": 999}}

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "555", "weight": 1.0, "ended_at": 1000}], 1000, 0),
        ):
            applied = apply_watch_credits(
                config=self._config(),
                counters=counters,
                media_type="movie",
                watched_ids=set(),
                media_info_cache=media_info_cache,
                plex=plex,
                cache_dir=str(tmp_path),
            )

        assert applied == 1
        # recency_decay disabled -> multiplier 1.0, no rating/rewatch applied
        # at all -> counter should be exactly the credit's own weight.
        assert counters["genres"]["drama"] == pytest.approx(1.0)
        assert 999 in counters["tmdb_ids"]

    def test_partial_tier_weight_applied(self, tmp_path):
        counters = create_empty_counters("movie")
        plex = Mock()
        plex.fetchItem.return_value = _plex_movie_item(rating_key=555)
        media_info_cache = {"555": {"genres": ["drama"]}}

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "555", "weight": 0.4, "ended_at": 1000}], 1000, 0),
        ):
            apply_watch_credits(
                config=self._config(),
                counters=counters,
                media_type="movie",
                watched_ids=set(),
                media_info_cache=media_info_cache,
                plex=plex,
                cache_dir=str(tmp_path),
            )

        assert counters["genres"]["drama"] == pytest.approx(0.4)

    def test_sampled_tier_negative_weight_reduces_counter(self, tmp_path):
        """A negative (sampled, -0.3) credit must produce a NEGATIVE
        contribution, capped the same way bad_ratings.cap_penalty caps any
        other negative signal (utils.counters._apply_capped_weight)."""
        counters = create_empty_counters("movie")
        counters["genres"]["drama"] = 10.0  # pre-existing strong positive signal
        plex = Mock()
        plex.fetchItem.return_value = _plex_movie_item(rating_key=555)
        media_info_cache = {"555": {"genres": ["drama"]}}

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "555", "weight": -0.3, "ended_at": 1000}], 1000, 0),
        ):
            apply_watch_credits(
                config=self._config(),
                counters=counters,
                media_type="movie",
                watched_ids=set(),
                media_info_cache=media_info_cache,
                plex=plex,
                cache_dir=str(tmp_path),
            )

        # Reduced, but never below cap_penalty (0.5) * the pre-existing value
        assert counters["genres"]["drama"] < 10.0
        assert counters["genres"]["drama"] >= 10.0 * 0.5

    def test_recency_multiplier_is_applied(self, tmp_path):
        """A credit far in the past must be weighted less than one that just
        happened, via the same recency_decay config every other watched item
        uses."""
        media_info_cache = {"555": {"genres": ["drama"]}}
        plex = Mock()
        plex.fetchItem.return_value = _plex_movie_item(rating_key=555)

        config = {
            "curacast": {"enabled": True, "url": "http://x", "api_key": "k"},
            "recency_decay": {
                "enabled": True,
                "days_0_30": 1.0,
                "days_31_90": 0.5,
                "days_91_180": 0.25,
                "days_180_plus": 0.1,
            },
            "negative_signals": {"bad_ratings": {"cap_penalty": 0.5}},
        }

        import time

        now_ms = int(time.time() * 1000)
        old_ms = now_ms - (200 * 24 * 60 * 60 * 1000)  # 200 days ago

        counters_recent = create_empty_counters("movie")
        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "555", "weight": 1.0, "ended_at": now_ms}], now_ms, 0),
        ):
            apply_watch_credits(config, counters_recent, "movie", set(), media_info_cache, plex, str(tmp_path))

        counters_old = create_empty_counters("movie")
        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "555", "weight": 1.0, "ended_at": old_ms}], old_ms, 0),
        ):
            apply_watch_credits(config, counters_old, "movie", set(), media_info_cache, plex, str(tmp_path))

        assert counters_old["genres"]["drama"] < counters_recent["genres"]["drama"]


class TestApplyWatchCreditsDedup:
    def test_movie_already_in_watched_ids_is_skipped(self, tmp_path):
        counters = create_empty_counters("movie")
        plex = Mock()
        plex.fetchItem.return_value = _plex_movie_item(rating_key=555)
        media_info_cache = {"555": {"genres": ["drama"]}}

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "555", "weight": 1.0, "ended_at": 1000}], 1000, 0),
        ):
            applied = apply_watch_credits(
                config={"curacast": {"enabled": True, "url": "http://x", "api_key": "k"}},
                counters=counters,
                media_type="movie",
                watched_ids={555},  # already counted via normal Plex history
                media_info_cache=media_info_cache,
                plex=plex,
                cache_dir=str(tmp_path),
            )

        assert applied == 0
        assert counters["genres"]["drama"] == 0

    def test_tv_episode_dedups_against_show_id_not_episode_id(self, tmp_path):
        counters = create_empty_counters("tv")
        plex = Mock()
        plex.fetchItem.return_value = _plex_episode_item(rating_key=9001, grandparent_rating_key=42)
        media_info_cache = {"42": {"genres": ["comedy"]}}

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "9001", "weight": 1.0, "ended_at": 1000}], 1000, 0),
        ):
            applied = apply_watch_credits(
                config={"curacast": {"enabled": True, "url": "http://x", "api_key": "k"}},
                counters=counters,
                media_type="tv",
                watched_ids={42},  # the SHOW is already watched via Plex history
                media_info_cache=media_info_cache,
                plex=plex,
                cache_dir=str(tmp_path),
            )

        assert applied == 0
        assert counters["genres"]["comedy"] == 0


class TestApplyWatchCreditsExclusion:
    """Credits at/above curacast.exclude_at_weight (default 0.8) must mark
    their resolved item as watched for recommendation exclusion - not just
    contribute to scoring counters. See apply_watch_credits' own docstring
    for why this REQUIRES the caller's real, live watched_ids set (movie.py/
    tv.py pass self.watched_ids, never their own local snapshot)."""

    def test_weight_above_threshold_adds_to_watched_ids(self, tmp_path):
        watched_ids = set()
        plex = Mock()
        plex.fetchItem.return_value = _plex_movie_item(rating_key=555)

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "555", "weight": 1.0, "ended_at": 1000}], 1000, 0),
        ):
            apply_watch_credits(
                config={"curacast": {"enabled": True, "url": "http://x", "api_key": "k"}},
                counters=create_empty_counters("movie"),
                media_type="movie",
                watched_ids=watched_ids,
                media_info_cache={"555": {"genres": ["drama"]}},
                plex=plex,
                cache_dir=str(tmp_path),
            )

        assert 555 in watched_ids

    def test_weight_exactly_at_threshold_adds_to_watched_ids(self, tmp_path):
        """>= is inclusive: exactly 0.8 ("substantial") must exclude."""
        watched_ids = set()
        plex = Mock()
        plex.fetchItem.return_value = _plex_movie_item(rating_key=555)

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "555", "weight": 0.8, "ended_at": 1000}], 1000, 0),
        ):
            apply_watch_credits(
                config={"curacast": {"enabled": True, "url": "http://x", "api_key": "k"}},
                counters=create_empty_counters("movie"),
                media_type="movie",
                watched_ids=watched_ids,
                media_info_cache={"555": {"genres": ["drama"]}},
                plex=plex,
                cache_dir=str(tmp_path),
            )

        assert 555 in watched_ids

    def test_weight_below_threshold_does_not_add(self, tmp_path):
        """A 'partial' (0.4) credit means they bailed early - stays recommendable."""
        watched_ids = set()
        plex = Mock()
        plex.fetchItem.return_value = _plex_movie_item(rating_key=555)

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "555", "weight": 0.4, "ended_at": 1000}], 1000, 0),
        ):
            apply_watch_credits(
                config={"curacast": {"enabled": True, "url": "http://x", "api_key": "k"}},
                counters=create_empty_counters("movie"),
                media_type="movie",
                watched_ids=watched_ids,
                media_info_cache={"555": {"genres": ["drama"]}},
                plex=plex,
                cache_dir=str(tmp_path),
            )

        assert watched_ids == set()

    def test_sampled_negative_credit_never_excludes(self, tmp_path):
        watched_ids = set()
        plex = Mock()
        plex.fetchItem.return_value = _plex_movie_item(rating_key=555)

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "555", "weight": -0.3, "ended_at": 1000}], 1000, 0),
        ):
            apply_watch_credits(
                config={"curacast": {"enabled": True, "url": "http://x", "api_key": "k"}},
                counters=create_empty_counters("movie"),
                media_type="movie",
                watched_ids=watched_ids,
                media_info_cache={"555": {"genres": ["drama"]}},
                plex=plex,
                cache_dir=str(tmp_path),
            )

        assert watched_ids == set()

    def test_threshold_is_configurable(self, tmp_path):
        """Lowering exclude_at_weight admits a credit the default would not."""
        watched_ids = set()
        plex = Mock()
        plex.fetchItem.return_value = _plex_movie_item(rating_key=555)
        config = {
            "curacast": {"enabled": True, "url": "http://x", "api_key": "k", "exclude_at_weight": 0.3},
        }

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "555", "weight": 0.4, "ended_at": 1000}], 1000, 0),
        ):
            apply_watch_credits(
                config=config,
                counters=create_empty_counters("movie"),
                media_type="movie",
                watched_ids=watched_ids,
                media_info_cache={"555": {"genres": ["drama"]}},
                plex=plex,
                cache_dir=str(tmp_path),
            )

        assert 555 in watched_ids

    def test_raising_threshold_excludes_a_default_qualifying_credit(self, tmp_path):
        watched_ids = set()
        plex = Mock()
        plex.fetchItem.return_value = _plex_movie_item(rating_key=555)
        config = {
            "curacast": {"enabled": True, "url": "http://x", "api_key": "k", "exclude_at_weight": 0.95},
        }

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "555", "weight": 0.8, "ended_at": 1000}], 1000, 0),
        ):
            apply_watch_credits(
                config=config,
                counters=create_empty_counters("movie"),
                media_type="movie",
                watched_ids=watched_ids,
                media_info_cache={"555": {"genres": ["drama"]}},
                plex=plex,
                cache_dir=str(tmp_path),
            )

        assert watched_ids == set()

    def test_exclusion_uses_the_exact_object_passed_in_not_a_copy(self, tmp_path):
        """The whole point: mutation must be visible to the caller's own
        reference, not just to a local variable inside apply_watch_credits."""
        caller_watched_ids = set()
        plex = Mock()
        plex.fetchItem.return_value = _plex_movie_item(rating_key=555)

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "555", "weight": 1.0, "ended_at": 1000}], 1000, 0),
        ):
            apply_watch_credits(
                config={"curacast": {"enabled": True, "url": "http://x", "api_key": "k"}},
                counters=create_empty_counters("movie"),
                media_type="movie",
                watched_ids=caller_watched_ids,
                media_info_cache={"555": {"genres": ["drama"]}},
                plex=plex,
                cache_dir=str(tmp_path),
            )

        # Same object identity, mutated in place - not a replaced reference.
        assert 555 in caller_watched_ids

    def test_exclusion_applies_even_without_a_cache_hit(self, tmp_path):
        """Exclusion is a factual "did they watch it" question, independent
        of whether TMDB/library metadata happens to be cached for scoring."""
        watched_ids = set()
        plex = Mock()
        plex.fetchItem.return_value = _plex_movie_item(rating_key=555)

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "555", "weight": 1.0, "ended_at": 1000}], 1000, 0),
        ):
            applied = apply_watch_credits(
                config={"curacast": {"enabled": True, "url": "http://x", "api_key": "k"}},
                counters=create_empty_counters("movie"),
                media_type="movie",
                watched_ids=watched_ids,
                media_info_cache={},  # no metadata cached for 555
                plex=plex,
                cache_dir=str(tmp_path),
            )

        assert 555 in watched_ids
        assert applied == 0  # nothing scored - no metadata to score with

    def test_dedup_still_works_within_the_same_batch_after_exclusion(self, tmp_path):
        """Two credits for the same show in one fetch: the first crosses the
        exclusion threshold and is added to watched_ids; the second (same
        resolved item_id) must then be deduped, not double-counted."""
        counters = create_empty_counters("tv")
        plex = Mock()
        plex.fetchItem.return_value = _plex_episode_item(rating_key=9001, grandparent_rating_key=42)
        watched_ids = set()

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=(
                [
                    {"program_key": "9001", "weight": 1.0, "ended_at": 1000},
                    {"program_key": "9001", "weight": 1.0, "ended_at": 2000},
                ],
                2000,
                0,
            ),
        ):
            applied = apply_watch_credits(
                config={
                    "curacast": {"enabled": True, "url": "http://x", "api_key": "k"},
                    "recency_decay": {"enabled": False},
                },
                counters=counters,
                media_type="tv",
                watched_ids=watched_ids,
                media_info_cache={"42": {"genres": ["comedy"]}},
                plex=plex,
                cache_dir=str(tmp_path),
            )

        assert 42 in watched_ids
        # Only the FIRST credit is counted - the second is deduped against
        # the id the first one just added to watched_ids.
        assert applied == 1
        assert counters["genres"]["comedy"] == pytest.approx(1.0)

    def test_tv_episode_exclusion_marks_the_whole_show_matching_history_semantics(self, tmp_path):
        """fetch_plex_watch_history_shows() (utils/plex.py) already adds the
        SHOW's grandparentRatingKey to watched_ids for a single watched
        episode - curatarr's watched_ids has always been show-granularity
        for TV, never episode-level. A curacast episode credit must match
        that exactly: one substantial-or-better episode credit marks the
        WHOLE SHOW watched, same as one Plex-history episode does."""
        watched_ids = set()
        plex = Mock()
        plex.fetchItem.return_value = _plex_episode_item(rating_key=9001, grandparent_rating_key=42)

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "9001", "weight": 1.0, "ended_at": 1000}], 1000, 0),
        ):
            apply_watch_credits(
                config={"curacast": {"enabled": True, "url": "http://x", "api_key": "k"}},
                counters=create_empty_counters("tv"),
                media_type="tv",
                watched_ids=watched_ids,
                media_info_cache={"42": {"genres": ["comedy"]}},
                plex=plex,
                cache_dir=str(tmp_path),
            )

        assert watched_ids == {42}  # the SHOW id, never the episode's own ratingKey (9001)


class TestApplyWatchCreditsResolution:
    def test_episode_credit_resolved_to_show_counters(self, tmp_path):
        counters = create_empty_counters("tv")
        plex = Mock()
        plex.fetchItem.return_value = _plex_episode_item(rating_key=9001, grandparent_rating_key=42)
        media_info_cache = {"42": {"genres": ["comedy"], "tmdb_id": 777}}

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "9001", "weight": 1.0, "ended_at": 1000}], 1000, 0),
        ):
            applied = apply_watch_credits(
                config={
                    "curacast": {"enabled": True, "url": "http://x", "api_key": "k"},
                    "recency_decay": {"enabled": False},
                },
                counters=counters,
                media_type="tv",
                watched_ids=set(),
                media_info_cache=media_info_cache,
                plex=plex,
                cache_dir=str(tmp_path),
            )

        assert applied == 1
        assert counters["genres"]["comedy"] == pytest.approx(1.0)
        assert 777 in counters["tmdb_ids"]

    def test_unresolvable_rating_key_is_skipped_not_raised(self, tmp_path, caplog):
        import logging

        counters = create_empty_counters("movie")
        plex = Mock()
        plex.fetchItem.side_effect = Exception("Not found")
        media_info_cache = {"555": {"genres": ["drama"]}}

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "555", "weight": 1.0, "ended_at": 1000}], 1000, 0),
        ):
            with caplog.at_level(logging.DEBUG, logger="curatarr"):
                applied = apply_watch_credits(
                    config={"curacast": {"enabled": True, "url": "http://x", "api_key": "k"}},
                    counters=counters,
                    media_type="movie",
                    watched_ids=set(),
                    media_info_cache=media_info_cache,
                    plex=plex,
                    cache_dir=str(tmp_path),
                )

        assert applied == 0
        assert counters["genres"]["drama"] == 0

    def test_cache_miss_is_skipped_not_raised(self, tmp_path):
        counters = create_empty_counters("movie")
        plex = Mock()
        plex.fetchItem.return_value = _plex_movie_item(rating_key=555)

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "555", "weight": 1.0, "ended_at": 1000}], 1000, 0),
        ):
            applied = apply_watch_credits(
                config={"curacast": {"enabled": True, "url": "http://x", "api_key": "k"}},
                counters=counters,
                media_type="movie",
                watched_ids=set(),
                media_info_cache={},  # ratingKey 555 not in cache
                plex=plex,
                cache_dir=str(tmp_path),
            )

        assert applied == 0

    def test_wrong_media_type_credit_is_skipped(self, tmp_path):
        """A movie recommender run must ignore episode credits (and vice
        versa) - each recommender only processes the credits relevant to it,
        the other type is picked up by that recommender's own independent
        cursor."""
        counters = create_empty_counters("movie")
        plex = Mock()
        plex.fetchItem.return_value = _plex_episode_item(rating_key=9001, grandparent_rating_key=42)

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "9001", "weight": 1.0, "ended_at": 1000}], 1000, 0),
        ):
            applied = apply_watch_credits(
                config={"curacast": {"enabled": True, "url": "http://x", "api_key": "k"}},
                counters=counters,
                media_type="movie",
                watched_ids=set(),
                media_info_cache={"42": {"genres": ["comedy"]}},
                plex=plex,
                cache_dir=str(tmp_path),
            )

        assert applied == 0

    def test_movie_credit_skipped_on_a_tv_run(self, tmp_path):
        """The reverse of test_wrong_media_type_credit_is_skipped: a TV
        recommender run must ignore movie credits too."""
        counters = create_empty_counters("tv")
        plex = Mock()
        plex.fetchItem.return_value = _plex_movie_item(rating_key=555)

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "555", "weight": 1.0, "ended_at": 1000}], 1000, 0),
        ):
            applied = apply_watch_credits(
                config={"curacast": {"enabled": True, "url": "http://x", "api_key": "k"}},
                counters=counters,
                media_type="tv",
                watched_ids=set(),
                media_info_cache={"555": {"genres": ["drama"]}},
                plex=plex,
                cache_dir=str(tmp_path),
            )

        assert applied == 0

    def test_episode_with_no_parent_show_is_skipped(self, tmp_path):
        counters = create_empty_counters("tv")
        plex = Mock()
        item = Mock()
        item.type = "episode"
        item.ratingKey = 9001
        item.grandparentRatingKey = None
        plex.fetchItem.return_value = item

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=([{"program_key": "9001", "weight": 1.0, "ended_at": 1000}], 1000, 0),
        ):
            applied = apply_watch_credits(
                config={"curacast": {"enabled": True, "url": "http://x", "api_key": "k"}},
                counters=counters,
                media_type="tv",
                watched_ids=set(),
                media_info_cache={},
                plex=plex,
                cache_dir=str(tmp_path),
            )

        assert applied == 0

    def test_credit_missing_program_key_or_weight_is_skipped(self, tmp_path):
        counters = create_empty_counters("movie")
        plex = Mock()

        with patch(
            "utils.curacast.get_watch_credits",
            return_value=(
                [
                    {"program_key": None, "weight": 1.0, "ended_at": 1000},
                    {"program_key": "555", "weight": None, "ended_at": 1000},
                ],
                1000,
                0,
            ),
        ):
            applied = apply_watch_credits(
                config={"curacast": {"enabled": True, "url": "http://x", "api_key": "k"}},
                counters=counters,
                media_type="movie",
                watched_ids=set(),
                media_info_cache={"555": {"genres": ["drama"]}},
                plex=plex,
                cache_dir=str(tmp_path),
            )

        assert applied == 0
        plex.fetchItem.assert_not_called()


class TestApplyWatchCreditsCursor:
    def test_cursor_persisted_and_reused_across_calls(self, tmp_path):
        counters = create_empty_counters("movie")
        plex = Mock()
        plex.fetchItem.return_value = _plex_movie_item(rating_key=555)
        media_info_cache = {"555": {"genres": ["drama"]}}
        config = {"curacast": {"enabled": True, "url": "http://x", "api_key": "k"}}

        with patch("utils.curacast.get_watch_credits") as mock_get:
            mock_get.return_value = ([{"program_key": "555", "weight": 1.0, "ended_at": 1000}], 12345, 7)
            apply_watch_credits(config, counters, "movie", set(), media_info_cache, plex, str(tmp_path))
            assert mock_get.call_args.kwargs["since"] == 0  # first run starts at 0
            assert mock_get.call_args.kwargs["since_id"] == 0

            mock_get.return_value = ([], 12345, 7)
            apply_watch_credits(config, counters, "movie", set(), media_info_cache, plex, str(tmp_path))
            assert mock_get.call_args.kwargs["since"] == 12345  # second run resumes from persisted cursor
            assert mock_get.call_args.kwargs["since_id"] == 7

    def test_failed_fetch_does_not_move_cursor(self, tmp_path):
        counters = create_empty_counters("movie")
        plex = Mock()
        config = {"curacast": {"enabled": True, "url": "http://x", "api_key": "k"}}

        with patch("utils.curacast.get_watch_credits", return_value=([], None, None)):
            apply_watch_credits(config, counters, "movie", set(), {}, plex, str(tmp_path))

        assert _load_cursor(str(tmp_path), "movie") == (0, 0)


class TestApplyWatchCreditsUsernameFiltering:
    def test_blank_config_username_falls_back_to_plex_username(self, tmp_path):
        plex = Mock()
        config = {"curacast": {"enabled": True, "url": "http://x", "api_key": "k", "username": ""}}

        with patch("utils.curacast.get_watch_credits", return_value=([], None, None)) as mock_get:
            apply_watch_credits(
                config,
                create_empty_counters("movie"),
                "movie",
                set(),
                {},
                plex,
                str(tmp_path),
                plex_username="testuser",
            )

        assert mock_get.call_args.kwargs["username"] == "testuser"

    def test_explicit_config_username_overrides_plex_username(self, tmp_path):
        plex = Mock()
        config = {"curacast": {"enabled": True, "url": "http://x", "api_key": "k", "username": "curacast_viewer"}}

        with patch("utils.curacast.get_watch_credits", return_value=([], None, None)) as mock_get:
            apply_watch_credits(
                config,
                create_empty_counters("movie"),
                "movie",
                set(),
                {},
                plex,
                str(tmp_path),
                plex_username="testuser",
            )

        assert mock_get.call_args.kwargs["username"] == "curacast_viewer"
