"""Tests for error hierarchy."""

import pytest

from meganova.errors import MeganovaError, APIError, AuthenticationError, RateLimitError


class TestMeganovaError:
    def test_basic_message(self):
        err = MeganovaError("something broke")
        assert str(err) == "something broke"
        assert err.code is None
        assert err.status is None

    def test_with_code_and_status(self):
        err = MeganovaError("bad", code="invalid_request", status=400)
        assert err.code == "invalid_request"
        assert err.status == 400

    def test_is_exception(self):
        assert issubclass(MeganovaError, Exception)


class TestAPIError:
    def test_inherits_meganova_error(self):
        assert issubclass(APIError, MeganovaError)

    def test_with_status(self):
        err = APIError("not found", status=404)
        assert err.status == 404


class TestAuthenticationError:
    def test_inherits_meganova_error(self):
        assert issubclass(AuthenticationError, MeganovaError)

    def test_with_401(self):
        err = AuthenticationError("invalid key", code="auth_error", status=401)
        assert err.status == 401
        assert err.code == "auth_error"


class TestRateLimitError:
    def test_inherits_meganova_error(self):
        assert issubclass(RateLimitError, MeganovaError)

    def test_with_429(self):
        err = RateLimitError("slow down", status=429)
        assert err.status == 429

    def test_can_be_caught_as_meganova_error(self):
        with pytest.raises(MeganovaError):
            raise RateLimitError("too fast", status=429)

    def test_can_be_caught_specifically(self):
        with pytest.raises(RateLimitError):
            raise RateLimitError("too fast", status=429)
