"""Tests for testdata_ai.faker_bridge — apply_faker_fields."""
from unittest.mock import MagicMock, patch

import pytest

from testdata_ai.faker_bridge import apply_faker_fields


@pytest.fixture
def fake_faker_cls():
    """Return a mock Faker class whose instances expose a few standard methods."""
    fake = MagicMock()
    fake.email.return_value = "test@example.com"
    fake.phone_number.return_value = "+48 123 456 789"
    fake.first_name.return_value = "Anna"

    cls = MagicMock(return_value=fake)
    return cls, fake


class TestApplyFakerFields:

    def test_basic_overwrite(self, fake_faker_cls):
        cls, fake = fake_faker_cls
        records = [{"name": "Jan", "email": "old@x.com"}]
        with patch("testdata_ai.faker_bridge.Faker", cls):
            result = apply_faker_fields(records, {"email": "faker:email"})
        assert result[0]["email"] == "test@example.com"
        assert result[0]["name"] == "Jan"  # AI value preserved

    def test_multiple_fields(self, fake_faker_cls):
        cls, fake = fake_faker_cls
        records = [{"name": "Jan", "email": "old@x.com", "phone": "000"}]
        with patch("testdata_ai.faker_bridge.Faker", cls):
            result = apply_faker_fields(
                records,
                {"email": "faker:email", "phone": "faker:phone_number"},
            )
        assert result[0]["email"] == "test@example.com"
        assert result[0]["phone"] == "+48 123 456 789"

    def test_multiple_records(self, fake_faker_cls):
        cls, fake = fake_faker_cls
        records = [{"email": "a@a.com"}, {"email": "b@b.com"}]
        with patch("testdata_ai.faker_bridge.Faker", cls):
            result = apply_faker_fields(records, {"email": "faker:email"})
        assert len(result) == 2
        assert fake.email.call_count == 2

    def test_original_records_unchanged(self, fake_faker_cls):
        cls, fake = fake_faker_cls
        original = {"email": "orig@x.com", "name": "Jan"}
        records = [original]
        with patch("testdata_ai.faker_bridge.Faker", cls):
            apply_faker_fields(records, {"email": "faker:email"})
        assert original["email"] == "orig@x.com"  # shallow copy — original untouched

    def test_locale_passed_to_faker(self, fake_faker_cls):
        cls, fake = fake_faker_cls
        records = [{"email": "x@x.com"}]
        with patch("testdata_ai.faker_bridge.Faker", cls):
            apply_faker_fields(records, {"email": "faker:email"}, locale="pl_PL")
        cls.assert_called_once_with("pl_PL")

    def test_no_locale_uses_default(self, fake_faker_cls):
        cls, fake = fake_faker_cls
        records = [{"email": "x@x.com"}]
        with patch("testdata_ai.faker_bridge.Faker", cls):
            apply_faker_fields(records, {"email": "faker:email"}, locale=None)
        cls.assert_called_once_with()  # Faker() without args

    def test_unknown_method_raises(self, fake_faker_cls):
        cls, fake = fake_faker_cls
        fake.nonexistent_xyz_method = None  # not a callable
        # getattr returns None → should raise ValueError
        with patch("testdata_ai.faker_bridge.Faker", cls):
            with pytest.raises(ValueError, match="Faker has no method"):
                apply_faker_fields([{"f": 1}], {"f": "faker:nonexistent_xyz_method"})

    def test_invalid_spec_no_method_raises(self):
        """'faker' without ':method' is invalid at apply time too."""
        # The regex won't match — ValueError from format check
        with pytest.raises(ValueError, match="Invalid provider spec"):
            # Need Faker to be importable, so just patch it
            with patch("testdata_ai.faker_bridge.Faker", MagicMock()):
                apply_faker_fields([{"f": 1}], {"f": "faker"})

    def test_invalid_spec_bad_format_raises(self):
        with pytest.raises(ValueError, match="Invalid provider spec"):
            with patch("testdata_ai.faker_bridge.Faker", MagicMock()):
                apply_faker_fields([{"f": 1}], {"f": "random:email"})

    def test_faker_not_installed_raises_import_error(self):
        with patch("testdata_ai.faker_bridge.Faker", None):
            with pytest.raises(ImportError, match="pip install"):
                apply_faker_fields([{"email": "x"}], {"email": "faker:email"})
