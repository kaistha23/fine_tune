import pytest

from credit_risk.settings import settings


@pytest.fixture(autouse=True)
def isolated_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "review_store_path", tmp_path / "reviews.sqlite3")
    monkeypatch.setattr(settings, "audit_store_path", tmp_path / "audit.sqlite3")
    # Individual tests may configure identities; restore settings after each test.
    monkeypatch.setattr(settings, "reviewers", dict(settings.reviewers))
    monkeypatch.setattr(settings, "service_token", settings.service_token)

    from credit_risk import api
    from credit_risk.feedback_store import FeedbackStore
    from credit_risk.interaction_store import InteractionStore

    monkeypatch.setattr(api, "feedback_store", FeedbackStore(tmp_path / "audit.sqlite3"))
    monkeypatch.setattr(api, "interaction_store", InteractionStore(tmp_path / "audit.sqlite3"))
