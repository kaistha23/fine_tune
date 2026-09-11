from credit_risk.audit_store import AuditStore
from credit_risk.schemas import FeedbackRecord


class FeedbackStore:
    def __init__(self, path):
        self.path = path
        self.store = AuditStore(path)

    def append(self, record):
        self.store.append("feedback", record.interaction_id, record.model_dump_json())

    def read_all(self):
        return [FeedbackRecord.model_validate_json(s) for s in self.store.read("feedback")]
