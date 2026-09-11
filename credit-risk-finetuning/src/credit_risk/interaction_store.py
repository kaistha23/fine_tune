from credit_risk.audit_store import AuditStore
from credit_risk.schemas import InteractionRecord


class InteractionNotFound(LookupError):
    pass


class InteractionStore:
    def __init__(self, path):
        self.path = path
        self.store = AuditStore(path)

    def append(self, record):
        self.store.append("interaction", record.interaction_id, record.model_dump_json())

    def get(self, interaction_id):
        rows = self.store.read("interaction", interaction_id)
        if not rows:
            raise InteractionNotFound(interaction_id)
        return InteractionRecord.model_validate_json(rows[-1])
