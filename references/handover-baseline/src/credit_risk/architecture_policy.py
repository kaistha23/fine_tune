from __future__ import annotations

from pathlib import Path

import yaml


class ArchitecturePolicyError(PermissionError):
    pass


class ArchitecturePolicy:
    """Fail-closed service capability policy loaded from an immutable YAML file."""

    def __init__(self, path: str | Path, expected_version: str | None = None):
        with Path(path).open("r", encoding="utf-8") as handle:
            self.data = yaml.safe_load(handle)
        self.version = str(self.data.get("version", ""))
        if expected_version and self.version != expected_version:
            raise ArchitecturePolicyError(
                f"Architecture policy version {self.version!r} does not match "
                f"required version {expected_version!r}"
            )
        if self.data.get("default_deny") is not True:
            raise ArchitecturePolicyError("Architecture policy must be default-deny")

    def require(self, role: str, capability: str) -> None:
        role_policy = self.data.get("roles", {}).get(role)
        if not role_policy:
            raise ArchitecturePolicyError(f"Unknown service role: {role}")
        if capability in role_policy.get("denied", []):
            raise ArchitecturePolicyError(f"Capability {capability!r} is denied for {role!r}")
        if capability not in role_policy.get("allowed", []):
            raise ArchitecturePolicyError(
                f"Capability {capability!r} is not explicitly allowed for {role!r}"
            )


