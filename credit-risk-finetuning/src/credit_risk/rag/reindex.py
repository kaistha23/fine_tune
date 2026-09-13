"""Build fresh collections and verify all citations before producing a switchable policy."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import fields
from pathlib import Path

import yaml

from credit_risk.rag.ingest import DocumentMeta, chunk_document
from credit_risk.rag.index import QdrantPolicyIndex, from_payload, validate_chunk_ids
from credit_risk.review_store import digest
from credit_risk.schemas import Jurisdiction


def rebuild(documents, index, namespaces):
    if any(index.client.collection_exists(name) for name in namespaces.values()):
        raise ValueError("Reindex requires entirely new collections; old collections are retained")
    chunks = []
    for document in documents:
        meta = dict(document["meta"])
        from datetime import date
        for key in ("effective_from", "effective_to"):
            if meta.get(key):
                meta[key] = date.fromisoformat(meta[key])
        meta["jurisdiction"] = Jurisdiction(meta["jurisdiction"])
        if set(meta) - {f.name for f in fields(DocumentMeta)}:
            raise ValueError("Unknown document metadata")
        chunks.extend(chunk_document(document["text"], DocumentMeta(**meta)))
    if not chunks:
        raise ValueError("No source chunks")
    validate_chunk_ids(chunks)
    if any(c.jurisdiction.value not in namespaces for c in chunks):
        raise ValueError("Missing jurisdiction namespace")
    index.upsert(chunks, namespaces=namespaces)
    observed = []
    for name in namespaces.values():
        index.ensure_collection(name)
        offset = None
        while True:
            points, offset = index.client.scroll(collection_name=name, offset=offset,
                                                  limit=256, with_payload=True)
            observed.extend(from_payload(p.payload) for p in points)
            if offset is None:
                break
    expected = {c.evidence_id: c.model_dump(mode="json") for c in chunks}
    actual = {c.evidence_id: c.model_dump(mode="json") for c in observed}
    if len(observed) != len(expected) or expected != actual:
        raise ValueError("Reindex verification failed; do not switch collections")
    return {"version": 2, "namespaces": namespaces,
            "embedding_signature": index.embedder.signature,
            "document_counts": dict(Counter(c.document_id for c in observed)),
            "chunk_count": len(observed), "citation_payload_hash": digest(actual),
            "source_hash": digest(documents)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True, type=Path, help="JSONL {text, meta}")
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--policy", default=Path("configs/retrieval.yaml"), type=Path)
    parser.add_argument("--out", required=True, type=Path, help="New directory for manifest and policy")
    args = parser.parse_args()
    if args.out.exists() or not re.fullmatch(r"[a-z0-9_]+", args.suffix):
        parser.error("New output directory and lowercase alphanumeric suffix required")
    from credit_risk.settings import settings
    from credit_risk.rag.factory import build_embedder
    if not settings.qdrant_url or settings.offline_test_mode:
        raise ValueError("Configured Qdrant and semantic embeddings required")
    policy = yaml.safe_load(args.policy.read_text())
    old = dict(policy["namespaces"])
    new = {k: f"{v}_v2_{args.suffix}" for k, v in old.items()}
    documents = [json.loads(line) for line in args.documents.read_text().splitlines() if line.strip()]
    manifest = rebuild(documents, QdrantPolicyIndex(settings.qdrant_url, build_embedder(settings)), new)
    manifest["previous_namespaces"] = old
    args.out.mkdir()
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    policy.update(version="2.0.0", namespaces=new)
    (args.out / "retrieval.yaml").write_text(yaml.safe_dump(policy, sort_keys=False))
    print(json.dumps({"verified": True, "policy": str(args.out / "retrieval.yaml")}))


if __name__ == "__main__":
    main()
