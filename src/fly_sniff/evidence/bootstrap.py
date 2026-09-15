from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .ledger import EvidenceLedger
from .schema import EntityRef, EvidenceClass, EvidenceFact, EvidenceRecord

PROTOCOL = "evidence-bootstrap-v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_ledger(config: dict[str, Any], *, root: str | Path = ".") -> EvidenceLedger:
    if config.get("protocol") != PROTOCOL:
        raise ValueError("unexpected evidence bootstrap protocol")
    dataset = str(config["dataset"])
    root_path = Path(root)
    records: list[EvidenceRecord] = []

    for row in config["records"]:
        authority_path = root_path / str(row["authority_path"])
        if not authority_path.exists():
            raise FileNotFoundError(f"missing evidence authority: {authority_path}")
        authority_payload = json.loads(authority_path.read_text())
        authority_dataset = authority_payload.get("dataset")
        if authority_dataset is not None and str(authority_dataset) != dataset:
            raise ValueError(
                f"authority dataset mismatch for {authority_path}: "
                f"expected {dataset}, got {authority_dataset}"
            )
        authority_sha = file_sha256(authority_path)
        authority_kind = authority_payload.get("authority_kind")
        facts = tuple(
            EvidenceFact(
                statement=str(fact["statement"]),
                evidence_class=EvidenceClass(str(fact["evidence_class"])),
                authority=str(row["authority_path"]),
                confidence=str(fact["confidence"]),
                caveat=None if fact.get("caveat") is None else str(fact["caveat"]),
                authority_sha256=authority_sha,
                metadata={
                    **dict(fact.get("metadata", {})),
                    "authority_kind": authority_kind,
                    "bootstrap_protocol": PROTOCOL,
                },
            )
            for fact in row["facts"]
        )
        record = EvidenceRecord(
            entity=EntityRef.from_dict(dict(row["entity"])),
            facts=facts,
            forbidden_claims=tuple(str(x) for x in row.get("forbidden_claims", [])),
            allowed_claims=tuple(str(x) for x in row.get("allowed_claims", [])),
        )
        records.append(record)

    ledger = EvidenceLedger(records=tuple(records))
    ledger.validate()
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EvidenceLedger v1 from frozen authority mappings")
    parser.add_argument("config", nargs="?", default="configs/evidence_bootstrap_v1.json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="authority/evidence-ledger-v1.json")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    ledger = build_ledger(config, root=args.root)
    output = ledger.save(args.output)
    payload = ledger.to_dict()
    print(f"{output}")
    print(f"records={len(ledger.records)} sha256={payload['ledger_sha256']}")


if __name__ == "__main__":
    main()
