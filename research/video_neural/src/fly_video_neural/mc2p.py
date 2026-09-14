from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SESSION_PATTERN = re.compile(r"^(?P<animal>.+)_(?P<trial>\d{3})$")


@dataclass(frozen=True)
class MC2PSession:
    session_id: str
    animal_id: str
    trial_id: str
    path: str
    behavior_video: str
    synchronization: str
    neural_dff: str
    pose: str | None
    inverse_kinematics: str | None
    rest_mask: str | None
    roi_traces: str | None
    neural_representation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first(root: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def parse_session_identity(path: str | Path) -> tuple[str, str]:
    name = Path(path).name
    match = SESSION_PATTERN.match(name)
    if not match:
        raise ValueError(f"MC2P session directory must end in _NNN trial id: {name!r}")
    return match.group("animal"), match.group("trial")


def inspect_session(path: str | Path) -> MC2PSession:
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"MC2P session is not a directory: {root}")
    animal, trial = parse_session_identity(root)
    video = _first(root, ("*behData*camera_1.mp4", "*camera_1.mp4"))
    sync = _first(root, ("indices.npy", "sync_indices.pkl", "*sync*indices*.pkl"))
    dff = _first(root, ("2p_dff.mm", "*2p_dff*.mm", "*2p_dff*.npy"))
    pose = _first(root, ("pose_result.pkl", "*pose_result*.mm", "*pose_result*.pkl"))
    inverse = _first(root, ("pose_result_inverse_kinematics.pkl", "*inverse*kinematics*.pkl"))
    rest = _first(root, ("rest.npy",))
    roi = _first(root, ("*roi_traces*.npy", "*roi_traces*.mm", "*roi_traces*.npz"))
    missing = [
        name
        for name, value in (("behavior video", video), ("synchronization", sync), ("measured dF/F", dff))
        if value is None
    ]
    if missing:
        raise ValueError(f"MC2P session {root.name} missing required files: {missing}")
    representation = "segmented_roi_traces" if roi is not None else "two_photon_dff_imaging"
    return MC2PSession(
        session_id=root.name,
        animal_id=animal,
        trial_id=trial,
        path=str(root.resolve()),
        behavior_video=str(video.resolve()),
        synchronization=str(sync.resolve()),
        neural_dff=str(dff.resolve()),
        pose=str(pose.resolve()) if pose else None,
        inverse_kinematics=str(inverse.resolve()) if inverse else None,
        rest_mask=str(rest.resolve()) if rest else None,
        roi_traces=str(roi.resolve()) if roi else None,
        neural_representation=representation,
    )


def discover_sessions(root: str | Path) -> list[MC2PSession]:
    base = Path(root)
    if not base.is_dir():
        raise ValueError(f"MC2P root is not a directory: {base}")
    sessions: list[MC2PSession] = []
    errors: list[str] = []
    for child in sorted(path for path in base.iterdir() if path.is_dir()):
        if SESSION_PATTERN.match(child.name) is None:
            continue
        try:
            sessions.append(inspect_session(child))
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError("invalid MC2P session directories:\n" + "\n".join(errors[:20]))
    if not sessions:
        raise ValueError("no valid MC2P session directories discovered")
    return sessions


def build_manifest(root: str | Path) -> dict[str, Any]:
    sessions = discover_sessions(root)
    animals = sorted({session.animal_id for session in sessions})
    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": "mc2p_v1",
        "root": str(Path(root).resolve()),
        "animal_count": len(animals),
        "session_count": len(sessions),
        "animals": animals,
        "neural_representations": sorted({s.neural_representation for s in sessions}),
        "sessions": [session.to_dict() for session in sessions],
        "pickle_policy": "paths may be indexed, but pickle synchronization/pose files are not deserialized by this command",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload
