from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .graph import GraphBundle


@dataclass(frozen=True)
class ShowcaseSpec:
    key: str
    title: str
    archetype: str
    required_roles: tuple[str, ...]
    optional_roles: tuple[str, ...]
    public_question: str
    scientific_use: str
    status: str


SPECS: dict[str, ShowcaseSpec] = {
    "odor-choice": ShowcaseSpec(
        key="odor-choice",
        title="Left or right?",
        archetype="native-behavior",
        required_roles=("odor_left", "odor_right", "steer_left", "steer_right"),
        optional_roles=(),
        public_question="Can the wiring turn toward the antenna that smells more odor?",
        scientific_use="Smallest bilateral sensory-to-steering sanity assay.",
        status="implemented",
    ),
    "odor-plume": ShowcaseSpec(
        key="odor-plume",
        title="Who Farted? / Plume Hunt",
        archetype="native-behavior",
        required_roles=(
            "odor_left",
            "odor_right",
            "wind_forward",
            "wind_backward",
            "wind_left",
            "wind_right",
            "steer_left",
            "steer_right",
        ),
        optional_roles=(),
        public_question="Can real fly wiring find the hidden odor source better than scrambled wiring?",
        scientific_use="Primary causal topology benchmark under intermittent turbulent odor.",
        status="implemented-environment-circuit-pending",
    ),
    "visual-replay": ShowcaseSpec(
        key="visual-replay",
        title="Connectome Cinema",
        archetype="open-loop-stimulus",
        required_roles=("vision_left", "vision_right"),
        optional_roles=("steer_left", "steer_right"),
        public_question="What does modeled MaleCNS activity do while the fly watches this clip?",
        scientific_use="Replayable stimulus-response visualization. Activity is not perception or understanding.",
        status="adapter-ready-role-map-pending",
    ),
    "game-control": ShowcaseSpec(
        key="game-control",
        title="Fly Arcade",
        archetype="closed-loop-game",
        required_roles=("vision_left", "vision_right", "steer_left", "steer_right"),
        optional_roles=("move_forward", "action_primary"),
        public_question="What happens when game observations drive fly sensory channels and fly readouts drive buttons?",
        scientific_use="Interface stress test. Engineered sensory and motor mappings must be shown explicitly.",
        status="adapter-contract-ready-world-adapter-pending",
    ),
    "loom-escape": ShowcaseSpec(
        key="loom-escape",
        title="Incoming!",
        archetype="native-behavior",
        required_roles=(
            "loom_size_left",
            "loom_size_right",
            "loom_velocity_left",
            "loom_velocity_right",
            "steer_left",
            "steer_right",
        ),
        optional_roles=("escape",),
        public_question="Can connectome-constrained circuitry turn away from an approaching object?",
        scientific_use=(
            "Direct-hit versus near-miss visuomotor assay with intact/rewired topology controls. "
            "Size and expansion-speed channels remain explicit modeled adapter features."
        ),
        status="implemented-environment-role-trace-pending",
    ),
}


def assess(bundle: GraphBundle, spec: ShowcaseSpec) -> dict:
    available = set(bundle.roles)
    missing_required = [role for role in spec.required_roles if role not in available]
    present_optional = [role for role in spec.optional_roles if role in available]
    qualified = bool(
        bundle.manifest and bundle.manifest.get("qualification_status") == "qualified"
    )
    return {
        "experiment": asdict(spec),
        "graph": {
            "qualification_status": (
                bundle.manifest.get("qualification_status") if bundle.manifest else "unsealed"
            ),
            "roles": sorted(available),
        },
        "missing_required_roles": missing_required,
        "present_optional_roles": present_optional,
        "interface_ready": not missing_required,
        "male_cns_claim_allowed": qualified and not missing_required,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List showcase archetypes or audit a GraphBundle against one adapter contract"
    )
    parser.add_argument("--list", action="store_true", help="list built-in showcase experiments")
    parser.add_argument("--experiment", choices=sorted(SPECS))
    parser.add_argument("--graph", help="GraphBundle directory to audit")
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()

    if args.list:
        payload = [asdict(SPECS[key]) for key in sorted(SPECS)]
    else:
        if not args.experiment or not args.graph:
            parser.error("use --list or provide both --experiment and --graph")
        bundle = GraphBundle.load(args.graph)
        payload = assess(bundle, SPECS[args.experiment])

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
