from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.patches import FancyBboxPatch

from .freeze import current_git_ref
from .olfactory_door import sha256_file
from .olfactory_e006_audit import _canonical_sha


HERO_FILENAME = "o002-hero-4x5.png"
DEEP_DIVE_FILENAME = "o002-scientific-deep-dive.png"
CLASS_RECALL_FILENAME = "o002-class-recall.png"
ROADMAP_FILENAME = "o002-next-stage-roadmap.png"
SOCIAL_FILENAME = "o002-social.mp4"
RECEIPT_FILENAME = "o002-visual-receipt.json"
SUMMARY_FILENAME = "SUMMARY.txt"
BUNDLE_FILENAME = "o002-showcase.zip"

BG = "#07111f"
PANEL = "#f7fafc"
INK = "#10243e"
MUTED = "#5b6b7f"
ACCENT = "#0ea5e9"
ACCENT_2 = "#f59e0b"
ACCENT_3 = "#8b5cf6"
GOOD = "#16a34a"
WARN = "#ca8a04"
GRID = "#d8e2ec"


@dataclass(frozen=True)
class O002VisualData:
    v1_dir: Path
    v2_dir: Path
    v3_dir: Path
    v1: dict[str, Any]
    v2: dict[str, Any]
    v3: dict[str, Any]
    coordinates: pd.DataFrame
    metadata: pd.DataFrame
    holdout: pd.DataFrame
    subspace: pd.DataFrame
    confusion: pd.DataFrame
    direction_class: pd.DataFrame

    @property
    def study_id(self) -> str:
        return str(self.v1["selected_study"]["study_id"])

    @property
    def eligible_odors(self) -> int:
        return int(self.v2["sample"]["eligible_odors"])

    @property
    def responding_units(self) -> int:
        return int(self.v2["sample"]["responding_units"])

    @property
    def eligible_classes(self) -> list[str]:
        return [str(x) for x in self.v2["sample"]["eligible_classes"]]


def _validated_receipt(
    path: Path,
    *,
    protocol: str,
    require_development_only: bool = True,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing receipt: {path}")
    payload = json.loads(path.read_text())
    if payload.get("protocol") != protocol:
        raise ValueError(f"{path.name} has unexpected protocol {payload.get('protocol')!r}")
    observed = str(payload.get("receipt_sha256", ""))
    unhashed = dict(payload)
    unhashed.pop("receipt_sha256", None)
    if observed != _canonical_sha(unhashed):
        raise ValueError(f"{path.name} canonical receipt hash mismatch")
    if require_development_only:
        if payload.get("development_only") is not True:
            raise ValueError(f"{path.name} must remain development-only")
        if payload.get("confirmatory_use_allowed") is not False:
            raise ValueError(f"{path.name} unexpectedly permits confirmatory use")
    return payload


def _verify_output_hash(receipt: dict[str, Any], key: str, path: Path) -> None:
    spec = receipt.get("outputs", {}).get(key)
    if spec is None:
        raise ValueError(f"receipt is missing output hash authority for {key}")
    expected = str(spec.get("sha256", ""))
    if not expected:
        raise ValueError(f"receipt output {key} lacks sha256")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{path.name} hash mismatch: {observed} != {expected}")


def load_o002_visual_data(
    v1_dir: str | Path,
    v2_dir: str | Path,
    v3_dir: str | Path,
) -> O002VisualData:
    v1_dir = Path(v1_dir).expanduser().resolve()
    v2_dir = Path(v2_dir).expanduser().resolve()
    v3_dir = Path(v3_dir).expanduser().resolve()

    v1 = _validated_receipt(
        v1_dir / "o002-development-receipt.json",
        protocol="o002-within-study-development-v1",
    )
    v2 = _validated_receipt(
        v2_dir / "o002-robustness-receipt.json",
        protocol="o002-coding-robustness-development-v2",
    )
    v3 = _validated_receipt(
        v3_dir / "o002-stability-receipt.json",
        protocol="o002-subspace-stability-development-v3",
    )

    if str(v2["input"]["v1_receipt_sha256"]) != str(v1["receipt_sha256"]):
        raise ValueError("O002 v2 lineage does not point to supplied O002 v1")
    if str(v3["input"]["v1_receipt_sha256"]) != str(v1["receipt_sha256"]):
        raise ValueError("O002 v3 lineage does not point to supplied O002 v1")
    if str(v3["input"]["v2_receipt_sha256"]) != str(v2["receipt_sha256"]):
        raise ValueError("O002 v3 lineage does not point to supplied O002 v2")

    coordinate_path = v1_dir / "o002-pca-coordinates.csv"
    metadata_path = v1_dir / "o002-odor-metadata.csv"
    holdout_path = v3_dir / "o002-v3-balanced-holdout.csv"
    subspace_path = v3_dir / "o002-v3-subspace-curve.csv"
    confusion_path = v3_dir / "o002-v3-direction-only-confusion.csv"
    direction_class_path = v3_dir / "o002-v3-direction-only-per-class.csv"

    for path in (
        coordinate_path,
        metadata_path,
        holdout_path,
        subspace_path,
        confusion_path,
        direction_class_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen O002 visualization input: {path}")

    _verify_output_hash(v1, "pca_coordinates", coordinate_path)
    _verify_output_hash(v1, "odor_metadata", metadata_path)
    _verify_output_hash(v3, "balanced_holdout", holdout_path)
    _verify_output_hash(v3, "subspace_curve", subspace_path)
    _verify_output_hash(v3, "direction_only_confusion", confusion_path)
    _verify_output_hash(v3, "direction_only_per_class", direction_class_path)

    coordinates = pd.read_csv(coordinate_path, index_col=0)
    metadata = pd.read_csv(metadata_path, index_col=0)
    holdout = pd.read_csv(holdout_path)
    subspace = pd.read_csv(subspace_path)
    confusion = pd.read_csv(confusion_path, index_col=0)
    direction_class = pd.read_csv(direction_class_path)

    coordinates.index = coordinates.index.astype(str)
    metadata.index = metadata.index.astype(str)
    confusion.index = confusion.index.astype(str)
    confusion.columns = confusion.columns.astype(str)

    required_coord = {"PC1", "PC2"}
    if not required_coord.issubset(coordinates.columns):
        raise ValueError("O002 coordinates require PC1 and PC2")
    required_meta = {"odor_name", "odor_class"}
    if not required_meta.issubset(metadata.columns):
        raise ValueError("O002 metadata require odor_name and odor_class")
    required_holdout = {
        "full_pattern_accuracy",
        "direction_only_accuracy",
        "identity_erased_sorted_accuracy",
    }
    if not required_holdout.issubset(holdout.columns):
        raise ValueError("O002 holdout table lacks required accuracy columns")
    required_subspace = {
        "components",
        "full_pattern_balanced_accuracy",
        "direction_only_balanced_accuracy",
    }
    if not required_subspace.issubset(subspace.columns):
        raise ValueError("O002 subspace table lacks required columns")
    if not {"odor_class", "recall"}.issubset(direction_class.columns):
        raise ValueError("O002 per-class table requires odor_class and recall")

    return O002VisualData(
        v1_dir=v1_dir,
        v2_dir=v2_dir,
        v3_dir=v3_dir,
        v1=v1,
        v2=v2,
        v3=v3,
        coordinates=coordinates,
        metadata=metadata,
        holdout=holdout,
        subspace=subspace,
        confusion=confusion,
        direction_class=direction_class,
    )


def _joined_geometry(data: O002VisualData) -> pd.DataFrame:
    joined = data.coordinates.join(
        data.metadata[["odor_name", "odor_class"]],
        how="inner",
    )
    joined["odor_name"] = joined["odor_name"].fillna("").astype(str)
    joined["odor_class"] = joined["odor_class"].fillna("unlabeled").astype(str)
    eligible = set(data.eligible_classes)
    return joined[joined["odor_class"].isin(eligible)].copy()


def _coding_values(data: O002VisualData) -> tuple[list[str], list[float], float, float]:
    amp = data.v2["amplitude_only"]
    labels = [
        "Full\npattern",
        "Direction\nonly",
        "Identity\nerased",
        "Mean\namplitude",
        "L2\namplitude",
        "Peak-to-\npeak",
    ]
    values = [
        float(data.v2["full_pattern"]["balanced_accuracy"]),
        float(data.v2["direction_only"]["balanced_accuracy"]),
        float(data.v2["identity_erased_sorted_profile"]["balanced_accuracy"]),
        float(amp["mean_response_balanced_accuracy"]),
        float(amp["l2_norm_balanced_accuracy"]),
        float(amp["peak_to_peak_balanced_accuracy"]),
    ]
    null = data.v2["channel_identity_shuffle_null"]
    return labels, values, float(null["mean"]), float(null["q95"])


def _figure_header(fig: plt.Figure, data: O002VisualData, *, title: str, subtitle: str) -> None:
    fig.text(
        0.055,
        0.965,
        title,
        ha="left",
        va="top",
        color="white",
        fontsize=25,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.925,
        subtitle,
        ha="left",
        va="top",
        color="#cbeeff",
        fontsize=11.5,
    )
    fig.text(
        0.055,
        0.885,
        (
            f"{data.eligible_odors} odors  •  {data.responding_units} measured response channels  •  "
            f"{len(data.eligible_classes)} eligible chemical classes"
        ),
        ha="left",
        va="top",
        color="#9edcf7",
        fontsize=9.5,
        fontweight="bold",
    )
    fig.text(
        0.945,
        0.945,
        "REAL MEASURED DATA\nDEVELOPMENT STUDY",
        ha="right",
        va="top",
        color="#9edcf7",
        fontsize=9,
        fontweight="bold",
        linespacing=1.35,
    )


def _style_panel(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=9)
    ax.grid(alpha=0.15, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def _draw_geometry(ax: plt.Axes, data: O002VisualData, *, legend: bool = True) -> None:
    frame = _joined_geometry(data)
    classes = sorted(frame["odor_class"].unique())
    cmap = plt.get_cmap("tab10")
    for index, label in enumerate(classes):
        block = frame[frame["odor_class"] == label]
        ax.scatter(
            block["PC1"],
            block["PC2"],
            s=33,
            alpha=0.76,
            label=label,
            color=cmap(index % 10),
            edgecolors="none",
        )
    geosmin = frame[frame["odor_name"].str.contains("geosmin", case=False, na=False)]
    for _, row in geosmin.iterrows():
        ax.annotate(
            str(row["odor_name"]),
            (float(row["PC1"]), float(row["PC2"])),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            color=INK,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if legend and len(classes) <= 12:
        ax.legend(fontsize=6.2, frameon=False, ncol=1, loc="best")


def _draw_coding(ax: plt.Axes, data: O002VisualData) -> None:
    labels, values, null_mean, null_q95 = _coding_values(data)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=[ACCENT, ACCENT, ACCENT_3, ACCENT_2, ACCENT_2, ACCENT_2])
    ax.axhline(null_mean, linestyle="--", linewidth=1.4, color=MUTED, label="channel-shuffle mean")
    ax.axhline(null_q95, linestyle=":", linewidth=1.7, color=ACCENT, label="shuffle 95th percentile")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Balanced accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.legend(fontsize=6.7, frameon=False, loc="upper right")
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK,
            fontweight="bold",
        )


def _draw_holdout(ax: plt.Axes, data: O002VisualData) -> None:
    holdout = data.holdout
    ax.boxplot(
        [
            holdout["full_pattern_accuracy"].to_numpy(),
            holdout["direction_only_accuracy"].to_numpy(),
            holdout["identity_erased_sorted_accuracy"].to_numpy(),
        ],
        tick_labels=["Full", "Direction", "Identity erased"],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#e6f6fd", "edgecolor": INK},
        medianprops={"color": ACCENT_2, "linewidth": 1.8},
        whiskerprops={"color": INK},
        capprops={"color": INK},
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy")


def _draw_subspace(ax: plt.Axes, data: O002VisualData) -> None:
    sub = data.subspace
    _, _, _, null_q95 = _coding_values(data)
    ax.plot(
        sub["components"],
        sub["full_pattern_balanced_accuracy"],
        marker="o",
        markersize=4,
        color=ACCENT,
        label="Full pattern",
    )
    ax.plot(
        sub["components"],
        sub["direction_only_balanced_accuracy"],
        marker="o",
        markersize=4,
        color=ACCENT_2,
        label="Direction only",
    )
    ax.axhline(null_q95, linestyle=":", linewidth=1.5, color=MUTED, label="shuffle 95th percentile")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("PCA components")
    ax.set_ylabel("Balanced accuracy")
    ax.legend(fontsize=6.8, frameon=False)


def _takeaway_text(data: O002VisualData) -> tuple[str, str, str]:
    _, values, _, null_q95 = _coding_values(data)
    direction = values[1]
    full = values[0]
    identity_erased = values[2]
    direction_holdout = float(data.holdout["direction_only_accuracy"].mean())
    return (
        (
            f"Direction-normalized responses score {direction:.2f} vs "
            f"{full:.2f} for the full pattern in this frozen development study."
        ),
        (
            f"Erasing stable channel identity reduces class information "
            f"({identity_erased:.2f}); shuffle q95 is {null_q95:.2f}."
        ),
        (
            f"The direction-only signal persists across {len(data.holdout)} paired holdouts "
            f"(mean {direction_holdout:.2f}) and low-rank projections."
        ),
    )


def render_hero(data: O002VisualData, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10.8, 13.5), dpi=150, facecolor=BG)
    _figure_header(
        fig,
        data,
        title="WHAT DOES A FLY'S ODOR SPACE LOOK LIKE?",
        subtitle="Measured DoOR physiology reveals structured chemical-class geometry.",
    )
    grid = fig.add_gridspec(
        3,
        2,
        left=0.06,
        right=0.94,
        top=0.82,
        bottom=0.10,
        hspace=0.35,
        wspace=0.26,
        height_ratios=[1.0, 1.0, 0.32],
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    for ax, title in (
        (ax_a, "A  •  Population odor geometry"),
        (ax_b, "B  •  What information carries class structure?"),
        (ax_c, f"C  •  Stable across {len(data.holdout)} paired holdouts"),
        (ax_d, "D  •  A low-dimensional code"),
    ):
        _style_panel(ax, title)

    _draw_geometry(ax_a, data)
    _draw_coding(ax_b, data)
    _draw_holdout(ax_c, data)
    _draw_subspace(ax_d, data)

    callout = fig.add_subplot(grid[2, :])
    callout.axis("off")
    takeaways = _takeaway_text(data)
    for index, text in enumerate(takeaways):
        x = 0.015 + index * 0.33
        box = FancyBboxPatch(
            (x, 0.12),
            0.305,
            0.76,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            transform=callout.transAxes,
            facecolor="#eaf7fd",
            edgecolor="#cae8f6",
            linewidth=1.0,
        )
        callout.add_patch(box)
        callout.text(
            x + 0.015,
            0.50,
            text,
            transform=callout.transAxes,
            ha="left",
            va="center",
            fontsize=8.4,
            color=INK,
            wrap=True,
            fontweight="bold",
        )

    fig.text(
        0.055,
        0.048,
        "WHAT THIS SHOWS",
        color="#7dd3fc",
        fontsize=8,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.025,
        "Within-study chemical-class structure in measured sensory responses.",
        color="#d6e3ef",
        fontsize=7.5,
    )
    fig.text(
        0.52,
        0.048,
        "WHAT THIS DOES NOT SHOW",
        color="#fcd34d",
        fontsize=8,
        fontweight="bold",
    )
    fig.text(
        0.52,
        0.025,
        "Odor identity, behavioral valence, receptor-specific mechanism, cross-study generalization, or connectome effects.",
        color="#d6e3ef",
        fontsize=7.0,
    )

    fig.savefig(output, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


def render_class_recall(data: O002VisualData, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = data.direction_class.sort_values("recall", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    ax.barh(frame["odor_class"].astype(str), frame["recall"].astype(float), color=ACCENT)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Direction-only leave-one-odor-out recall")
    ax.set_ylabel("Source chemical class")
    ax.set_title("Which chemical classes are easiest to separate?", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.18)
    for y, value in enumerate(frame["recall"].astype(float)):
        ax.text(min(0.97, value + 0.015), y, f"{value:.2f}", va="center", fontsize=8)
    fig.text(
        0.5,
        0.01,
        "O002 development study • class-wise recall is descriptive, not a confirmatory biological inference.",
        ha="center",
        fontsize=7.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def render_deep_dive(data: O002VisualData, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 10), dpi=150)
    grid = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.32)
    ax_geo = fig.add_subplot(grid[0, 0])
    ax_conf = fig.add_subplot(grid[0, 1])
    ax_class = fig.add_subplot(grid[0, 2])
    ax_code = fig.add_subplot(grid[1, 0])
    ax_holdout = fig.add_subplot(grid[1, 1])
    ax_subspace = fig.add_subplot(grid[1, 2])

    for ax, title in (
        (ax_geo, "Population geometry"),
        (ax_conf, "Direction-only confusion"),
        (ax_class, "Per-class recall"),
        (ax_code, "Coding decomposition"),
        (ax_holdout, "Paired holdout stability"),
        (ax_subspace, "Low-dimensional retention"),
    ):
        _style_panel(ax, title)

    _draw_geometry(ax_geo, data, legend=False)
    image = ax_conf.imshow(data.confusion.to_numpy(), aspect="auto")
    ax_conf.set_xticks(np.arange(len(data.confusion.columns)))
    ax_conf.set_xticklabels(data.confusion.columns, rotation=45, ha="right", fontsize=6.5)
    ax_conf.set_yticks(np.arange(len(data.confusion.index)))
    ax_conf.set_yticklabels(data.confusion.index, fontsize=6.5)
    ax_conf.set_xlabel("Predicted class")
    ax_conf.set_ylabel("Source class")
    fig.colorbar(image, ax=ax_conf, fraction=0.046, pad=0.04, label="Odor count")

    recall = data.direction_class.sort_values("recall", ascending=True)
    ax_class.barh(recall["odor_class"].astype(str), recall["recall"].astype(float), color=ACCENT)
    ax_class.set_xlim(0.0, 1.0)
    ax_class.set_xlabel("Recall")

    _draw_coding(ax_code, data)
    _draw_holdout(ax_holdout, data)
    _draw_subspace(ax_subspace, data)

    fig.suptitle(
        "O002 measured olfactory representation study • scientific deep dive",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Development-only. Repeated holdouts are stability diagnostics, not independent biological replicates. "
            "No receptor-specific, valence, cross-study, or connectome-topology claim is made."
        ),
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def render_roadmap(data: O002VisualData, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150, facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.text(
        0.05,
        0.90,
        "FROM SENSORY STRUCTURE TO MECHANISM",
        transform=ax.transAxes,
        color="white",
        fontsize=23,
        fontweight="bold",
    )
    ax.text(
        0.05,
        0.84,
        "What is measured now, what is still gated, and what comes next.",
        transform=ax.transAxes,
        color="#b7d9ea",
        fontsize=11,
    )

    stages = [
        (
            0.05,
            "1",
            "MEASURED NOW",
            "O002 sensory geometry",
            f"{data.eligible_odors} odors × {data.responding_units} channels\n"
            "chemical-class structure + stability",
            GOOD,
        ),
        (
            0.29,
            "2",
            "AUTHORITY GATE",
            "E001 / E002",
            "freeze primary physiology authority\n"
            "resolve Or56a/FlyWire cohort identity",
            WARN,
        ),
        (
            0.53,
            "3",
            "NEXT MECHANISTIC TEST",
            "O003",
            "odor conflict + temporal computation\n"
            "matched models + prespecified lesions",
            ACCENT_2,
        ),
        (
            0.77,
            "4",
            "NEXT BEHAVIORAL TEST",
            "O004",
            "plume evidence → steering\n"
            "intact topology vs matched nulls",
            ACCENT,
        ),
    ]

    for x, number, status, title, body, color in stages:
        rect = FancyBboxPatch(
            (x, 0.31),
            0.19,
            0.40,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            transform=ax.transAxes,
            facecolor="#0d1c2f",
            edgecolor=color,
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + 0.018, 0.65, number, transform=ax.transAxes, color=color, fontsize=19, fontweight="bold")
        ax.text(x + 0.018, 0.59, status, transform=ax.transAxes, color=color, fontsize=7.5, fontweight="bold")
        ax.text(x + 0.018, 0.51, title, transform=ax.transAxes, color="white", fontsize=11, fontweight="bold")
        ax.text(x + 0.018, 0.39, body, transform=ax.transAxes, color="#b7c7d9", fontsize=8, linespacing=1.5)

    ax.text(
        0.05,
        0.16,
        "The visual system may preview later stages, but claims advance only when their independent evidence gates do.",
        transform=ax.transAxes,
        color="#d8e7f2",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.text(
        0.05,
        0.09,
        "DEVELOPMENT ROADMAP • NOT A CONNECTOME-TOPOLOGY RESULT",
        transform=ax.transAxes,
        color="#fcd34d",
        fontsize=8.5,
        fontweight="bold",
    )
    fig.savefig(output, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


def render_social_video(
    data: O002VisualData,
    output: str | Path,
    *,
    fps: int = 30,
    seconds: int = 16,
) -> Path:
    if fps < 1 or seconds < 4:
        raise ValueError("social video requires fps >= 1 and seconds >= 4")
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg is required for the O002 social MP4")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = _joined_geometry(data)
    labels, values, _, null_q95 = _coding_values(data)
    total_frames = fps * seconds

    fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.07, 0.12, 0.86, 0.70])
    title_ax = fig.add_axes([0.05, 0.84, 0.90, 0.13])
    title_ax.axis("off")

    def draw(index: int):
        ax.clear()
        title_ax.clear()
        title_ax.axis("off")
        t = index / fps

        if t < 4:
            title_ax.text(
                0.5,
                0.68,
                "WHAT DOES A FLY'S ODOR SPACE LOOK LIKE?",
                ha="center",
                color="white",
                fontsize=25,
                fontweight="bold",
            )
            title_ax.text(
                0.5,
                0.26,
                f"{data.eligible_odors} measured odors • {data.responding_units} response channels",
                ha="center",
                color="#bfe8fb",
                fontsize=11,
            )
            ax.set_facecolor("#f7fafc")
            classes = sorted(frame["odor_class"].unique())
            fraction = min(1.0, max(0.0, t / 2.6))
            cmap = plt.get_cmap("tab10")
            for class_index, label in enumerate(classes):
                block = frame[frame["odor_class"] == label]
                count = max(1, int(np.ceil(len(block) * fraction)))
                show = block.iloc[:count]
                ax.scatter(
                    show["PC1"],
                    show["PC2"],
                    s=55,
                    alpha=0.78,
                    color=cmap(class_index % 10),
                    label=label,
                )
            ax.set_title("Measured population response geometry", fontsize=16, fontweight="bold")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.legend(fontsize=7, frameon=False)
        elif t < 8:
            title_ax.text(
                0.5,
                0.68,
                "WHAT CARRIES THE CLASS STRUCTURE?",
                ha="center",
                color="white",
                fontsize=25,
                fontweight="bold",
            )
            title_ax.text(
                0.5,
                0.26,
                (
                    f"Direction-normalized score {values[1]:.2f} • "
                    f"full pattern {values[0]:.2f} • shuffle q95 {null_q95:.2f}"
                ),
                ha="center",
                color="#bfe8fb",
                fontsize=11,
            )
            ax.set_facecolor("#f7fafc")
            bars = ax.bar(np.arange(len(labels)), values, color=[ACCENT, ACCENT, ACCENT_3, ACCENT_2, ACCENT_2, ACCENT_2])
            ax.axhline(null_q95, linestyle=":", color=MUTED, label="shuffle q95")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Balanced accuracy")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, fontsize=9)
            for bar, value in zip(bars, values, strict=True):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}", ha="center", fontsize=9)
            ax.legend(frameon=False)
        elif t < 12:
            title_ax.text(
                0.5,
                0.68,
                "DOES IT SURVIVE RESAMPLING?",
                ha="center",
                color="white",
                fontsize=25,
                fontweight="bold",
            )
            title_ax.text(
                0.5,
                0.26,
                f"{len(data.holdout)} paired balanced holdouts",
                ha="center",
                color="#bfe8fb",
                fontsize=11,
            )
            ax.set_facecolor("#f7fafc")
            _draw_holdout(ax, data)
            ax.set_title("Direction-only structure stays strongest across resamples", fontsize=15, fontweight="bold")
        else:
            ax.axis("off")
            title_ax.text(
                0.5,
                0.70,
                "MEASURED FLY OLFACTION HAS STRUCTURE",
                ha="center",
                color="white",
                fontsize=25,
                fontweight="bold",
            )
            title_ax.text(
                0.5,
                0.28,
                "The next question is whether independently qualified biological wiring explains or exploits it.",
                ha="center",
                color="#bfe8fb",
                fontsize=10.5,
            )
            ax.text(
                0.5,
                0.70,
                "O002",
                transform=ax.transAxes,
                ha="center",
                color=ACCENT,
                fontsize=45,
                fontweight="bold",
            )
            ax.text(
                0.5,
                0.53,
                "sensory geometry ✓",
                transform=ax.transAxes,
                ha="center",
                color="white",
                fontsize=20,
                fontweight="bold",
            )
            ax.text(
                0.5,
                0.38,
                "E001 / E002 authority → O003 / O004 mechanism",
                transform=ax.transAxes,
                ha="center",
                color="#cfe8f5",
                fontsize=13,
            )
            ax.text(
                0.5,
                0.20,
                "Development study • no connectome-topology claim",
                transform=ax.transAxes,
                ha="center",
                color="#fcd34d",
                fontsize=10,
                fontweight="bold",
            )
        return []

    movie = animation.FuncAnimation(
        fig,
        draw,
        frames=total_frames,
        interval=1000 / fps,
        blit=False,
    )
    try:
        movie.save(output, writer=animation.FFMpegWriter(fps=fps, bitrate=5500))
    finally:
        plt.close(fig)
    return output


def _safe_git_ref() -> str:
    try:
        return current_git_ref()
    except Exception:
        return "unknown"


def _deterministic_zip(output_dir: Path, members: list[Path], bundle_path: Path) -> None:
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(members, key=lambda item: item.name):
            info = zipfile.ZipInfo(path.name)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def render_showcase(
    v1_dir: str | Path,
    v2_dir: str | Path,
    v3_dir: str | Path,
    *,
    output_dir: str | Path,
    video: bool = False,
    force: bool = False,
    fps: int = 30,
    seconds: int = 16,
) -> dict[str, Any]:
    data = load_o002_visual_data(v1_dir, v2_dir, v3_dir)
    output = Path(output_dir).expanduser().resolve()
    known = [
        HERO_FILENAME,
        DEEP_DIVE_FILENAME,
        CLASS_RECALL_FILENAME,
        ROADMAP_FILENAME,
        SOCIAL_FILENAME,
        RECEIPT_FILENAME,
        SUMMARY_FILENAME,
        BUNDLE_FILENAME,
    ]
    if output.exists():
        existing = [output / name for name in known if (output / name).exists()]
        if existing and not force:
            raise ValueError(
                "refusing to overwrite existing O002 visualization artifacts; "
                "use --force only to regenerate presentation outputs"
            )
        if force:
            for path in existing:
                if path.is_file():
                    path.unlink()
    output.mkdir(parents=True, exist_ok=True)

    hero = render_hero(data, output / HERO_FILENAME)
    deep = render_deep_dive(data, output / DEEP_DIVE_FILENAME)
    class_recall = render_class_recall(data, output / CLASS_RECALL_FILENAME)
    roadmap = render_roadmap(data, output / ROADMAP_FILENAME)

    social: Path | None = None
    if video:
        social = render_social_video(
            data,
            output / SOCIAL_FILENAME,
            fps=fps,
            seconds=seconds,
        )

    rendered = [hero, deep, class_recall, roadmap]
    if social is not None:
        rendered.append(social)

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "o002-visual-showcase-v1",
        "program_id": "olfactory-computation-v0",
        "status": "presentation_complete_development_only",
        "development_only": True,
        "confirmatory_use_allowed": False,
        "scientific_metrics_recomputed": False,
        "selection_from_visual_outcomes_allowed": False,
        "renderer_git_ref": _safe_git_ref(),
        "input": {
            "v1_receipt_sha256": data.v1["receipt_sha256"],
            "v2_receipt_sha256": data.v2["receipt_sha256"],
            "v3_receipt_sha256": data.v3["receipt_sha256"],
            "study_id": data.study_id,
        },
        "sample": {
            "eligible_odors": data.eligible_odors,
            "responding_units": data.responding_units,
            "eligible_classes": data.eligible_classes,
        },
        "outputs": {
            path.name: {
                "sha256": sha256_file(path),
                "byte_count": path.stat().st_size,
            }
            for path in rendered
        },
        "claim_boundary": (
            "These files are receipt-driven visualizations of already-frozen O002 development outputs. "
            "The renderer does not recompute scientific metrics or promote evidence. The visuals may "
            "describe within-study chemical-class geometry, coding decompositions, paired-resampling "
            "stability, and low-dimensional retention. They do not establish odor identity, behavioral "
            "valence, receptor-specific mechanism, cross-study generalization, or connectome-topology "
            "effects, and they do not authorize O003/O004."
        ),
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    receipt_path = output / RECEIPT_FILENAME
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    takeaways = _takeaway_text(data)
    summary = [
        "O002 VISUAL SHOWCASE V1",
        f"status: {receipt['status']}",
        f"study: {data.study_id}",
        f"eligible_odors: {data.eligible_odors}",
        f"responding_units: {data.responding_units}",
        f"eligible_classes: {len(data.eligible_classes)}",
        f"renderer_git_ref: {receipt['renderer_git_ref']}",
        f"receipt_sha256: {receipt['receipt_sha256']}",
        "",
        "TAKEAWAYS",
        *[f"- {text}" for text in takeaways],
        "",
        "OUTPUTS",
        *[f"- {path.name}: {sha256_file(path)}" for path in rendered],
        "",
        "CLAIM BOUNDARY",
        receipt["claim_boundary"],
    ]
    summary_path = output / SUMMARY_FILENAME
    summary_path.write_text("\n".join(summary) + "\n")

    bundle = output / BUNDLE_FILENAME
    members = rendered + [receipt_path, summary_path]
    _deterministic_zip(output, members, bundle)

    return {
        "output_dir": str(output),
        "hero": str(hero),
        "deep_dive": str(deep),
        "class_recall": str(class_recall),
        "roadmap": str(roadmap),
        "social_video": str(social) if social else None,
        "receipt": str(receipt_path),
        "summary": str(summary_path),
        "bundle": str(bundle),
        "receipt_sha256": receipt["receipt_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render receipt-verified O002 scientific and social showcase artifacts"
    )
    parser.add_argument("v1", help="O002 v1 development artifact directory")
    parser.add_argument("v2", help="O002 v2 robustness artifact directory")
    parser.add_argument("v3", help="O002 v3 stability artifact directory")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--video", action="store_true", help="also render the 4:5 MP4 (requires ffmpeg)")
    parser.add_argument("--force", action="store_true", help="replace only known presentation outputs")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=int, default=16)
    args = parser.parse_args(argv)

    result = render_showcase(
        args.v1,
        args.v2,
        args.v3,
        output_dir=args.output,
        video=args.video,
        force=args.force,
        fps=args.fps,
        seconds=args.seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
