"""
morphology.py — Neural stem cell morphology simulation (radial glia-like).

Simulates NSC morphology at different states using matplotlib + numpy.
Generates PNG images and returns quantitative morphological features.
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from matplotlib.collections import LineCollection
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MorphologyFeatures:
    """Quantitative morphological features of a single cell."""
    cell_state: str
    soma_area: float
    soma_center: Tuple[float, float]
    processes: List[dict] = field(default_factory=list)
    branch_count: int = 0
    total_process_length: float = 0.0
    longest_process_length: float = 0.0
    eccentricity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "cell_state": self.cell_state,
            "soma_area": round(self.soma_area, 2),
            "soma_center": [round(v, 2) for v in self.soma_center],
            "num_processes": len(self.processes),
            "branch_count": self.branch_count,
            "total_process_length": round(self.total_process_length, 2),
            "longest_process_length": round(self.longest_process_length, 2),
            "eccentricity": round(self.eccentricity, 3),
        }


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def _bezier_curve(p0, p1, p2, p3, num_points=80):
    """Cubic Bezier curve from p0 to p3 with control points p1, p2."""
    t = np.linspace(0, 1, num_points)
    x = (1 - t)**3 * p0[0] + 3*(1 - t)**2 * t * p1[0] \
        + 3*(1 - t) * t**2 * p2[0] + t**3 * p3[0]
    y = (1 - t)**3 * p0[1] + 3*(1 - t)**2 * t * p1[1] \
        + 3*(1 - t) * t**2 * p2[1] + t**3 * p3[1]
    return np.column_stack([x, y])


def _compute_length(points):
    """Compute total arclength of a polyline."""
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))


def _random_sign():
    return 1 if np.random.rand() > 0.5 else -1


# ---------------------------------------------------------------------------
# Process / branch generators
# ---------------------------------------------------------------------------

def _make_radial_fiber(
    start, angle_deg, length, bend_strength=0.3, num_points=100, endfoot_radius=6.0,
):
    """
    A single long radial fiber with gentle curvature, ending in an endfoot.
    Returns (points, endfoot_pos, endfoot_radius).
    """
    angle_rad = math.radians(angle_deg)
    # Control points for a gentle curve
    p0 = np.array(start)
    mid_dir = np.array([math.cos(angle_rad), math.sin(angle_rad)])
    p3 = p0 + mid_dir * length

    # Offset 1st control point perpendicularly for curvature
    perp = np.array([-mid_dir[1], mid_dir[0]])
    offset = perp * bend_strength * length * (_random_sign() * 0.5 + 0.5 * np.random.rand())

    p1 = p0 + mid_dir * length * 0.2 + offset
    p2 = p0 + mid_dir * length * 0.6 + offset * 0.5

    points = _bezier_curve(p0, p1, p2, p3, num_points)
    endfoot_pos = points[-1]
    return points, endfoot_pos, endfoot_radius


def _make_short_process(start, angle_deg, length, num_points=20):
    """A short unbranched process (filopodium-like)."""
    angle_rad = math.radians(angle_deg)
    p0 = np.array(start)
    direction = np.array([math.cos(angle_rad), math.sin(angle_rad)])
    jitter = np.random.randn(2) * 0.08 * length
    p3 = p0 + direction * length + jitter
    p1 = p0 + direction * length * 0.3 + np.random.randn(2) * 0.1 * length
    p2 = p0 + direction * length * 0.7 + np.random.randn(2) * 0.08 * length
    return _bezier_curve(p0, p1, p2, p3, num_points)


def _make_dendritic_branch(start, angle_deg, length, depth=0, max_depth=2):
    """
    Recursively generate a dendritic tree. Returns list of (parent_points, children_branches).
    """
    angle_rad = math.radians(angle_deg)
    p0 = np.array(start)
    direction = np.array([math.cos(angle_rad), math.sin(angle_rad)])
    endpoint = p0 + direction * length + np.random.randn(2) * 0.05 * length
    cp1 = p0 + direction * length * 0.3 + np.random.randn(2) * 0.12 * length
    cp2 = p0 + direction * length * 0.6 + np.random.randn(2) * 0.10 * length
    points = _bezier_curve(p0, cp1, cp2, endpoint, 30)

    branches = []
    if depth < max_depth and length > 8:
        for _ in range(np.random.randint(1, 3)):
            child_angle = angle_deg + _random_sign() * (20 + np.random.rand() * 40)
            child_len = length * (0.5 + np.random.rand() * 0.4)
            child_branches = _make_dendritic_branch(
                endpoint, child_angle, child_len, depth + 1, max_depth
            )
            branches.append(child_branches)
    return (points, branches)


def _flatten_branches(tree, collection=None):
    """Flatten recursive dendritic tree into a list of point arrays."""
    if collection is None:
        collection = []
    points, children = tree
    collection.append(points)
    for child in children:
        _flatten_branches(child, collection)
    return collection


# ---------------------------------------------------------------------------
# Main cell builders
# ---------------------------------------------------------------------------

def _build_qnsc(center=(0, 0)):
    """
    Quiescent NSC (qNSC): small soma, 1 long radial fiber with endfoot.
    """
    cx, cy = center
    soma_a, soma_b = 5.0, 6.5   # small elliptical soma
    angle_soma = 15.0  # tilt degrees

    processes = []
    # One long radial fiber pointing upward (apical)
    pts, ef_pos, ef_r = _make_radial_fiber(
        (cx, cy + soma_b * 0.7), -90, 70, bend_strength=0.15, endfoot_radius=5.0
    )
    processes.append({"type": "radial_fiber", "points": pts,
                       "endfoot_pos": ef_pos, "endfoot_radius": ef_r})

    # A few tiny filopodia
    for ang in np.random.choice(np.linspace(-150, 150, 12), size=3, replace=False):
        if abs(ang) < 30:  # avoid overlap with main fiber
            continue
        pts = _make_short_process((cx, cy), ang, 5 + np.random.rand() * 6)
        processes.append({"type": "filopodium", "points": pts})

    return _assemble_cell("qNSC", cx, cy, soma_a, soma_b, angle_soma, processes)


def _build_ansc(center=(0, 0)):
    """
    Activated NSC (aNSC): larger soma, 1-2 long radial fibers, many short processes.
    """
    cx, cy = center
    soma_a, soma_b = 8.0, 10.0
    angle_soma = 10.0

    processes = []
    # Apical radial fiber
    pts, ef_pos, ef_r = _make_radial_fiber(
        (cx, cy + soma_b * 0.6), -90, 85, bend_strength=0.18, endfoot_radius=5.5
    )
    processes.append({"type": "radial_fiber", "points": pts,
                       "endfoot_pos": ef_pos, "endfoot_radius": ef_r})

    # Additional basal process
    pts2, ef_pos2, ef_r2 = _make_radial_fiber(
        (cx, cy - soma_b * 0.6), 90, 50, bend_strength=0.12, endfoot_radius=3.0
    )
    processes.append({"type": "basal_process", "points": pts2,
                       "endfoot_pos": ef_pos2, "endfoot_radius": ef_r2})

    # Many short processes/filopodia
    for ang in np.random.choice(np.linspace(-180, 180, 24), size=10, replace=False):
        if -45 < ang < 45:  # avoid apical region
            continue
        pts = _make_short_process((cx, cy), ang, 8 + np.random.rand() * 12)
        processes.append({"type": "filopodium", "points": pts})

    return _assemble_cell("aNSC", cx, cy, soma_a, soma_b, angle_soma, processes)


def _build_tap(center=(0, 0)):
    """
    Transit Amplifying Progenitor (TAP): round soma, almost no processes.
    """
    cx, cy = center
    soma_a, soma_b = 7.5, 8.0   # nearly circular
    angle_soma = 0.0

    processes = []
    # Just a few very short stubby processes
    for ang in np.random.choice(np.linspace(0, 360, 8), size=4, replace=False):
        pts = _make_short_process((cx, cy), ang, 3 + np.random.rand() * 4)
        processes.append({"type": "stub", "points": pts})

    return _assemble_cell("TAP", cx, cy, soma_a, soma_b, angle_soma, processes)


def _build_neuron(center=(0, 0)):
    """
    Differentiated Neuron: complex dendritic tree + single axon.
    """
    cx, cy = center
    soma_a, soma_b = 9.0, 11.0
    angle_soma = 20.0

    processes = []

    # Axon (long, thin, extends downward)
    pts, _, _ = _make_radial_fiber(
        (cx, cy - soma_b * 0.6), 90, 100, bend_strength=0.25, endfoot_radius=2.0
    )
    processes.append({"type": "axon", "points": pts,
                       "endfoot_pos": pts[-1], "endfoot_radius": 2.0})

    # Dendritic tree (multiple branches upward and sideways)
    for base_angle in [-60, -20, 30, 70, 110, 150]:
        length = 25 + np.random.rand() * 30
        tree = _make_dendritic_branch(
            (cx, cy + soma_b * 0.4), base_angle, length, depth=0, max_depth=2
        )
        branch_points = _flatten_branches(tree)
        processes.append({"type": "dendrite_tree", "points_list": branch_points})

    return _assemble_cell("Neuron", cx, cy, soma_a, soma_b, angle_soma, processes)


# ---------------------------------------------------------------------------
# Common assembly
# ---------------------------------------------------------------------------

def _assemble_cell(state, cx, cy, soma_a, soma_b, angle_soma, processes):
    """
    Assemble a complete cell: compute morphological features, return dict
    with drawing instructions + features.
    """
    soma_ellipse = Ellipse(xy=(cx, cy), width=soma_a * 2, height=soma_b * 2,
                           angle=angle_soma)

    # Compute features
    total_len = 0.0
    longest_len = 0.0
    branch_count = 0

    for proc in processes:
        if "points" in proc:
            l = _compute_length(proc["points"])
            total_len += l
            if l > longest_len:
                longest_len = l
        if "points_list" in proc:
            for pts in proc["points_list"]:
                l = _compute_length(pts)
                total_len += l
                if l > longest_len:
                    longest_len = l
                # Count branching points (endpoints of each branch segment)
                branch_count += 1

    eccentricity = math.sqrt(1 - (min(soma_a, soma_b)**2 / max(soma_a, soma_b)**2))

    features = MorphologyFeatures(
        cell_state=state,
        soma_area=math.pi * soma_a * soma_b,
        soma_center=(cx, cy),
        processes=processes,
        branch_count=branch_count,
        total_process_length=total_len,
        longest_process_length=longest_len,
        eccentricity=eccentricity,
    )

    return {
        "soma_ellipse": soma_ellipse,
        "processes": processes,
        "features": features,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_STATE_COLORS = {
    "qNSC":  {"soma": "#4A90D9", "process": "#4A90D9", "endfoot": "#2E6EB5"},
    "aNSC":  {"soma": "#E67E22", "process": "#E67E22", "endfoot": "#C06818"},
    "TAP":   {"soma": "#27AE60", "process": "#27AE60", "endfoot": "#1E8449"},
    "Neuron":{"soma": "#8E44AD", "process": "#8E44AD", "endfoot": "#6C3483"},
}


def _plot_cell(ax, cell_data, title, xlim=(-80, 60), ylim=(-70, 80)):
    """Draw a single cell on the given axes."""
    colors = _STATE_COLORS.get(cell_data["features"].cell_state, _STATE_COLORS["qNSC"])

    # Draw processes
    for proc in cell_data["processes"]:
        if "points" in proc:
            pts = proc["points"]
            ax.plot(pts[:, 0], pts[:, 1], color=colors["process"],
                    linewidth=1.8 if proc["type"] in ("radial_fiber", "axon") else 0.8,
                    alpha=0.85, zorder=1)
            # Endfoot
            if "endfoot_pos" in proc and proc["endfoot_radius"] > 2:
                ef = plt.Circle(proc["endfoot_pos"], proc["endfoot_radius"],
                                color=colors["endfoot"], alpha=0.7, zorder=2)
                ax.add_patch(ef)
        elif "points_list" in proc:
            for pts in proc["points_list"]:
                ax.plot(pts[:, 0], pts[:, 1], color=colors["process"],
                        linewidth=0.9, alpha=0.75, zorder=1)

    # Draw soma
    soma = cell_data["soma_ellipse"]
    ax.add_patch(Ellipse(
        xy=soma.center, width=soma.width, height=soma.height,
        angle=soma.angle,
        facecolor=colors["soma"], edgecolor="#333333",
        linewidth=1.2, alpha=0.85, zorder=3
    ))
    # Nucleus (darker center)
    nucleus = Ellipse(
        xy=soma.center, width=soma.width * 0.45, height=soma.height * 0.45,
        angle=soma.angle,
        facecolor="#1a1a2e", edgecolor="none", alpha=0.35, zorder=4
    )
    ax.add_patch(nucleus)

    # Styling
    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.axis("off")


def _plot_feature_table(fig, ax, features_list):
    """Render a summary table of morphological features."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    headers = ["State", "Soma Area", "Processes", "Branches", "Total Len.", "Longest", "Eccentricity"]
    cell_text = []
    for f in features_list:
        d = f.to_dict()
        cell_text.append([
            d["cell_state"],
            str(d["soma_area"]),
            str(d["num_processes"]),
            str(d["branch_count"]),
            f'{d["total_process_length"]:.1f}',
            f'{d["longest_process_length"]:.1f}',
            f'{d["eccentricity"]:.3f}',
        ])

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=[0.12, 0.12, 0.12, 0.12, 0.14, 0.14, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif key[0] % 2 == 0:
            cell.set_facecolor("#f2f3f4")
        else:
            cell.set_facecolor("#ffffff")
    ax.set_title("Morphological Features", fontsize=11, fontweight="bold", pad=10)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_nsc_morphology(output_path: Optional[str] = "nsc_morphology.png",
                        seed: Optional[int] = 42) -> List[dict]:
    """
    Generate a 2x2 panel of NSC morphology at four different states +
    a summary feature table. Returns a list of feature dictionaries.

    Parameters
    ----------
    output_path : str or None
        PNG file path. If None, no file is saved (useful for in-memory use).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    list of dict
        Morphological features for each cell state.
    """
    if seed is not None:
        np.random.seed(seed)

    builders = [
        ("qNSC – Quiescent NSC", _build_qnsc),
        ("aNSC – Activated NSC", _build_ansc),
        ("TAP – Transit Amplifying", _build_tap),
        ("Neuron – Differentiated", _build_neuron),
    ]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.15],
                          hspace=0.15, wspace=0.10)

    features_list = []
    cell_data_list = []

    for i, (title, builder) in enumerate(builders):
        row, col = divmod(i, 2)
        # Adjust layout: first two cells in row 0, next two in row 1
        ax_idx = i
        ax = fig.add_subplot(2, 4, ax_idx + 1)
        cell_data = builder()
        cell_data_list.append(cell_data)
        features_list.append(cell_data["features"])
        _plot_cell(ax, cell_data, title)

    # Feature table spanning the last column (both rows)
    ax_table = fig.add_subplot(gs[:, 3])
    _plot_feature_table(fig, ax_table, features_list)

    # Global title
    fig.suptitle("Neural Stem Cell Morphology – Radial Glia-like Lineage",
                 fontsize=14, fontweight="bold", y=0.98)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
    else:
        plt.close(fig)

    return [f.to_dict() for f in features_list]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    features = plot_nsc_morphology()
    print("\n=== NSC Morphology Features ===")
    for f in features:
        print(f"  {f['cell_state']:>8s} | soma_area={f['soma_area']:>6.1f}  "
              f"processes={f['num_processes']:>2d}  branches={f['branch_count']:>2d}  "
              f"total_len={f['total_process_length']:>6.1f}  "
              f"longest={f['longest_process_length']:>6.1f}  "
              f"ecc={f['eccentricity']:.3f}")
    print(f"\nOutput saved to: nsc_morphology.png")
