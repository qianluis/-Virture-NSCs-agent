"""
SVZ Neural Stem Cell Niche Spatial Model (ABM)
===============================================
Agent-based model of the subventricular zone (SVZ) neurogenic niche.
Features:
  - 2D grid with B (NSC), C (TAP), A (Neuroblast), E (Ependymal) cell types
  - Notch-Delta lateral inhibition signaling between neighbors
  - Cell division: symmetric (B->B+B) and asymmetric (B->B+C)
  - Cell migration: A cells toward RMS (rightward) direction
  - Visualization: grid snapshots, population dynamics timeline
  - Output: PNG figures + JSON statistics

Usage:
    python -c "from core.niche_model import run_niche_simulation; run_niche_simulation()"
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import json
import os
from collections import Counter

# ── Cell Type Constants ──────────────────────────────────────────────────────
EMPTY, B_NSC, C_TAP, A_NB, E_EP = 0, 1, 2, 3, 4

CELL_TYPE_NAMES = {
    EMPTY: 'Empty',
    B_NSC: 'B (NSC)',
    C_TAP: 'C (TAP)',
    A_NB: 'A (Neuroblast)',
    E_EP: 'E (Ependymal)',
}

CELL_COLORS_HEX = {
    EMPTY: '#ffffff',
    B_NSC: '#1976D2',   # Blue – stem cells
    C_TAP: '#388E3C',   # Green – transit-amplifying progenitors
    A_NB:  '#F57C00',   # Orange – neuroblasts
    E_EP:  '#7B1FA2',   # Purple – ependymal cells
}

CELL_CMAP = ListedColormap([CELL_COLORS_HEX[i] for i in range(5)])

# ── Default Simulation Parameters ────────────────────────────────────────────
DEFAULT_PARAMS = {
    'height': 64,
    'width': 64,
    'n_steps': 100,
    'n_b_cells_init': 30,
    'n_c_cells_init': 15,
    'n_a_cells_init': 5,
    'b_division_prob': 0.03,
    'asymmetric_ratio': 0.3,
    'c_division_prob': 0.06,
    'c_diff_prob': 0.04,
    'a_migrate_prob': 0.6,
    'a_death_prob': 0.005,
    'max_age': 200,
    'seed': 42,
    'notch_iters': 3,
}

# ── Legend Proxy Artists ─────────────────────────────────────────────────────
LEGEND_PATCHES = [
    mpatches.Patch(color=CELL_COLORS_HEX[EMPTY], label='Empty'),
    mpatches.Patch(color=CELL_COLORS_HEX[B_NSC], label='B (NSC)'),
    mpatches.Patch(color=CELL_COLORS_HEX[C_TAP], label='C (TAP)'),
    mpatches.Patch(color=CELL_COLORS_HEX[A_NB],  label='A (Neuroblast)'),
    mpatches.Patch(color=CELL_COLORS_HEX[E_EP],  label='E (Ependymal)'),
]


# =============================================================================
#   SVZ Niche Model
# =============================================================================
class SVZNicheModel:
    """Agent-based model of the SVZ neural stem cell niche on a 2D grid."""

    def __init__(self, params=None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        H, W = self.params['height'], self.params['width']
        np.random.seed(self.params['seed'])

        self.H = H
        self.W = W
        self.step = 0

        # Grid-based cell state arrays
        self.type = np.full((H, W), EMPTY, dtype=np.int32)
        self.delta = np.zeros((H, W), dtype=np.float64)
        self.notch = np.zeros((H, W), dtype=np.float64)
        self.age = np.zeros((H, W), dtype=np.int32)

        # History (recorded per step)
        self.history_counts = []   # list[dict]
        self.history_snapshots = []  # list[np.ndarray] — grid snapshots

        self._seed_cells()
        self._update_signaling()

    # ── Initialisation ───────────────────────────────────────────────────

    def _seed_cells(self):
        """Place initial cell populations on the grid."""
        H, W = self.H, self.W

        # Ependymal cells line the top row (ventricle wall)
        self.type[0, :] = E_EP

        # ── Type B (NSC) — scattered in rows 1–12 ──
        b_positions = []
        for _ in range(self.params['n_b_cells_init']):
            for _ in range(20):  # retry if taken
                y = np.random.randint(1, 13)
                x = np.random.randint(0, W)
                if self.type[y, x] == EMPTY:
                    self.type[y, x] = B_NSC
                    self.age[y, x] = np.random.randint(0, 50)
                    b_positions.append((y, x))
                    break

        # ── Type C (TAP) — adjacent to B cells ──
        placed = 0
        attempts = 0
        target = self.params['n_c_cells_init']
        while placed < target and attempts < 2000 and b_positions:
            attempts += 1
            by, bx = b_positions[np.random.randint(len(b_positions))]
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            dy, dx = dirs[np.random.randint(len(dirs))]
            ny, nx = by + dy, bx + dx
            if 0 <= ny < H and 0 <= nx < W and self.type[ny, nx] == EMPTY:
                self.type[ny, nx] = C_TAP
                self.age[ny, nx] = 0
                placed += 1

        # ── Type A (Neuroblast) — scattered deeper in niche ──
        for _ in range(self.params['n_a_cells_init']):
            for _ in range(50):
                y = np.random.randint(3, 18)
                x = np.random.randint(0, W)
                if self.type[y, x] == EMPTY:
                    self.type[y, x] = A_NB
                    self.age[y, x] = 0
                    break

    # ── Notch-Delta Lateral Inhibition ──────────────────────────────────

    @staticmethod
    def _neighbor_offsets():
        """4-connected von Neumann neighborhood."""
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _neighbor_coords(self, y, x):
        """Yield (y, x) for valid 4-connected neighbours."""
        for dy, dx in self._neighbor_offsets():
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.H and 0 <= nx < self.W:
                yield ny, nx

    def _update_signaling(self):
        """Iterate Notch-Delta lateral inhibition toward equilibrium.

        For each occupied cell:
            notch = mean(neighbor_delta)
            delta = tanh(1 / (notch + 0.1))

        This creates a classic lateral-inhibition pattern: cells with
        high neighbour-Delta have high Notch and therefore low Delta,
        and vice versa.
        """
        H, W = self.H, self.W
        n_iters = self.params['notch_iters']
        sigmoid = lambda v: np.tanh(1.0 / (v + 0.1))

        for _ in range(n_iters):
            new_notch = np.zeros_like(self.notch)
            for y in range(H):
                for x in range(W):
                    if self.type[y, x] == EMPTY:
                        continue
                    s = 0.0
                    count = 0
                    for ny, nx in self._neighbor_coords(y, x):
                        s += self.delta[ny, nx]
                        count += 1
                    new_notch[y, x] = s / count if count else 0.0

            self.notch = new_notch
            # Only signal-responsive cells update Delta
            sig_cells = (self.type == B_NSC) | (self.type == C_TAP) | (self.type == A_NB)
            self.delta[sig_cells] = sigmoid(self.notch[sig_cells])
            self.delta[~sig_cells] = 0.0

    # ── Time Step ────────────────────────────────────────────────────────

    def step_simulation(self):
        """Advance the model by one time step."""
        self.step += 1
        H, W, prm = self.H, self.W, self.params

        # 1. Update Notch-Delta signaling
        self._update_signaling()

        # 2. B cell (NSC) division
        b_divisions = []
        for y in range(H):
            for x in range(W):
                if self.type[y, x] != B_NSC:
                    continue
                # Higher Notch → higher division probability (stemness maintenance)
                div_prob = prm['b_division_prob'] * (1.0 + self.notch[y, x])
                if np.random.random() > div_prob:
                    continue
                empty_nb = [(ny, nx) for ny, nx in self._neighbor_coords(y, x)
                            if self.type[ny, nx] == EMPTY]
                if not empty_nb:
                    continue
                ny, nx = empty_nb[np.random.randint(len(empty_nb))]
                b_divisions.append((y, x, ny, nx))

        for y, x, ny, nx in b_divisions:
            if self.type[ny, nx] != EMPTY:
                continue
            self.age[y, x] = 0  # parent resets age
            if np.random.random() < prm['asymmetric_ratio']:
                self.type[ny, nx] = C_TAP   # B → B + C
            else:
                self.type[ny, nx] = B_NSC   # B → B + B
            self.age[ny, nx] = 0

        # 3. C cell (TAP) division and differentiation
        c_actions = []
        for y in range(H):
            for x in range(W):
                if self.type[y, x] != C_TAP:
                    continue
                # Differentiation: C → A
                if np.random.random() < prm['c_diff_prob']:
                    c_actions.append((y, x, 'diff'))
                    continue
                # Division: C → C + C
                if np.random.random() < prm['c_division_prob']:
                    empty_nb = [(ny, nx) for ny, nx in self._neighbor_coords(y, x)
                                if self.type[ny, nx] == EMPTY]
                    if empty_nb:
                        ny, nx = empty_nb[np.random.randint(len(empty_nb))]
                        c_actions.append((y, x, 'div', ny, nx))

        for action in c_actions:
            if action[2] == 'diff':
                y, x = action[0], action[1]
                if self.type[y, x] == C_TAP:
                    self.type[y, x] = A_NB
                    self.age[y, x] = 0
            elif action[2] == 'div':
                y, x, ny, nx = action[0], action[1], action[3], action[4]
                if self.type[y, x] == C_TAP and self.type[ny, nx] == EMPTY:
                    self.type[ny, nx] = C_TAP
                    self.age[ny, nx] = 0
                    self.age[y, x] = 0

        # 4. A cell (Neuroblast) migration toward RMS (rightward)
        # Process columns right-to-left to avoid double-moving the same cell
        for y in range(H):
            for x in range(W - 2, -1, -1):
                if self.type[y, x] != A_NB:
                    continue
                if np.random.random() > prm['a_migrate_prob']:
                    continue
                # Primary: move right
                if self.type[y, x + 1] == EMPTY:
                    self._move_cell(y, x, y, x + 1)
                # Backup: diag-right (up or down) if blocked
                elif self.type[y, x + 1] != EMPTY:
                    for dy in [-1, 1]:
                        ny, nx = y + dy, x + 1
                        if 0 <= ny < H and self.type[ny, nx] == EMPTY:
                            self._move_cell(y, x, ny, nx)
                            break

        # 5. Cell death (A cell apoptosis + age-induced death)
        for y in range(H):
            for x in range(W):
                if self.type[y, x] == A_NB and np.random.random() < prm['a_death_prob']:
                    self._clear_cell(y, x)
                elif self.type[y, x] in (B_NSC, C_TAP, A_NB) and self.age[y, x] > prm['max_age']:
                    self._clear_cell(y, x)

        # 6. Age all living cells
        mask = self.type > EMPTY
        self.age[mask] += 1

        # 7. Record history
        self._record()

    # ── Helpers ─────────────────────────────────────────────────────────

    def _move_cell(self, sy, sx, ty, tx):
        """Move cell from (sy,sx) to (ty,tx); target must be empty."""
        self.type[ty, tx] = self.type[sy, sx]
        self.age[ty, tx] = self.age[sy, sx]
        self.delta[ty, tx] = self.delta[sy, sx]
        self.notch[ty, tx] = self.notch[sy, sx]
        self._clear_cell(sy, sx)

    def _clear_cell(self, y, x):
        """Empty the cell at (y, x) and reset properties."""
        self.type[y, x] = EMPTY
        self.age[y, x] = 0
        self.delta[y, x] = 0.0
        self.notch[y, x] = 0.0

    # ── Recording ───────────────────────────────────────────────────────

    def _record(self):
        """Record current counts and grid snapshot."""
        counts = Counter(self.type.flatten())
        self.history_counts.append({
            'step': self.step,
            'B_NSC': int(counts[B_NSC]),
            'C_TAP': int(counts[C_TAP]),
            'A_NB': int(counts[A_NB]),
            'E_EP': int(counts[E_EP]),
            'total': int(np.sum(self.type > EMPTY)),
        })
        self.history_snapshots.append(self.type.copy())

    # ── Run ─────────────────────────────────────────────────────────────

    def run(self, n_steps=None):
        """Run the simulation for *n_steps* iterations (default: params value).

        Returns the history_counts list.
        """
        if n_steps is None:
            n_steps = self.params['n_steps']
        self._record()  # t=0
        for i in range(n_steps):
            self.step_simulation()
        return self.history_counts

    # =====================================================================
    #   Visualization
    # =====================================================================

    def _add_legend(self, ax, **kwargs):
        """Add cell-type legend to the right of the axis."""
        ax.legend(
            handles=LEGEND_PATCHES,
            bbox_to_anchor=(1.02, 1.0),
            loc='upper left',
            frameon=True,
            **kwargs,
        )

    def plot_grid(self, step=None, ax=None, title=None, show_legend=True):
        """Plot the 2D cell grid as a colour-coded image.

        Parameters
        ----------
        step : int or None
            Which step from history to plot.  If None, plots the current state.
        ax : Axes or None
        title : str or None
        show_legend : bool
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        else:
            fig = ax.figure

        data = self.history_snapshots[step] if step is not None else self.type
        ax.imshow(data, cmap=CELL_CMAP, vmin=0, vmax=4,
                  aspect='equal', interpolation='nearest')
        s = self.history_counts[step]['step'] if step is not None else self.step
        ax.set_title(title or f'SVZ Niche — Step {s}')
        ax.set_xlabel('X (→ RMS direction)')
        ax.set_ylabel('Y (Ventricle → Parenchyma)')
        if show_legend:
            self._add_legend(ax, fontsize=9)
        return fig

    def plot_timeline(self, figsize=(10, 4)):
        """Plot cell-type population counts over the full simulation."""
        counts = self.history_counts
        steps = [c['step'] for c in counts]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(steps, [c['B_NSC'] for c in counts],
                color=CELL_COLORS_HEX[B_NSC], label='B (NSC)', linewidth=2)
        ax.plot(steps, [c['C_TAP'] for c in counts],
                color=CELL_COLORS_HEX[C_TAP], label='C (TAP)', linewidth=2)
        ax.plot(steps, [c['A_NB'] for c in counts],
                color=CELL_COLORS_HEX[A_NB], label='A (Neuroblast)', linewidth=2)
        ax.plot(steps, [c['E_EP'] for c in counts],
                color=CELL_COLORS_HEX[E_EP], label='E (Ependymal)', linewidth=2, linestyle='--')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cell Count')
        ax.set_title('SVZ Niche — Cell Type Population Dynamics')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_multi_snapshot(self, n_frames=5, figsize=(16, 3.5)):
        """Plot grid snapshots at evenly spaced time points."""
        total = len(self.history_snapshots)
        indices = [int(i * (total - 1) / max(n_frames - 1, 1)) for i in range(min(n_frames, total))]

        fig, axes = plt.subplots(1, len(indices), figsize=figsize)
        if len(indices) == 1:
            axes = [axes]
        for ax, idx in zip(axes, indices):
            self.plot_grid(step=idx, ax=ax,
                           title=f'Step {self.history_counts[idx]["step"]}',
                           show_legend=False)
        # Legend on the right of the last subplot
        axes[-1].legend(
            handles=LEGEND_PATCHES,
            bbox_to_anchor=(1.02, 1.0),
            loc='upper left',
            fontsize=9,
            frameon=True,
        )
        fig.tight_layout()
        return fig

    # =====================================================================
    #   Save Outputs
    # =====================================================================

    def save_results(self, output_dir='niche_output'):
        """Save all figures and statistics JSON to *output_dir*."""
        os.makedirs(output_dir, exist_ok=True)

        # ── Initial grid ──
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        self.plot_grid(step=0, ax=ax, title='SVZ Niche — Initial State')
        fig.savefig(os.path.join(output_dir, 'initial_grid.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ── Final grid ──
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        self.plot_grid(step=len(self.history_snapshots) - 1, ax=ax,
                       title=f'SVZ Niche — Final (Step {self.step})')
        fig.savefig(os.path.join(output_dir, 'final_grid.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ── Multi-frame snapshot ──
        fig = self.plot_multi_snapshot()
        fig.savefig(os.path.join(output_dir, 'niche_snapshots.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ── Population timeline ──
        fig = self.plot_timeline()
        fig.savefig(os.path.join(output_dir, 'population_timeline.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ── Statistics JSON ──
        stats = {
            'params': {k: v for k, v in self.params.items()},
            'final_step': self.step,
            'history': self.history_counts,
            'final_counts': self.history_counts[-1] if self.history_counts else {},
        }
        with open(os.path.join(output_dir, 'niche_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

        print(f'✅ Results saved to {output_dir}/')
        print(f'   - initial_grid.png')
        print(f'   - final_grid.png')
        print(f'   - niche_snapshots.png')
        print(f'   - population_timeline.png')
        print(f'   - niche_stats.json')
        return stats


# =============================================================================
#   Entry-point
# =============================================================================
def run_niche_simulation(params=None, output_dir='niche_output'):
    """Run the SVZ niche simulation and save all outputs."""
    model = SVZNicheModel(params)
    model.run()
    model.save_results(output_dir)
    return model


if __name__ == '__main__':
    run_niche_simulation()
