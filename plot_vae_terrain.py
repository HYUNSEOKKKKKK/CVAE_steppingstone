import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def Rz_deg(yaw_deg: float) -> np.ndarray:
    th = np.deg2rad(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)


def make_box_vertices(center, width, height, thickness, yaw_deg) -> np.ndarray:
    cx, cy, cz = map(float, center)
    w = float(width)
    h = float(height)
    t = float(thickness)

    dx = w / 2.0
    dy = h / 2.0
    dz = t / 2.0

    local_vertices = np.array([
        [-dx, -dy, -dz],
        [ dx, -dy, -dz],
        [ dx,  dy, -dz],
        [-dx,  dy, -dz],
        [-dx, -dy,  dz],
        [ dx, -dy,  dz],
        [ dx,  dy,  dz],
        [-dx,  dy,  dz],
    ], dtype=float)

    R = Rz_deg(yaw_deg)
    world_vertices = (R @ local_vertices.T).T
    world_vertices += np.array([cx, cy, cz], dtype=float)

    return world_vertices


def box_faces_from_vertices(V: np.ndarray):
    return [
        [V[0], V[1], V[2], V[3]],
        [V[4], V[5], V[6], V[7]],
        [V[0], V[1], V[5], V[4]],
        [V[1], V[2], V[6], V[5]],
        [V[2], V[3], V[7], V[6]],
        [V[3], V[0], V[4], V[7]],
    ]


def set_axes_equal(ax):
    x_limits = np.array(ax.get_xlim3d())
    y_limits = np.array(ax.get_ylim3d())
    z_limits = np.array(ax.get_zlim3d())

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max(x_range, y_range, z_range)

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_stones_3d_from_json(
    json_path: str | Path,
    terrain_key: Optional[str] = None,
    thickness: float = 0.04,
    face_alpha: float = 0.55,
    edgecolor: str = "k",
    show_yaw_arrow: bool = False,
    save_path: Optional[str | Path] = None,
):
    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if terrain_key is None:
        terrain_key = next(iter(data.keys()))

    if terrain_key not in data:
        raise KeyError(f"'{terrain_key}' not found in {json_path}. Available keys: {list(data.keys())}")

    block = data[terrain_key]
    stones: List[Dict[str, Any]] = block["stones"]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    n = len(stones)
    centers_z = np.array([float(s["center"][2]) for s in stones], dtype=float)
    z_floor = float(np.min(centers_z) - thickness / 2.0 - 0.1)
    use_colorbar = False
    if n >= 3:
        mid_z = centers_z[1:-1]
        z_min, z_max = float(mid_z.min()), float(mid_z.max())
        use_colorbar = z_max > z_min
        if not use_colorbar:
            z_min, z_max = z_min - 0.5, z_max + 0.5
    else:
        z_min, z_max = 0.0, 1.0

    cmap = plt.get_cmap("gray")
    norm = Normalize(vmin=z_min - 0.01, vmax=z_max + 0.01)

    def gray_by_z(z):
        if n < 3:
            return (0.65, 0.65, 0.65, 1.0)
        return cmap(norm(z))

    all_vertices = []

    for idx, s in enumerate(stones):
        cx, cy, cz = map(float, s["center"])
        w = float(s["width"])
        h = float(s["height"])
        yaw = float(s.get("yaw_deg", 0.0))

        if idx == 0:
            facecolor = (0.0, 0.7, 0.2, 1.0)
        elif idx == n - 1:
            facecolor = (0.9, 0.1, 0.1, 1.0)
        else:
            facecolor = gray_by_z(cz)

        V = make_box_vertices((cx, cy, cz), w, h, thickness, yaw)

        # --- shadow 추가 ---
        shadow = V[[0, 1, 2, 3]].copy()  # stone의 아랫면 4개 꼭짓점 사용
        shadow[:, 2] = z_floor  # 바닥 높이로 투영
        shadow[:, 0] += 0.015  # x 방향 살짝 이동
        shadow[:, 1] -= 0.015  # y 방향 살짝 이동

        shadow_poly = Poly3DCollection(
            [shadow],
            facecolors=(0.0, 0.0, 0.0, 0.12),  # 검정, 투명하게
            edgecolors='none'
        )
        ax.add_collection3d(shadow_poly)

        # --- 원래 stone ---
        faces = box_faces_from_vertices(V)
        poly = Poly3DCollection(
            faces,
            facecolors=facecolor,
            edgecolors=edgecolor,
            linewidths=0.8,
            alpha=face_alpha,
        )
        ax.add_collection3d(poly)
        all_vertices.append(V)

        if show_yaw_arrow:
            arrow_len = 0.45 * max(w, h)
            dir_vec = Rz_deg(yaw) @ np.array([arrow_len, 0.0, 0.0])
            ax.quiver(
                cx, cy, cz + thickness * 0.55,
                dir_vec[0], dir_vec[1], 0.0,
                length=1.0,
                normalize=False,
                color="tab:blue",
                linewidth=1.5,
            )

    if all_vertices:
        P = np.vstack(all_vertices)
        rng_x = np.ptp(P[:, 0])
        rng_y = np.ptp(P[:, 1])
        rng_z = np.ptp(P[:, 2])
        pad = 0.01 * max(rng_x, rng_y, rng_z, 1e-6)

        ax.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad)
        ax.set_ylim(P[:, 1].min() - pad, P[:, 1].max() + pad)
        ax.set_zlim(z_floor - 0.01, P[:, 2].max() + pad)

    set_axes_equal(ax)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Generated Stepping Stones ({terrain_key})", pad=18)
    ax.set_box_aspect((1, 1, 1.1))
    ax.grid(False)
    ax.view_init(elev=22, azim=-120)
    ax.set_axis_off()
    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    except Exception:
        pass

    legend_patches = []
    if n >= 1:
        legend_patches.append(Patch(facecolor=(0.0, 0.7, 0.2), edgecolor="k", label="First stone"))
    if n >= 2:
        legend_patches.append(Patch(facecolor=(0.9, 0.1, 0.1), edgecolor="k", label="Last stone"))

    if legend_patches:
        ax.legend(handles=legend_patches, loc="upper right")

    if n >= 3:
        if use_colorbar:
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])
            cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Height (z)", rotation=90)
        else:
            ax.text2D(
                0.98, 0.92,
                "Middle stones: constant height",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9
            )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    plt.show()


if __name__ == "__main__":
    json_path = Path("vae_terrain.json")

    plot_stones_3d_from_json(
        json_path=json_path,
        terrain_key="terrain",
        thickness=0.04,
        face_alpha=0.55,
        edgecolor="k",
        show_yaw_arrow=False,
        save_path="vae_run_plot.png",
    )