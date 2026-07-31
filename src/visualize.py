import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


def plot_2d_trajectory(
    points_brut, points_corrige, output_path,
    title, invert_y=True, equal_aspect=True,
):
    cx_brut = [p[0] for p in points_brut]
    cy_brut = [p[1] for p in points_brut]
    cx_corr = [p[0] for p in points_corrige]
    cy_corr = [p[1] for p in points_corrige]
    n = len(points_brut)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    sc0 = axes[0].scatter(cx_brut, cy_brut, c=range(n), cmap="viridis", s=18)
    axes[0].plot(cx_brut, cy_brut, "-", color="gray", alpha=0.4, linewidth=1)
    axes[0].set_title(f"Trajectoire brute (pixels) - {title}")
    if invert_y:
        axes[0].invert_yaxis()
    if equal_aspect:
        axes[0].set_aspect("equal")
    fig.colorbar(sc0, ax=axes[0], label="frame (temps)")

    sc1 = axes[1].scatter(cx_corr, cy_corr, c=range(n), cmap="viridis", s=18)
    axes[1].plot(cx_corr, cy_corr, "-", color="gray", alpha=0.4, linewidth=1)
    axes[1].set_title(f"Trajectoire corrigee (distorsion enlevee) - {title}")
    if invert_y:
        axes[1].invert_yaxis()
    if equal_aspect:
        axes[1].set_aspect("equal")
    fig.colorbar(sc1, ax=axes[1], label="frame (temps)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_trajectory_overlay(points, background, output_path):
    bg = background.copy()
    pts = np.array([(int(p[0]), int(p[1])) for p in points], dtype=np.int32)
    n = len(pts)
    for i in range(1, n):
        couleur = (
            int(255 * (1 - i / n)),
            int(255 * (i / n)),
            0,
        )
        cv2.line(bg, tuple(pts[i - 1]), tuple(pts[i]), couleur, 2)
    for p in pts:
        cv2.circle(bg, tuple(p), 3, (0, 0, 255), -1)
    cv2.imwrite(output_path, bg)


def plot_3d_trajectory(points_3d, output_path):
    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]
    n = len(points_3d)
    couleurs = range(n)

    fig = plt.figure(figsize=(13, 10))

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    sc = ax3d.scatter(X, Y, Z, c=couleurs, cmap="viridis", s=15)
    ax3d.plot(X, Y, Z, color="gray", alpha=0.4, linewidth=1)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("Trajectoire 3D")
    fig.colorbar(sc, ax=ax3d, shrink=0.6, label="frame (temps)")

    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.scatter(X, Y, c=couleurs, cmap="viridis", s=12)
    ax_xy.plot(X, Y, color="gray", alpha=0.4, linewidth=1)
    ax_xy.set_title("Vue de dessus (XY)")
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.set_aspect("equal")

    ax_xz = fig.add_subplot(2, 2, 3)
    ax_xz.scatter(X, Z, c=couleurs, cmap="viridis", s=12)
    ax_xz.plot(X, Z, color="gray", alpha=0.4, linewidth=1)
    ax_xz.set_title("Vue de face (XZ)")
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z")
    ax_xz.set_aspect("equal")

    ax_yz = fig.add_subplot(2, 2, 4)
    ax_yz.scatter(Y, Z, c=couleurs, cmap="viridis", s=12)
    ax_yz.plot(Y, Z, color="gray", alpha=0.4, linewidth=1)
    ax_yz.set_title("Vue de cote (YZ)")
    ax_yz.set_xlabel("Y")
    ax_yz.set_ylabel("Z")
    ax_yz.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_3d_interactive(points_3d, frame_names, output_path):
    if not _PLOTLY_AVAILABLE:
        print(
            "plotly non installe : graphique interactif non genere. "
            "Installe-le avec 'pip install plotly' pour l'obtenir."
        )
        return

    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]
    n = len(points_3d)

    fig = go.Figure(data=[
        go.Scatter3d(
            x=X, y=Y, z=Z,
            mode="markers+lines",
            marker=dict(
                size=4,
                color=list(range(n)),
                colorscale="Viridis",
                colorbar=dict(title="frame (temps)"),
            ),
            line=dict(color="gray", width=2),
            text=frame_names,
            hovertemplate=(
                "frame=%{text}<br>"
                "X=%{x:.2f}<br>Y=%{y:.2f}<br>Z=%{z:.2f}"
                "<extra></extra>"
            ),
        )
    ])
    fig.update_layout(
        title="Trajectoire 3D (interactive)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.write_html(output_path)
