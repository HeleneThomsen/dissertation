#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:17:17 2026

@author: au605715
"""

"""
3D illustration of two embedding vectors with:
- projection-based depth fading for arrows and endpoint markers
- dashed drop lines and landing markers on the XZ floor (data Y = 0)
- spherical interpolation (slerp) for the theta arc
- equal 3D aspect ratio and custom projected 3D arrows
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch


class Arrow3D(FancyArrowPatch):
    """A 3D arrow projected into Matplotlib's 2D drawing plane."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return float(np.min(zs))

    def draw(self, renderer):
        self.do_3d_projection(renderer)
        super().draw(renderer)


def to_plot(v):
    """Map data (X, Y, Z) to Matplotlib axes (X, Z, Y)."""
    v = np.asarray(v)
    if v.ndim == 1:
        return np.array([v[0], v[2], v[1]])
    return v[:, [0, 2, 1]]


def slerp(v1, v2, t):
    """Spherical linear interpolation between two vectors."""
    v1n, v2n = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
    omega = np.arccos(dot)
    if omega < 1e-6:
        return np.outer(1 - t, v1n) + np.outer(t, v2n)
    sin_omega = np.sin(omega)
    a = np.sin((1 - t) * omega) / sin_omega
    b = np.sin(t * omega) / sin_omega
    return np.outer(a, v1n) + np.outer(b, v2n)


# 1. Sample 3D embedding vectors
embedding_a = np.array([3.0, 5.0, 6.0])
embedding_b = np.array([4.2, 1.7, 1.7])  # shortened bottom vector
origin = np.zeros(3)
vectors = (embedding_a, embedding_b)

# 2. Set up plot and camera before computing projected depths
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(0, 7)
ax.set_ylim(0, 7)
ax.set_zlim(0, 7)
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=18, azim=-60)
fig.canvas.draw()  # finalizes ax.get_proj()


# Projection-depth helpers. Larger projected z is treated as nearer.
def projected_depth(data_point):
    p = to_plot(data_point)
    return float(proj3d.proj_transform(p[0], p[1], p[2], ax.get_proj())[2])


# Use arrow midpoints and tips together to establish a stable scene depth range.
depth_samples = [projected_depth(origin)]
for vec in vectors:
    depth_samples.extend([projected_depth(0.5 * vec), projected_depth(vec)])
d_min, d_max = min(depth_samples), max(depth_samples)


def depth_style(data_point, alpha_range=(0.48, 1.0), lw_range=(1.35, 2.55)):
    """Return (alpha, linewidth) from normalized projected camera depth."""
    d = projected_depth(data_point)
    near = (d - d_min) / max(d_max - d_min, 1e-12)
    near = float(np.clip(near, 0.0, 1.0))
    alpha = alpha_range[0] + near * (alpha_range[1] - alpha_range[0])
    linewidth = lw_range[0] + near * (lw_range[1] - lw_range[0])
    return alpha, linewidth


# 3. Bounding cube
cube_min, cube_max = 0, 7
corners = np.array([[x, y, z]
                    for x in (cube_min, cube_max)
                    for y in (cube_min, cube_max)
                    for z in (cube_min, cube_max)])
edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
         if np.sum(corners[i] != corners[j]) == 1]

for i, j in edges:
    p1, p2 = to_plot(corners[i]), to_plot(corners[j])
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color="gray", linewidth=0.7, linestyle=":", alpha=0.6)

# Faint grid lines on the three cube faces meeting at the origin
grid_faces = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
for var1, var2, fixed in grid_faces:
    for val in range(cube_min, cube_max + 1):
        p_start, p_end = np.zeros(3), np.zeros(3)
        p_start[var1] = p_end[var1] = val
        p_start[var2], p_end[var2] = cube_min, cube_max
        p1, p2 = to_plot(p_start), to_plot(p_end)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color="gray", linewidth=0.5, linestyle=":", alpha=0.55)

        p_start2, p_end2 = np.zeros(3), np.zeros(3)
        p_start2[var2] = p_end2[var2] = val
        p_start2[var1], p_end2[var1] = cube_min, cube_max
        p1b, p2b = to_plot(p_start2), to_plot(p_end2)
        ax.plot([p1b[0], p2b[0]], [p1b[1], p2b[1]], [p1b[2], p2b[2]],
                color="gray", linewidth=0.5, linestyle=":", alpha=0.55)

# 4. Drop lines to the XZ floor. In data coordinates, that floor is Y = 0.
for vec in vectors:
    floor_point = np.array([vec[0], 0.0, vec[2]])
    shadow_alpha, _ = depth_style(0.5 * (vec + floor_point),
                                  alpha_range=(0.28, 0.52),
                                  lw_range=(0.8, 1.1))
    tip_p, floor_p = to_plot(vec), to_plot(floor_point)
    ax.plot([tip_p[0], floor_p[0]], [tip_p[1], floor_p[1]],
            [tip_p[2], floor_p[2]],
            color="0.30", linewidth=1.0, linestyle=(0, (3, 3)),
            alpha=shadow_alpha, zorder=2)
    ax.scatter(*floor_p, color="0.25", s=23, marker="o",
               alpha=min(shadow_alpha + 0.12, 0.68),
               depthshade=False, zorder=3)

# 5. Depth-faded arrows and endpoints
origin_p = to_plot(origin)
for vec in vectors:
    vec_p = to_plot(vec)
    arrow_alpha, arrow_lw = depth_style(0.55 * vec)
    dot_alpha, _ = depth_style(vec, alpha_range=(0.55, 1.0), lw_range=(1, 1))

    arrow = Arrow3D([origin_p[0], vec_p[0]],
                    [origin_p[1], vec_p[1]],
                    [origin_p[2], vec_p[2]],
                    mutation_scale=18, lw=arrow_lw, arrowstyle="-|>",
                    color="black", alpha=arrow_alpha, zorder=5)
    ax.add_artist(arrow)
    ax.scatter(*vec_p, color="black", s=50, alpha=dot_alpha,
               depthshade=False, zorder=6)

# 6. Labels
def label_pos(vec, other_vec, radial_extra=0.9, perp_scale=0.45):
    unit = vec / np.linalg.norm(vec)
    helper = np.array([0, 0, 1]) if abs(unit[2]) < 0.9 else np.array([1, 0, 0])
    perp = np.cross(vec, helper)
    perp = perp / np.linalg.norm(perp)
    if np.dot(perp, other_vec) > 0:
        perp = -perp
    return vec + unit * radial_extra + perp * perp_scale


label_a_pos = label_pos(embedding_a, embedding_b)
label_b_pos = np.array([5.1, 0.9, 1.8])

ax.text(*to_plot(label_a_pos), "Embedding A", fontsize=16, ha="left")
ax.text(*to_plot(label_b_pos), "Embedding B", fontsize=16, ha="left")

# 7. Theta arc
t = np.linspace(0, 1, 40)
arc_radius = 1.5
arc_points = to_plot(slerp(embedding_a, embedding_b, t) * arc_radius)
ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2],
        color="red", linewidth=2, zorder=7)

mid_arc = to_plot(slerp(embedding_a, embedding_b, np.array([0.5]))[0]
                  * arc_radius * 1.15)
ax.text(*mid_arc, r"$\theta$", color="red", fontsize=16,
        fontweight="bold", zorder=8)

# 8. Styling
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.grid(False)
ax.set_axis_off()

for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_visible(False)
    axis.pane.set_edgecolor("lightgray")

plt.tight_layout()

plt.savefig("/Users/au605715/Documents/GitHub/embeddings_3d_depth_shadow.png",
            dpi=200, bbox_inches="tight")
plt.show()