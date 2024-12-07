from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plots)

# Helper functions
def _generate_polygon(vertices: int, radius: float = 1, center: Tuple[float, float] = (0, 0)) -> List[Tuple[float, float]]:
    """
    Generate the vertices of a regular polygon.

    Arguments:
        vertices: Number of vertices of the polygon.
        radius: Radius of the polygon.
        center: (x, y) coordinates of the center.

    Returns:
        List of (x, y) coordinates for the polygon vertices.
    """
    angles = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
    return [
        (center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle))
        for angle in angles
    ]


def generate_sphere(radius: float, resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the coordinates of a sphere.

    Arguments:
        radius: Radius of the sphere.
        resolution: Resolution of the sphere mesh.

    Returns:
        Tuple of 3D arrays (x, y, z) representing the sphere.
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


# Endpoints
def plot_unwound_necklace(beads: List[str], cuts: Optional[List[int]] = []) -> None:
    n = len(beads)
    x = np.arange(0, n)
    y = np.zeros(n) 

    # Initialize plot
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.set_aspect('auto')
    ax.axis('off')

    # Plot the beads
    marker_size = 20
    for i in range(n):
        ax.plot(x[i], y[i], 'o', markersize=marker_size, color=beads[i], markeredgecolor='black')

    # Plot the lines connecting beads (necklace string)
    for i in range(n):
        next_idx = (i + 1 if i + 1 < n else i)
        if cuts and i in cuts:
            if i < n - 1:  # Ensure we're not at the last bead
                x_gap = (x[i] + x[i + 1]) / 2
                ax.plot([x_gap, x_gap], [-0.5, 0.5], color='blue', linestyle='-', alpha=0.8, linewidth=2)
        else:
            edge_color = 'gray'
            ax.plot([x[i], x[next_idx]], [y[i], y[next_idx]], color=edge_color, linewidth=1.5)

    plt.show()

def plot_wound_necklace(beads: List[str], cuts: Optional[List[int]] = []) -> None:
    n = len(beads)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    x = np.cos(angles)
    y = np.sin(angles)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('auto')
    ax.axis('off')

    # Plot the beads
    marker_size = 30
    for i in range(n):
        ax.plot(x[i], y[i], 'o', markersize=marker_size, color=beads[i], markeredgecolor='gray')

    # Plot the lines connecting beads (necklace string)
    for i in range(n):
        next_idx = (i + 1) % n
        if cuts and i in cuts:
            # Get midpoint of the gap, which will be the midpoint of our line segement
            angle_mid = (angles[i] + angles[(i + 1) % n]) / 2
            x_mid = np.cos(angle_mid)
            y_mid = np.sin(angle_mid)

            scale_factor = 0.3
            x_start, y_start = (1 + scale_factor) * x_mid, (1 + scale_factor) * y_mid
            x_end, y_end = (1 - scale_factor) * x_mid, (1 - scale_factor) * y_mid
            ax.plot([x_start, x_end], [y_start, y_end], color='blue', linewidth=1.5)
            
        else:
            edge_color = 'black'
            ax.plot([x[i], x[next_idx]], [y[i], y[next_idx]], color=edge_color, linewidth=1.5)

    plt.show()


def rotating_sphere(save_path: str = "rotating_sphere.gif", fps: int = 20) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    radius = 1
    x, y, z = generate_sphere(radius)
    ax.plot_surface(x, y, z, color="blue", alpha=0.6, edgecolor="white")

    def update(frame: int) -> None:
        ax.view_init(elev=30, azim=frame)

    ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)
    ani.save(save_path, writer=PillowWriter(fps=fps))
    print(f"Animation saved as {save_path}...")


def shrinking_sphere_and_2d(save_path: str = "shrinking_sphere_and_2d.gif", num_frames: int = 100, fps: int = 20) -> None:
    fig = plt.figure(figsize=(10, 5))

    # Left: Sphere with shrinking circle
    ax_sphere = fig.add_subplot(121, projection="3d")
    radius = 1
    x, y, z = generate_sphere(radius)

    # Right: 2D shrinking polygon
    ax_2d = fig.add_subplot(122)
    polygon_vertices = _generate_polygon(12, radius=5)

    def update(frame: int) -> None:
        ax_sphere.cla()
        ax_2d.cla()

        shrink_factor = 1 - frame / num_frames
        shrinking_circle_z = np.cos(np.pi / 2 * shrink_factor)
        shrinking_circle_radius = np.sqrt(1 - shrinking_circle_z**2)

        circle_angle = np.linspace(0, 2 * np.pi, 100)
        circle_x = shrinking_circle_radius * np.cos(circle_angle)
        circle_y = shrinking_circle_radius * np.sin(circle_angle)
        circle_z = np.full_like(circle_x, shrinking_circle_z)

        ax_sphere.plot_surface(x, y, z, color="blue", alpha=0.3, edgecolor="white")
        ax_sphere.plot(circle_x, circle_y, circle_z, color="red", linewidth=2)

        ax_sphere.set_xlim([-1.5, 1.5])
        ax_sphere.set_ylim([-1.5, 1.5])
        ax_sphere.set_zlim([-1.5, 1.5])
        ax_sphere.set_box_aspect([1, 1, 1])
        ax_sphere.axis("off")

        # Shrinking 2D polygon
        shrunken_vertices = [
            (shrink_factor * vx, shrink_factor * vy) for vx, vy in polygon_vertices
        ]
        x_2d, y_2d = zip(*shrunken_vertices)
        ax_2d.fill(x_2d, y_2d, color="lightblue", alpha=0.6, edgecolor="blue")
        ax_2d.set_xlim(-6, 6)
        ax_2d.set_ylim(-6, 6)
        ax_2d.set_aspect("equal")

    ani = FuncAnimation(fig, update, frames=num_frames, interval=50)
    ani.save(save_path, writer=PillowWriter(fps=fps))
    print(f"Animation saved as {save_path}...")


def borsuk_ulum(save_path: str = "borsuk_ulum.gif", num_frames: int = 180, fps: int = 35) -> None:
    fig = plt.figure(figsize=(10, 5))
    ax_sphere = fig.add_subplot(121, projection="3d")
    ax_plane = fig.add_subplot(122)

    radius = 1
    x, y, z = generate_sphere(radius)

    def update(frame: int) -> None:
        ax_sphere.cla()
        ax_plane.cla()

        ax_sphere.plot_surface(x, y, z, color="blue", alpha=0.6, edgecolor="white")
        theta = np.pi * frame / num_frames
        phi = 2 * np.pi * frame / num_frames

        px1 = radius * np.sin(theta) * np.cos(phi)
        py1 = radius * np.sin(theta) * np.sin(phi)
        pz1 = radius * np.cos(theta)

        px2, py2, pz2 = -px1, -py1, -pz1

        ax_sphere.scatter([px1, px2], [py1, py2], [pz1, pz2], color=["red", "green"])
        ax_plane.scatter([px1, px2], [py1, py2], color=["red", "green"])

    ani = FuncAnimation(fig, update, frames=num_frames, interval=50)
    ani.save(save_path, writer=PillowWriter(fps=fps))
    print(f"Animation saved as {save_path}...")


def plot_2d_area_in_3d(vertices: List[Tuple[float, float]], z_height: float = 0, color: str = "lightblue", alpha: float = 0.6) -> None:
    if vertices[0] != vertices[-1]:
        vertices.append(vertices[0])

    x, y = zip(*vertices)
    z = [z_height] * len(vertices)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    verts = [list(zip(x, y, z))]
    poly = Poly3DCollection(verts, facecolor=color, alpha=alpha)
    ax.add_collection3d(poly)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_zlim(z_height, z_height + 1)
    plt.show()
