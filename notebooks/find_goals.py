import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def main(
    trajectory_path,
    image_path,
    eps=0.15,
    min_samples=10,
    start_row=1,
    step=1,
    image_size=(200, 200),
):
    """
    Analyze trajectory and find goals using DBSCAN clustering.

    Args:
        trajectory_path: Path to the trajectory CSV file
        image_path: Path to the background image file
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        start_row: Starting row for trajectory subsample
        step: Sampling step for trajectory
        image_size: Size to resize the image (width, height)
    """

    # Load trajectory
    try:
        print(f"Trajectory path: {trajectory_path}")
        print(f"Loading trajectory from {trajectory_path}...")
        trajectory = pd.read_csv(trajectory_path)
    except FileNotFoundError:
        print(f"Error: Trajectory file not found at {trajectory_path}")
        sys.exit(1)

    trajectory = trajectory[["x", "y"]]
    destination_file = trajectory_path[:-4] + '_center.csv'

    # Drop first rows and sample
    trajectory_drop = trajectory.iloc[start_row::step]

    # DBSCAN Clustering
    df = trajectory_drop[["x", "y"]].to_numpy()
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(df)

    labels = db.labels_

    unique_clusters = set(labels) - {-1}

    # Compute cluster centers (where person stays the most)
    cluster_centers = {}

    for cluster_id in unique_clusters:
        points = df[labels == cluster_id]
        center = points.mean(axis=0)
        cluster_centers[cluster_id] = center

    for cid, ctr in cluster_centers.items():
        print(f"  Cluster {cid}: center at {ctr}")

    # ------------------------------------------
    # Visualization - Clusters only
    # ------------------------------------------
    plt.figure(figsize=(6, 6))

    # Plot each cluster
    for cluster_id in unique_clusters:
        pts = df[labels == cluster_id]
        plt.scatter(pts[:, 0], pts[:, 1], label=f"Cluster {cluster_id}")

    # Plot noise
    noise = df[labels == -1]
    if len(noise) > 0:
        plt.scatter(noise[:, 0], noise[:, 1], color="gray", label="Noise")

    # Plot centers
    for cid, ctr in cluster_centers.items():
        plt.scatter(ctr[0], ctr[1], marker="x", s=200, linewidths=3, color="black")

    plt.legend()
    plt.title("Clusters of 'Still' Behavior in Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")
    if "hall" in trajectory_path.lower():
        plt.xlim(-3.1, 3.5)
        plt.ylim(-11, 11.26)
    elif "apartment" in trajectory_path.lower():
        plt.xlim(-10, 0)
        plt.ylim(-6.5, -0.2)
    elif "break" in trajectory_path.lower():
        plt.xlim(-6.7, 7)
        plt.ylim(-2.8, 5)
    plt.grid(True)
    plt.savefig(
        trajectory_path[:-4] + "_trajectory.png",
        dpi=300,
        bbox_inches="tight",
        transparent=False,
    )
    # plt.show()

    # ------------------------------------------
    # Load and visualize image with trajectory
    # ------------------------------------------
    import matplotlib.image as mpimg
    from PIL import Image

    # Load the image
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)

    img_resized = img.resize(image_size)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(img_resized)
    ax1.set_title("Background Image")

    for cluster_id in unique_clusters:
        pts = df[labels == cluster_id]
        ax2.scatter(pts[:, 0], pts[:, 1], label=f"Cluster {cluster_id}")

    # Plot noise
    noise = df[labels == -1]
    if len(noise) > 0:
        ax2.scatter(noise[:, 0], noise[:, 1], color="gray", label="Noise")

    # Plot centers
    for cid, ctr in cluster_centers.items():
        ax2.scatter(ctr[0], ctr[1], marker="x", s=200, linewidths=3, color="black")
    

    with open(destination_file, 'w') as file:
        # Write the new line to the file, including a newline character for formatting
        file.write("x,y\n")
        for cid, ctr in cluster_centers.items():
            x, y = ctr
            text = f"{x},{y}\n"
            file.write(text)

    ax2.set_title("Trajectory with Detected Goals")
    ax2.legend()

    if "hall" in trajectory_path.lower():
        ax2.set_xlim(-3.1, 3.5)
        ax2.set_ylim(-11, 11.26)
    elif "apartment" in trajectory_path.lower():
        ax2.set_xlim(-10, 0)
        ax2.set_ylim(-6.5, -0.2)
    elif "break" in trajectory_path.lower():
        ax2.set_xlim(-6.7, 7)
        ax2.set_ylim(-2.8, 5)

    plt.tight_layout()
    plt.savefig(
        trajectory_path[:-4] + "_comparison.png",
        dpi=300,
        bbox_inches="tight",
        transparent=False,
    )
    # plt.show()


if __name__ == "__main__":
    # python find_goals.py ../trajectory/pedestrian_positions_20251124_064305.csv ../worlds/.apartment_cropped.jpg --eps 0.2 --min-samples 15 --start-row 3500 --step 10 --image-size 300 300
    parser = argparse.ArgumentParser(
        description="Find goals from trajectory using DBSCAN clustering"
    )
    parser.add_argument("trajectory", help="Path to trajectory CSV file")
    parser.add_argument("image", help="Path to background image file")
    parser.add_argument(
        "--eps",
        type=float,
        default=0.15,
        help="DBSCAN epsilon parameter (default: 0.15)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="DBSCAN min_samples parameter (default: 10)",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=3500,
        help="Starting row for trajectory subsample (default: 3500)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Sampling step for trajectory (default: 10)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[200, 200],
        help="Image resize dimensions width height (default: 200 200)",
    )

    args = parser.parse_args()

    main(
        trajectory_path=args.trajectory,
        image_path=args.image,
        eps=args.eps,
        min_samples=args.min_samples,
        start_row=args.start_row,
        step=args.step,
        image_size=tuple(args.image_size),
    )
