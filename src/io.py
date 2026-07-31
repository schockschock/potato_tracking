import csv
import glob
import os
from pathlib import Path


def read_images(directory, extension="*.tif"):
    if isinstance(directory, Path):
        directory = str(directory)
    return sorted(glob.glob(os.path.join(directory, extension)))


def read_trajectory_2d_csv(path):
    lignes = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lignes.append({
                "frame": row["frame"],
                "cx": float(row["cx_corrige"]),
                "cy": float(row["cy_corrige"]),
            })
    return lignes


def write_trajectory_2d_csv(path, trajectory):
    fieldnames = [
        "frame", "cx_brut", "cy_brut", "cx_corrige", "cy_corrige", "aire_px",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trajectory)


def write_trajectory_3d_csv(path, points_3d, frame_names):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "X", "Y", "Z"])
        for name, pt in zip(frame_names, points_3d):
            writer.writerow([name, pt[0], pt[1], pt[2]])


def write_segmentation_summary(path, results, background_path=None):
    with open(path, "w", encoding="utf-8") as f:
        if background_path is not None:
            f.write(f"Image de fond : {os.path.basename(background_path)}\n\n")
        f.write(f"{'nom':<30} {'aire_px':>10} {'bbox (x,y,w,h)'}\n")
        for r in results:
            f.write(
                f"{r['nom']:<30} {r['aire_px']:>10.0f} {r['bbox']}\n"
            )
