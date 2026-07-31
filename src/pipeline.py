"""
Pipeline complet : acquisition -> segmentation -> trajectoire 2D -> 3D -> sorties.

Utilisation en tant que module :
    from src.pipeline import run_pipeline, PipelineParams
    run_pipeline(data_dir, calibration_yml, output_dir, params=...)

Utilisation autonome :
    python -m src.pipeline
    python -m src.pipeline --data-dir /path/to/data --calib config/camera_parameters.yaml
"""

import os

import cv2
import numpy as np

from .camera import load_stereo_calibration, undistort_point
from .config import PipelineParams
from .io import read_images, write_trajectory_2d_csv, write_trajectory_3d_csv
from .segmentation import (
    apply_morphology,
    compute_centroid,
    find_largest_contour,
    segment_by_difference,
)
from .visualize import (
    plot_2d_trajectory,
    plot_3d_interactive,
    plot_3d_trajectory,
    plot_trajectory_overlay,
)


def _track_2d_camera(image_dir, intrinsics, background_path, params):
    images = read_images(image_dir)
    if not images:
        print(f"Aucune image trouvee dans {image_dir}")
        return [], [], []

    bg_img = cv2.imread(background_path)
    if bg_img is None:
        raise RuntimeError(f"Impossible de lire l'image de fond : {background_path}")

    trajectoire = []
    points_brut = []
    points_corrige = []

    for path in images:
        if path == background_path:
            continue
        nom = os.path.splitext(os.path.basename(path))[0]

        img = cv2.imread(path)
        if img is None:
            print("Erreur lecture :", path)
            continue

        masque = segment_by_difference(
            img, bg_img,
            threshold=params.threshold,
        )
        masque = apply_morphology(masque, kernel_size=params.morph_kernel_size)

        contour = find_largest_contour(masque, min_area=params.min_area)
        if contour is None:
            print("  -> ignoree :", nom)
            continue

        centroide = compute_centroid(contour)
        if centroide is None:
            continue

        cx, cy = centroide
        cx_c, cy_c = undistort_point(cx, cy, intrinsics.K, intrinsics.dist)
        aire = cv2.contourArea(contour)

        trajectoire.append({
            "frame": nom,
            "cx_brut": cx, "cy_brut": cy,
            "cx_corrige": cx_c, "cy_corrige": cy_c,
            "aire_px": aire,
        })
        points_brut.append((cx, cy))
        points_corrige.append((cx_c, cy_c))

    return trajectoire, points_brut, points_corrige


def run_pipeline(data_dir, calibration_yml, output_dir, params=None):
    if params is None:
        params = PipelineParams()

    if isinstance(data_dir, str):
        from pathlib import Path
        data_dir = Path(data_dir)
    if isinstance(calibration_yml, str):
        from pathlib import Path
        calibration_yml = Path(calibration_yml)
    if isinstance(output_dir, str):
        from pathlib import Path
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    stereo = load_stereo_calibration(str(calibration_yml))

    gauche_dir = str(data_dir / "CamG")
    droite_dir = str(data_dir / "CamD")

    images_g = read_images(gauche_dir)
    images_d = read_images(droite_dir)
    print(f"Images gauche : {len(images_g)}")
    print(f"Images droite : {len(images_d)}")

    if not images_g or not images_d:
        raise RuntimeError(
            f"Aucune image trouvee dans {gauche_dir} ou {droite_dir}"
        )

    bg_gauche = images_g[0]
    bg_droite = images_d[0]
    print("Image de fond gauche :", os.path.basename(bg_gauche))
    print("Image de fond droite :", os.path.basename(bg_droite))

    # --- 2D trajectories ---
    print("\n--- Trajectoires 2D ---")
    traj_g, pts_brut_g, pts_corr_g = _track_2d_camera(
        gauche_dir, stereo.left, bg_gauche, params,
    )
    traj_d, pts_brut_d, pts_corr_d = _track_2d_camera(
        droite_dir, stereo.right, bg_droite, params,
    )

    if not traj_g or not traj_d:
        raise RuntimeError("Echec du suivi 2D sur au moins une camera")

    print(f"Gauche : {len(traj_g)} frames")
    print(f"Droite : {len(traj_d)} frames")

    # Save 2D CSVs
    os.makedirs(str(output_dir / "trajectoire_gauche"), exist_ok=True)
    os.makedirs(str(output_dir / "trajectoire_droite"), exist_ok=True)

    csv_g = str(output_dir / "trajectoire_gauche" / "trajectoire_2d.csv")
    csv_d = str(output_dir / "trajectoire_droite" / "trajectoire_2d.csv")
    write_trajectoire_2d_csv(csv_g, traj_g)
    write_trajectoire_2d_csv(csv_d, traj_d)

    # 2D plots
    plot_2d_trajectory(
        pts_brut_g, pts_corr_g,
        str(output_dir / "trajectoire_gauche" / "trajectoire_2d.png"),
        title="gauche",
    )
    plot_2d_trajectory(
        pts_brut_d, pts_corr_d,
        str(output_dir / "trajectoire_droite" / "trajectoire_2d.png"),
        title="droite",
    )

    bg_img_g = cv2.imread(bg_gauche)
    bg_img_d = cv2.imread(bg_droite)
    if bg_img_g is not None:
        plot_trajectory_overlay(
            pts_corr_g, bg_img_g,
            str(output_dir / "trajectoire_gauche" / "trajectoire_2d_overlay.png"),
        )
    if bg_img_d is not None:
        plot_trajectory_overlay(
            pts_corr_d, bg_img_d,
            str(output_dir / "trajectoire_droite" / "trajectoire_2d_overlay.png"),
        )

    # --- Match frames ---
    print("\n--- Appariement des frames ---")
    dict_g = {t["frame"]: t for t in traj_g}
    dict_d = {t["frame"]: t for t in traj_d}
    noms_communs = sorted(f for f in dict_g if f in dict_d)

    if noms_communs:
        print(f"Appariement par nom : {len(noms_communs)} paires")
        paires = [
            (nom, (dict_g[nom]["cx_corrige"], dict_g[nom]["cy_corrige"]),
                   (dict_d[nom]["cx_corrige"], dict_d[nom]["cy_corrige"]))
            for nom in noms_communs
        ]
    else:
        n = min(len(traj_g), len(traj_d))
        print(f"AVERTISSEMENT : appariement par index ({n} frames)")
        paires = [
            (traj_g[i]["frame"],
             (traj_g[i]["cx_corrige"], traj_g[i]["cy_corrige"]),
             (traj_d[i]["cx_corrige"], traj_d[i]["cy_corrige"]))
            for i in range(n)
        ]

    if not paires:
        raise RuntimeError("Aucune paire de points a trianguler")

    # --- 3D triangulation ---
    print("\n--- Triangulation 3D ---")
    P_G = stereo.left.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P_D = stereo.right.K @ np.hstack([
        stereo.R, stereo.T.reshape(3, 1),
    ])

    pts_G = np.array([[p[1][0], p[1][1]] for p in paires], dtype=np.float64).T
    pts_D = np.array([[p[2][0], p[2][1]] for p in paires], dtype=np.float64).T

    points_4d = cv2.triangulatePoints(P_G, P_D, pts_G, pts_D)
    points_3d = (points_4d[:3] / points_4d[3]).T
    frame_names = [p[0] for p in paires]

    print(f"Points 3D obtenus : {len(points_3d)}")

    # Save 3D CSV
    csv_3d = str(output_dir / "trajectoire_3d.csv")
    write_trajectoire_3d_csv(csv_3d, points_3d, frame_names)

    # 3D plots
    plot_3d_trajectory(
        points_3d, str(output_dir / "trajectoire_3d.png"),
    )
    plot_3d_interactive(
        points_3d, frame_names,
        str(output_dir / "trajectoire_3d.html"),
    )

    print("\n====================")
    print(f"Pipeline termine. Resultats dans : {output_dir}/")
    print("====================")

    return output_dir


if __name__ == "__main__":
    import argparse
    from .config import get_data_dir, get_output_dir

    parser = argparse.ArgumentParser(
        description="Pipeline complet de suivi 3D de pomme de terre",
    )
    parser.add_argument(
        "--data-dir",
        default=str(get_data_dir() or "data"),
        help="Dossier contenant CamG/ et CamD/",
    )
    parser.add_argument(
        "--calib",
        default="config/camera_parameters.yaml",
        help="Fichier YAML de calibration stereo",
    )
    parser.add_argument(
        "--output",
        default=str(get_output_dir() or "output"),
        help="Dossier de sortie",
    )
    parser.add_argument(
        "--threshold", type=int, default=20,
        help="Seuil de difference pour soustraction de fond",
    )
    parser.add_argument(
        "--morph-kernel", type=int, default=7,
        help="Taille du noyau morphologique",
    )
    parser.add_argument(
        "--min-area", type=int, default=500,
        help="Aire minimale de l'objet (px)",
    )
    args = parser.parse_args()

    params = PipelineParams(
        threshold=args.threshold,
        morph_kernel_size=args.morph_kernel,
        min_area=args.min_area,
    )

    run_pipeline(
        data_dir=args.data_dir,
        calibration_yml=args.calib,
        output_dir=args.output,
        params=params,
    )
