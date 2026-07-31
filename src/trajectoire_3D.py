"""
Trajectoire 3D de la pomme de terre, obtenue par triangulation stereo a
partir des deux trajectoires 2D (gauche et droite) deja calculees par
trajectoire_2D.py.

Principe :
  - Pour chaque frame, on dispose d'un point 2D corrige dans l'image gauche
    et d'un point 2D corrige dans l'image droite (colonnes cx_corrige/cy_corrige
    des CSV generes par trajectoire_2D.py).
  - On construit les matrices de projection :
        P_G = K_G . [I | 0]                (camera gauche = reference)
        P_D = K_D . [R | T]                (camera droite, relative a gauche)
  - cv2.triangulatePoints(P_G, P_D, pts_G, pts_D) donne le point 3D
    (en coordonnees homogenes) dans le repere de la camera gauche.

Appariement des frames entre gauche et droite :
  - Par defaut, appariement par nom de frame identique (colonne "frame").
  - Si aucun nom ne correspond, appariement par position (index) avec avertissement.

Sorties (dans DOSSIER_SORTIE) :
  trajectoire_3d.csv         : frame, X, Y, Z
  trajectoire_3d.png         : trajectoire 3D + projections XY/XZ/YZ
  trajectoire_3d.html        : trajectoire 3D INTERACTIVE (plotly)

Utilisation :
    python -m src.trajectoire_3D

A ADAPTER : la section PARAMETRES ci-dessous.
"""

import os

import cv2
import numpy as np

from .camera import load_stereo_calibration
from .io import read_trajectory_2d_csv, write_trajectory_3d_csv
from .visualize import plot_3d_interactive, plot_3d_trajectory

# ==========================================================================
# PARAMETRES A ADAPTER
# ==========================================================================

CALIBRATION_YML = "CalibrationStereo.yml"

CSV_GAUCHE = os.path.join("trajectoire", "trajectoire_gauche", "trajectoire_2d.csv")
CSV_DROITE = os.path.join("trajectoire", "trajectoire_droite", "trajectoire_2d.csv")

DOSSIER_SORTIE = "trajectoire_3d"

INVERSER_RT = False


# ==========================================================================
# APPARIEMENT DES FRAMES
# ==========================================================================
def apparier_frames(traj_g, traj_d):
    dict_g = {t["frame"]: t for t in traj_g}
    dict_d = {t["frame"]: t for t in traj_d}
    noms_communs = [f for f in dict_g if f in dict_d]

    paires = []
    if noms_communs:
        print(
            f"Appariement par nom de frame : "
            f"{len(noms_communs)} correspondances trouvees."
        )
        for nom in sorted(noms_communs):
            g, d = dict_g[nom], dict_d[nom]
            paires.append((nom, (g["cx"], g["cy"]), (d["cx"], d["cy"])))
    else:
        n = min(len(traj_g), len(traj_d))
        print(
            "AVERTISSEMENT : aucun nom de frame commun entre gauche et droite. "
            f"Appariement par position (index) sur {n} frames. "
            "Verifie que les deux dossiers d'images sont bien synchronises."
        )
        for i in range(n):
            g, d = traj_g[i], traj_d[i]
            paires.append((g["frame"], (g["cx"], g["cy"]), (d["cx"], d["cy"])))

    return paires


# ==========================================================================
# TRIANGULATION
# ==========================================================================
def construire_projections(K_G, K_D, R, T, inverser=False):
    if inverser:
        R = R.T
        T = -R @ T
    P_G = K_G @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P_D = K_D @ np.hstack([R, T.reshape(3, 1)])
    return P_G, P_D


def trianguler(paires, P_G, P_D):
    pts_G = np.array(
        [[p[1][0], p[1][1]] for p in paires], dtype=np.float64,
    ).T
    pts_D = np.array(
        [[p[2][0], p[2][1]] for p in paires], dtype=np.float64,
    ).T
    points_4d = cv2.triangulatePoints(P_G, P_D, pts_G, pts_D)
    points_3d = (points_4d[:3] / points_4d[3]).T
    return points_3d


# ==========================================================================
# PROGRAMME PRINCIPAL
# ==========================================================================
if __name__ == "__main__":
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    stereo = load_stereo_calibration(CALIBRATION_YML)

    traj_g = read_trajectory_2d_csv(CSV_GAUCHE)
    traj_d = read_trajectory_2d_csv(CSV_DROITE)
    print(f"Trajectoire gauche : {len(traj_g)} frames")
    print(f"Trajectoire droite : {len(traj_d)} frames")

    paires = apparier_frames(traj_g, traj_d)
    if not paires:
        raise RuntimeError("Aucune paire de points gauche/droite a trianguler.")

    P_G, P_D = construire_projections(
        stereo.left.K, stereo.right.K, stereo.R, stereo.T,
        inverser=INVERSER_RT,
    )
    points_3d = trianguler(paires, P_G, P_D)
    frame_names = [p[0] for p in paires]

    csv_path = os.path.join(DOSSIER_SORTIE, "trajectoire_3d.csv")
    write_trajectory_3d_csv(csv_path, points_3d, frame_names)
    print("Trajectoire 3D sauvegardee dans", csv_path)

    png_path = os.path.join(DOSSIER_SORTIE, "trajectoire_3d.png")
    plot_3d_trajectory(points_3d, png_path)
    print("Graphique sauvegarde dans", png_path)

    html_path = os.path.join(DOSSIER_SORTIE, "trajectoire_3d.html")
    plot_3d_interactive(points_3d, frame_names, html_path)
    print("Graphique interactif sauvegarde dans", html_path)

    n = len(points_3d)
    print(f"\nTriangulation terminee : {n} points 3D")
    print(f"Resultats dans : {DOSSIER_SORTIE}/")
    print("====================")

    import matplotlib.pyplot as plt
    plt.show()
