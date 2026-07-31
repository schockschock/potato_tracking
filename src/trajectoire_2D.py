"""
Trajectoire 2D (par camera) de la pomme de terre - gauche ET droite.

Ce script combine la segmentation (methode "difference") et le calibrage
camera (K, dist issus de CalibrationStereo.yml).

Pour chaque cote (gauche / droite), et pour chaque image du dossier
correspondant (traitee comme une frame temporelle, dans l'ordre alphabetique
des noms de fichiers), on :
  1) segmente la pomme de terre (masque + contour principal)
  2) calcule son centroide (moments du contour)
  3) corrige la distorsion optique du centroide

La premiere image (triee) de chaque dossier sert automatiquement d'image de
fond pour ce cote.

Utilisation :
    python -m src.trajectoire_2D

Sorties, regroupees dans un seul dossier DOSSIER_SORTIE :
  trajectoire/
    trajectoire_gauche/
      trajectoire_2d.csv
      trajectoire_2d.png
      trajectoire_2d_overlay.png
    trajectoire_droite/
      trajectoire_2d.csv
      trajectoire_2d.png
      trajectoire_2d_overlay.png

A ADAPTER : la section PARAMETRES ci-dessous.
"""

import os

import cv2

from .camera import load_camera_intrinsics, undistort_point
from .io import read_images, write_trajectory_2d_csv
from .segmentation import (
    apply_morphology,
    compute_centroid,
    find_largest_contour,
    segment_by_difference,
)
from .visualize import plot_2d_trajectory, plot_trajectory_overlay

# ==========================================================================
# PARAMETRES A ADAPTER
# ==========================================================================

DOSSIER_ENTREE_GAUCHE = "CamG"
DOSSIER_ENTREE_DROITE = "CamD"
EXTENSION = "*.tif"

SEUIL_DIFFERENCE = 20
TAILLE_NOYAU_MORPHO = 7
AIRE_MINIMALE = 500

CALIBRATION_YML = "CalibrationStereo.yml"

DOSSIER_SORTIE = "trajectoire"


# ==========================================================================
# SEGMENTATION + CENTROIDE
# ==========================================================================
def segmenter_et_centrer(path, img_fond):
    img = cv2.imread(path)
    if img is None:
        print("Erreur lecture :", path)
        return None

    masque = segment_by_difference(img, img_fond, threshold=SEUIL_DIFFERENCE)
    masque = apply_morphology(masque, kernel_size=TAILLE_NOYAU_MORPHO)

    contour = find_largest_contour(masque, min_area=AIRE_MINIMALE)
    if contour is None:
        print("Aucun objet trouve (ou trop petit) :", path)
        return None

    centroide = compute_centroid(contour)
    if centroide is None:
        return None

    cx, cy = centroide
    aire = cv2.contourArea(contour)
    return cx, cy, aire


# ==========================================================================
# TRAITEMENT D'UN COTE (une camera : gauche ou droite)
# ==========================================================================
def traiter_camera(dossier_entree, camera, nom_sortie):
    dossier_sortie_cote = os.path.join(DOSSIER_SORTIE, nom_sortie)
    os.makedirs(dossier_sortie_cote, exist_ok=True)

    print(f"\n=== {nom_sortie} (camera {camera}) ===")
    print(f"Dossier : {dossier_entree}")

    images = read_images(dossier_entree, EXTENSION)
    if not images:
        print(f"Aucune image trouvee dans {dossier_entree}, cote ignore.")
        return

    img_fond = cv2.imread(images[0])
    if img_fond is None:
        raise RuntimeError(f"Impossible de lire l'image de fond : {images[0]}")
    print("Image de fond utilisee :", os.path.basename(images[0]))
    images = images[1:]

    intrinsics = load_camera_intrinsics(CALIBRATION_YML, camera)
    print(f"Images trouvees : {len(images)}")

    trajectoire = []
    points_brut = []
    points_corrige = []

    for path in images:
        nom = os.path.splitext(os.path.basename(path))[0]
        res = segmenter_et_centrer(path, img_fond)
        if res is None:
            print("  -> ignoree :", nom)
            continue

        cx, cy, aire = res
        cx_c, cy_c = undistort_point(
            cx, cy, intrinsics.K, intrinsics.dist,
        )

        trajectoire.append({
            "frame": nom,
            "cx_brut": cx, "cy_brut": cy,
            "cx_corrige": cx_c, "cy_corrige": cy_c,
            "aire_px": aire,
        })
        points_brut.append((cx, cy))
        points_corrige.append((cx_c, cy_c))
        print(
            f"{nom}: brut=({cx:.1f},{cy:.1f})  corrige=({cx_c:.1f},{cy_c:.1f})"
        )

    print(f"\n{len(trajectoire)}/{len(images)} frames valides ({nom_sortie})")

    if trajectoire:
        csv_path = os.path.join(dossier_sortie_cote, "trajectoire_2d.csv")
        write_trajectoire_2d_csv(csv_path, trajectoire)
        print("Trajectoire sauvegardee dans", csv_path)

        plot_path = os.path.join(dossier_sortie_cote, "trajectoire_2d.png")
        plot_2d_trajectory(points_brut, points_corrige, plot_path, title=nom_sortie)
        print("Graphique sauvegarde dans", plot_path)

        overlay_path = os.path.join(
            dossier_sortie_cote, "trajectoire_2d_overlay.png",
        )
        plot_trajectory_overlay(points_corrige, img_fond, overlay_path)
        print("Overlay sauvegarde dans", overlay_path)
    else:
        print(f"Aucune trajectoire a tracer ({nom_sortie}).")


# ==========================================================================
# PROGRAMME PRINCIPAL
# ==========================================================================
if __name__ == "__main__":
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    traiter_camera(DOSSIER_ENTREE_GAUCHE, "G", "trajectoire_gauche")
    traiter_camera(DOSSIER_ENTREE_DROITE, "D", "trajectoire_droite")
    print("\n====================")
    print("Trajectoires 2D terminees pour les deux cotes")
    print(
        f"Resultats dans : {DOSSIER_SORTIE}/trajectoire_gauche/ "
        f"et {DOSSIER_SORTIE}/trajectoire_droite/"
    )
    print("====================")
