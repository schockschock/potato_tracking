"""
Segmentation d'un objet sur toutes les images de deux dossiers (camera
gauche et camera droite).

Pour chaque cote, la PREMIERE image du dossier (ordre alphabetique) est
utilisee automatiquement comme image de fond (scene VIDE, sans objet).

Deux methodes disponibles :
  1) SOUSTRACTION DE FOND (recommandee si possible)
  2) GRABCUT (repli si pas d'image de fond disponible)

Tous les resultats (gauche + droite) sont enregistres dans le meme
dossier de sortie DOSSIER_SORTIE, avec un sous-dossier par cote :

  segmentation/
    segmentation_gauche/
      masques/
      objets_extraits/
      apercu/
      resume.txt
    segmentation_droite/
      masques/
      objets_extraits/
      apercu/
      resume.txt

Utilisation :
    python -m src.segment_object

A ADAPTER : la section PARAMETRES ci-dessous.
"""

import os

import cv2

from .io import read_images, write_segmentation_summary
from .segmentation import (
    apply_morphology,
    find_largest_contour,
    make_filled_mask,
    segment_by_difference,
    segment_by_grabcut,
)

# ==========================================================================
# PARAMETRES A ADAPTER
# ==========================================================================

DOSSIER_ENTREE_GAUCHE = "CamG"
DOSSIER_ENTREE_DROITE = "CamD"
EXTENSION = "*.tif"

DOSSIER_SORTIE = "segmentation"

METHODE = "difference"

SEUIL_DIFFERENCE = 20

RECT_INITIAL_PROPORTION = (0.05, 0.40, 0.90, 0.55)
GRABCUT_ITERATIONS = 5

TAILLE_NOYAU_MORPHO = 7
AIRE_MINIMALE = 500


# ==========================================================================
# SEGMENTATION D'UNE IMAGE
# ==========================================================================
def segmenter_image(path, img_fond=None):
    img = cv2.imread(path)
    if img is None:
        print("Erreur lecture :", path)
        return None

    if METHODE == "difference":
        masque = segment_by_difference(
            img, img_fond, threshold=SEUIL_DIFFERENCE,
        )
    else:
        masque = segment_by_grabcut(
            img,
            rect_proportion=RECT_INITIAL_PROPORTION,
            iterations=GRABCUT_ITERATIONS,
        )

    masque = apply_morphology(masque, kernel_size=TAILLE_NOYAU_MORPHO)

    contour = find_largest_contour(masque, min_area=AIRE_MINIMALE)
    if contour is None:
        print("Aucun objet trouve :", path)
        return None

    masque_final = make_filled_mask(contour, masque)
    return img, masque_final, contour


# ==========================================================================
# TRAITEMENT D'UN DOSSIER (une camera / un cote)
# ==========================================================================
def traiter_dossier(dossier_entree, nom_sortie):
    dossier_sortie_cote = os.path.join(DOSSIER_SORTIE, nom_sortie)
    os.makedirs(dossier_sortie_cote, exist_ok=True)
    os.makedirs(os.path.join(dossier_sortie_cote, "masques"), exist_ok=True)
    os.makedirs(
        os.path.join(dossier_sortie_cote, "objets_extraits"), exist_ok=True,
    )
    os.makedirs(os.path.join(dossier_sortie_cote, "apercu"), exist_ok=True)

    images = read_images(dossier_entree, EXTENSION)
    print(f"\n=== {nom_sortie} ===")
    print(f"Dossier : {dossier_entree}")
    print(f"Methode : {METHODE}")
    print(f"Images trouvees : {len(images)}")

    if not images:
        print("Aucune image trouvee, cote ignore.")
        return

    img_fond = None
    image_fond_path = None
    if METHODE == "difference":
        image_fond_path = images[0]
        img_fond = cv2.imread(image_fond_path)
        if img_fond is None:
            raise RuntimeError(
                f"Impossible de lire l'image de fond : {image_fond_path}"
            )
        print(f"Image de fond (auto) : {os.path.basename(image_fond_path)}")

    resultats = []

    for path in images:
        if METHODE == "difference" and path == image_fond_path:
            continue

        nom = os.path.splitext(os.path.basename(path))[0]
        print("Traitement :", os.path.basename(path))

        res = segmenter_image(path, img_fond)
        if res is None:
            print("  -> ignoree")
            continue

        img, masque, contour = res
        x, y, w, h = cv2.boundingRect(contour)
        aire = cv2.contourArea(contour)

        cv2.imwrite(
            os.path.join(dossier_sortie_cote, "masques", f"{nom}_masque.png"),
            masque,
        )

        objet_seul = cv2.bitwise_and(img, img, mask=masque)
        objet_recadre = objet_seul[y:y + h, x:x + w]
        cv2.imwrite(
            os.path.join(
                dossier_sortie_cote, "objets_extraits", f"{nom}_objet.png",
            ),
            objet_recadre,
        )

        apercu = img.copy()
        cv2.drawContours(apercu, [contour], -1, (0, 255, 0), 3)
        cv2.rectangle(apercu, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(
            os.path.join(dossier_sortie_cote, "apercu", f"{nom}_apercu.png"),
            apercu,
        )

        resultats.append({
            "nom": nom, "aire_px": aire, "bbox": (x, y, w, h),
        })
        print(
            f"  -> objet trouve : aire={aire:.0f}px, "
            f"bbox=({x},{y},{w},{h})"
        )

    nb_traitees = len(images) - (1 if METHODE == "difference" else 0)
    print(
        f"\nSegmentation terminee ({nom_sortie}) : "
        f"{len(resultats)}/{nb_traitees} images traitees avec succes"
    )
    print(f"Resultats dans : {dossier_sortie_cote}/")

    write_segmentation_summary(
        os.path.join(dossier_sortie_cote, "resume.txt"),
        resultats,
        background_path=image_fond_path,
    )
    print("Recapitulatif sauvegarde dans resume.txt")


# ==========================================================================
# PROGRAMME PRINCIPAL
# ==========================================================================
if __name__ == "__main__":
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    traiter_dossier(DOSSIER_ENTREE_GAUCHE, "segmentation_gauche")
    traiter_dossier(DOSSIER_ENTREE_DROITE, "segmentation_droite")
    print("\n====================")
    print("Segmentation terminee pour les deux cotes")
    print(
        f"Resultats dans : {DOSSIER_SORTIE}/segmentation_gauche/ "
        f"et {DOSSIER_SORTIE}/segmentation_droite/"
    )
    print("====================")
