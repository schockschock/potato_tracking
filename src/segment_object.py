"""
Segmentation d'un objet sur toutes les images de deux dossiers (camera
gauche et camera droite).

Pour chaque cote, la PREMIERE image du dossier (ordre alphabetique) est
utilisee automatiquement comme image de fond (scene VIDE, sans objet).
Elle est quand meme traitee comme les autres (elle sera juste ignoree
au moment de l'extraction, puisque sa difference avec elle-meme est nulle).

Deux methodes disponibles :
  1) SOUSTRACTION DE FOND (recommandee si possible) : necessite une image
     de la scene VIDE (sans objet), prise avec le meme cadrage/eclairage.
     Tres robuste : peu importe la couleur/luminosite de l'objet, seul ce
     qui differe du fond est detecte.
  2) GRABCUT (repli si pas d'image de fond disponible) : segmentation par
     rectangle initial + affinement automatique des couleurs/textures.
     Necessite d'ajuster RECT_INITIAL a la zone approximative de l'objet.

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

A ADAPTER : la section PARAMETRES ci-dessous.
"""

import cv2
import numpy as np
import glob
import os

# ==========================================================================
# PARAMETRES A ADAPTER
# ==========================================================================

DOSSIER_ENTREE_GAUCHE = "CamG"   # dossier images camera gauche
DOSSIER_ENTREE_DROITE = "CamD"   # dossier images camera droite
EXTENSION = "*.tif"                               # extension des images

DOSSIER_SORTIE = "segmentation"    # dossier de sortie commun (gauche + droite)

METHODE = "difference"             # "difference" ou "grabcut"

# --- Pour la methode "difference" ---
# L'image de fond n'est plus specifiee a la main : c'est automatiquement
# la premiere image (triee) de chaque dossier d'entree.
SEUIL_DIFFERENCE = 20                 # si trop de faux positifs (bruit du capteur,
                                    # variations d'eclairage), a diminuer si
                                    # l'objet n'est pas assez detecte

# --- Pour la methode "grabcut" ---
# Rectangle initial (x, y, largeur, hauteur) EN PROPORTION de l'image
# (0.0 a 1.0), englobant largement la zone ou se trouve l'objet.
# A ajuster en regardant une image : plus c'est precis, meilleur le resultat.
RECT_INITIAL_PROPORTION = (0.05, 0.40, 0.90, 0.55)  # (x, y, largeur, hauteur)
GRABCUT_ITERATIONS = 5

# --- Commun aux deux methodes ---
TAILLE_NOYAU_MORPHO = 7            # nettoyage morphologique du masque
AIRE_MINIMALE = 500                # aire minimale (px) pour garder un contour


# ==========================================================================
# METHODE 1 : SOUSTRACTION DE FOND
# ==========================================================================
def segmenter_par_difference(img, img_fond):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris_fond = cv2.cvtColor(img_fond, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gris, gris_fond)
    _, masque = cv2.threshold(diff, SEUIL_DIFFERENCE, 255, cv2.THRESH_BINARY)

    return masque


# ==========================================================================
# METHODE 2 : GRABCUT
# ==========================================================================
def segmenter_par_grabcut(img):
    h, w = img.shape[:2]
    x, y, rw, rh = RECT_INITIAL_PROPORTION
    rect = (int(x*w), int(y*h), int(rw*w), int(rh*h))

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT)

    return np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')


# ==========================================================================
# SEGMENTATION D'UNE IMAGE (commun)
# ==========================================================================
def segmenter_image(path, img_fond=None):
    img = cv2.imread(path)
    if img is None:
        print("Erreur lecture :", path)
        return None

    if METHODE == "difference":
        masque = segmenter_par_difference(img, img_fond)
    else:
        masque = segmenter_par_grabcut(img)

    noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TAILLE_NOYAU_MORPHO, TAILLE_NOYAU_MORPHO))
    masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, noyau)
    masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, noyau)

    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Aucun objet trouve :", path)
        return None

    contour_principal = max(contours, key=cv2.contourArea)
    aire = cv2.contourArea(contour_principal)

    if aire < AIRE_MINIMALE:
        print(f"Objet trop petit ({aire:.0f} px), ignore :", path)
        return None

    masque_final = np.zeros_like(masque)
    cv2.drawContours(masque_final, [contour_principal], -1, 255, thickness=cv2.FILLED)

    return img, masque_final, contour_principal


# ==========================================================================
# TRAITEMENT D'UN DOSSIER (une camera / un cote)
# ==========================================================================
def traiter_dossier(dossier_entree, nom_sortie):
    """
    Traite toutes les images de `dossier_entree` et ecrit les resultats
    dans DOSSIER_SORTIE/nom_sortie/. La premiere image (triee) du dossier
    sert automatiquement d'image de fond pour la methode "difference".
    """
    dossier_sortie_cote = os.path.join(DOSSIER_SORTIE, nom_sortie)
    os.makedirs(dossier_sortie_cote, exist_ok=True)
    os.makedirs(os.path.join(dossier_sortie_cote, "masques"), exist_ok=True)
    os.makedirs(os.path.join(dossier_sortie_cote, "objets_extraits"), exist_ok=True)
    os.makedirs(os.path.join(dossier_sortie_cote, "apercu"), exist_ok=True)

    images = sorted(glob.glob(os.path.join(dossier_entree, EXTENSION)))
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
        # La premiere image du dossier sert de reference (fond vide)
        image_fond_path = images[0]
        img_fond = cv2.imread(image_fond_path)
        if img_fond is None:
            raise RuntimeError(f"Impossible de lire l'image de fond : {image_fond_path}")
        print(f"Image de fond (auto) : {os.path.basename(image_fond_path)}")

    resultats = []

    for path in images:
        # On ne traite pas l'image de fond elle-meme comme un objet a extraire
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

        cv2.imwrite(os.path.join(dossier_sortie_cote, "masques", f"{nom}_masque.png"), masque)

        objet_seul = cv2.bitwise_and(img, img, mask=masque)
        objet_recadre = objet_seul[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(dossier_sortie_cote, "objets_extraits", f"{nom}_objet.png"), objet_recadre)

        apercu = img.copy()
        cv2.drawContours(apercu, [contour], -1, (0, 255, 0), 3)
        cv2.rectangle(apercu, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(dossier_sortie_cote, "apercu", f"{nom}_apercu.png"), apercu)

        resultats.append({"nom": nom, "aire_px": aire, "bbox": (x, y, w, h)})
        print(f"  -> objet trouve : aire={aire:.0f}px, bbox=({x},{y},{w},{h})")

    nb_traitees = len(images) - (1 if METHODE == "difference" else 0)
    print(f"\nSegmentation terminee ({nom_sortie}) : {len(resultats)}/{nb_traitees} images traitees avec succes")
    print(f"Resultats dans : {dossier_sortie_cote}/")

    with open(os.path.join(dossier_sortie_cote, "resume.txt"), "w", encoding="utf-8") as f:
        if image_fond_path is not None:
            f.write(f"Image de fond : {os.path.basename(image_fond_path)}\n\n")
        f.write(f"{'nom':<30} {'aire_px':>10} {'bbox (x,y,w,h)'}\n")
        for r in resultats:
            f.write(f"{r['nom']:<30} {r['aire_px']:>10.0f} {r['bbox']}\n")
    print("Recapitulatif sauvegarde dans resume.txt")


# ==========================================================================
# PROGRAMME PRINCIPAL
# ==========================================================================
os.makedirs(DOSSIER_SORTIE, exist_ok=True)

traiter_dossier(DOSSIER_ENTREE_GAUCHE, "segmentation_gauche")
traiter_dossier(DOSSIER_ENTREE_DROITE, "segmentation_droite")

print("\n====================")
print("Segmentation terminee pour les deux cotes")
print(f"Resultats dans : {DOSSIER_SORTIE}/segmentation_gauche/ et {DOSSIER_SORTIE}/segmentation_droite/")
print("====================")