"""
L’objectif de ce code est de quantifier la trajectoire de la pomme de terre à partir de la trajectoire 3D,
des données de calibration stéréo contenues dans le fichier CalibrationStereo.yml, ainsi que des données relatives
au plan de la surface du banc d’essai enregistrées dans le fichier calibration_plan.txt.

Le code permet d’obtenir, pour chaque image, les coordonnées (X), (Y) et (Z) de la pomme de terre dans le repère 3D,
ainsi que de déterminer sa vitesse et son accélération au cours de la chute.

Deux méthodes sont utilisées pour calculer la vitesse et l’accélération :
Méthode brut : le calcul est effectué directement entre deux images successives. La vitesse est déterminée à partir
de la variation de position entre deux images divisée par l’intervalle de temps (dt). Cette méthode ne repose sur aucune
hypothèse physique et permet de conserver au maximum les variations présentes dans les données expérimentales.

Méthode fit : une courbe, correspondant au modèle d’une chute libre, est ajustée sur plusieurs points de la trajectoire
afin de réduire l’influence du bruit de mesure. Cette méthode n’est appliquée que lorsqu’un nombre suffisant de points
est disponible pour obtenir un ajustement pertinent.

En sortie, le code génère un tableur regroupant l’ensemble de ces données 

A ADAPTER : la section PARAMETRES ci-dessous.
"""

import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (necessaire pour projection="3d")

try:
    from scipy.signal import savgol_filter
    SCIPY_DISPONIBLE = True
except ImportError:
    SCIPY_DISPONIBLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================================================
# PARAMETRES A ADAPTER
# ==========================================================================

CSV_TRAJECTOIRE_3D = os.path.join(BASE_DIR, "trajectoire_3d", "trajectoire_3d.csv")
FICHIER_CALIBRATION_PLAN = os.path.join(BASE_DIR, "calibration_plan.txt")

FPS = 10
DT = 1.0 / FPS

FACTEUR_MM_VERS_M = 0.001
UNITE_DISTANCE = "m"

DISTANCE_POINT_IMPACT_M = 0.32

# Lissage (utilise seulement pour les colonnes *_brut, methode differences
# finies -- garde a titre de comparaison avec l'ancienne methode).
LISSAGE_ACTIF = True
LISSAGE_FENETRE = 7
LISSAGE_ORDRE = 3

# Degre du polynome ajuste sur chaque segment (2 = acceleration constante,
# modele physique standard pour une chute/vol libre). Ne pas depasser 2 sauf
# si le segment a beaucoup de points et une justification physique claire.
DEGRE_POLYNOME = 2

# MODELE PHYSIQUE CONTRAINT (recommande quand il y a peu de points, ex.
# seulement 2-3 frames avant l'impact : un fit libre de degre 2 sur si peu
# de points n'a AUCUN pouvoir de lissage, il repasse exactement par les
# points bruites -> acceleration absurde. Ici on impose l'acceleration
# verticale = -g (chute libre / vol libre, aucune force sauf gravite une
# fois l'objet en l'air) et on ne cherche plus que la position et la
# vitesse initiales (2 inconnues au lieu de 3 par axe) : beaucoup plus
# robuste avec peu de points.
# Necessite que l'axe Y soit bien aligne avec la verticale reelle (verifier
# apres la correction d'inclinaison faite dans Trajectoire_3D.py) et que Y
# augmente vers le haut (ce qui est la convention actuelle du pipeline).
MODELE_PHYSIQUE_ACTIF = True
G = 9.81  # m/s^2
# Direction de la gravite : ESTIMEE AUTOMATIQUEMENT depuis les donnees (voir
# estimer_vecteur_gravite ci-dessous) -- plus besoin de supposer un axe fixe
# (Y, etc.), ce qui ne serait valable que si Trajectoire_3D.py avait fait
# une rotation d'alignement, ce que la version actuelle du pipeline ne fait
# plus.

# DETECTION DE LA ZONE D'IMPACT : basee sur le changement de DIRECTION du
# deplacement 3D (voir detecter_zone_impact ci-dessous), PAS sur le signe
# ou le minimum d'un axe particulier -- ca marche donc meme si le plan de
# reference (inclinaison) est mal calibre ou si l'acquisition est differente
# d'une fois sur l'autre.
#   SEUIL_COS_IMPACT : en dessous de ce cosinus entre deux deplacements
#       consecutifs, on considere que la direction est instable (= zone de
#       contact). 0.3 = ~72 degres d'ecart tolere avant de considerer que
#       c'est un vrai changement de direction. Baisser si des frames de vol
#       libre normales sont exclues a tort ; monter si des frames de contact
#       douteuses passent encore le filtre.
#   LONGUEUR_MIN_STABLE : nombre de transitions consecutives redevenues
#       stables necessaires pour considerer que la zone de contact est
#       terminee (evite de couper la zone trop tot sur un faux stable
#       isole).
SEUIL_COS_IMPACT = 0.3
LONGUEUR_MIN_STABLE = 3

DOSSIER_SORTIE = os.path.join(BASE_DIR, "cinematique_3d")


# ==========================================================================
# LECTURE DE LA TRAJECTOIRE 3D
# ==========================================================================
def lire_trajectoire_3d(path_csv):
    if not os.path.isfile(path_csv):
        raise FileNotFoundError(
            f"Fichier introuvable : {path_csv}\n"
            "Verifie que Trajectoire_3D.py a bien ete execute et a genere ce CSV."
        )
    frames, X, Y, Z = [], [], [], []
    with open(path_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(row["frame"])
            X.append(float(row["X"]))
            Y.append(float(row["Y"]))
            Z.append(float(row["Z"]))
    return frames, np.array(X), np.array(Y), np.array(Z)


def extraire_numero_frame(nom):
    m = re.search(r"\d+", nom)
    return int(m.group()) if m else None


def calculer_temps(frames):
    numeros = [extraire_numero_frame(f) for f in frames]
    if all(n is not None for n in numeros):
        numeros = np.array(numeros, dtype=np.float64)
        t = (numeros - numeros[0]) / FPS
        if np.any(np.diff(numeros) != 1):
            print("Des frames manquantes ont ete detectees : le temps reel "
                  "(base sur le numero de frame) est utilise.")
        return t
    print("AVERTISSEMENT : numero de frame non extractible, on suppose un "
          "espacement regulier de 1 frame entre chaque ligne du CSV.")
    return np.arange(len(frames)) * DT


def lisser_positions(X, Y, Z, fenetre, ordre):
    n = len(X)
    fenetre_eff = min(fenetre, n if n % 2 == 1 else n - 1)
    if fenetre_eff <= ordre:
        print(f"AVERTISSEMENT : pas assez de points ({n}) pour lisser. Lissage ignore.")
        return X, Y, Z
    if fenetre_eff != fenetre:
        print(f"Fenetre de lissage reduite a {fenetre_eff} points.")
    return (savgol_filter(X, fenetre_eff, ordre),
            savgol_filter(Y, fenetre_eff, ordre),
            savgol_filter(Z, fenetre_eff, ordre))


def derivee(valeurs, t):
    n = len(t)
    d = np.zeros_like(valeurs, dtype=np.float64)
    if n < 2:
        return d
    d[0] = (valeurs[1] - valeurs[0]) / (t[1] - t[0])
    for i in range(1, n - 1):
        d[i] = (valeurs[i + 1] - valeurs[i - 1]) / (t[i + 1] - t[i - 1])
    d[-1] = (valeurs[-1] - valeurs[-2]) / (t[-1] - t[-2])
    return d


# ==========================================================================
# NOUVEAU : detection de l'impact + ajustement polynomial par segment
# ==========================================================================
def detecter_zone_impact(X, Y, Z, seuil_cos=0.3, longueur_min_stable=3):
    """
    Detecte la ZONE de contact/impact a partir du changement de DIRECTION du
    deplacement en 3D -- independamment de X/Y/Z pris isolement, donc
    independant d'une eventuelle erreur de calibrage du plan de reference
    (inclinaison mal corrigee, profondeur qui "fuit" dans un autre axe,
    etc.). C'est generique : fonctionne pareil quelle que soit l'acquisition
    et l'orientation du repere.

    Principe : on calcule les vecteurs deplacement 3D consecutifs
    d[i] = P[i+1] - P[i], puis le cosinus de l'angle entre deplacements
    consecutifs d[i] et d[i+1]. Tant que la pomme de terre vole dans une
    direction stable (avant OU apres l'impact), ce cosinus est proche de 1
    (meme direction). Au moment du contact, le masque de segmentation se
    deforme et/ou la trajectoire change reellement de sens -> le cosinus
    chute (proche de 0 ou negatif) sur une ou plusieurs frames consecutives.

    On identifie la PREMIERE zone contigue ou ce cosinus tombe sous
    `seuil_cos`, suivie d'au moins `longueur_min_stable` transitions a
    nouveau stables (cosinus >= seuil_cos) : c'est la zone d'impact/contact,
    a exclure des deux segments (avant/apres) car les positions y sont peu
    fiables.

    Renvoie (idx_avant_fin, idx_apres_debut) : indices (inclus) delimitant
    respectivement la fin du segment "avant" et le debut du segment
    "apres". Si aucune instabilite n'est detectee (trajectoire jugee
    continue, ex. simple chute libre filmee sans le rebond), renvoie
    (n-1, n) : tout est considere comme "avant", pas de segment "apres".
    """
    P = np.stack([X, Y, Z], axis=1)
    d = np.diff(P, axis=0)  # n-1 vecteurs deplacement
    n = len(P)

    normes = np.linalg.norm(d, axis=1)
    cos = np.full(len(d) - 1, np.nan)
    for i in range(len(d) - 1):
        denom = normes[i] * normes[i + 1]
        cos[i] = np.dot(d[i], d[i + 1]) / denom if denom > 1e-12 else 1.0

    # cos[i] concerne la transition "autour" du point d'indice i+1.
    instable = cos < seuil_cos

    if not np.any(instable):
        print("Aucun changement de direction significatif detecte : la "
              "trajectoire est traitee comme un seul segment continu "
              "(pas de zone de contact identifiee).")
        return n - 1, n

    debut_zone = int(np.argmax(instable)) + 1  # +1 : cos[i] -> point i+1

    # etendre la zone tant que c'est instable OU que la stabilite qui suit
    # n'a pas encore dure `longueur_min_stable` transitions d'affilee
    fin_zone = debut_zone
    i = debut_zone - 1  # index dans cos correspondant au point debut_zone
    run_stable = 0
    while i < len(cos):
        if cos[i] < seuil_cos:
            run_stable = 0
            fin_zone = i + 1
        else:
            run_stable += 1
            if run_stable >= longueur_min_stable:
                break
        i += 1

    idx_avant_fin = max(debut_zone - 1, 0)
    idx_apres_debut = min(fin_zone + 1, n)
    return idx_avant_fin, idx_apres_debut


def ajuster_segment(t_seg, pos_seg, degre):
    """
    Ajuste un polynome de degre `degre` a `pos_seg` (array 1D) en fonction
    de `t_seg` par moindres carres. Renvoie les coefficients (numpy.poly1d)
    pour la position, la vitesse (derivee 1) et l'acceleration (derivee 2).
    Renvoie (None, None, None) si pas assez de points pour ce degre.
    """
    if len(t_seg) < degre + 1:
        return None, None, None
    coeffs = np.polyfit(t_seg, pos_seg, degre)
    p = np.poly1d(coeffs)
    v = p.deriv(1)
    a = p.deriv(2)
    return p, v, a


def ajuster_segment_physique(t_seg, pos_seg, a_connue):
    """
    Ajuste position(t) = p0 + v0.t + 0.5.a_connue.t^2 avec a_connue FIXE
    (connue a priori), en cherchant seulement p0 et v0 par moindres carres
    lineaires. Beaucoup plus robuste que polyfit degre 2 quand il y a peu
    de points, car on ne cherche plus que 2 inconnues au lieu de 3.
    Necessite au moins 2 points.
    """
    if len(t_seg) < 2:
        return None, None
    y_ajuste = pos_seg - 0.5 * a_connue * t_seg**2
    A = np.vstack([np.ones_like(t_seg), t_seg]).T
    (p0, v0), *_ = np.linalg.lstsq(A, y_ajuste, rcond=None)
    p = np.poly1d([0.5 * a_connue, v0, p0])
    v = p.deriv(1)
    return p, v


def lire_calibration_plan(path):
    """
    Lit un fichier de calibration produit par calibrer_plan.py (normale du
    plateau mesuree UNE FOIS via des reperes fixes, independamment de
    chaque trajectoire -- bien plus fiable que d'estimer la gravite depuis
    le mouvement de chaque acquisition individuelle, surtout quand l'objet
    roule plutot que de voler librement). Renvoie (normale unitaire,
    centre) ou (None, None) si le fichier n'existe pas.
    """
    if not os.path.isfile(path):
        return None, None
    valeurs = {}
    with open(path, "r", encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne or ligne.startswith("#"):
                continue
            cle, val = ligne.split("=")
            valeurs[cle] = float(val)
    normale = np.array([valeurs["normale_x"], valeurs["normale_y"], valeurs["normale_z"]])
    normale = normale / np.linalg.norm(normale)
    centre = np.array([valeurs["centre_x"], valeurs["centre_y"], valeurs["centre_z"]])
    return normale, centre


def estimer_vecteur_gravite(t, X, Y, Z, idx_avant_fin, idx_apres_debut, g):
    """
    Estime le VECTEUR gravite (magnitude fixee a g, direction determinee a
    partir des donnees) via la NORMALE DU PLAN DE ROULEMENT, pas via la
    courbure du mouvement.

    IMPORTANT : le segment "apres impact" n'est PAS forcement en vol libre
    -- si l'objet roule sur le plateau (frottement/contact continu, pas de
    rebond), son acceleration ne reflete plus la gravite seule, et une
    estimation par courbure (ajustement d'une parabole) donnerait une
    direction fausse. En revanche, un objet qui ROULE garde son centre a
    peu pres dans un plan parallele au plateau -- la NORMALE de ce plan,
    elle, reste une mesure fiable de la verticale reelle, quel que soit le
    regime de mouvement (roulement ou rebond) dans ce segment.

    On ajuste donc un plan (moindres carres, SVD) sur les points du segment
    le plus long, et on prend sa normale comme direction de la gravite.
    """
    n = len(t)
    segments = []
    if idx_avant_fin + 1 >= 3:
        segments.append(("avant", slice(0, idx_avant_fin + 1)))
    if n - idx_apres_debut >= 3:
        segments.append(("apres", slice(idx_apres_debut, n)))

    if not segments:
        print("AVERTISSEMENT : pas assez de points dans un seul segment pour "
              "estimer la direction de la gravite depuis les donnees -> "
              "repli sur (0,-g,0) (axe Y), A VERIFIER visuellement.")
        return np.array([0.0, -g, 0.0])

    nom_segment, sl = max(segments, key=lambda s: s[1].stop - s[1].start)
    points = np.stack([X[sl], Y[sl], Z[sl]], axis=1)

    centre = points.mean(axis=0)
    pts_centres = points - centre
    _, valeurs_singulieres, vt = np.linalg.svd(pts_centres)
    normale = vt[-1]
    normale = normale / np.linalg.norm(normale)

    # Residu du plan (a quel point les points sont-ils vraiment coplanaires ?)
    # -- utile pour juger si ce segment est bien en roulement/plan stable.
    residu_plan = valeurs_singulieres[-1] / np.sqrt(len(points))

    # Le signe de la normale est arbitraire (SVD) -- on le fixe pour qu'elle
    # pointe vers le "haut" reel, en utilisant le fait que les points d'AVANT
    # l'impact (donc plus haut, encore en l'air) doivent etre du cote positif.
    if idx_avant_fin + 1 >= 1:
        point_avant = np.array([X[0], Y[0], Z[0]])
        if np.dot(point_avant - centre, normale) < 0:
            normale = -normale

    vecteur_gravite = -normale * g  # la gravite pointe vers le BAS, oppose au "haut"

    print(f"Direction de la gravite estimee via la normale du plan ajuste sur "
          f"le segment '{nom_segment}' ({sl.stop - sl.start} points, residu au "
          f"plan = {residu_plan:.4f} m) : {-normale} (magnitude fixee a g={g}) -> "
          f"vecteur utilise : {vecteur_gravite}.")
    return vecteur_gravite


def analyser_segment_physique(nom_segment, t_seg, X_seg, Y_seg, Z_seg, vecteur_gravite):
    """
    Comme analyser_segment, mais impose l'acceleration = vecteur_gravite
    (fixe, meme direction et magnitude sur tout le segment). Beaucoup plus
    robuste avec peu de points (2-3) qu'un fit libre de degre 2.
    """
    ax_, ay_, az_ = vecteur_gravite

    px, vx = ajuster_segment_physique(t_seg, X_seg, ax_)
    py, vy = ajuster_segment_physique(t_seg, Y_seg, ay_)
    pz, vz = ajuster_segment_physique(t_seg, Z_seg, az_)

    if px is None:
        print(f"[{nom_segment}] Pas assez de points ({len(t_seg)}) meme pour "
              f"le modele physique contraint (minimum 2). Segment ignore.")
        return None

    t0, t1 = t_seg[0], t_seg[-1]
    v_debut = np.array([vx(t0), vy(t0), vz(t0)])
    v_fin = np.array([vx(t1), vy(t1), vz(t1)])

    print(f"\n--- Segment '{nom_segment}' - MODELE PHYSIQUE "
          f"(acceleration fixee a {vecteur_gravite} m/s2) "
          f"({len(t_seg)} points, t=[{t0:.4f}s ; {t1:.4f}s]) ---")
    print(f"  Vitesse en debut de segment : {v_debut} {UNITE_DISTANCE}/s "
          f"(norme {np.linalg.norm(v_debut):.3f})")
    print(f"  Vitesse en fin de segment   : {v_fin} {UNITE_DISTANCE}/s "
          f"(norme {np.linalg.norm(v_fin):.3f})")

    return {
        "t": t_seg, "X": px(t_seg), "Y": py(t_seg), "Z": pz(t_seg),
        "vx": vx(t_seg), "vy": vy(t_seg), "vz": vz(t_seg),
        "ax": np.full_like(t_seg, ax_),
        "ay": np.full_like(t_seg, ay_),
        "az": np.full_like(t_seg, az_),
        "v_debut": v_debut, "v_fin": v_fin,
        "a_const": np.array(vecteur_gravite),
    }


def ajuster_segment_direction_fixee(t_seg, positions_seg, direction):
    """
    Ajuste position(t) = P0 + V0.t + 0.5*a*direction*t^2 avec `direction`
    FIXEE (vecteur unitaire), mais `a` (la magnitude de l'acceleration)
    LIBRE, mesuree par moindres carres -- PAS supposee egale a g.

    Interet par rapport a ajuster_segment_physique (qui fixe aussi la
    magnitude) : on garde la possibilite de comparer ensuite la magnitude
    mesuree a g=9.81 pour verifier que le calcul est coherent -- fixer la
    magnitude a priori empecherait cette verification, ce qui est tout
    l'interet de comparer a une constante physique connue.

    Inconnues : P0 (3), V0 (3), a (1) = 7 au total, pour 3*N equations
    (N points x 3 coordonnees) -> necessite N >= 3 points. Utiliser
    ajuster_segment_physique (magnitude fixee) en dessous de ce seuil.
    """
    positions_seg = np.asarray(positions_seg)
    N = len(t_seg)
    if N < 3:
        return None

    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)

    # Systeme lineaire empile sur les 3 coordonnees x,y,z pour chaque point :
    # pos_axe(t) = P0_axe + V0_axe*t + 0.5*a*direction_axe*t^2
    lignes_A, valeurs_b = [], []
    for i, t_i in enumerate(t_seg):
        for axe in range(3):
            ligne = np.zeros(7)  # [P0x,P0y,P0z, V0x,V0y,V0z, a]
            ligne[axe] = 1.0
            ligne[3 + axe] = t_i
            ligne[6] = 0.5 * direction[axe] * t_i**2
            lignes_A.append(ligne)
            valeurs_b.append(positions_seg[i, axe])

    A = np.array(lignes_A)
    b = np.array(valeurs_b)
    solution, *_ = np.linalg.lstsq(A, b, rcond=None)
    P0, V0, a_scalaire = solution[:3], solution[3:6], solution[6]

    vecteur_acceleration = a_scalaire * direction

    def position(t):
        t = np.atleast_1d(t)
        return P0[None, :] + V0[None, :] * t[:, None] + 0.5 * vecteur_acceleration[None, :] * (t**2)[:, None]

    def vitesse(t):
        t = np.atleast_1d(t)
        return V0[None, :] + vecteur_acceleration[None, :] * t[:, None]

    return position, vitesse, vecteur_acceleration


def analyser_segment_direction_fixee(nom_segment, t_seg, X_seg, Y_seg, Z_seg, direction):
    positions_seg = np.stack([X_seg, Y_seg, Z_seg], axis=1)
    resultat = ajuster_segment_direction_fixee(t_seg, positions_seg, direction)
    if resultat is None:
        return None
    position, vitesse, vecteur_acceleration = resultat

    t0, t1 = t_seg[0], t_seg[-1]
    v_debut = vitesse(t0)[0]
    v_fin = vitesse(t1)[0]
    pos_seg = position(t_seg)

    print(f"\n--- Segment '{nom_segment}' - DIRECTION FIXEE, magnitude MESUREE "
          f"({len(t_seg)} points, t=[{t0:.4f}s ; {t1:.4f}s]) ---")
    print(f"  Acceleration mesuree : {vecteur_acceleration} m/s2 "
          f"(norme {np.linalg.norm(vecteur_acceleration):.3f}, a comparer a g=9.81)")
    print(f"  Vitesse en debut de segment : {v_debut} {UNITE_DISTANCE}/s "
          f"(norme {np.linalg.norm(v_debut):.3f})")
    print(f"  Vitesse en fin de segment   : {v_fin} {UNITE_DISTANCE}/s "
          f"(norme {np.linalg.norm(v_fin):.3f})")

    return {
        "t": t_seg, "X": pos_seg[:, 0], "Y": pos_seg[:, 1], "Z": pos_seg[:, 2],
        "vx": vitesse(t_seg)[:, 0], "vy": vitesse(t_seg)[:, 1], "vz": vitesse(t_seg)[:, 2],
        "ax": np.full(len(t_seg), vecteur_acceleration[0]),
        "ay": np.full(len(t_seg), vecteur_acceleration[1]),
        "az": np.full(len(t_seg), vecteur_acceleration[2]),
        "v_debut": v_debut, "v_fin": v_fin,
        "a_const": vecteur_acceleration,
    }


def analyser_segment(nom_segment, t_seg, X_seg, Y_seg, Z_seg, degre):

    """
    Ajuste X, Y, Z sur un segment et affiche/renvoie vitesse+acceleration
    a chaque extremite du segment (utile pour "juste avant" / "juste apres"
    l'impact).
    """
    px, vx, ax = ajuster_segment(t_seg, X_seg, degre)
    py, vy, ay = ajuster_segment(t_seg, Y_seg, degre)
    pz, vz, az = ajuster_segment(t_seg, Z_seg, degre)

    if px is None:
        print(f"[{nom_segment}] Pas assez de points ({len(t_seg)}) pour un "
              f"ajustement de degre {degre}. Segment ignore.")
        return None

    t0, t1 = t_seg[0], t_seg[-1]
    v_debut = np.array([vx(t0), vy(t0), vz(t0)])
    v_fin = np.array([vx(t1), vy(t1), vz(t1)])
    a_const = np.array([ax(0) if np.isscalar(ax(0)) else ax(0),
                         ay(0), az(0)])  # constante si degre=2

    print(f"\n--- Segment '{nom_segment}' ({len(t_seg)} points, "
          f"t=[{t0:.4f}s ; {t1:.4f}s]) ---")
    print(f"  Vitesse en debut de segment : {v_debut} {UNITE_DISTANCE}/s "
          f"(norme {np.linalg.norm(v_debut):.3f})")
    print(f"  Vitesse en fin de segment   : {v_fin} {UNITE_DISTANCE}/s "
          f"(norme {np.linalg.norm(v_fin):.3f})")
    if degre == 2:
        print(f"  Acceleration (constante sur le segment) : {a_const} "
              f"{UNITE_DISTANCE}/s2 (norme {np.linalg.norm(a_const):.3f})")

    return {
        "t": t_seg, "X": px(t_seg), "Y": py(t_seg), "Z": pz(t_seg),
        "vx": vx(t_seg), "vy": vy(t_seg), "vz": vz(t_seg),
        "ax": np.full_like(t_seg, ax(0)) if degre == 2 else ax(t_seg),
        "ay": np.full_like(t_seg, ay(0)) if degre == 2 else ay(t_seg),
        "az": np.full_like(t_seg, az(0)) if degre == 2 else az(t_seg),
        "v_debut": v_debut, "v_fin": v_fin, "a_const": a_const,
    }


# ==========================================================================
# PROGRAMME PRINCIPAL
# ==========================================================================
os.makedirs(DOSSIER_SORTIE, exist_ok=True)

frames, X, Y, Z = lire_trajectoire_3d(CSV_TRAJECTOIRE_3D)
n = len(frames)
print(f"Trajectoire 3D chargee : {n} frames")

X, Y, Z = X * FACTEUR_MM_VERS_M, Y * FACTEUR_MM_VERS_M, Z * FACTEUR_MM_VERS_M
X_brut, Y_brut, Z_brut = X.copy(), Y.copy(), Z.copy()

if n < 3:
    raise RuntimeError("Pas assez de points (minimum 3) pour calculer vitesse et acceleration.")

t = calculer_temps(frames)

# --- Methode ORIGINALE : differences finies sur positions lissees ---
if LISSAGE_ACTIF and SCIPY_DISPONIBLE:
    Xs, Ys, Zs = lisser_positions(X, Y, Z, LISSAGE_FENETRE, LISSAGE_ORDRE)
else:
    Xs, Ys, Zs = X, Y, Z

vx = derivee(Xs, t); vy = derivee(Ys, t); vz = derivee(Zs, t)
vitesse = np.sqrt(vx**2 + vy**2 + vz**2)
ax_ = derivee(vx, t); ay_ = derivee(vy, t); az_ = derivee(vz, t)
acceleration = np.sqrt(ax_**2 + ay_**2 + az_**2)

# --- NOUVELLE METHODE : detection de la zone d'impact (direction 3D) + fit avant/apres ---
idx_avant_fin, idx_apres_debut = detecter_zone_impact(
    X, Y, Z, seuil_cos=SEUIL_COS_IMPACT, longueur_min_stable=LONGUEUR_MIN_STABLE)

if idx_apres_debut > idx_avant_fin + 1:
    print(f"\nZone de contact detectee entre les frames "
          f"{frames[idx_avant_fin]} et {frames[idx_apres_debut]} "
          f"(frames {frames[idx_avant_fin + 1] if idx_avant_fin + 1 < idx_apres_debut else '-'} "
          f"a {frames[idx_apres_debut - 1]} exclues du calcul).")
else:
    print("\nAucune zone de contact franche detectee (trajectoire continue).")

idx_avant = slice(0, idx_avant_fin + 1)
idx_apres = slice(idx_apres_debut, n)
# Pour les graphiques : centre approximatif de la zone d'impact, en temps
t_impact_approx = 0.5 * (t[min(idx_avant_fin, n - 1)] + t[min(idx_apres_debut, n - 1)])

SEUIL_POINTS_FIT_LIBRE = 2 * DEGRE_POLYNOME + 2

resultat_avant = None
resultat_apres = None

vecteur_gravite = None
if MODELE_PHYSIQUE_ACTIF:
    normale_calibree, centre_calibre = lire_calibration_plan(FICHIER_CALIBRATION_PLAN)
    if normale_calibree is not None:
        # Signe : la normale doit pointer vers le "haut" reel -- on utilise
        # le premier point (avant impact, donc plus haut, encore en l'air)
        # par rapport au centre du plan calibre pour fixer le signe.
        point_avant = np.array([X[0], Y[0], Z[0]])
        if np.dot(point_avant - centre_calibre, normale_calibree) < 0:
            normale_calibree = -normale_calibree
        vecteur_gravite = -normale_calibree * G
        print(f"Direction de la gravite lue depuis {FICHIER_CALIBRATION_PLAN} "
              f"(calibration dediee, independante de cette trajectoire) : "
              f"{-normale_calibree} -> vecteur utilise : {vecteur_gravite}.")
    else:
        print(f"AVERTISSEMENT : {FICHIER_CALIBRATION_PLAN} introuvable -- "
              f"repli sur une estimation depuis le mouvement de CETTE trajectoire "
              f"(moins fiable si l'objet roule apres l'impact). Lance calibrer_plan.py "
              f"une fois pour ce montage pour eviter cet avertissement.")
        vecteur_gravite = estimer_vecteur_gravite(t, X, Y, Z, idx_avant_fin, idx_apres_debut, G)

n_avant = idx_avant.stop
if n_avant >= SEUIL_POINTS_FIT_LIBRE:
    resultat_avant = analyser_segment(
        "AVANT impact", t[idx_avant], X[idx_avant], Y[idx_avant], Z[idx_avant], DEGRE_POLYNOME)
elif MODELE_PHYSIQUE_ACTIF and n_avant >= 3:
    print(f"\n[AVANT impact] {n_avant} points : pas assez pour un fit libre "
          f"degre {DEGRE_POLYNOME}, mais assez pour mesurer la magnitude de "
          f"l'acceleration (direction fixee).")
    resultat_avant = analyser_segment_direction_fixee(
        "AVANT impact", t[idx_avant], X[idx_avant], Y[idx_avant], Z[idx_avant],
        vecteur_gravite / np.linalg.norm(vecteur_gravite))
elif MODELE_PHYSIQUE_ACTIF and n_avant >= 2:
    print(f"\n[AVANT impact] Seulement {n_avant} points : meme la magnitude de "
          f"l'acceleration n'est pas mesurable (direction fixee + magnitude "
          f"libre = 7 inconnues, minimum 3 points necessaires) -> magnitude "
          f"ASSUMEE = g (non verifiee ici, faute de donnees suffisantes).")
    resultat_avant = analyser_segment_physique(
        "AVANT impact", t[idx_avant], X[idx_avant], Y[idx_avant], Z[idx_avant], vecteur_gravite)
else:
    print(f"\n[AVANT impact] Pas assez de points avant l'impact ({n_avant}).")

n_apres = n - idx_apres.start
if n_apres >= SEUIL_POINTS_FIT_LIBRE:
    resultat_apres = analyser_segment(
        "APRES impact", t[idx_apres], X[idx_apres], Y[idx_apres], Z[idx_apres], DEGRE_POLYNOME)
elif MODELE_PHYSIQUE_ACTIF and n_apres >= 3:
    print(f"\n[APRES impact] {n_apres} points : pas assez pour un fit libre "
          f"degre {DEGRE_POLYNOME}, mais assez pour mesurer la magnitude de "
          f"l'acceleration (direction fixee).")
    resultat_apres = analyser_segment_direction_fixee(
        "APRES impact", t[idx_apres], X[idx_apres], Y[idx_apres], Z[idx_apres],
        vecteur_gravite / np.linalg.norm(vecteur_gravite))
elif MODELE_PHYSIQUE_ACTIF and n_apres >= 2:
    print(f"\n[APRES impact] Seulement {n_apres} points : meme la magnitude de "
          f"l'acceleration n'est pas mesurable (direction fixee + magnitude "
          f"libre = 7 inconnues, minimum 3 points necessaires) -> magnitude "
          f"ASSUMEE = g (non verifiee ici, faute de donnees suffisantes).")
    resultat_apres = analyser_segment_physique(
        "APRES impact", t[idx_apres], X[idx_apres], Y[idx_apres], Z[idx_apres], vecteur_gravite)
else:
    print(f"\n[APRES impact] Pas assez de points apres l'impact ({n_apres}).")

if resultat_avant is not None:
    print(f"\n>>> Vitesse JUSTE AVANT l'impact (extrapolee) : "
          f"{np.linalg.norm(resultat_avant['v_fin']):.3f} {UNITE_DISTANCE}/s")
if resultat_apres is not None:
    print(f">>> Vitesse JUSTE APRES l'impact (extrapolee)  : "
          f"{np.linalg.norm(resultat_apres['v_debut']):.3f} {UNITE_DISTANCE}/s")

# ==========================================================================
# SAUVEGARDE CSV (brut = differences finies, fit = polynomial par segment)
# ==========================================================================
# ==========================================================================
# SAUVEGARDE CSV (brut = differences finies frame par frame, fit = modele
# polynomial/physique ajuste sur tout le segment avant/apres -> constante
# sur le segment, beaucoup plus fiable pour l'analyse physique)
# ==========================================================================
vx_fit = np.full(n, np.nan); vy_fit = np.full(n, np.nan); vz_fit = np.full(n, np.nan)
ax_fit = np.full(n, np.nan); ay_fit = np.full(n, np.nan); az_fit = np.full(n, np.nan)

if resultat_avant is not None:
    vx_fit[idx_avant] = resultat_avant["vx"]; vy_fit[idx_avant] = resultat_avant["vy"]; vz_fit[idx_avant] = resultat_avant["vz"]
    ax_fit[idx_avant] = resultat_avant["ax"]; ay_fit[idx_avant] = resultat_avant["ay"]; az_fit[idx_avant] = resultat_avant["az"]
if resultat_apres is not None:
    vx_fit[idx_apres] = resultat_apres["vx"]; vy_fit[idx_apres] = resultat_apres["vy"]; vz_fit[idx_apres] = resultat_apres["vz"]
    ax_fit[idx_apres] = resultat_apres["ax"]; ay_fit[idx_apres] = resultat_apres["ay"]; az_fit[idx_apres] = resultat_apres["az"]

vitesse_fit = np.sqrt(vx_fit**2 + vy_fit**2 + vz_fit**2)
acceleration_fit = np.sqrt(ax_fit**2 + ay_fit**2 + az_fit**2)

csv_path = os.path.join(DOSSIER_SORTIE, "cinematique_3d.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "t_s", "X_m", "Y_m", "Z_m",
                      "vitesse_brut_m_s", "acceleration_brut_m_s2",
                      "vitesse_fit_m_s", "acceleration_fit_m_s2",
                      "segment"])
    for i in range(n):
        if i <= idx_avant_fin:
            segment = "avant"
        elif i >= idx_apres_debut:
            segment = "apres"
        else:
            segment = "contact_exclu"
        writer.writerow([
            frames[i], t[i], X[i], Y[i], Z[i],
            vitesse[i], acceleration[i],
            vitesse_fit[i], acceleration_fit[i],
            segment,
        ])
print("\nCinematique (brut + fit) sauvegardee dans", csv_path)
print("-> Utilise les colonnes '*_fit' pour l'analyse (constantes sur "
      "chaque segment, beaucoup moins bruitees que les colonnes '*_brut', "
      "gardees seulement a titre de comparaison).")

resume_path = os.path.join(DOSSIER_SORTIE, "resume_impact.txt")
with open(resume_path, "w", encoding="utf-8") as f:
    f.write(f"Zone de contact : entre les frames {frames[idx_avant_fin]} "
            f"et {frames[min(idx_apres_debut, n - 1)]}\n")
    if resultat_avant is not None:
        f.write(f"Vitesse juste AVANT impact : {resultat_avant['v_fin']} m/s "
                f"(norme {np.linalg.norm(resultat_avant['v_fin']):.4f})\n")
        f.write(f"Acceleration (segment avant) : {resultat_avant['a_const']} m/s2 "
                f"(norme {np.linalg.norm(resultat_avant['a_const']):.4f})\n")
    if resultat_apres is not None:
        f.write(f"Vitesse juste APRES impact : {resultat_apres['v_debut']} m/s "
                f"(norme {np.linalg.norm(resultat_apres['v_debut']):.4f})\n")
        f.write(f"Acceleration (segment apres) : {resultat_apres['a_const']} m/s2 "
                f"(norme {np.linalg.norm(resultat_apres['a_const']):.4f})\n")
print("Resume impact sauvegarde dans", resume_path)

# ==========================================================================
# GRAPHIQUES
# ==========================================================================
fig = plt.figure(figsize=(14, 10))
fig.suptitle(f"Cinematique 3D \u2013 zone de contact entre les frames "
             f"{frames[idx_avant_fin]} et {frames[min(idx_apres_debut, n - 1)]}",
             fontsize=11, color="dimgray")

ax_v = fig.add_subplot(2, 2, 1)
ax_v.plot(t, vitesse, color="tab:blue", alpha=0.5, label="brut (diff. finies)")
if resultat_avant is not None:
    ax_v.plot(resultat_avant["t"],
               np.sqrt(resultat_avant["vx"]**2 + resultat_avant["vy"]**2 + resultat_avant["vz"]**2),
               color="tab:green", linewidth=2, label="fit avant impact")
if resultat_apres is not None:
    ax_v.plot(resultat_apres["t"],
               np.sqrt(resultat_apres["vx"]**2 + resultat_apres["vy"]**2 + resultat_apres["vz"]**2),
               color="tab:orange", linewidth=2, label="fit apres impact")
ax_v.axvline(t_impact_approx, color="black", linestyle="--", alpha=0.5, label="zone impact")
ax_v.set_title("Vitesse (norme)")
ax_v.set_xlabel("temps (s)"); ax_v.set_ylabel(f"vitesse ({UNITE_DISTANCE}/s)")
ax_v.legend(fontsize=8); ax_v.grid(alpha=0.3)

ax_a = fig.add_subplot(2, 2, 2)
ax_a.plot(t, acceleration, color="tab:red", alpha=0.5, label="brut (diff. finies)")
if resultat_avant is not None:
    ax_a.plot(resultat_avant["t"],
               np.sqrt(resultat_avant["ax"]**2 + resultat_avant["ay"]**2 + resultat_avant["az"]**2),
               color="tab:green", linewidth=2, label="fit avant impact")
if resultat_apres is not None:
    ax_a.plot(resultat_apres["t"],
               np.sqrt(resultat_apres["ax"]**2 + resultat_apres["ay"]**2 + resultat_apres["az"]**2),
               color="tab:orange", linewidth=2, label="fit apres impact")
ax_a.axvline(t_impact_approx, color="black", linestyle="--", alpha=0.5, label="zone impact")
ax_a.set_title("Acceleration (norme)")
ax_a.set_xlabel("temps (s)"); ax_a.set_ylabel(f"acceleration ({UNITE_DISTANCE}/s\u00b2)")
ax_a.legend(fontsize=8); ax_a.grid(alpha=0.3)

ax_pos = fig.add_subplot(2, 2, 3)
ax_pos.plot(t, Y, "o-", color="gray", alpha=0.6, label="Y brut")
ax_pos.axvline(t_impact_approx, color="black", linestyle="--", alpha=0.5, label="zone impact")
ax_pos.set_title("Hauteur Y au cours du temps (verification visuelle de l'impact)")
ax_pos.set_xlabel("temps (s)"); ax_pos.set_ylabel(f"Y ({UNITE_DISTANCE})")
ax_pos.legend(fontsize=8); ax_pos.grid(alpha=0.3)

ax_3d = fig.add_subplot(2, 2, 4, projection="3d")
sc = ax_3d.scatter(X, Y, Z, c=vitesse, cmap="plasma", s=20)
ax_3d.plot(X, Y, Z, color="gray", alpha=0.3, linewidth=1)
idx_debut_marq = min(idx_avant_fin + 1, n - 1)
idx_fin_marq = min(idx_apres_debut, n - 1)
ax_3d.scatter([X[idx_debut_marq], X[idx_fin_marq]],
              [Y[idx_debut_marq], Y[idx_fin_marq]],
              [Z[idx_debut_marq], Z[idx_fin_marq]],
              color="black", s=60, marker="x", label="zone impact")
ax_3d.set_xlabel("X"); ax_3d.set_ylabel("Y"); ax_3d.set_zlabel("Z")
ax_3d.set_title("Trajectoire 3D coloree par vitesse")
fig.colorbar(sc, ax=ax_3d, shrink=0.6, label=f"vitesse ({UNITE_DISTANCE}/s)")

plt.tight_layout()
png_path = os.path.join(DOSSIER_SORTIE, "cinematique_3d.png")
plt.savefig(png_path, dpi=150)
print("Graphiques sauvegardes dans", png_path)

print("\n====================")
print(f"Zone de contact : frames {frames[idx_avant_fin]} a "
      f"{frames[min(idx_apres_debut, n - 1)]}")
if resultat_avant is not None:
    print(f"Vitesse avant impact : {np.linalg.norm(resultat_avant['v_fin']):.3f} {UNITE_DISTANCE}/s")
if resultat_apres is not None:
    print(f"Vitesse apres impact : {np.linalg.norm(resultat_apres['v_debut']):.3f} {UNITE_DISTANCE}/s")
print(f"Resultats dans : {DOSSIER_SORTIE}/")
print("====================")

plt.show()