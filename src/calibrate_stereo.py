import cv2
import numpy as np
import glob
import os

# ==========================
# PARAMETRES
# ==========================
pattern_size = (7, 7)   # coins internes du damier
square_size = 30        # mm, taille réelle d'une case du damier

# Crop trouvé précédemment sur l'image réduite (600x600)
x0 = 0
y0 = 250
crop_width = 600
crop_height = 400

# Taille des images Ximea MQ042MG-CM
IMG_SIZE = (2048, 2048)  # (w, h)

OUTPUT_YML = "CalibrationStereo.yml"

# ==========================
# POINTS DU DAMIER 3D
# ==========================
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# listes finales
objpoints = []      # points 3D (communs aux deux caméras)
imgpoints_G = []    # points 2D caméra gauche
imgpoints_D = []    # points 2D caméra droite


# ==========================
# FONCTION DETECTION
# ==========================
def detect_chessboard(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erreur lecture :", path)
        return None

    h, w = img.shape

    # réduction uniquement pour retrouver le damier
    img_small = cv2.resize(img, (600, 600))

    # crop dans l'image réduite
    crop = img_small[y0:y0 + crop_height, x0:x0 + crop_width]

    ret, corners = cv2.findChessboardCornersSB(crop, pattern_size)
    if not ret:
        print("Damier non trouvé :", path)
        return None

    # Passage crop -> image réduite
    corners[:, 0] += x0
    corners[:, 1] += y0

    # Passage image réduite -> image réelle
    scale_x = w / 600
    scale_y = h / 600
    corners[:, 0] *= scale_x
    corners[:, 1] *= scale_y

    # Affinage sub-pixel sur l'image réelle (findChessboardCornersSB est déjà
    # précis, mais on peut resserrer encore avec cornerSubPix si besoin)
    corners = corners.reshape(-1, 1, 2).astype(np.float32)
    return corners


# ==========================
# TRAITEMENT DES IMAGES
# ==========================
images_G = sorted(glob.glob("CameraG/*.tif"))
images_D = sorted(glob.glob("CameraD/*.tif"))

print("Images gauche :", len(images_G))
print("Images droite :", len(images_D))

if len(images_G) != len(images_D):
    print("Attention : nombre d'images différent !")

valid_names_G = []
valid_names_D = []

for imgG, imgD in zip(images_G, images_D):
    print("\nTraitement :")
    print(imgG)
    print(imgD)

    corners_G = detect_chessboard(imgG)
    corners_D = detect_chessboard(imgD)

    if corners_G is not None and corners_D is not None:
        objpoints.append(objp)
        imgpoints_G.append(corners_G)
        imgpoints_D.append(corners_D)
        valid_names_G.append(imgG)
        valid_names_D.append(imgD)
        print("paire valide")
    else:
        print("paire ignorée")

print("\n====================")
print("Détection terminée")
print("====================")
print("Nombre de couples valides :", len(objpoints))

if len(objpoints) < 5:
    raise RuntimeError(
        "Pas assez de paires valides pour une calibration fiable "
        "(minimum recommandé : 10-15 paires, plusieurs angles/orientations)."
    )


# ==========================
# LISTE DETAILLEE DES PAIRES VALIDES / REJETEES
# ==========================
noms_rejetes_G = [f for f in images_G if f not in valid_names_G]
noms_rejetes_D = [f for f in images_D if f not in valid_names_D]

print("\n=== Paires VALIDES ===")
for g, d in zip(valid_names_G, valid_names_D):
    print(f"  {os.path.basename(g)}  <->  {os.path.basename(d)}")

print("\n=== Images REJETEES (côté gauche) ===")
for f in noms_rejetes_G:
    print(" ", os.path.basename(f))

print("\n=== Images REJETEES (côté droite) ===")
for f in noms_rejetes_D:
    print(" ", os.path.basename(f))

print(f"\nTotal : {len(valid_names_G)} paires valides / {len(images_G)} paires traitées")

# Sauvegarde de la liste dans un fichier texte pour archivage
with open("paires_valides.txt", "w", encoding="utf-8") as f:
    f.write("=== Paires VALIDES ===\n")
    for g, d in zip(valid_names_G, valid_names_D):
        f.write(f"{g} ; {d}\n")
    f.write("\n=== Images REJETEES (gauche) ===\n")
    for name in noms_rejetes_G:
        f.write(f"{name}\n")
    f.write("\n=== Images REJETEES (droite) ===\n")
    for name in noms_rejetes_D:
        f.write(f"{name}\n")

print("Liste des paires sauvegardée dans 'paires_valides.txt'")


# ==========================
# CALIBRATION INTRINSEQUE - CAMERA GAUCHE
# ==========================
print("\n=== Calibration intrinsèque - Caméra GAUCHE ===")
flags = (
    cv2.CALIB_FIX_ASPECT_RATIO |
    cv2.CALIB_ZERO_TANGENT_DIST |
    cv2.CALIB_FIX_K3
)
K_init = np.array([
    [6000,0,1024],
    [0,6000,1024],
    [0,0,1]
],dtype=np.float64)


rms_G, K_G, dist_G, rvecs_G, tvecs_G = cv2.calibrateCamera(
    objpoints,
    imgpoints_G,
    IMG_SIZE,
    K_init,
    None,
    flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS
)
print("RMS caméra gauche :", rms_G)
print("Matrice intrinsèque K_G :\n", K_G)
print("Coefficients de distorsion (G) :\n", dist_G.ravel())


# ==========================
# CALIBRATION INTRINSEQUE - CAMERA DROITE
# ==========================
print("\n=== Calibration intrinsèque - Caméra DROITE ===")
flags = (
    cv2.CALIB_FIX_ASPECT_RATIO |
    cv2.CALIB_ZERO_TANGENT_DIST |
    cv2.CALIB_FIX_K3
)
K_init = np.array([
    [6000,0,1024],
    [0,6000,1024],
    [0,0,1]
],dtype=np.float64)

rms_D, K_D, dist_D, rvecs_D, tvecs_D = cv2.calibrateCamera(
    objpoints,
    imgpoints_D,
    IMG_SIZE,
    K_init,
    None,
    flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS
)
print("RMS caméra droite :", rms_D)
print("Matrice intrinsèque K_D :\n", K_D)
print("Coefficients de distorsion (D) :\n", dist_D.ravel())


# ==========================
# CALIBRATION STEREO
# ==========================
print("\n=== Calibration stéréo ===")

stereo_flags = cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

rms_stereo, K_G2, dist_G2, K_D2, dist_D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_G,
    imgpoints_D,
    K_G, dist_G,
    K_D, dist_D,
    IMG_SIZE,
    criteria=criteria_stereo,
    flags=stereo_flags,
)

print("RMS stéréo :", rms_stereo)
print("\nMatrice de rotation R (gauche -> droite) :\n", R)
print("\nVecteur de translation T (mm) :\n", T.ravel())
print("\nMatrice essentielle E :\n", E)
print("\nMatrice fondamentale F :\n", F)


# ==========================
# RECTIFICATION STEREO
# ==========================
print("\n=== Rectification stéréo ===")

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K_G, dist_G,
    K_D, dist_D,
    IMG_SIZE,
    R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=1,  # 0 = crop max (pas de zones noires), 1 = garde tout le champ
)
print("ROI gauche :", roi1)
print("ROI droite :", roi2)

print("R1 (rotation rectification gauche) :\n", R1)
print("\nR2 (rotation rectification droite) :\n", R2)
print("\nP1 (matrice de projection gauche) :\n", P1)
print("\nP2 (matrice de projection droite) :\n", P2)
print("\nMatrice Q (reprojection 3D) :\n", Q)


# ==========================
# CARTES DE RECTIFICATION (undistort + remap)
# ==========================
map1_G, map2_G = cv2.initUndistortRectifyMap(
    K_G, dist_G, R1, P1, IMG_SIZE, cv2.CV_16SC2
)
map1_D, map2_D = cv2.initUndistortRectifyMap(
    K_D, dist_D, R2, P2, IMG_SIZE, cv2.CV_16SC2
)


# ==========================
# ERREUR MOYENNE DE REPROJECTION (calibration stéréo)
# ==========================
def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist):
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints_proj, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], K, dist
        )
        error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2)
        total_error += error ** 2
        total_points += len(objpoints[i])
    return np.sqrt(total_error / total_points)


err_G = compute_reprojection_error(objpoints, imgpoints_G, rvecs_G, tvecs_G, K_G, dist_G)
err_D = compute_reprojection_error(objpoints, imgpoints_D, rvecs_D, tvecs_D, K_D, dist_D)

print("\n=== Erreurs de reprojection ===")
print("Erreur moyenne caméra gauche :", err_G)
print("Erreur moyenne caméra droite :", err_D)
print("RMS stéréo global (retourné par stereoCalibrate) :", rms_stereo)


# ==========================
# TRIANGULATION 3D (point correspondant G/D -> coordonnées 3D en mm)
# ==========================
def triangulate_point(pt_G, pt_D, P1, P2):
    """
    pt_G, pt_D : tuples (x, y) en pixels, coordonnées dans les images
                 RECTIFIEES (après remap avec map1_G/map2_G et map1_D/map2_D).
    P1, P2     : matrices de projection issues de stereoRectify.
    Retourne les coordonnées (X, Y, Z) en mm dans le repère de la caméra gauche.
    """
    pt_G = np.array(pt_G, dtype=np.float64).reshape(2, 1)
    pt_D = np.array(pt_D, dtype=np.float64).reshape(2, 1)

    point_4d = cv2.triangulatePoints(P1, P2, pt_G, pt_D)
    point_3d = point_4d[:3] / point_4d[3]  # normalisation homogène
    return point_3d.ravel()  # [X, Y, Z] en mm


def triangulate_from_disparity(pt_G, disparity, Q):
    """
    Alternative utilisant la matrice Q : pratique si on a déjà calculé
    une carte de disparité dense (ex. avec StereoSGBM) plutôt qu'un
    point unique.
    pt_G       : (x, y) en pixels dans l'image gauche RECTIFIEE
    disparity  : valeur de disparité (x_gauche - x_droite) en pixels
                 pour ce point
    Q          : matrice de reprojection 3D (issue de stereoRectify)
    """
    x, y = pt_G
    vec = np.array([x, y, disparity, 1.0], dtype=np.float64)
    point_4d = Q @ vec
    point_3d = point_4d[:3] / point_4d[3]
    return point_3d  # [X, Y, Z] en mm


# Exemple d'utilisation (à adapter avec de vrais points cliqués/détectés
# sur les images rectifiées) :
#
# pt_G_example = (1024, 1024)
# pt_D_example = (980, 1024)
# xyz = triangulate_point(pt_G_example, pt_D_example, P1, P2)
# print("Point 3D reconstruit (mm) :", xyz)


# ==========================
# SAUVEGARDE DES PARAMETRES (format YAML, standard OpenCV)
# ==========================
fs = cv2.FileStorage(OUTPUT_YML, cv2.FILE_STORAGE_WRITE)

# Intrinsèques individuels
fs.write("K_G", K_G)
fs.write("dist_G", dist_G)
fs.write("K_D", K_D)
fs.write("dist_D", dist_D)

# Résultats stéréo
fs.write("R", R)
fs.write("T", T)
fs.write("E", E)
fs.write("F", F)

# Rectification
fs.write("R1", R1)
fs.write("R2", R2)
fs.write("P1", P1)
fs.write("P2", P2)
fs.write("Q", Q)
fs.write("roi1", np.array(roi1))
fs.write("roi2", np.array(roi2))

# Cartes de rectification (undistort + remap), utiles pour ne pas
# les recalculer à chaque fois
fs.write("map1_G", map1_G)
fs.write("map2_G", map2_G)
fs.write("map1_D", map1_D)
fs.write("map2_D", map2_D)

# Erreurs et métadonnées
fs.write("rms_G", float(rms_G))
fs.write("rms_D", float(rms_D))
fs.write("rms_stereo", float(rms_stereo))
fs.write("img_size", np.array(IMG_SIZE))

fs.release()
print(f"\nParamètres sauvegardés dans '{OUTPUT_YML}'")


# ==========================
# VERIFICATION VISUELLE : IMAGES RECTIFIEES
# ==========================
def show_rectified_pair(pathG, pathD, n_lines=15):
    imgG = cv2.imread(pathG)
    imgD = cv2.imread(pathD)
    if imgG is None or imgD is None:
        return

    rectG = cv2.remap(imgG, map1_G, map2_G, cv2.INTER_LINEAR)
    rectD = cv2.remap(imgD, map1_D, map2_D, cv2.INTER_LINEAR)

    # Redimensionner pour affichage
    disp_h = 600
    scale = disp_h / rectG.shape[0]
    rectG_disp = cv2.resize(rectG, None, fx=scale, fy=scale)
    rectD_disp = cv2.resize(rectD, None, fx=scale, fy=scale)

    combined = np.hstack((rectG_disp, rectD_disp))

    # Lignes épipolaires horizontales pour vérifier l'alignement
    h_disp = combined.shape[0]
    for y in range(0, h_disp, h_disp // n_lines):
        cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

    cv2.imshow("Verification rectification (G | D)", combined)
    print("Appuyer sur une touche pour passer à la paire suivante (ESC pour quitter)")
    key = cv2.waitKey(0)
    return key


print("\n=== Vérification visuelle des images rectifiées ===")
for pathG, pathD in zip(valid_names_G, valid_names_D):
    key = show_rectified_pair(pathG, pathD)
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()

print("\n====================")
print("Calibration terminée avec succès")
print("====================")