"""
Point d'entree CLI pour le pipeline de suivi 3D de pomme de terre.

Lit les variables DATA_DIR et OUTPUT_DIR du fichier .env si present,
puis les surcharge par les arguments de ligne de commande.

Utilisation :
    python -m src.tracking
    python -m src.tracking --data-dir /path/to/data
    python -m src.tracking --data-dir /path/to/data --calib config/camera_parameters.yaml --output results/
"""

import argparse
from pathlib import Path

from .config import get_data_dir, get_output_dir, PipelineParams
from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Suivi 3D de pomme de terre par vision stereo",
    )
    parser.add_argument(
        "--data-dir",
        default=str(get_data_dir() or "data"),
        help="Dossier contenant les sous-dossiers CamG/ et CamD/",
    )
    parser.add_argument(
        "--calib",
        default="config/camera_parameters.yaml",
        help="Fichier YAML de calibration stereo OpenCV",
    )
    parser.add_argument(
        "--output",
        default=str(get_output_dir() or "output"),
        help="Dossier de sortie pour les resultats",
    )
    parser.add_argument(
        "--threshold", type=int, default=20,
        help="Seuil de difference pour la soustraction de fond",
    )
    parser.add_argument(
        "--morph-kernel", type=int, default=7,
        help="Taille du noyau pour les operations morphologiques",
    )
    parser.add_argument(
        "--min-area", type=int, default=500,
        help="Aire minimale de l objet en pixels",
    )

    args = parser.parse_args()

    params = PipelineParams(
        threshold=args.threshold,
        morph_kernel_size=args.morph_kernel,
        min_area=args.min_area,
    )

    run_pipeline(
        data_dir=Path(args.data_dir),
        calibration_yml=Path(args.calib),
        output_dir=Path(args.output),
        params=params,
    )


if __name__ == "__main__":
    main()
