"""
common.py — éléments communs à toutes les pages de l'application.

Ce fichier centralise deux types d'éléments :
1. Les constantes et tableaux de résultats utilisés dans les pages.
2. Les petites fonctions réutilisables pour éviter de répéter le même code.

Objectif pédagogique : garder un code simple, lisible et facile à modifier.
"""


import pandas as pd
import streamlit as st
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image

# =========================
# GLOBAL STYLE
# =========================

def apply_global_style():
    st.markdown("""
    <style>

    /* ================= SIDEBAR ================= */

    section[data-testid="stSidebar"] ul li div{
        font-size:18px !important;
        font-weight:500;
    }

    section[data-testid="stSidebar"] ul li{
        margin-bottom:8px;
    }

    /* ================= PAGE ================= */

    .block-container{
        padding-top:2rem;
        padding-bottom:2rem;
    }

    /* ================= TITLES ================= */

    .main-title{
        font-size:36px;
        font-weight:700;
        margin-bottom:20px;
    }

    .section-title{
        font-size:24px;
        font-weight:600;
        margin-top:20px;
        margin-bottom:10px;
    }

    /* ================= TEXT ================= */

    .text{
        font-size:20px;
        line-height:1.6;
    }

    </style>
    """, unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# Chemins du projet
# --------------------------------------------------------------------------------------
# BASE_DIR correspond au dossier où se trouve common.py.
BASE_DIR = Path(__file__).resolve().parent

# Dossier où sont stockées les figures utilisées par les pages.
IMAGE_DIR = BASE_DIR / "images"

# Dossier où l'utilisateur peut déposer le modèle entraîné.
MODEL_DIR = BASE_DIR / "models"

# Nom du modèle attendu pour la démonstration externe.
MODEL_FILENAME = "final_vgg16_fixed_splits.keras"

# Taille d'entrée utilisée pour la démonstration externe.
# Elle peut être modifiée si le modèle final a été entraîné avec une autre taille.
MODEL_INPUT_SIZE = (224, 224)

# Les 4 classes diagnostiques du dataset.
CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

# Couleurs utilisées dans les graphiques.
CLASS_COLORS = {
    "COVID": "#E4572E",
    "Lung_Opacity": "#F3A712",
    "Normal": "#17A398",
    "Viral Pneumonia": "#3B6EA5",
}

def image_path(filename):
    """
    Retourne le chemin complet d'une image située dans le dossier images/.

    Exemple : image_path("vgg16_confusion_matrix.png")
    renvoie le chemin complet vers images/vgg16_confusion_matrix.png.
    """
    return IMAGE_DIR / filename


def show_image(filename, caption=None):
    """
    Affiche une image du dossier images/ si elle existe.

    Si l'image est absente, l'application ne plante pas : elle affiche un message d'information.
    C'est pratique pendant le développement, quand certaines figures ne sont pas encore copiées.
    """
    path = image_path(filename)
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Image non trouvée : images/{filename}")


def default_model_path():
    """
    Retourne le chemin attendu du modèle Keras final.

    Le fichier n'est pas inclus dans l'archive. Il faut le copier manuellement dans models/.
    """
    return MODEL_DIR / MODEL_FILENAME


def model_status_message():
    """
    Vérifie si le modèle entraîné est présent dans le dossier models/.

    Retour :
    - True + message si le fichier existe.
    - False + message si le fichier est absent.
    """
    path = default_model_path()
    if path.exists():
        return True, f"Modèle trouvé : {path}"
    return False, f"Modèle absent. Placez le fichier ici : {path}"


def input_fingerprint(image_bytes, apply_clahe=False, apply_mask=False, mask_bytes=None, **kwargs):
    """
    Crée une signature courte des choix de l'utilisateur.

    Pourquoi cette fonction est utile ?
    Streamlit recharge la page à chaque action. Cette signature permet de savoir
    si l'utilisateur a changé l'image, activé CLAHE ou ajouté un masque.

    Si quelque chose change, on efface l'ancienne prédiction pour éviter
    d'afficher un résultat qui ne correspond plus à l'image visible.

    Paramètres :
    - image_bytes : contenu de la radiographie chargée ;
    - apply_clahe : True si CLAHE est demandé ;
    - apply_mask : True si un masque pulmonaire est demandé ;
    - mask_bytes : contenu du masque, si l'utilisateur en fournit un ;
    - **kwargs : permet d'ignorer sans erreur d'éventuelles options futures.

    Remarque débutant :
    Le paramètre **kwargs rend la fonction plus robuste. Si une page Streamlit
    envoie une option supplémentaire, l'application ne se bloque pas pour autant.
    """
    hasher = hashlib.sha256()

    # Signature de la radiographie principale.
    hasher.update(image_bytes)

    # Signature des options de prétraitement.
    hasher.update(str(apply_clahe).encode("utf-8"))
    hasher.update(str(apply_mask).encode("utf-8"))

    # Signature du masque si un fichier masque a été chargé.
    if mask_bytes is not None:
        hasher.update(mask_bytes)

    return hasher.hexdigest()


def apply_simple_clahe(gray_array):
    """
    Applique CLAHE si OpenCV est installé.

    Pour garder l'application simple, OpenCV n'est pas obligatoire dans requirements.txt.
    Si OpenCV n'est pas installé, on renvoie l'image sans CLAHE.
    """
    try:
        import cv2
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_array)
    except Exception:
        return gray_array


def preprocess_external_image(image, apply_clahe=False, apply_mask=False, mask_image=None, **kwargs):
    """
    Prépare une radiographie externe avant prédiction.

    Cette fonction reste volontairement simple pour être facile à comprendre.
    Elle reproduit les grandes étapes utilisées dans l'étude, mais en version
    adaptée à une image externe chargée par l'utilisateur.

    Étapes appliquées :
    1. conversion de la radiographie en niveaux de gris ;
    2. application optionnelle de CLAHE pour améliorer le contraste local ;
    3. application optionnelle d'un masque pulmonaire si l'utilisateur en fournit un ;
    4. redimensionnement à la taille attendue par le modèle ;
    5. normalisation entre 0 et 1 ;
    6. conversion en RGB, car VGG16 travaille avec 3 canaux.

    Paramètres :
    - image : radiographie chargée par l'utilisateur ;
    - apply_clahe : True si l'utilisateur coche CLAHE ;
    - apply_mask : True si l'utilisateur souhaite appliquer un masque ;
    - mask_image : image du masque pulmonaire, si elle existe ;
    - **kwargs : permet d'ignorer sans erreur d'éventuelles options futures.

    Retour :
    - batch : tableau NumPy prêt pour model.predict() ;
    - stages : dictionnaire contenant les images intermédiaires affichées dans Streamlit.
    """
    # 1. Conversion de la radiographie en niveaux de gris.
    gray = image.convert("L")
    gray_array = np.array(gray)

    # 2. Application optionnelle de CLAHE.
    if apply_clahe:
        processed_array = apply_simple_clahe(gray_array)
    else:
        processed_array = gray_array.copy()

    # Image intermédiaire après CLAHE ou sans CLAHE.
    processed = Image.fromarray(processed_array.astype("uint8"))

    # On prépare un dictionnaire pour stocker toutes les étapes visibles.
    stages = {
        "gray": gray,
        "processed": processed,
    }

    # 3. Application optionnelle du masque pulmonaire.
    # Le masque doit être une image blanche/noire ou proche :
    # - blanc = zone gardée ;
    # - noir = zone supprimée.
    if apply_mask and mask_image is not None:
        mask_gray = mask_image.convert("L")

        # Le masque doit avoir la même taille que la radiographie avant multiplication.
        mask_gray = mask_gray.resize(gray.size)
        mask_array = np.array(mask_gray)

        # Création d'un masque binaire simple.
        # Les pixels > 127 sont considérés comme appartenant aux poumons.
        binary_mask = (mask_array > 127).astype("uint8")

        # On garde les pixels pulmonaires et on met le reste en noir.
        processed_array = processed_array * binary_mask

        # Images intermédiaires affichables dans la page Streamlit.
        stages["mask"] = mask_gray
        stages["masked"] = Image.fromarray(processed_array.astype("uint8"))

    # 4. Redimensionnement pour obtenir la taille attendue par le modèle.
    final_image = Image.fromarray(processed_array.astype("uint8"))
    resized = final_image.resize(MODEL_INPUT_SIZE)

    # 5. VGG16 attend une image à 3 canaux.
    # Comme la radiographie est en niveaux de gris, on duplique simplement le canal.
    rgb = resized.convert("RGB")

    # 6. Normalisation simple des pixels entre 0 et 1.
    array = np.array(rgb).astype("float32") / 255.0

    # 7. Ajout de la dimension batch.
    # Exemple : (224, 224, 3) devient (1, 224, 224, 3).
    batch = np.expand_dims(array, axis=0)

    # Dernières étapes affichables.
    stages["resized"] = resized
    stages["rgb"] = rgb

    return batch, stages



# def load_keras_model(model_path):
#     """
#     Charge le modèle Keras sauvegardé.

#     TensorFlow n'est importé qu'à l'intérieur de cette fonction pour que l'application
#     de présentation puisse fonctionner même si TensorFlow n'est pas installé.
#     """
#     from tensorflow import keras
#     return keras.models.load_model(model_path)

@st.cache_resource(show_spinner=False)
def load_keras_model(model_path):
    """
    Charge le modèle Keras utilisé par la page de démonstration.

    Le modèle VGG16 contient une couche Lambda appelée
    'vgg16_preprocess'. Cette couche utilise la fonction
    preprocess_input de VGG16.

    Lors du chargement, cette fonction doit être fournie
    explicitement à Keras avec custom_objects.
    """

    # Import local :
    # TensorFlow n'est importé que lorsque la page de prédiction
    # a réellement besoin du modèle.
    from tensorflow import keras

    # Fonction utilisée dans la couche Lambda du modèle sauvegardé.
    from tensorflow.keras.applications.vgg16 import preprocess_input

    model = keras.models.load_model(
        model_path,

        # On indique à Keras quelle fonction correspond au nom
        # "preprocess_input" enregistré dans le fichier .keras.
        custom_objects={
            "preprocess_input": preprocess_input,
        },

        # Pour une simple prédiction, il n'est pas nécessaire
        # de restaurer l'optimizer, la loss et les métriques.
        compile=False,

        # Nécessaire pour certains modèles contenant une Lambda.
        # À utiliser seulement parce que le modèle est le vôtre
        # et provient d'une source de confiance.
        safe_mode=False,
    )

    return model

def predict_probabilities(model, batch):
    """
    Calcule les probabilités de chaque classe avec le modèle chargé.

    Retour : dictionnaire du type {"COVID": 0.81, "Normal": 0.10, ...}.
    """
    predictions = model.predict(batch, verbose=0)[0]
    return {label: float(prob) for label, prob in zip(CLASSES, predictions)}


# --------------------------------------------------------------------------------------
# Partie I — Exploration des données (page 2)
# --------------------------------------------------------------------------------------
CLASS_COUNTS = pd.DataFrame(
    {
        "Catégorie": ["Normal", "Lung_Opacity", "COVID", "Viral Pneumonia"],
        "Images": [10192, 6012, 3616, 1345],
        "Pourcentage": [48.2, 28.4, 17.1, 6.4],
    }
)
TOTAL_IMAGES = int(CLASS_COUNTS["Images"].sum())

LUMINOSITY_STATS = pd.DataFrame(
    {
        "Catégorie": ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"],
        "Luminosité moyenne": [139.5, 127.0, 127.0, 127.0],
        "Écart-type (contraste)": [56, 58, 57, 58],
    }
)

OUTLIER_THRESHOLDS = {"sombre": 75, "clair": 175}

TTEST_RESULTS = {
    "COVID": {"mean": 141.48, "std": 20.70, "n": 50},
    "Normal": {"mean": 129.52, "std": 27.27, "n": 50},
    "t_stat": 2.45,
    "p_value": 0.0163,
}

# --------------------------------------------------------------------------------------
# Partie I — Prétraitement (page 3)
# --------------------------------------------------------------------------------------
PREPROCESSING_COMPARISON = pd.DataFrame(
    {
        "Approche": ["Baseline", "CLAHE", "Masque", "CLAHE + Masque"],
        "Variance intra-classe": [0.0077, 0.0042, 0.0091, 0.0041],
        "Std inter-classe": [0.032, 0.024, 0.035, 0.029],
        "p-value (t-test)": [9.3e-7, 2.2e-8, 1.6e-12, 5.6e-14],
    }
)

PIPELINE_STEPS = [
    "Chargement de l'image en niveaux de gris (conversion si nécessaire)",
    "Application de CLAHE (clipLimit = 2.0, tileGridSize = 8×8)",
    "Chargement et application du masque pulmonaire correspondant",
    "Normalisation des valeurs dans l'intervalle [0, 1]",
    "Augmentation de données (classes minoritaires) + sous-échantillonnage (classe majoritaire)",
]

# --------------------------------------------------------------------------------------
# Partie II — Machine Learning (page 4)
# --------------------------------------------------------------------------------------
DATASET_SPLIT = pd.DataFrame(
    {
        "Split": ["Train", "Validation", "Test", "Total"],
        "COVID": [2531, 543, 542, 3616],
        "Normal": [7134, 1529, 1529, 10192],
        "Lung_Opacity": [4208, 902, 902, 6012],
        "Viral_Pneumonia": [942, 201, 202, 1345],
        "Total": [14815, 3175, 3175, 21165],
    }
)

ML_RESULTS = pd.DataFrame(
    [
        ["LR (pixels)", "Pixels", "64×64", 0.65, 0.61, 0.27, None],
        ["LR (HOG)", "HOG", "64×64", 0.72, 0.72, 0.63, 0.703],
        ["LR + SMOTE (HOG)", "HOG", "64×64", 0.72, 0.72, 0.62, None],
        ["Random Forest", "Pixels", "64×64", 0.65, 0.62, 0.39, 0.63],
        ["Random Forest (HOG)", "HOG", "64×64", 0.62, 0.60, 0.46, 0.60],
        ["KNN + PCA", "PCA(200)", "64×64", 0.68, 0.63, 0.29, 0.620],
        ["Boosting (Hist GB) + PCA", "PCA(200)", "128×128", 0.71, 0.69, 0.44, 0.656],
        ["XGBoost + PCA", "PCA(200)", "128×128", 0.74, 0.72, 0.40, None],
        ["SVC (SVM) + PCA (64)", "PCA(200)", "64×64", 0.75, 0.74, 0.57, 0.728],
        ["SVC (SVM) + PCA (128)", "PCA(200)", "128×128", 0.75, 0.75, 0.62, None],
    ],
    columns=["Modèle", "Features", "Resize", "Accuracy", "F1-macro", "Recall COVID", "CV F1-macro"],
)

ML_F1_BY_CLASS = pd.DataFrame(
    {
        "Modèle": ["LR (pix)", "LR (HOG)", "LR+SMOTE", "RF", "KNN", "Boosting", "XGBoost", "SVC/SVM (64)", "SVC/SVM (128)"],
        "COVID": [0.33, 0.55, 0.55, 0.44, 0.39, 0.46, 0.48, 0.54, 0.56],
        "Lung_Opacity": [0.58, 0.70, 0.70, 0.66, 0.60, 0.68, 0.71, 0.73, 0.73],
        "Normal": [0.75, 0.79, 0.79, 0.77, 0.77, 0.80, 0.82, 0.83, 0.83],
        "Viral Pneumonia": [0.77, 0.83, 0.84, 0.79, 0.77, 0.83, 0.84, 0.86, 0.88],
    }
).set_index("Modèle")

ML_COVID_RECALL_PRECISION = pd.DataFrame(
    {
        "Modèle": ["LR (pix)", "LR (HOG)", "LR+SMOTE", "RF", "KNN", "Boosting", "XGBoost", "SVC/SVM (64)", "SVC/SVM (128)"],
        "Recall COVID": [0.27, 0.63, 0.62, 0.39, 0.29, 0.44, 0.40, 0.57, 0.62],
        "Precision COVID": [0.41, 0.49, 0.49, 0.52, 0.59, 0.48, 0.61, 0.51, 0.51],
    }
)

ML_CONF_MATRIX = pd.DataFrame(
    [
        [448, 166, 96, 13],
        [162, 854, 178, 9],
        [218, 100, 1651, 69],
        [6, 15, 14, 234],
    ],
    index=CLASSES,
    columns=CLASSES,
)

# --------------------------------------------------------------------------------------
# Partie III — Deep Learning (page 5)
# --------------------------------------------------------------------------------------
DL_RESULTS = pd.DataFrame(
    [
        ["VGG16", "Non", "Non", 0.886, 0.880, 0.885, 0.810, 0.814],
        ["VGG16", "Oui", "Oui", 0.877, 0.875, 0.877, 0.825, 0.802],
        ["CNN custom", "Non", "Non", 0.878, 0.873, 0.877, 0.699, 0.790],
        ["CNN custom", "Oui", "Oui", 0.875, 0.872, 0.875, 0.792, 0.794],
        ["ResNet50", "Non", "Non", 0.859, 0.855, 0.859, 0.773, 0.749],
        ["EfficientNetB0", "Non", "Non", 0.855, 0.846, 0.854, 0.745, 0.759],
        ["DenseNet121", "Non", "Non", 0.839, 0.830, 0.841, 0.780, 0.710],
        ["EfficientNetB0", "Oui", "Oui", 0.827, 0.821, 0.829, 0.780, 0.697],
        ["ResNet50", "Oui", "Oui", 0.830, 0.815, 0.827, 0.664, 0.700],
        ["DenseNet121", "Oui", "Oui", 0.787, 0.769, 0.793, 0.760, 0.630],
    ],
    columns=["Modèle", "CLAHE", "Data Augmentation", "Accuracy", "Macro-F1", "F1-weighted", "Recall COVID", "F1 COVID"],
)
DL_RESULTS["Variante"] = DL_RESULTS["Modèle"] + " (" + (DL_RESULTS["CLAHE"].eq("Oui").map({True: "T/T", False: "F/F"})) + ")"

DL_BASELINE = pd.DataFrame(
    {
        "Modèle": ["Dense Baseline", "CNN personnalisé*", "EfficientNetB0*", "ResNet50*", "VGG16*"],
        "Accuracy": [0.688, 0.878, 0.855, 0.859, 0.886],
        "F1 COVID": [0.434, 0.790, 0.759, 0.749, 0.814],
    }
)

DL_CNN_CM = pd.DataFrame(
    [[379, 64, 98, 1], [20, 1418, 85, 6], [19, 77, 805, 1], [0, 9, 7, 186]],
    index=["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"],
    columns=["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"],
)
DL_VGG_CM = pd.DataFrame(
    [[439, 72, 28, 3], [28, 1461, 33, 7], [68, 103, 729, 2], [1, 15, 1, 185]],
    index=["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"],
    columns=["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"],
)

ROC_AUC = pd.DataFrame(
    {
        "Classe": ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia", "Macro", "Weighted", "Micro"],
        "CNN custom": [0.969, 0.972, 0.966, 0.998, 0.977, 0.972, 0.980],
        "VGG16": [0.969, 0.975, 0.969, 0.998, 0.978, 0.974, 0.983],
    }
)
