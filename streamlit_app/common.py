"""
common.py — données, constantes et helpers partagés par toutes les pages
de l'application "Classification de Radiographies Pulmonaires — COVID-19".

Toutes les valeurs numériques sont reprises du compte rendu final du projet.
"""

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# Équipe / méta
# --------------------------------------------------------------------------------------
TEAM = ["Houssein Abbouchi", "Romuald Crochat", "Sareh Moghaddam", "Mathilde L'Hommelet"]
PROJECT_TITLE = "Classification de Radiographies Pulmonaires"
PROJECT_SUBTITLE = "Projet I.A. — Modélisation Machine Learning & Deep Learning"
GITHUB_URL = "https://github.com/mar25-iia2-radiographies/mar25_radiographies_pulmonaires_covid19"

CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
CLASS_COLORS = {
    "COVID": "#E4572E",
    "Lung_Opacity": "#F3A712",
    "Normal": "#17A398",
    "Viral Pneumonia": "#3B6EA5",
}


def page_header(icon: str, title: str, caption: str | None = None):
    """Bandeau d'en-tête cohérent sur chaque page."""
    st.title(f"{icon} {title}")
    st.caption(f"{PROJECT_SUBTITLE} · " + " · ".join(TEAM))
    if caption:
        st.markdown(caption)
    st.divider()


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
