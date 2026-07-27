"""Page 1 — Introduction et contexte du projet (Section 1 du compte rendu)."""

import streamlit as st

from common import CLASSES, TOTAL_IMAGES, page_header

# Configuration Streamlit : titre, icône et largeur de la page.
st.set_page_config(page_title="Introduction — Radiographies COVID-19", page_icon="📖", layout="wide")
# En-tête commun : même présentation au début de chaque page.
page_header("📖", "Introduction et contexte du projet")

st.markdown(
    """
L'interprétation des radiographies thoraciques constitue un élément essentiel du diagnostic de
nombreuses pathologies pulmonaires. Parmi celles-ci, la pandémie de COVID-19 a mis en évidence
l'intérêt de disposer d'outils d'aide à la décision capables d'assister les radiologues dans
l'identification rapide des atteintes pulmonaires associées à l'infection.

Les progrès récents de l'intelligence artificielle, et plus particulièrement du Machine Learning
et du Deep Learning, ont démontré leur capacité à extraire automatiquement des caractéristiques
discriminantes à partir d'images médicales complexes. Toutefois, la performance de ces approches
dépend fortement de la qualité des données, du prétraitement appliqué ainsi que du choix de
l'architecture de classification.
    """
)

st.subheader("🎯 Objectif du projet")
st.markdown(
    f"""
Développer et évaluer un **pipeline complet de classification automatique** de radiographies
thoraciques à partir du dataset **COVID-19 Radiography Database** ({TOTAL_IMAGES:,} images) —
prédire l'appartenance d'une radiographie à l'une des **4 catégories** suivantes :
    """.replace(",", " ")
)

cols = st.columns(4)
descriptions = {
    "COVID": "Atteinte pulmonaire liée à l'infection COVID-19",
    "Normal": "Radiographie sans anomalie pulmonaire",
    "Lung_Opacity": "Opacités pulmonaires d'origines diverses (non-COVID)",
    "Viral Pneumonia": "Pneumonie virale (hors COVID-19)",
}
for col, cls in zip(cols, CLASSES):
    with col:
        with st.container(border=True):
            st.markdown(f"**{cls}**")
            st.caption(descriptions[cls])

st.subheader("🧭 Démarche expérimentale progressive")
st.markdown("Le projet suit une logique en trois phases successives :")

phase_cols = st.columns(3)
phases = [
    ("1️⃣", "Exploration & Prétraitement", "Analyse approfondie des données, comparaison de stratégies de prétraitement (CLAHE, masques pulmonaires)."),
    ("2️⃣", "Machine Learning classique", "Évaluation de plusieurs modèles fondés sur des représentations manuelles des images (pixels, HOG, PCA)."),
    ("3️⃣", "Deep Learning", "Comparaison d'architectures convolutionnelles (CNN custom, transfert d'apprentissage) pour maximiser la détection du COVID-19."),
]
for col, (num, title, desc) in zip(phase_cols, phases):
    with col:
        with st.container(border=True):
            st.markdown(f"### {num} {title}")
            st.markdown(desc)

st.subheader("❓ Question de recherche centrale")
st.info(
    "**Dans quelle mesure les techniques modernes de Machine Learning et de Deep Learning "
    "permettent-elles d'identifier de manière fiable les atteintes pulmonaires liées à la "
    "COVID-19 à partir de radiographies thoraciques, tout en conservant un niveau de "
    "généralisation compatible avec les exigences du domaine médical ?**"
)

st.markdown(
    """
Au-delà de la recherche de performances prédictives élevées, ce travail s'intéresse également
aux problématiques de **robustesse**, d'**interprétabilité** et de **généralisation**, qui
constituent des enjeux majeurs pour toute application d'intelligence artificielle en santé.
    """
)
