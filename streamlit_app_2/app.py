"""
app.py — Page d'accueil
Classification de Radiographies Pulmonaires (COVID-19)

Lancer avec : streamlit run app.py
Les autres sections sont dans le dossier pages/.

arborescence attendue à côté de ce fichier :
  app.py
  common.py
  pages/
    1_📖_Introduction.py
    2_📊_Exploration_des_données.py
    3_🧹_Prétraitement.py
    4_📈_Machine_Learning.py
    5_🧠_Deep_Learning.py
    6_🔥_Interprétabilité_Grad-CAM.py
    7_✅_Conclusion.py
    8_🩻_Démonstration_externe.py

"""

import streamlit as st

# Les constantes ci-dessous viennent de common.py pour éviter de les répéter.
from common import CLASSES, GITHUB_URL, PROJECT_SUBTITLE, PROJECT_TITLE, TEAM, TOTAL_IMAGES

# Configuration globale de la page d’accueil.
st.set_page_config(page_title="Radiographies pulmonaires — COVID-19", page_icon="🫁", layout="wide")

st.title(f"🫁 {PROJECT_TITLE}")
st.caption(PROJECT_SUBTITLE)
st.markdown("**Équipe :** " + " · ".join(TEAM))
st.divider()

st.markdown(
    f"""
Ce dashboard restitue le projet de classification automatique de radiographies thoraciques à
partir du **COVID-19 Radiography Database** — **{TOTAL_IMAGES:,}** radiographies réparties en
**4 catégories diagnostiques** : {", ".join(CLASSES)}.
    """.replace(",", " ")
)

st.markdown(
    """
> **Question de recherche :** dans quelle mesure les techniques modernes de Machine Learning et
> de Deep Learning permettent-elles d'identifier de manière fiable les atteintes pulmonaires
> liées à la COVID-19 à partir de radiographies thoraciques, tout en conservant un niveau de
> généralisation compatible avec les exigences du domaine médical ?
    """
)

st.subheader("📑 Sommaire")
st.markdown("Clique sur une section ci-dessous, ou utilise le menu à gauche pour naviguer :")

# Chaque tuple : (icône, titre affiché, description, chemin du fichier de page)
PAGES = [
    ("📖", "Introduction", "Contexte du projet, problématique, démarche en trois phases.",
     "pages/1_📖_Introduction.py"),
    ("📊", "Exploration des données", "Structure du dataset, histogrammes, outliers, test statistique.",
     "pages/2_📊_Exploration_des_données.py"),
    ("🧹", "Prétraitement", "CLAHE, masques pulmonaires, comparaison quantitative, pipeline retenu.",
     "pages/3_🧹_Prétraitement.py"),
    ("📈", "Machine Learning", "6 familles de modèles classiques, SVM/PCA meilleur modèle (F1-macro 0.75).",
     "pages/4_📈_Machine_Learning.py"),
    ("🧠", "Deep Learning", "6 architectures comparées, VGG16 retenu (F1-macro 0.880).",
     "pages/5_🧠_Deep_Learning.py"),
    ("🔥", "Interprétabilité (Grad-CAM)", "Explicabilité des prédictions avec les cartes Grad-CAM réelles.",
     "pages/6_🔥_Interprétabilité_Grad-CAM.py"),
    ("🩻", "Démonstration externe", "Tester une radiographie externe avec le modèle VGG16 si le fichier .keras est disponible.",
     "pages/8_🩻_Démonstration_externe.py"),
    ("✅", "Conclusion", "Synthèse, limites et pistes d'amélioration.",
     "pages/7_✅_Conclusion.py"),
]

# On affiche les pages sous forme de cartes, deux cartes par ligne.
cols = st.columns(2)
for i, (icon, name, desc, path) in enumerate(PAGES):
    with cols[i % 2]:
        with st.container(border=True):
            try:
                st.page_link(path, label=f"{icon} {name}", use_container_width=True)
            except Exception:
                # Repli si st.page_link indisponible (Streamlit < 1.31) ou fichier introuvable :
                # au moins afficher le titre, avec un rappel du chemin attendu.
                st.markdown(f"**{icon} {name}**")
                st.caption(f"⚠️ Lien indisponible — vérifie que `{path}` existe bien à côté de app.py.")
            st.caption(desc)

st.divider()
st.caption(f"Dataset : COVID-19 Radiography Database (Kaggle) · Repository : {GITHUB_URL}")
