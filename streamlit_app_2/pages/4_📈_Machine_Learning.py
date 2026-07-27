"""Page 4 — Modélisation Machine Learning classique (Sections 6-10 du compte rendu)."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import ML_CONF_MATRIX, ML_COVID_RECALL_PRECISION, ML_F1_BY_CLASS, ML_RESULTS, CLASSES, page_header

# Configuration Streamlit : titre, icône et largeur de la page.
st.set_page_config(page_title="Machine Learning — Radiographies COVID-19", page_icon="📈", layout="wide")
# En-tête commun : même présentation au début de chaque page.
page_header("📈", "Modélisation par Machine Learning classique")

st.markdown(
    """
Six familles de modèles ont été testées sur des features extraites après le pipeline de
prétraitement retenu (CLAHE + masque pulmonaire + normalisation). Deux stratégies d'extraction
de caractéristiques sont comparées : **pixels aplatis + PCA(200)** et **descripteurs HOG**.
La métrique principale est le **F1-macro**, complétée par le **recall COVID** — l'enjeu clinique
central de ce projet.
    """
)

col1, col2, col3 = st.columns(3)
col1.metric("Meilleur modèle ML", "SVC (SVM) + PCA", "F1-macro = 0.75")
col2.metric("Meilleur recall COVID (ML)", "0.63", "LR (HOG)")
col3.metric("Classe la plus difficile", "COVID", "F1 entre 0.33 et 0.56")

st.subheader("Tableau comparatif des modèles")
st.dataframe(
    ML_RESULTS,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=1, format="%.2f"),
        "F1-macro": st.column_config.ProgressColumn("F1-macro", min_value=0, max_value=1, format="%.2f"),
        "Recall COVID": st.column_config.ProgressColumn("Recall COVID", min_value=0, max_value=1, format="%.2f"),
        "CV F1-macro": st.column_config.NumberColumn("CV F1-macro", format="%.3f"),
    },
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("F1-macro vs Accuracy (test)")
    fig = go.Figure()
    fig.add_bar(name="F1-macro", x=ML_RESULTS["Modèle"], y=ML_RESULTS["F1-macro"], marker_color="#1f77b4")
    fig.add_bar(name="Accuracy", x=ML_RESULTS["Modèle"], y=ML_RESULTS["Accuracy"], marker_color="#93c5fd")
    fig.update_layout(barmode="group", xaxis_tickangle=-35, yaxis_range=[0, 0.85], height=430)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.subheader("Recall vs Precision — classe COVID")
    fig = go.Figure()
    fig.add_bar(name="Recall COVID", x=ML_COVID_RECALL_PRECISION["Modèle"], y=ML_COVID_RECALL_PRECISION["Recall COVID"], marker_color="#E4572E")
    fig.add_bar(name="Precision COVID", x=ML_COVID_RECALL_PRECISION["Modèle"], y=ML_COVID_RECALL_PRECISION["Precision COVID"], marker_color="#f4a582")
    fig.update_layout(barmode="group", xaxis_tickangle=-35, yaxis_range=[0, 0.85], height=430)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("F1-score par classe et par modèle")
fig = px.imshow(
    ML_F1_BY_CLASS,
    text_auto=".2f",
    color_continuous_scale="RdYlGn",
    aspect="auto",
    zmin=0.3, zmax=0.9,
)
fig.update_layout(height=420)
st.plotly_chart(fig, use_container_width=True)
st.caption("La classe COVID reste systématiquement en retrait (F1 entre 0.33 et 0.56), loin derrière Normal et Viral Pneumonia.")

st.subheader("Matrice de confusion — meilleur modèle ML (SVC/SVM + PCA, 128×128)")
cm_col, txt_col = st.columns([1.3, 1])
with cm_col:
    fig = px.imshow(
        ML_CONF_MATRIX,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Prédiction", y="Réalité", color="N"),
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
with txt_col:
    st.markdown(
        """
**Lecture de la matrice :**

- La confusion dominante est bilatérale entre **COVID** et **Lung_Opacity** (166 + 162 cas).
- Le modèle manque encore près de **40 %** des cas COVID (recall = 0.62).
- Viral Pneumonia est très bien séparée des autres classes.

Cette confusion structurelle COVID ↔ Lung_Opacity motive le passage au **Deep Learning**,
capable de capturer des patterns plus fins que les features de bas niveau (pixels/HOG).
        """
    )

with st.expander("Détail des 6 familles de modèles testées"):
    st.markdown(
        """
- **Régression Logistique** — baseline sur pixels bruts (non convergée, F1-macro 0.61), fortement
  améliorée avec HOG (+11 points, F1-macro 0.72). SMOTE n'apporte qu'un gain marginal.
- **Random Forest** — limité par la destruction de la structure spatiale (pixels aplatis) et par
  l'insuffisance des features HOG pour capturer la complexité des motifs COVID (F1-macro ≈ 0.60–0.62).
- **KNN + PCA(200)** — F1-macro 0.63 ; souffre sur COVID et Lung_Opacity, classes proches dans
  l'espace PCA.
- **SVC (SVM) à noyau RBF + PCA(200)** — **meilleur modèle ML classique** (F1-macro 0.75,
  accuracy 0.75), grâce à sa capacité à modéliser des frontières non linéaires.
- **Gradient Boosting (HistGradientBoosting)** — F1-macro 0.69, early stopping à 118 itérations.
- **XGBoost** — F1-macro 0.72 ; meilleure précision COVID (0.61) mais recall plus faible (0.40),
  modèle le plus « prudent ».
        """
    )

st.info(
    "**Conclusion Partie ML :** le SVC (SVM) + PCA(200) est le meilleur modèle classique "
    "(F1-macro = 0.75). L'extraction de features adaptée (HOG, PCA) est déterminante. "
    "Aucun modèle ML classique ne dépasse un F1 COVID de 0.56 — la confusion avec "
    "Lung_Opacity est structurelle et motive le recours au Deep Learning."
)
