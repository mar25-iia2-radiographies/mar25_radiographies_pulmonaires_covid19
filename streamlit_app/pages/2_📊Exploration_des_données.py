"""Page 2 — Exploration des données (Section 2 du compte rendu)."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (
    CLASS_COLORS,
    CLASS_COUNTS,
    LUMINOSITY_STATS,
    OUTLIER_THRESHOLDS,
    TOTAL_IMAGES,
    TTEST_RESULTS,
    page_header,
)

st.set_page_config(page_title="Exploration — Radiographies COVID-19", page_icon="📊", layout="wide")
page_header("📊", "Exploration des données")

st.markdown(
    f"""
Le dataset est téléchargé depuis Kaggle via l'API `kagglehub`, puis exploré récursivement.
Il contient **{TOTAL_IMAGES:,} radiographies** organisées en 4 catégories, chacune disposant de
deux sous-dossiers : `images/` (radiographies) et `masks/` (masques de segmentation pulmonaire
pré-calculés).
    """.replace(",", " ")
)

st.subheader("2.1 · Structure et répartition des classes")
c1, c2 = st.columns([1.2, 1])
with c1:
    fig = px.bar(
        CLASS_COUNTS.sort_values("Images"), x="Images", y="Catégorie", orientation="h",
        color="Catégorie", color_discrete_map=CLASS_COLORS, text="Images",
        height=350,
    )
    fig.update_layout(showlegend=False, xaxis_title="Nombre d'images", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.pie(
        CLASS_COUNTS, values="Images", names="Catégorie", color="Catégorie",
        color_discrete_map=CLASS_COLORS, height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

st.warning(
    "**Déséquilibre significatif :** la classe **Normal** (48,2 %) est **7,5 fois** plus "
    "représentée que **Viral Pneumonia** (6,4 %). Ce déséquilibre risque de biaiser "
    "l'apprentissage vers la classe majoritaire — il sera traité lors de la modélisation "
    "(pondération des classes, augmentation de données, sous-échantillonnage).",
    icon="⚠️",
)

st.subheader("2.2 · Vérification du format des images")
m1, m2, m3 = st.columns(3)
m1.metric("Images en niveaux de gris", "99,3 %")
m2.metric("Images en couleur (RGB)", "0,7 %", "140 images, Viral Pneumonia")
m3.metric("Dimension uniforme", "299 × 299 px")
st.caption(
    "L'uniformité des dimensions simplifie le pipeline : aucun redimensionnement préalable "
    "n'est nécessaire. Les 140 images en couleur sont converties en niveaux de gris au chargement."
)

st.subheader("2.3 · Histogrammes de niveaux de gris")
st.markdown(
    """
L'histogramme des niveaux de gris (0 = noir, 255 = blanc) caractérise le contraste, la
luminosité et la dynamique d'une image. Toutes les catégories présentent une **distribution
bimodale** typique des radiographies thoraciques : un premier pic vers les valeurs sombres
(fond, médiastin) et un second pic vers les valeurs moyennes/claires (parenchyme pulmonaire).

Les courbes sont globalement similaires entre catégories (protocoles d'acquisition homogènes),
avec un **décalage subtil** : les images COVID sont légèrement décalées vers les valeurs plus
claires — confirmé statistiquement en 2.6.
    """
)

st.subheader("2.4 · Luminosité par catégorie")
c1, c2 = st.columns([1, 1])
with c1:
    fig = px.bar(
        LUMINOSITY_STATS, x="Catégorie", y="Luminosité moyenne", color="Catégorie",
        color_discrete_map=CLASS_COLORS, text="Luminosité moyenne", height=350,
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_range=[0, 170])
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.markdown(
        """
**Trois métriques calculées par image :**
- **Luminosité moyenne** — les images COVID ont une moyenne plus élevée (**139,5**) que les
  autres catégories (125-129), différence statistiquement significative (cf. 2.6).
- **Écart-type (contraste)** — distributions similaires entre catégories, centrées ~55-60.
- **Luminosité maximale** — la plupart des images saturent proche de 255, avec quelques
  exceptions notables (< 200).
        """
    )

st.subheader("2.5 · Identification des outliers")
st.markdown(
    f"""
Les boxplots révèlent de nombreux points aberrants (outliers) — images dont la luminosité
moyenne s'écarte significativement de la distribution principale. Des seuils empiriques sont
retenus :
- **Outlier sombre** : luminosité moyenne < **{OUTLIER_THRESHOLDS['sombre']}**
- **Outlier clair** : luminosité moyenne > **{OUTLIER_THRESHOLDS['clair']}**
    """
)
st.info(
    "**Réflexion critique :** ces images ne sont pas nécessairement « mauvaises ». Elles "
    "contiennent potentiellement des informations diagnostiques valides mais souffrent d'un "
    "problème d'exposition à l'acquisition. La stratégie retenue est donc une **correction de "
    "contraste** plutôt qu'une exclusion systématique (détaillée page Prétraitement)."
)

st.subheader("2.6 · Test statistique t de Student — COVID vs Normal")
res = TTEST_RESULTS
c1, c2, c3 = st.columns(3)
c1.metric("COVID — moyenne (n=50)", f"{res['COVID']['mean']:.2f}", f"σ = {res['COVID']['std']:.2f}")
c2.metric("Normal — moyenne (n=50)", f"{res['Normal']['mean']:.2f}", f"σ = {res['Normal']['std']:.2f}")
c3.metric("p-value (test de Welch)", f"{res['p_value']:.4f}", f"T = {res['t_stat']:.2f}")

st.success(
    f"Avec une p-value = {res['p_value']:.4f} < 0,05, on **rejette H₀** : il existe une "
    "différence statistiquement significative entre les intensités moyennes des images COVID "
    f"et Normal (Δ ≈ {res['COVID']['mean'] - res['Normal']['mean']:.0f} unités de gris — les "
    "radiographies COVID sont en moyenne légèrement plus claires)."
)
st.caption(
    "**Nuance :** cette différence statistique ne garantit pas une classification parfaite. "
    "Elle suggère qu'un modèle pourrait exploiter cette caractéristique parmi d'autres — le "
    "contraste local et les patterns de texture restent les critères discriminants principaux."
)
