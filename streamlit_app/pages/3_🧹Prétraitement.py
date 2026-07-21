"""Page 3 — Prétraitement (Sections 3, 4 et 5 du compte rendu)."""

import plotly.express as px
import streamlit as st

from common import PIPELINE_STEPS, PREPROCESSING_COMPARISON, page_header

st.set_page_config(page_title="Prétraitement — Radiographies COVID-19", page_icon="🧹", layout="wide")
page_header("🧹", "Prétraitement des images")

st.markdown(
    """
Fort des observations de la phase exploratoire, le prétraitement vise trois objectifs
complémentaires : **corriger** les anomalies de contraste des outliers sans les supprimer,
**homogénéiser** les distributions intra-classe pour faciliter l'apprentissage, et **focaliser**
l'analyse sur les régions anatomiquement pertinentes (poumons).
    """
)

st.subheader("3.1 · Stratégie de traitement des outliers")
c1, c2 = st.columns(2)
with c1:
    st.error("**Suppression** — Exclure les images aberrantes ❌\n\nPerte d'information potentiellement diagnostique.")
with c2:
    st.success("**Correction (retenue)** — Égalisation d'histogramme ✅\n\nRedistribue les intensités sur [0-255], conserve l'information.")

st.subheader("3.2 · Égalisation d'histogramme globale")
st.markdown(
    """
`cv2.equalizeHist()` calcule la fonction de répartition cumulative (CDF) de l'histogramme et
l'utilise pour redistribuer les intensités. **Observation :** améliore significativement le
contraste des images sombres, mais **sur-corrige** les images claires (artefacts de saturation).
Une approche plus fine est nécessaire.
    """
)

st.subheader("3.3 · CLAHE — égalisation adaptative")
st.markdown(
    """
**CLAHE** (Contrast Limited Adaptive Histogram Equalization) divise l'image en tuiles et applique
l'égalisation localement, avec un paramètre `clipLimit` qui limite l'amplification du contraste
(évite le bruit).
    """
)
adv1, adv2, adv3 = st.columns(3)
adv1.info("✅ Préserve les détails locaux (structures pulmonaires fines)")
adv2.info("✅ Évite la sur-saturation des zones claires")
adv3.info("✅ Contrôle fin via le paramètre `clipLimit`")

st.subheader("3.4 – 3.5 · Masques de segmentation pulmonaire")
st.markdown(
    """
Le dataset fournit des **masques de segmentation pré-calculés** pour chaque radiographie,
délimitant précisément la zone des poumons. Leur application élimine les contributions du fond
(pixels très sombres) et des structures osseuses (pixels très clairs) : l'histogramme obtenu
est plus concentré et caractéristique de l'état des tissus pulmonaires, agissant comme une
**réduction de bruit** en supprimant les régions non diagnostiques (clavicules, cœur,
diaphragme, annotations textuelles).
    """
)

st.subheader("3.6 · Comparaison quantitative des 4 approches")
st.markdown(
    "Comparaison systématique sur un échantillon de 800 images (200 par classe), avec deux "
    "métriques : la **variance intra-classe** (homogénéité, plus faible = mieux) et la "
    "**p-value du test t COVID vs Normal** (séparabilité, plus faible = mieux)."
)

df = PREPROCESSING_COMPARISON.copy()
c1, c2 = st.columns([1, 1.2])
with c1:
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Variance intra-classe": st.column_config.NumberColumn(format="%.4f"),
            "Std inter-classe": st.column_config.NumberColumn(format="%.3f"),
            "p-value (t-test)": st.column_config.NumberColumn(format="%.2e"),
        },
    )
with c2:
    fig = px.bar(df, x="Approche", y="Variance intra-classe", color="Approche", height=340,
                 color_discrete_sequence=["#94a3b8", "#3B6EA5", "#F3A712", "#17A398"])
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.success(
    "**CLAHE + Masque** obtient la variance intra-classe la plus faible (0,0041) et la p-value "
    "la plus significative (5,6×10⁻¹⁴) pour le test COVID vs Normal — la **meilleure "
    "séparabilité**. L'effet est synergique : CLAHE seul réduit déjà fortement la variance, le "
    "masque seul l'augmente légèrement, mais combinés l'effet est positif."
)

st.divider()
st.subheader("✅ Pipeline de prétraitement retenu")
for i, step in enumerate(PIPELINE_STEPS, start=1):
    st.markdown(f"**{i}.** {step}")

with st.expander("Synthèse de la démarche et réflexions clés"):
    st.markdown(
        """
- **Ne pas supprimer les outliers a priori** — la correction par CLAHE récupère des images
  exploitables qui auraient été perdues par une exclusion systématique.
- **Combiner rehaussement de contraste et segmentation** — CLAHE homogénéise le contraste,
  le masque focalise l'analyse sur la région anatomiquement pertinente.
- **Le masque élimine le bruit contextuel** — en supprimant fond, os et annotations, le modèle
  se concentre sur les patterns pathologiques réels.
- **La validation quantitative guide les choix** — variance intra-classe et test t permettent
  d'éviter des décisions subjectives basées uniquement sur l'inspection visuelle.
        """
    )

st.subheader("5 · Stratégie de modélisation")
st.markdown(
    """
Les modèles ML classiques et les CNN partagent des principes communs d'équilibrage des classes :

- **Poids de classes (class weights)** — davantage d'importance aux classes minoritaires
  (COVID-19, Viral Pneumonia) lors de l'apprentissage.
- **Data augmentation contrôlée** — rotations et zooms d'amplitude volontairement limitée pour
  rester compatibles avec la posture réelle des patients, sans déformer la géométrie pulmonaire.
- **CLAHE appliqué ponctuellement** pour améliorer localement le contraste.

Cette combinaison vise à réduire l'impact du déséquilibre des classes et à favoriser une
meilleure généralisation sur les données de validation et de test.
    """
)
