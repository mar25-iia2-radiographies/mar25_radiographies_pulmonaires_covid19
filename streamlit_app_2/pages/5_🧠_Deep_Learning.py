"""Page 5 — Modélisation Deep Learning (Sections 11-20 du compte rendu)."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (
    DATASET_SPLIT,
    DL_BASELINE,
    DL_CNN_CM,
    DL_RESULTS,
    DL_VGG_CM,
    ROC_AUC,
    page_header,
)

# Configuration Streamlit : titre, icône et largeur de la page.
st.set_page_config(page_title="Deep Learning — Radiographies COVID-19", page_icon="🧠", layout="wide")
# En-tête commun : même présentation au début de chaque page.
page_header("🧠", "Modélisation par Deep Learning")

st.markdown(
    """
Objectif : identifier l'architecture offrant le **meilleur compromis** entre performance globale
et détection fiable des cas COVID-19. Protocole expérimental commun (split fixe, prétraitement
identique) pour garantir une comparaison équitable entre 6 architectures : **Dense (baseline)**,
**CNN personnalisé**, **DenseNet121**, **EfficientNetB0**, **ResNet50**, **VGG16**.
    """
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Modèle retenu", "VGG16")
col2.metric("Accuracy (F/F)", "88.6 %")
col3.metric("Macro-F1 (F/F)", "0.880")
col4.metric("Recall COVID (T/T)", "0.825", "meilleur de l'étude")

st.subheader("Jeu de données — split fixe (80/20 stratifié + validation)")
st.dataframe(DATASET_SPLIT, use_container_width=True, hide_index=True)

with st.expander("Pipeline de prétraitement & gestion du déséquilibre"):
    st.markdown(
        """
**Prétraitement commun :** masque pulmonaire → suppression hors poumons → redimensionnement
→ conversion RGB (architectures pré-entraînées) → normalisation.

**Deux variantes testées par architecture :**
- **F/F** — Sans CLAHE, sans Data Augmentation
- **T/T** — Avec CLAHE (clipLimit=2.0) + Data Augmentation (rotations/translations/zooms/luminosité,
  amplitudes limitées pour rester anatomiquement plausibles)

**Déséquilibre des classes :** poids de classes (class weights) calculés sur le train, pénalisant
davantage les erreurs sur COVID-19 et Viral Pneumonia.

**Métrique principale :** Macro F1-score (poids égal à chaque classe), complétée par le Recall COVID.
        """
    )

st.subheader("De la baseline Dense au CNN personnalisé")
fig = go.Figure()
fig.add_bar(name="Accuracy", x=DL_BASELINE["Modèle"], y=DL_BASELINE["Accuracy"], marker_color="#93c5fd")
fig.add_bar(name="F1 COVID", x=DL_BASELINE["Modèle"], y=DL_BASELINE["F1 COVID"], marker_color="#E4572E")
fig.update_layout(barmode="group", height=380, yaxis_range=[0, 1])
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Le passage du réseau Dense (Flatten, aucune structure spatiale) au CNN personnalisé "
    "apporte +19 points d'accuracy et +36 points de F1 COVID — la préservation de "
    "l'organisation spatiale de l'image est déterminante en imagerie médicale."
)

st.subheader("Résultats comparatifs complets — 5 architectures × 2 variantes (CLAHE/Augmentation)")
st.dataframe(
    DL_RESULTS[["Modèle", "CLAHE", "Data Augmentation", "Accuracy", "Macro-F1", "F1-weighted", "Recall COVID", "F1 COVID"]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0.6, max_value=1, format="%.3f"),
        "Macro-F1": st.column_config.ProgressColumn("Macro-F1", min_value=0.6, max_value=1, format="%.3f"),
        "F1-weighted": st.column_config.NumberColumn("F1-weighted", format="%.3f"),
        "Recall COVID": st.column_config.ProgressColumn("Recall COVID", min_value=0.5, max_value=1, format="%.3f"),
        "F1 COVID": st.column_config.NumberColumn("F1 COVID", format="%.3f"),
    },
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Accuracy & Macro-F1 par variante")
    fig = go.Figure()
    fig.add_bar(name="Accuracy", x=DL_RESULTS["Variante"], y=DL_RESULTS["Accuracy"], marker_color="#93c5fd")
    fig.add_bar(name="Macro-F1", x=DL_RESULTS["Variante"], y=DL_RESULTS["Macro-F1"], marker_color="#1f77b4")
    fig.update_layout(barmode="group", xaxis_tickangle=-35, height=430, yaxis_range=[0.6, 0.95])
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.subheader("Compromis Recall COVID / Macro-F1")
    fig = px.scatter(
        DL_RESULTS, x="Recall COVID", y="Macro-F1", color="Modèle", symbol="CLAHE",
        text="Variante", height=430,
    )
    fig.update_traces(textposition="top center", marker=dict(size=13))
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
st.caption(
    "VGG16 (F/F) offre le meilleur équilibre global ; VGG16 (T/T, CLAHE+Augmentation) obtient "
    "le meilleur recall COVID (0.825) de toute l'étude, au prix d'un léger recul du Macro-F1."
)

st.subheader("Influence du CLAHE + Data Augmentation, architecture par architecture")
st.markdown(
    """
| Architecture | Sans CLAHE/Aug (F/F) | Avec CLAHE/Aug (T/T) | Effet observé |
|---|---|---|---|
| **CNN custom** | Acc 0.878 · Macro-F1 0.873 | Acc 0.875 · Macro-F1 0.872 | Gain global quasi nul, mais recall COVID ↑ (0.699 → 0.792) |
| **EfficientNetB0** | Acc 0.855 · Macro-F1 0.846 | Acc 0.827 · Macro-F1 0.821 | Recall COVID ↑ (0.745 → 0.780), cohérence multiclasse ↓ |
| **DenseNet121** | Acc 0.839 · Macro-F1 0.830 | Acc 0.787 · Macro-F1 0.769 | Dégradation nette, y compris sur COVID |
| **ResNet50** | Acc 0.859 · Macro-F1 0.855 | Acc 0.830 · Macro-F1 0.815 | Dégradation globale et du recall COVID |
| **VGG16** | Acc 0.886 · Macro-F1 0.880 | Acc 0.877 · Macro-F1 0.875 | Quasi stable, meilleur recall COVID (0.825) |

**Enseignement clé :** contrairement à l'intuition, CLAHE + augmentation n'améliorent pas
systématiquement les performances. L'effet est architecture-dépendant — d'où l'intérêt d'une
validation expérimentale systématique plutôt que d'un choix a priori.
    """
)

st.subheader("Stabilité de l'apprentissage & matrices de confusion")
st.markdown(
    "Les meilleures courbes de validation (val_loss la plus faible et la plus stable) sont "
    "celles de **VGG16 F/F** (≈0.322) et **CNN custom F/F** (≈0.326), signe d'une bonne "
    "généralisation. ResNet50 F/F montre un début de surapprentissage (écart train/val croissant)."
)

cm1, cm2 = st.columns(2)
with cm1:
    st.markdown("**CNN personnalisé (F/F)**")
    fig = px.imshow(DL_CNN_CM, text_auto=True, color_continuous_scale="Blues",
                     labels=dict(x="Prédit", y="Réel", color="N"))
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)
with cm2:
    st.markdown("**VGG16 (F/F) — modèle final**")
    fig = px.imshow(DL_VGG_CM, text_auto=True, color_continuous_scale="Blues",
                     labels=dict(x="Prédit", y="Réel", color="N"))
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Erreur la plus critique : **COVID → Normal** (faux négatif). Erreur la plus fréquente : "
    "**COVID ↔ Lung_Opacity** (opacités pulmonaires visuellement proches). Viral Pneumonia "
    "reste la classe la mieux reconnue par les deux modèles."
)

st.subheader("Analyse ROC / AUC")
fig = go.Figure()
for model, color in [("CNN custom", "#1f77b4"), ("VGG16", "#E4572E")]:
    fig.add_bar(name=model, x=ROC_AUC["Classe"], y=ROC_AUC[model], marker_color=color)
fig.update_layout(barmode="group", height=380, yaxis_range=[0.9, 1.0])
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "AUC COVID quasi identique entre les deux modèles (CNN custom = 0.9693 vs VGG16 = 0.9694). "
    "L'avantage de VGG16 ne vient donc pas d'un meilleur pouvoir discriminant intrinsèque, mais "
    "de performances globales légèrement supérieures et d'une meilleure homogénéité inter-classes."
)

st.success(
    "**Choix retenu : VGG16 (Scénario A — performance multiclasse globale).** "
    "Meilleur Macro-F1 (0.880) et F1-weighted (0.885) de l'étude, courbes d'apprentissage "
    "stables, cartes Grad-CAM lisibles. La variante CLAHE+Augmentation (Scénario B) reste "
    "disponible si la priorité clinique est la réduction des faux négatifs COVID (recall 0.825)."
)
