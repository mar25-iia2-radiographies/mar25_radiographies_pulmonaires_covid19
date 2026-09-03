"""
Page 5 — Modélisation Deep Learning.

Cette page présente les résultats des modèles de Deep Learning testés.
Les modifications de l'étape actuelle concernent surtout deux points :
1. ajouter les courbes de validation dans la partie stabilité de l'apprentissage ;
2. remplacer la partie ROC/AUC par une lecture plus simple des performances par classe.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (
    apply_global_style,
    DATASET_SPLIT,
    DL_BASELINE,
    DL_CNN_CM,
    DL_RESULTS,
    DL_VGG_CM,
    show_image,
)


# Configuration générale de la page Streamlit.
st.set_page_config(
    page_title="Deep Learning — Radiographies COVID-19",
    page_icon="🧠",
    layout="wide",
)

# Application du style commun de l'application.
apply_global_style()

# Titre principal de la page.
st.markdown(
    """
    <div class="main-title">
    🧠 Modélisation par Deep Learning
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
Objectif : identifier l'architecture offrant le **meilleur compromis** entre performance globale
et détection fiable des cas COVID-19. Protocole expérimental commun (split fixe, prétraitement
identique) pour garantir une comparaison équitable entre 6 architectures : **Dense (baseline)**,
**CNN personnalisé**, **DenseNet121**, **EfficientNetB0**, **ResNet50**, **VGG16**.
    """
)

# Résumé très court des résultats principaux.
col1, col2, col3, col4 = st.columns(4)
col1.metric("Modèle retenu", "VGG16")
col2.metric("Accuracy (F/F)", "88.6 %")
col3.metric("Macro-F1 (F/F)", "0.880")
col4.metric("Recall COVID (T/T)", "0.825", "meilleur de l'étude")

# Présentation du split utilisé pour comparer les modèles.
st.subheader("Jeu de données — split fixe")
st.dataframe(DATASET_SPLIT, width="stretch", hide_index=True)

with st.expander("Pipeline de prétraitement & gestion du déséquilibre"):
    st.markdown(
        """
**Prétraitement commun :** masque pulmonaire → suppression hors poumons → redimensionnement
→ conversion RGB pour les architectures pré-entraînées → normalisation.

**Deux variantes testées par architecture :**
- **F/F** — Sans CLAHE, sans Data Augmentation
- **T/T** — Avec CLAHE + Data Augmentation

**Déséquilibre des classes :** des poids de classes sont utilisés pour pénaliser davantage
les erreurs sur les classes minoritaires.

**Métrique principale :** Macro-F1, car elle donne le même poids à chaque classe.
        """
    )

# Comparaison simple entre une baseline Dense et des modèles CNN.
st.subheader("De la baseline Dense au CNN personnalisé")
fig = go.Figure()
fig.add_bar(name="Accuracy", x=DL_BASELINE["Modèle"], y=DL_BASELINE["Accuracy"], marker_color="#93c5fd")
fig.add_bar(name="F1 COVID", x=DL_BASELINE["Modèle"], y=DL_BASELINE["F1 COVID"], marker_color="#E4572E")
fig.update_layout(barmode="group", height=380, yaxis_range=[0, 1])
st.plotly_chart(fig, width="stretch")
st.caption(
    "Le passage d'un réseau Dense à un CNN améliore fortement les résultats, car le CNN conserve "
    "l'organisation spatiale de l'image. C'est important pour reconnaître les structures pulmonaires."
)

# Tableau complet des architectures testées.
st.subheader("Résultats comparatifs complets — 5 architectures × 2 variantes")
st.dataframe(
    DL_RESULTS[["Modèle", "CLAHE", "Data Augmentation", "Accuracy", "Macro-F1", "F1-weighted", "Recall COVID", "F1 COVID"]],
    width="stretch",
    hide_index=True,
    column_config={
        "Accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0.6, max_value=1, format="%.3f"),
        "Macro-F1": st.column_config.ProgressColumn("Macro-F1", min_value=0.6, max_value=1, format="%.3f"),
        "F1-weighted": st.column_config.NumberColumn("F1-weighted", format="%.3f"),
        "Recall COVID": st.column_config.ProgressColumn("Recall COVID", min_value=0.5, max_value=1, format="%.3f"),
        "F1 COVID": st.column_config.NumberColumn("F1 COVID", format="%.3f"),
    },
)

# Deux graphiques de synthèse : performance globale et compromis COVID.
c1, c2 = st.columns(2)
with c1:
    st.subheader("Accuracy & Macro-F1 par variante")
    fig = go.Figure()
    fig.add_bar(name="Accuracy", x=DL_RESULTS["Variante"], y=DL_RESULTS["Accuracy"], marker_color="#93c5fd")
    fig.add_bar(name="Macro-F1", x=DL_RESULTS["Variante"], y=DL_RESULTS["Macro-F1"], marker_color="#1f77b4")
    fig.update_layout(barmode="group", xaxis_tickangle=-35, height=430, yaxis_range=[0.6, 0.95])
    st.plotly_chart(fig, width="stretch")

with c2:
    st.subheader("Compromis Recall COVID / Macro-F1")
    fig = px.scatter(
        DL_RESULTS,
        x="Recall COVID",
        y="Macro-F1",
        color="Modèle",
        symbol="CLAHE",
        text="Variante",
        height=430,
    )
    fig.update_traces(textposition="top center", marker=dict(size=13))
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, width="stretch")

st.caption(
    "VGG16 sans CLAHE ni augmentation offre le meilleur équilibre global. "
    "VGG16 avec CLAHE et augmentation obtient le meilleur rappel COVID."
)

# Lecture de l'effet du CLAHE et de la data augmentation.
st.subheader("Influence du CLAHE + Data Augmentation")
st.markdown(
    """
| Architecture | Sans CLAHE/Augmentation | Avec CLAHE/Augmentation | Effet observé |
|---|---|---|---|
| **CNN custom** | Acc 0.878 · Macro-F1 0.873 | Acc 0.875 · Macro-F1 0.872 | Résultat global stable, recall COVID meilleur |
| **EfficientNetB0** | Acc 0.855 · Macro-F1 0.846 | Acc 0.827 · Macro-F1 0.821 | Recall COVID meilleur, mais performance globale plus faible |
| **DenseNet121** | Acc 0.839 · Macro-F1 0.830 | Acc 0.787 · Macro-F1 0.769 | Dégradation nette |
| **ResNet50** | Acc 0.859 · Macro-F1 0.855 | Acc 0.830 · Macro-F1 0.815 | Dégradation globale |
| **VGG16** | Acc 0.886 · Macro-F1 0.880 | Acc 0.877 · Macro-F1 0.875 | Très stable, meilleur recall COVID |

**Conclusion :** CLAHE + augmentation ne sont pas automatiquement meilleurs. Leur intérêt dépend
de l'architecture et de l'objectif : performance globale ou priorité au rappel COVID.
    """
)

# -----------------------------------------------------------------------------
# Nouvelle version de la section stabilité.
# On ajoute ici les courbes de validation, comme demandé.
# -----------------------------------------------------------------------------
st.subheader("Stabilité de l'apprentissage & matrices de confusion")
st.markdown(
    """
Les courbes d'apprentissage permettent de vérifier si le modèle apprend correctement.
On regarde principalement :

- la **loss d'entraînement** : erreur mesurée sur les données d'apprentissage ;
- la **loss de validation** : erreur mesurée sur des données non utilisées pour ajuster les poids ;
- l'**accuracy de validation** : performance sur des données que le modèle ne voit pas directement pendant l'apprentissage.

Dans le cas du modèle VGG16 retenu, la validation reste globalement cohérente avec l'entraînement.
Cela montre que le modèle généralise correctement, même s'il existe naturellement un petit écart
entre train et validation.
    """
)

# Figure réelle des courbes de validation VGG16 présente dans le dossier images/.
show_image(
    "vgg16_learning_curves.png",
    caption="Courbes d'apprentissage du modèle VGG16 : loss et accuracy en entraînement/validation.",
)

st.info(
    "Lecture simple : si la courbe de validation s'améliore puis se stabilise, le modèle apprend. "
    "Si la courbe d'entraînement continue à s'améliorer alors que la validation se dégrade fortement, "
    "on suspecte du surapprentissage."
)

# Matrices de confusion : comparaison CNN personnalisé / VGG16.
cm1, cm2 = st.columns(2)
with cm1:
    st.markdown("**CNN personnalisé (F/F)**")
    fig = px.imshow(
        DL_CNN_CM,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Prédit", y="Réel", color="N"),
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")

with cm2:
    st.markdown("**VGG16 (F/F) — modèle final**")
    fig = px.imshow(
        DL_VGG_CM,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Prédit", y="Réel", color="N"),
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")

st.caption(
    "Les confusions les plus importantes concernent COVID et Lung_Opacity, car ces deux classes "
    "peuvent présenter des opacités pulmonaires visuellement proches."
)

# -----------------------------------------------------------------------------
# Remplacement de la partie ROC/AUC.
# On présente les résultats de façon plus directe : précision, rappel et F1-score.
# -----------------------------------------------------------------------------
st.subheader("Lecture simplifiée des performances par classe")
st.markdown(
    """
La courbe ROC/AUC est utile techniquement, mais elle n'est pas toujours intuitive à présenter.
Ici, on remplace cette lecture par trois métriques plus faciles à expliquer :

- **Précision** : quand le modèle prédit une classe, à quel point cette prédiction est fiable ;
- **Rappel** : parmi les vraies images de cette classe, combien sont retrouvées ;
- **F1-score** : compromis entre précision et rappel.
    """
)

# Classification report du modèle VGG16 retenu.
vgg16_report = pd.DataFrame(
    {
        "Classe": ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"],
        "Précision": [0.819, 0.885, 0.922, 0.939],
        "Rappel": [0.810, 0.956, 0.808, 0.916],
        "F1-score": [0.814, 0.919, 0.861, 0.927],
        "Support": [542, 1529, 902, 202],
    }
)

st.dataframe(
    vgg16_report,
    width="stretch",
    hide_index=True,
    column_config={
        "Précision": st.column_config.ProgressColumn("Précision", min_value=0, max_value=1, format="%.3f"),
        "Rappel": st.column_config.ProgressColumn("Rappel", min_value=0, max_value=1, format="%.3f"),
        "F1-score": st.column_config.ProgressColumn("F1-score", min_value=0, max_value=1, format="%.3f"),
    },
)

# Transformation du tableau en format long pour créer un graphique groupé.
report_long = vgg16_report.melt(
    id_vars="Classe",
    value_vars=["Précision", "Rappel", "F1-score"],
    var_name="Métrique",
    value_name="Score",
)

fig = px.bar(
    report_long,
    x="Classe",
    y="Score",
    color="Métrique",
    barmode="group",
    text="Score",
    height=430,
    title="Performances du modèle VGG16 par classe",
)
fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig.update_layout(yaxis_range=[0, 1.05])
st.plotly_chart(fig, width="stretch")

st.info(
    "Lecture pour la présentation : le modèle reconnaît très bien Normal et Viral Pneumonia. "
    "La classe COVID reste correcte, mais elle est plus difficile, surtout à cause de sa confusion "
    "avec Lung_Opacity."
)

# Conclusion de la page Deep Learning.
st.success(
    "**Choix retenu : VGG16 sans CLAHE ni augmentation.** "
    "C'est le meilleur compromis global de l'étude : Accuracy 0.886, Macro-F1 0.880 et F1-weighted 0.885. "
    "La variante VGG16 avec CLAHE + augmentation reste intéressante si l'objectif prioritaire est "
    "d'augmenter le rappel COVID."
)
