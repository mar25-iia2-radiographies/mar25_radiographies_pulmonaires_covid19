"""Page 3 — Prétraitement des images.

Cette page présente les choix de prétraitement utilisés avant la modélisation.
Elle reste volontairement simple : on utilise surtout les composants Streamlit
standards afin que le code soit facile à comprendre et à modifier.
"""

# Plotly sert uniquement pour le petit graphique de comparaison quantitative.
import plotly.express as px

# Streamlit sert à construire l'interface de la page.
import streamlit as st

# PIL permet de redimensionner les images du pipeline interactif.
from PIL import Image

# On importe les éléments communs pour éviter de répéter les tableaux et chemins.
from common import (
    PIPELINE_STEPS,
    PREPROCESSING_COMPARISON,
    image_path,
    page_header,
    show_image,
)


# --------------------------------------------------------------------------------------
# Configuration de la page
# --------------------------------------------------------------------------------------
# Cette configuration garde la même forme générale que la version enrichie 2.
st.set_page_config(page_title="Prétraitement — Radiographies COVID-19", page_icon="🧹", layout="wide")

# En-tête commun : même présentation au début de chaque page.
page_header("🧹", "Prétraitement des images")


# --------------------------------------------------------------------------------------
# Introduction de la page
# --------------------------------------------------------------------------------------
st.markdown(
    """
Fort des observations de la phase exploratoire, le prétraitement vise trois objectifs
complémentaires : **corriger** les anomalies de contraste des outliers sans les supprimer,
**homogénéiser** les distributions intra-classe pour faciliter l'apprentissage, et **focaliser**
l'analyse sur les régions anatomiquement pertinentes, c'est-à-dire principalement les poumons.
    """
)


# --------------------------------------------------------------------------------------
# 3.1 — Stratégie de traitement des outliers
# --------------------------------------------------------------------------------------
st.subheader("3.1 · Stratégie de traitement des outliers")

# Deux colonnes permettent de comparer visuellement les deux stratégies possibles.
c1, c2 = st.columns(2)
with c1:
    st.error(
        "**Suppression** — Exclure les images aberrantes ❌\n\n"
        "Cette solution est simple, mais elle peut entraîner une perte d'information "
        "potentiellement diagnostique."
    )
with c2:
    st.success(
        "**Correction (retenue)** — Égalisation d'histogramme ✅\n\n"
        "Cette solution conserve les images et tente de corriger les problèmes "
        "d'exposition ou de contraste."
    )


# --------------------------------------------------------------------------------------
# 3.2 — Égalisation globale de l'histogramme
# --------------------------------------------------------------------------------------
st.subheader("3.2 · Égalisation d'histogramme globale")
st.markdown(
    """
La fonction `cv2.equalizeHist()` d'OpenCV applique une égalisation globale. Elle calcule la
fonction de répartition cumulative de l'histogramme, puis l'utilise pour redistribuer les
intensités de pixels sur une plage plus large.

Cette méthode améliore souvent le contraste des images sombres, mais elle peut aussi
**sur-corriger** certaines images claires et créer des artefacts de saturation. C'est pour cette
raison qu'une approche locale, plus progressive, a ensuite été étudiée.
    """
)


# --------------------------------------------------------------------------------------
# 3.3 — CLAHE : amélioration locale du contraste
# --------------------------------------------------------------------------------------
st.subheader("3.3 · CLAHE : amélioration locale du contraste")
st.markdown(
    """
Le **CLAHE** (*Contrast Limited Adaptive Histogram Equalization*) est une méthode d'amélioration
locale du contraste. Contrairement à l'égalisation globale, le CLAHE découpe l'image en petites
zones, appelées tuiles, puis améliore le contraste localement dans chacune de ces zones.

Dans notre étude, le CLAHE est utilisé avec les paramètres classiques `clipLimit = 2.0` et
`tileGridSize = 8×8`. Le paramètre `clipLimit` limite l'amplification du contraste afin d'éviter
une augmentation excessive du bruit.

L'intérêt principal est de mieux faire ressortir certaines structures pulmonaires fines, tout en
conservant une image plus naturelle qu'avec une égalisation globale trop agressive.
    """
)

# Trois petits encadrés résument les avantages principaux de CLAHE.
adv1, adv2, adv3 = st.columns(3)
adv1.info("✅ Améliore le contraste local")
adv2.info("✅ Préserve mieux les détails anatomiques")
adv3.info("✅ Limite la sur-saturation grâce à `clipLimit`")

# Figure issue de la version enrichie : comparaison visuelle original / CLAHE.
show_image("Original_vs_CLAHE.png", "Comparaison : image originale et image après CLAHE")


# --------------------------------------------------------------------------------------
# 3.4 — Segmentation pulmonaire
# --------------------------------------------------------------------------------------
st.subheader("3.4 · Segmentation pulmonaire")
st.markdown(
    """
Le dataset fournit des **masques pulmonaires binaires** associés aux radiographies. Ces masques
permettent d'isoler la région des poumons et de réduire l'influence des zones non pertinentes :
fond de l'image, os, clavicules, diaphragme, cœur ou annotations éventuelles.

L'application du masque agit donc comme une réduction de bruit. Le modèle se concentre davantage
sur les tissus pulmonaires, c'est-à-dire la région anatomique la plus utile pour distinguer les
classes **COVID**, **Lung Opacity**, **Normal** et **Viral Pneumonia**.

Cette section corrige aussi la numérotation précédente : la segmentation pulmonaire est bien la
section **3.4**, tandis que le pipeline interactif devient la section **3.5**.
    """
)

# Figure issue de la version enrichie : image originale, masque et image masquée.
show_image("lung_segmentation_example.png", "Image originale — masque pulmonaire — image masquée")


# --------------------------------------------------------------------------------------
# 3.5 — Pipeline de prétraitement interactif
# --------------------------------------------------------------------------------------
st.subheader("3.5 · Pipeline de prétraitement interactif")
st.markdown(
    """
Cette partie reprend le pipeline interactif de la version enrichie. Elle permet de visualiser,
sur une même radiographie, l'effet de différentes étapes de prétraitement : image originale,
application du CLAHE, application du masque pulmonaire, puis combinaison **CLAHE + masque**.

L'objectif n'est pas de recalculer les images en direct, mais de montrer simplement les étapes
principales à l'aide d'exemples déjà générés.
    """
)

# L'utilisateur choisit les transformations qu'il veut afficher.
st.markdown("**Activer les transformations à afficher :**")
col1, col2, col3 = st.columns(3)
with col1:
    show_clahe = st.toggle("CLAHE", value=False, key="pretraitement_clahe")
with col2:
    show_mask = st.toggle("Masque pulmonaire", value=False, key="pretraitement_mask")
with col3:
    show_clahe_mask = st.toggle("CLAHE + Masque", value=False, key="pretraitement_clahe_mask")

# La carte "Originale" est toujours affichée.
cards = [
    {
        "title": "Originale",
        "text": "Radiographie de départ, avant les transformations principales.",
        "image": "original_image.png",
    }
]

# Les autres cartes sont ajoutées uniquement si l'utilisateur active les boutons.
if show_clahe:
    cards.append(
        {
            "title": "+ CLAHE",
            "text": "Amélioration locale du contraste.",
            "image": "clahe_image.png",
        }
    )

if show_mask:
    cards.append(
        {
            "title": "+ Masque",
            "text": "Isolation de la région pulmonaire.",
            "image": "masked_image.png",
        }
    )

if show_clahe_mask:
    cards.append(
        {
            "title": "+ CLAHE + Masque",
            "text": "Contraste amélioré puis focalisation sur les poumons.",
            "image": "clahe_masked_image.png",
        }
    )

# On crée autant de colonnes que de cartes, avec un maximum de 4 colonnes.
# Cette solution reste simple et lisible pour un niveau débutant.
columns = st.columns(len(cards))
for column, card in zip(columns, cards):
    with column:
        with st.container(border=True):
            st.markdown(f"**{card['title']}**")
            st.caption(card["text"])

            # On vérifie que l'image existe avant de l'ouvrir.
            path = image_path(card["image"])
            if path.exists():
                img = Image.open(path)
                img = img.resize((260, 260))
                st.image(img, use_container_width=True)
            else:
                st.warning(f"Image manquante : images/{card['image']}")


# --------------------------------------------------------------------------------------
# 3.6 — Comparaison quantitative des approches
# --------------------------------------------------------------------------------------
st.subheader("3.6 · Comparaison quantitative des 4 approches")
st.markdown(
    """
Pour choisir le pipeline le plus pertinent, quatre configurations ont été comparées sur un
échantillon de 800 images, soit 200 images par classe : **Baseline**, **CLAHE**, **Masque** et
**CLAHE + Masque**.

Deux indicateurs sont particulièrement utiles : la **variance intra-classe**, qui mesure
l'homogénéité des images d'une même classe, et la **p-value du test t COVID vs Normal**, qui donne
une indication de séparabilité statistique entre ces deux classes.
    """
)

# Copie du tableau pour éviter de modifier la constante originale du fichier common.py.
df = PREPROCESSING_COMPARISON.copy()

# Une colonne pour le tableau, une colonne pour le graphique.
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
    fig = px.bar(
        df,
        x="Approche",
        y="Variance intra-classe",
        color="Approche",
        height=340,
        color_discrete_sequence=["#94a3b8", "#3B6EA5", "#F3A712", "#17A398"],
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.success(
    "**CLAHE + Masque** obtient la variance intra-classe la plus faible (0,0041) et la p-value "
    "la plus significative (5,6×10⁻¹⁴) pour le test COVID vs Normal. Cette combinaison est donc "
    "retenue comme pipeline principal de prétraitement."
)


# --------------------------------------------------------------------------------------
# Pipeline retenu
# --------------------------------------------------------------------------------------
st.divider()
st.subheader("✅ Pipeline de prétraitement retenu")

# Affichage des étapes du pipeline final sous forme de liste numérotée.
for i, step in enumerate(PIPELINE_STEPS, start=1):
    st.markdown(f"**{i}.** {step}")

with st.expander("Synthèse de la démarche et réflexions clés"):
    st.markdown(
        """
- **Ne pas supprimer les outliers a priori** : la correction par CLAHE permet de récupérer des
  images exploitables qui auraient été perdues par une exclusion systématique.
- **Combiner rehaussement de contraste et segmentation** : le CLAHE homogénéise le contraste,
  tandis que le masque focalise l'analyse sur la région anatomiquement pertinente.
- **Limiter le bruit contextuel** : le masque réduit l'effet du fond, des os et des annotations.
- **Valider quantitativement les choix** : les métriques objectives évitent de se baser uniquement
  sur une inspection visuelle.
        """
    )


# --------------------------------------------------------------------------------------
# Stratégie de modélisation
# --------------------------------------------------------------------------------------
st.subheader("5 · Stratégie de modélisation")
st.markdown(
    """
Les modèles ML classiques et les CNN partagent des principes communs d'équilibrage des classes :

- **Poids de classes** : davantage d'importance est donnée aux classes minoritaires, notamment
  COVID-19 et Viral Pneumonia.
- **Data augmentation contrôlée** : les rotations, zooms et variations de contraste restent
  volontairement limités afin de ne pas déformer la géométrie pulmonaire.
- **CLAHE appliqué ponctuellement** : cette option est testée pour améliorer localement le
  contraste, mais son intérêt dépend de l'architecture utilisée.

Cette combinaison vise à réduire l'impact du déséquilibre des classes et à favoriser une meilleure
généralisation sur les données de validation et de test.
    """
)
