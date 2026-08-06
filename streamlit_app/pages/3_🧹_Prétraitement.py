import streamlit as st
import pandas as pd
from PIL import Image
from common import apply_global_style
# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Prétraitement",
    page_icon="🖌️",
    layout="wide"
)

apply_global_style()

# ================= CSS =========================

st.markdown("""
<style>
.pipeline-card-title{
    font-size:24px;
    font-weight:600;
    margin-bottom:10px;
}
.pipeline-title{
    min-height:80px;
    font-size:18px;
    font-weight:700;
}

.pipeline-text{
    min-height:55px;
    font-size:14px;
    line-height:1.5;
    overflow:hidden;
}

.info-box{
    background-color:#f6f8fc;
    padding:20px;
    border-radius:15px;
    margin-top:15px;
    margin-bottom:30px;
    font-size:16px;
    line-height:2;
}

.comment-box{
    background-color:#f6f8fc;
    padding:15px;
    border-radius:12px;
    margin-top:15px;
    margin-bottom:25px;
    font-size:15px;
    line-height:1.8;
}

.green-box{
    background:#f1faf3;
    border:1px solid #d5ead8;
    border-radius:15px;
    padding:20px;
    margin-top:30px;
    margin-bottom:20px;
    color:#35693d;
    font-size:15px;
    line-height:1.8;
}

div[data-testid="stVerticalBlockBorderWrapper"]{
    border-radius:18px;
    min-height:500px;
}

</style>
""", unsafe_allow_html=True)

# ===================================================
# TITLE
# ===================================================

st.markdown("""
<div class="main-title">
🖌️ Prétraitement des images
</div>
""", unsafe_allow_html=True)

# ===================================================
# INTRODUCTION
# ===================================================

st.markdown("""
<div class="info-box">
Le prétraitement a pour objectif d'homogénéiser les données d'entrée
et de faciliter l'apprentissage des modèles de Machine Learning
et de Deep Learning.
</div>
""", unsafe_allow_html=True)

# ===================================================
# TABLEAU RECAPITULATIF
# ===================================================

st.markdown("""
<div class="section-title">
📋 Techniques de prétraitement utilisées
</div>
""", unsafe_allow_html=True)

df = pd.DataFrame({
    "Technique": [
        "Resize",
        "Normalization",
        "Data Augmentation",
        "Class Weights",
        "Masques pulmonaires",
        "CLAHE"
    ],
    "Machine Learning": [
        "✅",
        "✅",
        "❌",
        "✅",
        "✅",
        "Optionnel"
    ],
    "Deep Learning": [
        "✅",
        "✅",
        "✅",
        "✅",
        "✅",
        "Optionnel"
    ]
})

col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True
    )

# ===================================================
# CLAHE
# ===================================================

st.markdown("""
<div class="section-title">
🔬 CLAHE : amélioration locale du contraste
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="comment-box">

Le CLAHE (Contrast Limited Adaptive Histogram Equalization)
a été étudié comme technique optionnelle d'amélioration du contraste.

Contrairement à l'égalisation globale de l'histogramme,
le CLAHE préserve davantage les structures anatomiques
et produit des résultats plus naturels.

</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    st.image(
        "images/Original_vs_CLAHE.png",
        caption="Comparaison : Image originale vs CLAHE",
        width="stretch"
    )

# ===================================================
# LUNG SEGMENTATION
# ===================================================

st.markdown("""
<div class="section-title">
🫁 Segmentation pulmonaire
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="comment-box">
Des masques pulmonaires binaires ont été utilisés afin d'isoler
uniquement la région des poumons avant l'entraînement des modèles.
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.image(
        "images/lung_segmentation_example.png",
        caption="Image originale – Masque pulmonaire – Image masquée",
        width="stretch"
    )

# ===================================================
# PIPELINE INTERACTIF
# ===================================================

st.markdown("""
<div class="section-title">
🫁 Pipeline de prétraitement interactif
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="comment-box">
Visualisez l'effet des différentes étapes de prétraitement appliquées à une même image.
</div>
""", unsafe_allow_html=True)

st.markdown("### Activer les transformations :")

col1, col2, col3 = st.columns(3)

with col1:
    show_clahe = st.toggle(
        "CLAHE",
        value=False,
        key="pretraitement_clahe"
    )

with col2:
    show_mask = st.toggle(
        "Masque pulmonaire",
        value=False,
        key="pretraitement_mask"
    )

with col3:
    show_clahe_mask = st.toggle(
        "CLAHE + Masque",
        value=False,
        key="pretraitement_clahe_mask"
    )

cards = [
    {
        "title": "Originale",
        "text": "Image redimensionnée et normalisée dans l'intervalle [0,1].",
        "image": "images/original_image.png"
    }
]

if show_clahe:
    cards.append({
        "title": "+ CLAHE",
        "text": "Amélioration locale du contraste.",
        "image": "images/clahe_image.png"
    })

if show_mask:
    cards.append({
        "title": "+ Masque",
        "text": "Application du masque pulmonaire.",
        "image": "images/masked_image.png"
    })

if show_clahe_mask:
    cards.append({
        "title": "+ CLAHE+Masque",
        "text": "Masque pulmonaire après CLAHE.",
        "image": "images/clahe_masked_image.png"
    })

n_cards = len(cards)

if n_cards == 1:
    columns = st.columns([3, 2, 3])
    display_columns = [columns[1]]

elif n_cards == 2:
    columns = st.columns([1, 2, 2, 1])
    display_columns = [columns[1], columns[2]]

elif n_cards == 3:
    columns = st.columns([0.5, 2, 2, 2, 0.5])
    display_columns = [columns[1], columns[2], columns[3]]

else:
    display_columns = st.columns(4)

for column, card in zip(display_columns, cards):

    with column:
        with st.container(border=True):

            st.markdown(
                f"""
                <div class="pipeline-card-title">
                    {card['title']}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="pipeline-text">
                    {card['text']}
                </div>
                """,
                unsafe_allow_html=True
            )

            img = Image.open(card["image"])
            img = img.resize((260, 260))

            st.image(
                img,
                width="stretch"
            )
st.markdown('</div>', unsafe_allow_html=True)
