import streamlit as st
from common import apply_global_style
# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Exploration des données",
    page_icon="📊",
    layout="wide"
)

apply_global_style()

# ================= CSS ============================

st.markdown("""
<style>
            
.outlier-subtitle{
    font-size:19px;
    font-weight:600;
    color:#1f4ea3;
    text-align:center;
    margin-top:25px;
    margin-bottom:20px;
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
    padding:12px;
    border-radius:10px;
    margin-top:15px;
    margin-bottom:25px;
    font-size:15px;
    line-height:1.7;
}

.interpret-box-blue{
    background-color:#f8fbff;
    border:2px solid #8bb6ff;
    border-radius:15px;
    padding:18px;
    margin-top:20px;
    font-size:15px;
    line-height:1.7;
}

.interpret-box-red{
    background-color:#fff9f9;
    border:2px solid #f4a0a0;
    border-radius:15px;
    padding:18px;
    margin-top:20px;
    font-size:15px;
    line-height:1.7;
}

.conclusion-box{
    background-color:#faf8ff;
    border:2px solid #d7b8ff;
    border-radius:15px;
    padding:18px;
    margin-top:30px;
    margin-bottom:30px;
    font-size:15px;
    line-height:1.7;
}

</style>
""", unsafe_allow_html=True)

# ===================================================
# TITLE
# ===================================================
st.markdown("""
<div class="main-title">
📊 Exploration des données
</div>
""", unsafe_allow_html=True)

# ===================================================
# GENERAL INFORMATION
# ===================================================
st.markdown("""
<div class="info-box">
• <b>Nombre total d'images :</b> 21 165<br>
• <b>Taille des images :</b> 299 × 299 pixels<br>
• <b>Taille des masques :</b> 299 × 299 pixels<br>
• <b>99,3 % des images</b> sont en niveaux de gris (Grayscale).<br>
• <b>0,7 % des images</b> (140 images de la classe Viral Pneumonia) sont au format RGB.
</div>
""", unsafe_allow_html=True)

# ===================================================
# CLASS DISTRIBUTION
# ===================================================
st.markdown("""
<div class="section-title">
📈 Répartition des images par catégorie
</div>
""", unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([1, 3, 1])

with col_center:
    st.image(
        "images/class_distribution.png",
        width=600
    )

st.markdown("""
<div class="comment-box">
La classe <b>Normal</b> est la plus représentée (48,2 %),
tandis que la classe <b>Viral Pneumonia</b> est la moins représentée (6,4 %).
</div>
""", unsafe_allow_html=True)

# ===================================================
# GRAY HISTOGRAMS
# ===================================================
st.markdown("""
<div class="section-title">
📉 Analyse des histogrammes de niveaux de gris
</div>
""", unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([1, 4, 1])

with col_center:
    st.image(
        "images/gray_histogram.png",
        width=750
    )

st.markdown("""
<div class="comment-box">
Les distributions des niveaux de gris diffèrent légèrement entre les catégories,
indiquant la présence de caractéristiques visuelles discriminantes.
Cependant, ces différences restent insuffisantes pour une séparation directe des classes.
</div>
""", unsafe_allow_html=True)

# ===================================================
# OUTLIERS
# ===================================================
st.markdown("""
<div class="section-title">
📊 Identification des outliers basée sur la luminosité moyenne
</div>
""", unsafe_allow_html=True)

left_col, right_col = st.columns(2)

# ===================================================
# LEFT COLUMN
# ===================================================
with left_col:
    st.markdown("""
    <div class="outlier-subtitle">
    1. Distribution de la luminosité moyenne
    </div>
    """, unsafe_allow_html=True)

    img_col_left, img_col_center, img_col_right = st.columns([1, 5, 1])

    with img_col_center:
        st.image(
            "images/mean_luminosity_boxplot.png",
            width=500
        )

    st.markdown("""
    <div class="interpret-box-blue">
    📖 <b>Interprétation</b>
                
    Le boxplot montre la distribution de la luminosité moyenne
    pour chaque catégorie.<br>
    Les seuils de <b>75</b> et <b>175</b> ont été utilisés
    pour identifier les images exceptionnellement sombres
    ou exceptionnellement claires.
    </div>
    """, unsafe_allow_html=True)

# ===================================================
# RIGHT COLUMN
# ===================================================
with right_col:
    st.markdown("""
    <div class="outlier-subtitle">
    2. Visualisation des outliers
    </div>
    """, unsafe_allow_html=True)

    st.image(
        "images/outliers_scatter.png",
        use_container_width=True
    )

    st.markdown("""
    <div class="interpret-box-red">
    🔎 <b>Interprétation</b>
                
    Chaque point représente une image.<br>
    • Axe horizontal : luminosité moyenne.<br>
    • Axe vertical : écart-type de luminosité.
    Les points rouges correspondent aux images dont la luminosité moyenne est :<br>
    • inférieure à <b>75</b> (images très sombres)<br>
    • supérieure à <b>175</b> (images très claires).
    </div>
    """, unsafe_allow_html=True)

# ===================================================
# EXEMPLES D'IMAGES ATYPIQUES
# ===================================================

st.markdown("""
<div class="section-title">
📸 Exemples d'images atypiques
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="comment-box">
Quelques exemples d'images présentant une luminosité
anormalement élevée ou faible.
</div>
""", unsafe_allow_html=True)


# ---------- images claires ----------
st.markdown("""
<div class="outlier-subtitle">
Images très claires (mean > 175)
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,5,1])

with col2:
    st.image(
        "images/outliers_bright_examples.png",
        use_container_width=True
    )


# ---------- images sombres ----------
st.markdown("""
<div class="outlier-subtitle">
Images très sombres (mean < 75)
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,5,1])

with col2:
    st.image(
        "images/outliers_dark_examples.png",
        use_container_width=True
    )


# ---------- interprétation ----------
st.markdown("""
<div class="conclusion-box">

💡 Ces images ne sont pas nécessairement à supprimer.
Elles peuvent contenir des informations diagnostiques utiles,
mais présentent un problème d'exposition ou de contraste.<br>
Une correction de contraste peut donc être préférable
à une suppression systématique.

</div>
""", unsafe_allow_html=True)