import streamlit as st
from pathlib import Path
from common import apply_global_style

st.set_page_config(
    page_title="Classification de Radiographies Pulmonaires",
    page_icon="🫁",
    layout="wide"
)

apply_global_style()

# ================= CSS =================

st.markdown("""
<style>

section[data-testid="stSidebar"] ul li div {
    font-size: 18px !important;
    font-weight: 500;
}

section[data-testid="stSidebar"] ul li {
    margin-bottom: 8px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 0rem;
}

.main-title {
    text-align: center;
    color: #0B4CC2;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}

.subtitle {
    text-align: center;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 18px;
}

.authors-title {
    text-align: center;
    color: #0B4CC2;
    font-size: 22px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 8px;
}

.authors {
    text-align: center;
    font-size: 18px;
    line-height: 1.45;
    margin-bottom: 12px;
}

.formation {
    text-align: center;
    color: #0B4CC2;
    font-size: 22px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 8px;
}

</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown(
    '<div class="main-title">Classification de Radiographies Pulmonaires</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Projet I.A. : Modélisation Machine Learning & Deep Learning</div>',
    unsafe_allow_html=True
)

BASE = Path(__file__).parent

st.write("BASE =", BASE)
st.write("Images folder exists :", (BASE / "images").exists())
st.write("Normal image exists :", (BASE / "images" / "Normal-10008.png").exists())

# ================= IMAGE =================
col1, col2, col3 = st.columns([2.3, 1.4, 2.3])

with col2:
    st.image(BASE / "images" / "Normal-10008.png")

# ================= AUTHORS =================
st.markdown(
    '<div class="authors-title">👥 Réalisé par :</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="authors">
        Houssein ABBOUCHI<br>
        Romuald CROCHAT<br>
        Mathilde L'HOMMELET<br>
        Sareh MAHMOUDIAN MOGHADDAM
    </div>
    """,
    unsafe_allow_html=True
)

# ================= FORMATION =================
st.markdown(
    '<div class="formation">📅 Formation Liora – Juillet 2026</div>',
    unsafe_allow_html=True
)

# ================= LIORA LOGO =================
logo_col1, logo_col2, logo_col3 = st.columns([2.4, 1.2, 2.4])

with logo_col2:
    st.image("images/OIP.jpg", width=150,)