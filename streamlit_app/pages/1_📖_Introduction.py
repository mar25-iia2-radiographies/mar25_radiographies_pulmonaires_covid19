import streamlit as st
from common import apply_global_style

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Introduction",
    page_icon="📖",
    layout="wide"
)

apply_global_style()

# ================= CSS =================

st.markdown("""
<style>

.class-box{
    width:200px;
    margin-top:8px;
    text-align:center;
    font-size:20px;
    font-weight:bold;
    padding:12px;
    border-radius:12px;
    box-sizing:border-box;
}

</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown(
    '<div class="main-title">📖 Introduction</div>',
    unsafe_allow_html=True
)

# ================= OBJECTIVE =================
st.markdown(
    '<div class="section-title">🎯 Objectif</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="text">
    Classification automatique des radiographies thoraciques
    à l'aide de techniques de Machine Learning et de Deep Learning.
    </div>
    """,
    unsafe_allow_html=True
)

# ================= DATASET =================
st.markdown(
    '<div class="section-title">📁 Base de données</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="text">
    COVID-19 Radiography Database (Kaggle).
    </div>
    """,
    unsafe_allow_html=True
)

# ================= CLASSES =================
st.markdown(
    '<div class="section-title">📌 Classes</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

# -------- COVID --------
with col1:
    st.image("images/COVID-1002.png",width=200)
    st.markdown(
        """
        <div class="class-box"
        style="background:#e8f3e8;color:#2e7d32;">
        COVID-19
        </div>
        """,
        unsafe_allow_html=True
    )

# -------- Lung Opacity --------
with col2:
    st.image("images/Lung_Opacity-10.png", width=200)
    st.markdown(
        """
        <div class="class-box"
        style="background:#fdf4d7;color:#a36b00;">
        Lung Opacity
        </div>
        """,
        unsafe_allow_html=True
    )

# -------- Viral Pneumonia --------
with col3:
    st.image("images/Viral Pneumonia-1007.png",width=200)
    st.markdown(
        """
        <div class="class-box"
        style="background:#e8eefc;color:#2952a3;">
        Viral Pneumonia
        </div>
        """,
        unsafe_allow_html=True
    )

# -------- Normal --------
with col4:
    st.image("images/Normal-10008.png",width=200)
    st.markdown(
        """
        <div class="class-box"
        style="background:#e8f3e8;color:#2e7d32;">
        Normal
        </div>
        """,
        unsafe_allow_html=True
    )