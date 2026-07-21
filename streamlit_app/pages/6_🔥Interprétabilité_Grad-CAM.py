"""Page 6 — Interprétabilité des prédictions / Grad-CAM (Section 17 du compte rendu)."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageFilter, ImageOps

from common import CLASS_COLORS, page_header

st.set_page_config(page_title="Grad-CAM — Radiographies COVID-19", page_icon="🔥", layout="wide")
page_header("🔥", "Interprétabilité des prédictions — Grad-CAM")

st.markdown(
    """
Dans un contexte médical, la performance seule ne suffit pas : il faut pouvoir **expliquer**
une décision. **Grad-CAM** (Gradient-weighted Class Activation Mapping) génère une carte
thermique mettant en évidence les régions de l'image ayant le plus influencé la décision du
réseau — permettant de vérifier que le modèle se concentre bien sur les zones pulmonaires
pertinentes plutôt que sur des artefacts.
    """
)

st.markdown(
    """
**Constat du rapport :** le modèle (CNN personnalisé comme VGG16) concentre l'essentiel de son
attention sur les zones pulmonaires, réduisant le risque d'appui sur des artefacts périphériques.
Les erreurs restantes concernent principalement la confusion **COVID / Lung_Opacity**, cohérente
avec la proximité radiologique réelle de ces deux pathologies (opacités diffuses, infiltrats
bilatéraux).
    """
)

st.divider()
st.subheader("🎛️ Démonstration interactive")
st.warning(
    "⚠️ **Mode démonstration.** Cette interface simule le rendu visuel d'une carte Grad-CAM "
    "à des fins de présentation (soutenance). Elle ne fait **pas** appel au modèle VGG16 "
    "réellement entraîné — la carte de chaleur et la prédiction affichées sont générées de "
    "façon illustrative à partir de l'image fournie, pas d'une inférence réelle.",
    icon="⚠️",
)

demo_col, param_col = st.columns([2, 1])
with param_col:
    backbone = st.selectbox("Architecture (démo)", ["VGG16 (modèle final)", "CNN personnalisé"])
    uploaded = st.file_uploader("Charger une radiographie thoracique", type=["png", "jpg", "jpeg"])
    run = st.button("🔍 Lancer l'analyse (démo)", type="primary", use_container_width=True)

with demo_col:
    if uploaded is not None:
        img = Image.open(uploaded).convert("L").resize((320, 320))
        if run:
            # --- Génération d'une carte de chaleur illustrative (non un vrai Grad-CAM) ---
            blurred = np.asarray(img.filter(ImageFilter.GaussianBlur(18))).astype(np.float32) / 255.0
            # zones "mi-tons" = approx tissu pulmonaire -> activation simulée plus forte
            activation = np.exp(-((blurred - 0.5) ** 2) / (2 * 0.18 ** 2))
            activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)

            heat = (activation * 255).astype(np.uint8)
            heat_img = Image.fromarray(heat).convert("L")
            heat_colored = ImageOps.colorize(heat_img, black="#000064", white="#ff2b00", mid="#ffd400")
            overlay = Image.blend(img.convert("RGB"), heat_colored, alpha=0.5)

            rng = np.random.default_rng(abs(hash(uploaded.name)) % (2**32))
            classes_demo = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
            probs = rng.dirichlet(alpha=[3, 3, 2, 1])
            pred_idx = int(np.argmax(probs))

            o1, o2 = st.columns(2)
            o1.image(img, caption="Radiographie chargée", use_container_width=True)
            o2.image(overlay, caption=f"Carte Grad-CAM simulée — {backbone}", use_container_width=True)

            st.markdown(f"**Classe prédite (simulée) : `{classes_demo[pred_idx]}`** — confiance {probs[pred_idx]*100:.1f} %")
            prob_df = pd.DataFrame({"Classe": classes_demo, "Probabilité": probs})
            fig = px.bar(prob_df, x="Classe", y="Probabilité", color="Classe",
                         color_discrete_map=CLASS_COLORS, height=300)
            fig.update_layout(showlegend=False, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.image(img, caption="Image chargée — clique sur « Lancer l'analyse (démo) »", use_container_width=True)
    else:
        st.info("Charge une radiographie (PNG/JPG) puis clique sur « Lancer l'analyse (démo) » pour voir le rendu de l'interface.")

st.caption(
    "💡 Pour une vraie démo en direct, il suffira de brancher ici le modèle VGG16 sauvegardé "
    "(`model.h5` / `.keras`) et une fonction Grad-CAM réelle sur la couche `block5_conv3` — "
    "l'interface (upload, affichage, barre de probabilités) est déjà prête à les recevoir."
)

st.divider()
st.subheader("📎 Exemples issus du compte rendu")
st.markdown("Quatre cas illustratifs analysés dans le rapport (couche Grad-CAM `block5_conv3` pour VGG16) :")

ex1, ex2 = st.columns(2)
with ex1:
    st.markdown("**✅ CNN personnalisé — COVID correctement classifié**")
    st.markdown("`Vrai = COVID` · `Prédit = COVID` · confiance = 1.000")
    st.markdown("**❌ CNN personnalisé — COVID mal classifié**")
    st.markdown("`Vrai = COVID` · `Prédit = Lung_Opacity` · confiance = 0.997")
with ex2:
    st.markdown("**✅ VGG16 — COVID correctement classifié**")
    st.markdown("`Vrai = COVID` · `Prédit = COVID` · confiance = 1.000")
    st.markdown("**❌ VGG16 — COVID mal classifié**")
    st.markdown("`Vrai = COVID` · `Prédit = Normal` · confiance = 0.991")

st.caption(
    "Les figures originales (heatmaps réelles) sont disponibles dans le compte rendu, section 17 "
    "(pages 52-54). Dépose-les dans un dossier `assets/` et remplace ce bloc par `st.image(...)` "
    "pour les afficher directement ici."
)
