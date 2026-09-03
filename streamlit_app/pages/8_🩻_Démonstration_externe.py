"""
Page 8 — Démonstration externe.

Objectif : charger une radiographie qui n'appartient pas au train, à la validation
ou au test, puis appliquer le même type de prétraitement avant une prédiction.

Modification demandée : ajouter la possibilité de charger aussi une image de masque
pulmonaire si l'utilisateur en possède une.

Important : la prédiction réelle fonctionne seulement si le modèle Keras entraîné
est placé dans le dossier models/.
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from common import (
    apply_global_style,
    CLASS_COLORS,
    default_model_path,
    input_fingerprint,
    load_keras_model,
    model_status_message,
    predict_probabilities,
    preprocess_external_image,
)


# Configuration de la page.
st.set_page_config(page_title="Démonstration externe", page_icon="🩻", layout="wide")

# Application du style commun de l'application.
apply_global_style()

# Titre principal.
st.markdown(
    """
    <div class="main-title">
    🩻 Démonstration externe
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="text">
    Tester une radiographie externe au dataset avec le modèle VGG16 retenu,
    si le fichier du modèle est disponible.
    </div>
    """,
    unsafe_allow_html=True,
)

# Message de prudence indispensable pour une application médicale expérimentale.
st.warning(
    "Cette page est destinée à la démonstration pédagogique. Elle ne fournit pas de diagnostic médical.",
    icon="⚠️",
)

# On vérifie si le fichier du modèle est présent dans le dossier models/.
model_found, model_message = model_status_message()

# Explication simple pour préparer le modèle.
with st.expander("Préparer le modèle pour activer la prédiction réelle", expanded=not model_found):
    st.markdown(
        f"""
Pour activer cette page, il faut placer le modèle entraîné ici :

```text
models/final_vgg16_fixed_splits.keras
```

Chemin actuellement attendu :

```text
{default_model_path()}
```

Puis installer les dépendances optionnelles :

```bash
pip install -r requirements-inference.txt
```
        """
    )

# Affichage clair de l'état du modèle.
if model_found:
    st.success(model_message)
else:
    st.info(model_message)

st.divider()

# -----------------------------------------------------------------------------
# 1. Chargement de la radiographie et options de prétraitement.
# -----------------------------------------------------------------------------
st.subheader("1. Charger une radiographie externe")

col_upload, col_options = st.columns([1.3, 1])

with col_upload:
    # Fichier principal : la radiographie à tester.
    uploaded_file = st.file_uploader(
        "Radiographie externe",
        type=["png", "jpg", "jpeg"],
        help="Image hors jeu d'entraînement, hors validation et hors test.",
    )

with col_options:
    # Option CLAHE : désactivée par défaut pour rester cohérent avec le VGG16 F/F retenu.
    apply_clahe = st.checkbox(
        "Appliquer CLAHE",
        value=False,
        help="À laisser désactivé pour la variante VGG16 principale sans CLAHE ni augmentation.",
    )

    # Nouvelle option : l'utilisateur peut indiquer qu'il possède un masque pulmonaire.
    has_mask = st.checkbox(
        "J'ai un masque pulmonaire pour cette image",
        value=False,
        help="À cocher seulement si vous avez une image de masque associée à la radiographie.",
    )

    # Confirmation pédagogique : l'image doit être externe au dataset d'étude.
    confirm_external = st.checkbox("Je confirme que cette image est externe au dataset.")

# Si l'utilisateur coche la case masque, on affiche un second chargeur de fichier.
mask_file = None
if has_mask:
    st.markdown("**Masque pulmonaire associé**")
    mask_file = st.file_uploader(
        "Image de masque pulmonaire",
        type=["png", "jpg", "jpeg"],
        help="Le masque doit correspondre à la radiographie chargée. Blanc = poumons, noir = fond.",
    )

# Si aucune radiographie n'est chargée, on arrête la page ici.
if uploaded_file is None:
    st.info("Charge une radiographie externe pour afficher le prétraitement et, si possible, la prédiction.")
    st.stop()

# Lecture de la radiographie chargée.
try:
    image_bytes = uploaded_file.getvalue()
    image = Image.open(uploaded_file)
except Exception as error:
    st.error(f"Impossible de lire la radiographie : {error}")
    st.stop()

# Lecture du masque si l'utilisateur en a fourni un.
mask_bytes = None
mask_image = None
apply_mask = False

if has_mask:
    if mask_file is None:
        # L'utilisateur a coché la case, mais n'a pas encore fourni le fichier masque.
        st.warning(
            "La case masque est cochée, mais aucune image de masque n'a été chargée. "
            "Le prétraitement sera affiché sans masque pour le moment."
        )
    else:
        try:
            mask_bytes = mask_file.getvalue()
            mask_image = Image.open(mask_file)
            apply_mask = True
        except Exception as error:
            st.error(f"Impossible de lire le masque pulmonaire : {error}")
            st.stop()

# Application du prétraitement défini dans common.py.
try:
    batch, stages = preprocess_external_image(
        image,
        apply_clahe=apply_clahe,
        apply_mask=apply_mask,
        mask_image=mask_image,
    )
except Exception as error:
    st.error(f"Erreur pendant le prétraitement : {error}")
    st.stop()

st.divider()
st.subheader("2. Contrôle visuel du prétraitement")

# On affiche les étapes principales pour que l'utilisateur voie ce que l'application fait.
# Si un masque est présent, on ajoute deux images : le masque et le résultat masqué.
if apply_mask:
    cols = st.columns(5)
else:
    cols = st.columns(3)

with cols[0]:
    st.image(stages["gray"], caption="Image en niveaux de gris", width="stretch")

with cols[1]:
    caption = "Après CLAHE" if apply_clahe else "Sans CLAHE"
    st.image(stages["processed"], caption=caption, width="stretch")

if apply_mask:
    with cols[2]:
        st.image(stages["mask"], caption="Masque chargé", width="stretch")
    with cols[3]:
        st.image(stages["masked"], caption="Image après masque", width="stretch")
    with cols[4]:
        st.image(stages["resized"], caption="Entrée modèle", width="stretch")
else:
    with cols[2]:
        st.image(stages["resized"], caption="Entrée modèle redimensionnée", width="stretch")

# Petite explication selon le cas.
if apply_mask:
    st.info(
        "Le masque a été appliqué avant le redimensionnement final. "
        "Les zones noires du masque sont supprimées et les zones blanches sont conservées."
    )
else:
    st.caption("Aucun masque n'est appliqué. La prédiction utilise seulement la radiographie prétraitée.")

st.divider()
st.subheader("3. Prédiction du modèle")

# La prédiction réelle est possible seulement si le modèle existe et si l'utilisateur confirme l'image externe.
can_predict = model_found and confirm_external

if not model_found:
    st.warning("Le modèle n'est pas encore présent. La page affiche donc seulement le prétraitement.")
elif not confirm_external:
    st.info("Coche la case de confirmation pour lancer la prédiction réelle.")

# Signature de l'entrée : image + options + masque éventuel.
# Cela évite d'afficher une ancienne prédiction si l'utilisateur change un fichier ou une option.
fingerprint = input_fingerprint(
    image_bytes,
    apply_clahe=apply_clahe,
    apply_mask=apply_mask,
    mask_bytes=mask_bytes,
)

if st.session_state.get("prediction_fingerprint") != fingerprint:
    st.session_state.pop("prediction_probabilities", None)
    st.session_state.pop("prediction_fingerprint", None)

# Bouton de prédiction.
if can_predict and st.button("Lancer la prédiction", type="primary", width="stretch"):
    try:
        with st.spinner("Chargement du modèle et calcul de la prédiction..."):
            model = load_keras_model(str(default_model_path()))
            probabilities = predict_probabilities(model, batch)

        # On mémorise les probabilités pour ne pas les perdre à chaque rafraîchissement Streamlit.
        st.session_state["prediction_probabilities"] = probabilities
        st.session_state["prediction_fingerprint"] = fingerprint

    except Exception as error:
        st.error(f"La prédiction a échoué : {error}")

# Si une prédiction est disponible, on l'affiche.
probabilities = st.session_state.get("prediction_probabilities")
if probabilities and st.session_state.get("prediction_fingerprint") == fingerprint:
    # Construction d'un tableau trié par probabilité décroissante.
    prob_df = pd.DataFrame(
        {
            "Classe": list(probabilities.keys()),
            "Probabilité": list(probabilities.values()),
        }
    ).sort_values("Probabilité", ascending=False)

    predicted_class = prob_df.iloc[0]["Classe"]
    confidence = float(prob_df.iloc[0]["Probabilité"])

    # Résumé numérique à gauche, graphique à droite.
    left, right = st.columns([1, 2])
    with left:
        st.metric("Classe prédite", predicted_class)
        st.metric("Confiance", f"{confidence:.1%}")
        st.caption("La confiance softmax n'est pas une certitude médicale.")

    with right:
        fig = px.bar(
            prob_df.sort_values("Probabilité"),
            x="Probabilité",
            y="Classe",
            orientation="h",
            color="Classe",
            color_discrete_map=CLASS_COLORS,
            text="Probabilité",
            title="Probabilités par classe",
        )
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig.update_layout(height=360, xaxis_range=[0, 1.05], showlegend=False)
        st.plotly_chart(fig, width="stretch")

# Note finale pour cadrer l'usage.
st.caption(
    "Cette page prépare l'intégration du modèle réel. Sans fichier .keras dans models/, aucune inférence réelle n'est exécutée."
)
