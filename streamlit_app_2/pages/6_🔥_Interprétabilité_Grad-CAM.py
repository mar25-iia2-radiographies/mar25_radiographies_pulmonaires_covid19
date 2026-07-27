"""
Page 6 — Interprétabilité Grad-CAM.

Cette page remplace la démonstration simulée par des images Grad-CAM réelles
copiées depuis la version « enrichie ».

Le style reste volontairement simple : composants Streamlit classiques,
pas de CSS personnalisé, et commentaires pour un niveau débutant.
"""

# streamlit permet de construire la page web.
import streamlit as st

# Fonctions communes : en-tête de page et affichage d'images.
from common import page_header, show_image


# Configuration de la page dans Streamlit.
st.set_page_config(page_title="Grad-CAM — Radiographies COVID-19", page_icon="🔥", layout="wide")

# En-tête commun à toutes les pages.
page_header("🔥", "Interprétabilité des prédictions — Grad-CAM")

# Introduction courte : pourquoi l'interprétabilité est importante.
st.markdown(
    """
Dans un contexte médical, la performance du modèle ne suffit pas. Il faut aussi vérifier
que le réseau regarde les **zones pulmonaires pertinentes** et non des éléments parasites
comme les bords, le fond, les annotations ou les artefacts d'acquisition.

La méthode **Grad-CAM** produit une carte de chaleur qui indique les régions de l'image
ayant le plus contribué à la prédiction du modèle.
    """
)

# Message de prudence : Grad-CAM aide à comprendre le modèle, mais ne remplace pas un diagnostic.
st.warning(
    "Grad-CAM est un outil d'interprétation du modèle. Il ne constitue pas une preuve clinique "
    "ni une localisation médicale validée des lésions.",
    icon="⚠️",
)

st.divider()

# Première figure globale issue de l'application enrichie.
st.subheader("1. Exemples sélectionnés sur le jeu de test")
st.markdown(
    """
La figure ci-dessous présente des exemples correctement et incorrectement classés.
Elle permet de visualiser rapidement les cas simples et les cas de confusion du modèle VGG16.
    """
)
show_image("vgg16_selected_examples.png", "Exemples corrects et incorrects par classe")

st.divider()

# Sélection de la classe à analyser.
st.subheader("2. Cartes Grad-CAM du modèle VGG16 retenu")

# Liste des classes disponibles. L'ordre correspond aux fichiers image copiés.
class_names = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]

# L'utilisateur choisit la classe à visualiser.
selected_class = st.selectbox("Classe à examiner", class_names)

# On récupère la position de la classe dans la liste.
index = class_names.index(selected_class)

# Pour chaque classe, deux images existent : un cas correct et un cas incorrect.
# Exemple : COVID -> vgg16_gradcam_ff_1.png et vgg16_gradcam_ff_2.png.
correct_image = f"vgg16_gradcam_ff_{2 * index + 1}.png"
incorrect_image = f"vgg16_gradcam_ff_{2 * index + 2}.png"

# Affichage en deux colonnes pour comparer rapidement les deux cas.
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Cas correctement classé**")
    show_image(correct_image, f"{selected_class} — prédiction correcte")

with col2:
    st.markdown("**Cas mal classé**")
    show_image(incorrect_image, f"{selected_class} — prédiction incorrecte")

# Aide à la lecture pour la présentation orale.
st.markdown(
    """
**Lecture attendue :**

- une carte pertinente doit se concentrer principalement sur les champs pulmonaires ;
- une attention forte sur les bords ou le fond peut indiquer un biais ;
- les cas mal classés aident à comprendre les confusions, notamment entre **COVID** et **Lung_Opacity**.
    """
)

st.divider()

# Bloc optionnel pour comparer avec la variante CLAHE + augmentation.
with st.expander("Comparer avec la variante VGG16 + CLAHE + augmentation"):
    st.markdown(
        """
Cette variante n'est pas le modèle principal retenu pour le meilleur Macro-F1,
mais elle obtient le meilleur rappel COVID. La comparaison des cartes Grad-CAM
permet d'observer si l'attention du modèle reste cohérente.
        """
    )

    # Même logique que précédemment, mais avec les fichiers tt au lieu de ff.
    correct_tt = f"vgg16_gradcam_tt_{2 * index + 1}.png"
    incorrect_tt = f"vgg16_gradcam_tt_{2 * index + 2}.png"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cas correctement classé — variante CLAHE/augmentation**")
        show_image(correct_tt, f"{selected_class} — variante CLAHE/augmentation, cas correct")
    with c2:
        st.markdown("**Cas mal classé — variante CLAHE/augmentation**")
        show_image(incorrect_tt, f"{selected_class} — variante CLAHE/augmentation, cas incorrect")

# Note finale pour rappeler l'origine des figures.
st.caption(
    "Figures reprises de la version enrichie : VGG16 sans CLAHE/augmentation et VGG16 avec CLAHE/augmentation."
)
