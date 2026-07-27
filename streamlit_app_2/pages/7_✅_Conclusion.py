"""Page 7 — Conclusion générale (Sections 18-20 du compte rendu)."""

import streamlit as st

from common import page_header

# Configuration Streamlit : titre, icône et largeur de la page.
st.set_page_config(page_title="Conclusion — Radiographies COVID-19", page_icon="✅", layout="wide")
# En-tête commun : même présentation au début de chaque page.
page_header("✅", "Conclusion générale")

st.markdown(
    """
Ce projet répondait à la question suivante :

> *Dans quelle mesure les techniques modernes de Machine Learning et de Deep Learning
> permettent-elles d'identifier de manière fiable les atteintes pulmonaires liées à la COVID-19
> à partir de radiographies thoraciques, tout en conservant un niveau de généralisation
> compatible avec les exigences du domaine médical ?*
    """
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Meilleur ML classique", "SVC/SVM", "F1-macro 0.75")
c2.metric("Meilleur Deep Learning", "VGG16", "F1-macro 0.880")
c3.metric("Gain DL vs ML", "+13 pts", "F1-macro")
c4.metric("AUC COVID (VGG16)", "0.969")

st.subheader("Ce que montre le projet")
st.markdown(
    """
- **La structure spatiale de l'image est déterminante.** Le passage des pixels aplatis / HOG (ML
  classique) aux convolutions (Deep Learning) apporte un gain massif : +19 points d'accuracy et
  +36 points de F1 COVID entre le réseau Dense (Flatten) et le CNN personnalisé.
- **VGG16 a été retenu comme modèle final**, pour son équilibre entre performance globale
  (accuracy 88.6 %, Macro-F1 0.880), sensibilité COVID, stabilité d'apprentissage et lisibilité
  des cartes Grad-CAM.
- **Le prétraitement n'est pas universellement bénéfique.** Contrairement à l'attente initiale,
  CLAHE + Data Augmentation n'améliorent pas systématiquement les performances : effet positif
  sur le recall COVID pour certaines architectures (VGG16, CNN custom, EfficientNetB0), mais
  dégradation nette pour d'autres (DenseNet121, ResNet50). Chaque choix méthodologique doit être
  validé expérimentalement plutôt que supposé bénéfique.
- **La confusion COVID ↔ Lung_Opacity est structurelle**, pas un simple défaut technique : les
  deux pathologies partagent des signatures radiologiques (opacités diffuses, infiltrats
  bilatéraux) réellement proches sur une radiographie standard.
- **L'explicabilité (Grad-CAM) confirme la cohérence clinique** du modèle : l'attention se
  concentre sur les zones pulmonaires, pas sur des artefacts périphériques.
    """
)

with st.expander("Limites et pistes d'amélioration"):
    st.markdown(
        """
- Déséquilibre persistant du dataset (Normal 7,5× plus représentée que Viral Pneumonia)
- Proximité radiologique intrinsèque entre COVID-19 et Lung Opacity
- Absence de validation croisée sur les modèles Deep Learning (split fixe)
- Dépendance à la qualité des masques de segmentation pulmonaire fournis
- Résolution limitée à 128×128 pixels (contrainte de ressources), au prix d'une perte
  d'informations fines
- Absence d'évaluation sur des données externes (autre hôpital, autre appareil)

**Pistes futures :** ajustement du learning rate en fine-tuning, calibration des seuils de
décision, pondération renforcée de la classe COVID, validation sur des cohortes externes avant
tout usage clinique.
        """
    )

st.success(
    "**En résumé :** les techniques de Deep Learning, et en particulier VGG16 combiné à "
    "Grad-CAM, permettent une détection fiable et interprétable des atteintes pulmonaires "
    "liées à la COVID-19 — mais avec des limites claires (confusion COVID/Lung_Opacity, "
    "absence de validation externe) qui interdisent, en l'état, un déploiement clinique "
    "sans validation complémentaire."
)

st.caption(
    "Dataset : COVID-19 Radiography Database (Kaggle). "
    "Repository : github.com/mar25-iia2-radiographies/mar25_radiographies_pulmonaires_covid19"
)
