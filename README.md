# Kosi Atlas — Cartographie du risque d'inondation

Prédiction de la susceptibilité aux inondations sur le **Kosi Megafan** (Bihar, Inde), par ensemble de Machine Learning classique (sans Deep Learning).

## Modèle

- **Architecture** : Stacking (RandomForest + SVM + XGBoost + méta-LogisticRegression)
- **Features** : 10 facteurs bruts + 5 dérivés
- **Entraînement** : 40 000 points équilibrés (20k inondés / 20k secs)
- **AUC-ROC** : 0.927 (test équilibré) · **Rappel spatial** : 76.3 % (carte complète)

## Résultats

- Carte de risque à 100 m de résolution
- 4 niveaux : Très faible / Faible / Élevé / Très élevé
- Interface Streamlit interactive (carte cliquable + import CSV + saisie manuelle)

## Structure
projet_kosi/
├── step1_raster_to_table.ipynb # Raster → Tableau
├── step3_improved.ipynb # Modélisation (Tier 1)
├── step7_cartography_improved.ipynb # Cartographie
├── prediction.ipynb # Tests de prédiction
├── streamlit_app.py # Interface Streamlit
├── data/processed/ # Modèles .pkl + CSV
├── outputs/ # Cartes .tif + PNG
└── assets/ # Images hero