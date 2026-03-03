# Data-scoring

## 📊 Évaluation du Risque de Crédit avec Machine Learning

Un projet complet d'analyse et de notation du risque de crédit utilisant des techniques avancées de machine learning et de science des données.

## 🎯 Vue d'ensemble

Ce projet développe un **modèle prédictif de scoring du risque de crédit** capable d'évaluer la probabilité de défaut de paiement d'un client. Le modèle permet aux institutions financières de prendre des décisions d'octroi de crédit plus éclairées et réduire les risques d'impayés.

## ✨ Caractéristiques principales

- **Analyse exploratoire complète** des données de crédit
- **Nettoyage et préparation** des données (gestion des valeurs manquantes, normalisation)
- **Ingénierie des features** pour améliorer la performance du modèle
- **Entraînement de multiples modèles** (Régression logistique, Random Forest, Gradient Boosting, etc.)
- **Évaluation et validation** avec plusieurs métriques (AUC-ROC, Précision, Recall, F1-Score)
- **Visualisations détaillées** pour l'interprétabilité des résultats
- **Génération automatisée** du notebook d'analyse

## 📁 Structure du projet

```
Data-scoring/
├── Credit_Risk_Scoring.ipynb          # Notebook principal d'analyse et modélisation
├── generate_notebook.py                # Script Python pour générer/mettre à jour le notebook
├── credit_risk_dataset.csv             # Dataset contenant les données de crédit
└── README.md                           # Ce fichier
```

## 📈 Dataset

- **Nom** : credit_risk_dataset.csv
- **Taille** : ~1.8 MB
- **Contenu** : Données historiques de demandes de crédit avec variables démographiques, financières et de crédit
- **Variables cibles** : Status de remboursement (défaut ou non)

## 🔬 Méthodologie

1. **Exploration des données** : Analyse statistique descriptive et visualisations
2. **Prétraitement** : Nettoyage, gestion des anomalies et normalisation
3. **Feature Engineering** : Création et sélection des variables pertinentes
4. **Modélisation** : Entraînement et comparaison de plusieurs algorithmes
5. **Validation** : Évaluation sur ensemble de test avec validation croisée
6. **Interprétabilité** : Analyse de l'importance des features et des prédictions

## 🚀 Installation

### Prérequis
- Python 3.8+
- pip ou conda

### Étapes

```bash
# Cloner le repository
git clone https://github.com/mickePY/Data-scoring.git
cd Data-scoring

# Installer les dépendances
pip install pandas numpy scikit-learn matplotlib seaborn jupyter notebook

# Lancer Jupyter
jupyter notebook
```

## 💻 Utilisation

### Exécuter l'analyse complète

1. Ouvrez `Credit_Risk_Scoring.ipynb` dans Jupyter Notebook
2. Exécutez les cellules séquentiellement pour :
   - Charger et explorer les données
   - Entraîner les modèles
   - Obtenir les prédictions et métriques

### Générer le notebook automatiquement

```bash
python generate_notebook.py
```

Ce script génère automatiquement le notebook avec les analyses et les résultats.

## 📊 Résultats

Le modèle atteint :
- **AUC-ROC** : Excellente discrimination entre les clients à risque et sans risque
- **Précision & Recall** : Bon équilibre entre faux positifs et faux négatifs
- **F1-Score** : Performance globale optimale pour l'application

## 🛠️ Technologies utilisées

- **Python** : Langage principal
- **Pandas** : Manipulation de données
- **NumPy** : Calculs numériques
- **Scikit-learn** : Machine Learning
- **Matplotlib & Seaborn** : Visualisations
- **Jupyter** : Notebooks interactifs

## 📝 Licence

Ce projet est disponible sous licence libre. N'hésitez pas à l'utiliser et à le modifier.

## 👤 Auteur

**mickePY** - Développement complet du projet de scoring du risque de crédit

---

**Dernière mise à jour** : Mars 2026