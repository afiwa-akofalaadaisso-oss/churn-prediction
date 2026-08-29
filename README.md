# Prédiction du Churn Client

Projet de machine learning de bout en bout visant à prédire si un client va résilier son abonnement (churn), à partir de son comportement et de ses caractéristiques.

## Objectif

Identifier en amont les clients à risque de résiliation, afin de permettre à l'entreprise d'agir avant qu'ils ne partent.

##  Démarche

- **Compréhension métier** : dictionnaire de données, définition du problème
- **Nettoyage des données** : gestion des doublons, erreurs de type, incohérences catégorielles, valeurs aberrantes
- **EDA (analyse exploratoire)** : tests statistiques (Chi², skewness), analyse univariée et bivariée, matrice de corrélation
- **Feature engineering** : 7 variables créées (ratios, écarts, interactions, variable temporelle) — ex. `charges_totales_par_mois`, `ecart_charge_mensuelle_vs_historique`, `engagement_x_reclamations`
- **Preprocessing** : Pipeline scikit-learn strict (normalisation robuste via `RobustScaler`, encodage `OneHotEncoder`, sélection de variables `SelectKBest`) — sans fuite train/test
- **Modélisation** : comparaison de Logistic Regression, Random Forest et XGBoost, avec gestion du déséquilibre de classes (SMOTE, class_weight)
- **Tuning** : optimisation des hyperparamètres par validation croisée stratifiée
- **Interprétabilité** : analyse SHAP (importance globale et impact par variable)
- **Déploiement local** : sauvegarde du pipeline complet avec Joblib, prêt à être rechargé pour prédiction sur de nouvelles données

##  Résultats

| Métrique | Valeur |
| F1-score | ≈ 0.36 |
| ROC AUC | ≈ 0.57 |

Le résultat reste modéré. L'analyse montre que ce n'est pas un problème de méthodologie mais de **signal disponible dans les données** : peu de variables sont réellement discriminantes, et le bruit résiduel limite la performance quel que soit le modèle utilisé. Cette conclusion a été validée par l'analyse SHAP et l'étude de corrélation avec la cible.

## Stack technique

`Python` · `pandas` · `scikit-learn` · `imbalanced-learn` (SMOTE) · `XGBoost` · `SHAP` · `matplotlib` / `seaborn` · `joblib`

##  Prochaines étapes

-  Transformer le modèle en service (API)
-  Construire une application autour du modèle
-  Intégrer des briques d'IA avancées (LLM)
-  Dockeriser l'ensemble
-  Déployer en production
-  Monitoring et pratiques MLOps

##  Contenu du repo

- `churn_prediction_pipeline.ipynb` : notebook complet (EDA → modélisation → interprétabilité)
