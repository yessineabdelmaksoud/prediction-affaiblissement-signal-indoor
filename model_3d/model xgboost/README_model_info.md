# Documentation du Modèle XGBoost - Structure model_info

## Vue d'ensemble
Le modèle XGBoost pour la prédiction du path loss indoor a été sauvegardé avec une structure `model_info` complète qui contient toutes les informations nécessaires pour utiliser le modèle.

## Structure du fichier model_info

Le fichier `xgboost_radio_propagation_model.pkl` contient un dictionnaire Python avec les clés suivantes :

### 1. `model` (XGBRegressor)
- **Type**: `xgboost.sklearn.XGBRegressor`
- **Description**: Le modèle XGBoost entraîné et optimisé
- **Usage**: Utilisez `model_info['model'].predict(data)` pour faire des prédictions

### 2. `feature_names` (list)
- **Type**: `list`
- **Contenu**: `['distance', 'numwall', 'etage', 'frequence']`
- **Description**: Liste ordonnée des features requises pour les prédictions

### 3. `feature_importance` (dict)
- **Type**: `dict`
- **Contenu**: Mapping feature_name -> importance_value
- **Exemple**: `{'distance': 0.061, 'numwall': 0.321, 'etage': 0.612, 'frequence': 0.006}`
- **Description**: Importance relative de chaque feature selon XGBoost

### 4. `hyperparameters` (dict)
- **Type**: `dict`
- **Description**: Tous les hyperparamètres du modèle XGBoost
- **Usage**: Peut être utilisé pour recréer un modèle identique

### 5. `metrics` (dict)
- **Type**: `dict`
- **Contenu**:
  - `rmse`: Root Mean Square Error sur le jeu de test
  - `mae`: Mean Absolute Error sur le jeu de test  
  - `r2_score`: Coefficient de détermination R²
- **Exemple**: `{'rmse': 16.081, 'mae': 8.992, 'r2_score': 0.8662}`

### 6. `dataset_info` (dict)
- **Type**: `dict`
- **Contenu**:
  - `n_samples`: Nombre total d'échantillons dans le dataset
  - `train_size`: Taille du jeu d'entraînement
  - `test_size`: Taille du jeu de test
  - `target_variable`: Nom de la variable cible

### 7. `creation_date` (str)
- **Type**: `str`
- **Format**: `'YYYY-MM-DD HH:MM:SS'`
- **Description**: Date et heure de création du modèle

### 8. `model_type` (str)
- **Type**: `str`
- **Valeur**: `'XGBoost Regressor'`
- **Description**: Type de modèle utilisé

### 9. `feature_engineering` (bool)
- **Type**: `bool`
- **Valeur**: `False`
- **Description**: Indique si du feature engineering a été appliqué

### 10. `notes` (str)
- **Type**: `str`
- **Description**: Notes supplémentaires sur le modèle

## Exemple d'utilisation

```python
import pickle
import pandas as pd

# Charger le modèle
with open('xgboost_radio_propagation_model.pkl', 'rb') as f:
    model_info = pickle.load(f)

# Vérifier les informations du modèle
print(f"Performance R²: {model_info['metrics']['r2_score']:.4f}")
print(f"Features requises: {model_info['feature_names']}")

# Faire une prédiction
input_data = pd.DataFrame({
    'distance': [20],
    'numwall': [2],
    'etage': [1], 
    'frequence': [2400]
})

# S'assurer que l'ordre des colonnes est correct
input_data = input_data[model_info['feature_names']]

# Prédiction
path_loss = model_info['model'].predict(input_data)[0]
print(f"Path loss prédit: {path_loss:.1f} dB")
```

## Features du modèle

Le modèle utilise uniquement les 4 colonnes originales du dataset :

1. **distance** (float): Distance entre émetteur et récepteur en mètres
2. **numwall** (int): Nombre de murs traversés par le signal
3. **etage** (int): Différence d'étage entre émetteur et récepteur  
4. **frequence** (int): Fréquence du signal en MHz (2400 ou 5000)

## Performance

- **RMSE**: ~16.1 dB
- **MAE**: ~9.0 dB  
- **R²**: ~0.866

## Remarques importantes

- Aucun feature engineering n'a été appliqué
- Le modèle a été optimisé avec GridSearchCV
- Validation croisée effectuée pour s'assurer de la robustesse
- Structure compatible avec les standards de sauvegarde de modèles ML
