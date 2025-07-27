# 📡 Analyseur de Pathloss Indoor

**Système de Prédiction et d'Optimisation WiFi pour Environnements Intérieurs**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Objectif du Projet

Ce projet développe une application intelligente capable de :

- **Prédire la perte de signal radio (pathloss)** dans des environnements intérieurs complexes à partir de plans architecturaux
- **Optimiser automatiquement le placement des points d'accès WiFi** pour assurer une couverture optimale
- **Analyser et visualiser la propagation radio** en 2D et 3D avec prise en compte des obstacles
- **Réduire les zones d'ombre** et améliorer la qualité de service WiFi indoor

## ✨ Fonctionnalités Principales

### 🖼️ Traitement d'Image Intelligent
- **Extraction automatique des murs** à partir de plans PNG
- **Détection des obstacles** et structures architecturales
- **Pipeline de traitement** : binarisation, nettoyage morphologique, détection de contours
- **Visualisation multi-couches** pour validation des résultats

### 🧠 Prédiction de Pathloss Hybride
- **Modèles ML pré-entraînés** : XGBoost 3D (RMSE: 16.08 dB), Régression 2D (RMSE: 5.92 dB)
- **Fallback théorique** : Formule de Friis avec atténuations matériaux
- **Calculs 2D/3D** avec prise en compte des étages et matériaux
- **Validation automatique** et gestion d'erreurs robuste

### 🎨 Visualisation Avancée
- **Cartes de chaleur 2D** avec matplotlib
- **Visualisations 3D interactives** avec Plotly
- **Rendu des trajets de propagation** et intersections avec obstacles
- **Classification par zones de qualité** (Excellent/Bon/Faible/Mauvais)

### ⚡ Optimisation Multi-Algorithmes
- **Algorithme Glouton (Greedy)** : Placement séquentiel avec amélioration marginale
- **Mélanges Gaussiens (GMM)** : Modélisation probabiliste avec EM
- **K-means Clustering** : Regroupement géométrique optimisé
- **Optimiseur Automatique** : Analyse géométrique + clustering adaptatif

## 🏗️ Architecture du Système

```
┌─────────────────────────────────────────────────────────────┐
│                    Interface Streamlit                     │
│                     (8 onglets)                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│  ImageProcessor     │  PathlossCalculator  │  Visualizer    │
│  ├─ Binarisation    │  ├─ ML Predictor     │  ├─ 2D Plots   │
│  ├─ Morphologie     │  ├─ Theoretical      │  ├─ 3D Models  │
│  └─ Détection       │  └─ Hybrid Logic     │  └─ Heatmaps   │
└─────────────────────┼───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│           Optimiseurs de Placement                         │
│  ├─ GreedyOptimizer3D    ├─ GMMOptimizer3D                │
│  ├─ AccessPointOptimizer ├─ AutoOptimizer3D               │
│  └─ HeatmapGenerator     └─ Visualization3D               │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prérequis
- Python 3.8+
- pip (gestionnaire de paquets Python)

### Installation des dépendances

```bash
# Cloner le repository
git clone https://github.com/username/prediction-affaiblissement-signal-indoor.git
cd prediction-affaiblissement-signal-indoor

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales
- `streamlit` - Interface web interactive
- `opencv-python` - Traitement d'image
- `numpy` - Calculs numériques
- `matplotlib` - Visualisations 2D
- `plotly` - Visualisations 3D interactives
- `scikit-learn` - Algorithmes ML et clustering
- `pandas` - Manipulation de données
- `Pillow` - Traitement d'images
- `scikit-image` - Traitement d'image avancé

## 💻 Utilisation

### Lancement de l'application

```bash
streamlit run app.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

### Interface utilisateur (8 onglets)

1. **📊 Pathloss 2D** - Analyse de propagation en 2D
2. **📈 Pathloss 3D** - Analyse de propagation en 3D
3. **🔥 Heatmap 2D** - Cartes de chaleur 2D
4. **🌡️ Heatmap 3D** - Visualisations thermiques 3D
5. **🎯 Optimisation 2D** - Placement optimal 2D
6. **🎯 Optimisation 3D** - Placement optimal 3D
7. **⚡ Auto-Optimisation 2D** - Optimisation automatique 2D
8. **⚡ Auto-Optimisation 3D** - Optimisation automatique 3D

### Workflow typique

1. **Upload du plan** : Charger un plan architectural (format PNG)
2. **Configuration** : Définir dimensions, fréquence, puissance émetteur
3. **Traitement** : Extraction automatique des murs et obstacles
4. **Analyse** : Calcul du pathloss et génération des visualisations
5. **Optimisation** : Placement optimal des points d'accès
6. **Export** : Sauvegarde des résultats et configurations

## 🔬 Algorithmes et Modèles

### Modèles de Prédiction ML

#### Modèle 2D
- **Type** : Régression linéaire
- **Features** : [distance, num_walls, frequency]
- **Performance** : RMSE = 5.92 dB, R² = 0.8515
- **Dataset** : 160,000 échantillons d'entraînement

#### Modèle 3D
- **Type** : XGBoost
- **Features** : [distance, numwall, etage, frequence]
- **Performance** : RMSE = 16.08 dB, MAE = 8.99 dB, R² = 0.8662
- **Dataset** : 200,000 échantillons total

### Formules Théoriques

#### Formule de Friis (espace libre)
```
PL_free = 20 × log₁₀(d) + 20 × log₁₀(f) + 32.45
```

#### Pathloss avec obstacles
```
PL_total = PL_free + N_walls × A_wall + N_floors × A_floor
```

Où :
- `d` : distance en km
- `f` : fréquence en MHz
- `N_walls` : nombre de murs traversés
- `A_wall` : atténuation par mur (6 dB par défaut)
- `N_floors` : différence d'étages
- `A_floor` : atténuation par étage (15 dB par défaut)

### Algorithmes d'Optimisation

#### 1. Algorithme Glouton (Greedy)
```python
while couverture < seuil_min and nb_aps < max_aps:
    meilleur_gain = 0
    for position in positions_candidates:
        gain = calculer_gain_couverture(position)
        if gain > meilleur_gain:
            meilleure_position = position
    placer_ap(meilleure_position)
```

#### 2. Mélanges Gaussiens (GMM)
```python
gmm = GaussianMixture(n_components=num_aps)
gmm.fit(coverage_points)
centers = gmm.means_
for center in centers:
    adjusted_pos = adjust_for_walls(center)
    ap_positions.append(adjusted_pos)
```

#### 3. K-means Clustering
```python
kmeans = KMeans(n_clusters=num_aps, init='k-means++')
cluster_centers = kmeans.fit(points).cluster_centers_
for center in cluster_centers:
    ap_positions.append(adjust_position(center))
```

## 📊 Résultats et Performance

### Métriques de Qualité
- **Excellent** : Pathloss ≤ 50 dB
- **Bon** : 50 < Pathloss ≤ 70 dB
- **Faible** : 70 < Pathloss ≤ 90 dB
- **Mauvais** : Pathloss > 90 dB

### Cas d'Usage
- **Appartements et bureaux** : Optimisation WiFi domestique/professionnel
- **Hôpitaux et écoles** : Couverture critique avec contraintes spécifiques
- **Centres commerciaux** : Couverture large surface avec obstacles complexes
- **Simulation pré-installation** : Validation avant déploiement réel

## 🛠️ Structure du Projet

```
prédiction-affaiblissement-signal-indoor/
├── app.py                          # Application Streamlit principale
├── image_processor.py              # Traitement et analyse d'images
├── pathloss_calculator.py          # Calculateur pathloss 2D
├── pathloss_calculator_3d.py       # Calculateur pathloss 3D
├── ml_pathloss_predictor_2d.py     # Prédicteur ML 2D
├── ml_pathloss_predictor_3d.py     # Prédicteur ML 3D
├── visualization.py                # Visualisations 2D
├── visualization_3d.py             # Visualisations 3D
├── heatmap_generator.py            # Générateur heatmaps 2D
├── heatmap_generator_3d.py         # Générateur heatmaps 3D
├── greedy_optimizer_3d.py          # Optimiseur glouton
├── gmm_optimizer_3d.py             # Optimiseur GMM
├── auto_optimizer_3d.py            # Optimiseur automatique
├── access_point_optimizer.py       # Orchestrateur optimisation
├── model_2d/                       # Modèles ML 2D
│   ├── pathloss_predictor.pkl
│   └── model_summary.txt
├── model_3d/                       # Modèles ML 3D
│   ├── xgboost_radio_propagation_model.pkl
│   └── README_model_info.md
├── rapport/                        # Documentation LaTeX
│   └── rapport.tex
├── requirements.txt                # Dépendances Python
└── README.md                       # Ce fichier
```

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 License

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👥 Auteurs

- **Équipe de développement** - Projet d'optimisation de réseaux WiFi indoor
- **Encadrement académique** - Université de Sfax, Faculté des Sciences

## 🙏 Remerciements

- Université de Sfax pour le support académique
- Communauté open-source pour les bibliothèques utilisées
- Contributeurs et testeurs du projet

---

**📧 Contact** : Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.

**🔗 Liens utiles** :
- [Documentation Streamlit](https://docs.streamlit.io/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Plotly Documentation](https://plotly.com/python/)
- [Scikit-learn Documentation](https://scikit-learn.org/)