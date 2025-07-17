#!/usr/bin/env python3
"""
Optimiseur GMM (Gaussian Mixture Model) pour placement de points d'accès WiFi 3D.

Ce module implémente un algorithme d'optimisation basé sur les mélanges gaussiens
pour optimiser le placement des points d'accès WiFi dans un environnement 3D.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class GMMOptimizer3D:
    """
    Optimiseur utilisant Gaussian Mixture Model + EM pour le placement de points d'accès WiFi 3D.
    
    L'algorithme utilise l'estimation par maximum de vraisemblance via l'algorithme EM
    pour modéliser la distribution des points à couvrir et placer optimalement les points d'accès.
    """
    
    def __init__(self, frequency: float):
        """
        Initialise l'optimiseur GMM 3D.
        
        Args:
            frequency: Fréquence de transmission en Hz
        """
        self.frequency = frequency
    
    def optimize_clustering_gmm_3d(self, coverage_points: List[Tuple[float, float, float]], 
                                  grid_info: Dict, longueur: float, largeur: float, hauteur_totale: float,
                                  target_coverage_db: float, min_coverage_percent: float, 
                                  power_tx: float, max_access_points: int) -> Tuple[Optional[Dict], Dict]:
        """
        Optimise le placement des points d'accès 3D avec GMM + EM.
        
        Args:
            coverage_points: Liste des points à couvrir [(x, y, z), ...]
            grid_info: Informations sur la grille
            longueur, largeur, hauteur_totale: Dimensions
            target_coverage_db: Niveau de signal minimum requis (dBm)
            min_coverage_percent: Pourcentage minimum de couverture requis
            power_tx: Puissance de transmission (dBm)
            max_access_points: Nombre maximum de points d'accès
            
        Returns:
            Tuple (configuration, analyse) ou (None, {}) si échec
        """
        
        if not coverage_points:
            return None, {}
        
        # Conversion en array numpy
        points_array = np.array(coverage_points)
        
        print(f"🧠 Optimisation GMM 3D: {len(coverage_points)} points à couvrir")
        print(f"📦 Volume: {longueur}m x {largeur}m x {hauteur_totale}m")
        
        best_config = None
        best_score = -1.0
        gmm_analysis = {}
        
        # Test différents nombres de composantes (points d'accès)
        max_components = min(max_access_points, 8)
        print(f"🔬 Test GMM: 1 à {max_components} composantes (objectif {min_coverage_percent}%)")
        
        for n_components in range(1, max_components + 1):
            try:
                # Configuration GMM
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',  # Covariances complètes pour flexibilité
                    max_iter=200,
                    n_init=5,
                    random_state=42,
                    tol=1e-4
                )
                
                # Ajustement du modèle
                gmm.fit(points_array)
                
                # Extraction des centres des composantes (moyennes)
                centers = gmm.means_
                
                # Ajustement des centres pour éviter les murs et contraintes 3D
                adjusted_centers = []
                for center in centers:
                    x, y, z = center
                    
                    # Contraintes de positionnement
                    x = np.clip(x, 1.0, longueur - 1.0)
                    y = np.clip(y, 1.0, largeur - 1.0)
                    z = np.clip(z, 0.5, hauteur_totale - 0.5)
                    
                    # Vérification des murs (projection 2D)
                    x_pixel = int(np.clip(x / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                    y_pixel = int(np.clip(y / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                    
                    # Si dans un mur, ajuster la position
                    if grid_info['walls_detected'][y_pixel, x_pixel] > 0:
                        # Déplacer vers le centre de la zone libre la plus proche
                        x = np.clip(x + np.random.uniform(-2, 2), 1.0, longueur - 1.0)
                        y = np.clip(y + np.random.uniform(-2, 2), 1.0, largeur - 1.0)
                    
                    adjusted_centers.append((x, y, z, power_tx))
                
                # Évaluation de cette configuration
                score, stats = self._evaluate_configuration_3d(
                    adjusted_centers, coverage_points, grid_info,
                    target_coverage_db, min_coverage_percent
                )
                
                # Métriques GMM
                gmm_metrics = {
                    'aic': gmm.aic(points_array),
                    'bic': gmm.bic(points_array),
                    'log_likelihood': gmm.score(points_array),
                    'converged': gmm.converged_,
                    'n_iter': gmm.n_iter_,
                    'covariances_shape': [cov.shape for cov in gmm.covariances_],
                    'weights': gmm.weights_.tolist()
                }
                
                gmm_analysis[n_components] = {
                    'centers': adjusted_centers,
                    'score': score,
                    'stats': stats,
                    'gmm_metrics': gmm_metrics,
                    'means': gmm.means_.tolist(),
                    'covariances': [cov.tolist() for cov in gmm.covariances_]
                }
                
                # Mise à jour du meilleur score
                if score > best_score:
                    best_score = score
                    best_config = {
                        'access_points': adjusted_centers,
                        'score': score,
                        'stats': stats,
                        'gmm_metrics': gmm_metrics,
                        'n_components': n_components
                    }
                
                # Affichage du progrès
                current_coverage = stats.get('coverage_percent', 0.0)
                print(f"🧠 {n_components} composantes: {current_coverage:.1f}% couverture (AIC: {gmm_metrics['aic']:.1f}, BIC: {gmm_metrics['bic']:.1f})")
                
                # Arrêt anticipé si objectif atteint
                if current_coverage >= min_coverage_percent:
                    print(f"✅ Objectif GMM {min_coverage_percent}% atteint avec {n_components} composantes ({current_coverage:.1f}%)")
                    break
                    
            except Exception as e:
                print(f"⚠️  Erreur GMM avec {n_components} composantes: {e}")
                gmm_analysis[n_components] = {
                    'error': str(e),
                    'score': 0.0,
                    'stats': {'coverage_percent': 0.0}
                }
                continue
        
        # Validation finale
        if best_config and best_config['stats']['coverage_percent'] < min_coverage_percent:
            print(f"⚠️  GMM 3D: Objectif {min_coverage_percent}% non atteint. Meilleur: {best_config['stats']['coverage_percent']:.1f}%")
            print(f"💡 Recommandation: Augmenter la puissance TX ou ajouter plus de points d'accès")
        
        return best_config, gmm_analysis
    
    def _evaluate_configuration_3d(self, access_points: List[Tuple[float, float, float, float]], 
                                  coverage_points: List[Tuple[float, float, float]], 
                                  grid_info: Dict, target_coverage_db: float, 
                                  min_coverage_percent: float) -> Tuple[float, Dict]:
        """
        Évalue la qualité d'une configuration de points d'accès 3D.
        
        Cette méthode doit être adaptée par la classe parent pour utiliser
        le bon calculateur de pathloss.
        
        Args:
            access_points: Liste des points d'accès [(x, y, z, power), ...]
            coverage_points: Points à couvrir
            grid_info: Informations sur la grille
            target_coverage_db: Signal minimal requis
            min_coverage_percent: Couverture minimale requise
            
        Returns:
            Tuple (score, statistiques)
        """
        # Cette méthode sera remplacée par la classe parent
        # qui a accès au calculateur de pathloss
        
        if len(access_points) == 0:
            return 0.0, {'covered_points': 0, 'total_points': len(coverage_points), 'coverage_percent': 0.0}
        
        # Calcul simple de distance pour évaluation de base
        covered_points = 0
        
        for point in coverage_points:
            x_rx, y_rx, z_rx = point
            best_distance = float('inf')
            
            for ap in access_points:
                x_tx, y_tx, z_tx, power_tx = ap
                distance_3d = np.sqrt((x_rx - x_tx)**2 + (y_rx - y_tx)**2 + (z_rx - z_tx)**2)
                
                if distance_3d < best_distance:
                    best_distance = distance_3d
            
            # Estimation simple: couvert si distance < seuil
            coverage_threshold = 15.0  # mètres
            if best_distance < coverage_threshold:
                covered_points += 1
        
        # Statistiques de base
        total_points = len(coverage_points)
        coverage_percent = (covered_points / total_points) * 100 if total_points > 0 else 0.0
        
        # Score favorisant d'abord l'atteinte de l'objectif
        num_aps = len(access_points)
        coverage_score = coverage_percent / 100.0
        
        if coverage_percent < min_coverage_percent:
            score = coverage_score * 2.0
            efficiency_penalty = num_aps * 0.01
            score -= efficiency_penalty
        else:
            score = 1.0 + coverage_score
            efficiency_penalty = (num_aps - 1) * 0.05
            score -= efficiency_penalty
        
        stats = {
            'covered_points': covered_points,
            'total_points': total_points,
            'coverage_percent': coverage_percent,
            'num_access_points': num_aps
        }
        
        return max(score, 0.0), stats
    
    def visualize_gmm_process_3d(self, config: Dict, analysis: Dict, 
                                coverage_points: List[Tuple[float, float, float]],
                                longueur: float, largeur: float, hauteur_totale: float) -> plt.Figure:
        """
        Visualise le processus d'optimisation GMM 3D.
        
        Args:
            config: Configuration des points d'accès
            analysis: Analyse du processus GMM
            coverage_points: Points de couverture
            longueur, largeur, hauteur_totale: Dimensions
            
        Returns:
            Figure matplotlib 3D
        """
        fig = plt.figure(figsize=(18, 12))
        
        # Graphique 1: Distribution 3D et centres GMM
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.set_title("Distribution 3D et Centres GMM", fontsize=14, fontweight='bold')
        
        # Points de couverture (échantillonnés pour performance)
        if len(coverage_points) > 1000:
            sample_indices = np.random.choice(len(coverage_points), 1000, replace=False)
            sample_points = [coverage_points[i] for i in sample_indices]
        else:
            sample_points = coverage_points
        
        if sample_points:
            points_array = np.array(sample_points)
            ax1.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], 
                       c='lightblue', s=8, alpha=0.4, label='Points à couvrir')
        
        # Points d'accès GMM
        access_points = config['access_points']
        colors = plt.cm.viridis(np.linspace(0, 1, len(access_points)))
        
        for i, (x, y, z, power) in enumerate(access_points):
            ax1.scatter(x, y, z, c=[colors[i]], s=300, marker='*', 
                       edgecolors='black', linewidth=2, zorder=5)
            ax1.text(x, y, z, f'AP{i+1}', fontsize=10, fontweight='bold')
        
        ax1.set_xlim(0, longueur)
        ax1.set_ylim(0, largeur)
        ax1.set_zlim(0, hauteur_totale)
        ax1.set_xlabel('Longueur (m)')
        ax1.set_ylabel('Largeur (m)')
        ax1.set_zlabel('Hauteur (m)')
        ax1.legend()
        
        # Graphique 2: Métriques GMM
        ax2 = fig.add_subplot(222)
        ax2.set_title("Métriques GMM par Nombre de Composantes", fontsize=14, fontweight='bold')
        
        n_components_list = []
        aic_values = []
        bic_values = []
        coverage_values = []
        
        for n_comp, data in analysis.items():
            if isinstance(n_comp, int) and 'gmm_metrics' in data:
                n_components_list.append(n_comp)
                aic_values.append(data['gmm_metrics']['aic'])
                bic_values.append(data['gmm_metrics']['bic'])
                coverage_values.append(data['stats']['coverage_percent'])
        
        if n_components_list:
            ax2_twin = ax2.twinx()
            
            line1 = ax2.plot(n_components_list, aic_values, 'b-o', linewidth=2, 
                           markersize=8, label='AIC')
            line2 = ax2.plot(n_components_list, bic_values, 'r-s', linewidth=2, 
                           markersize=8, label='BIC')
            line3 = ax2_twin.plot(n_components_list, coverage_values, 'g-^', linewidth=2, 
                                markersize=8, label='Couverture (%)', color='green')
            
            ax2.set_xlabel('Nombre de Composantes')
            ax2.set_ylabel('Critère d\'Information', color='black')
            ax2_twin.set_ylabel('Couverture (%)', color='green')
            
            ax2.tick_params(axis='y', labelcolor='black')
            ax2_twin.tick_params(axis='y', labelcolor='green')
            
            # Légendes combinées
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            ax2.grid(True, alpha=0.3)
        
        # Graphique 3: Distribution par étages
        ax3 = fig.add_subplot(223)
        ax3.set_title("Distribution des Points d'Accès par Étage", fontsize=14, fontweight='bold')
        
        if access_points:
            floors = [int(ap[2] // 2.7) + 1 for ap in access_points]  # Hauteur étage = 2.7m
            floor_counts = {}
            for floor in floors:
                floor_counts[floor] = floor_counts.get(floor, 0) + 1
            
            floors_list = sorted(floor_counts.keys())
            counts_list = [floor_counts[f] for f in floors_list]
            
            bars = ax3.bar(floors_list, counts_list, color='skyblue', alpha=0.7, edgecolor='black')
            
            # Ajout des valeurs sur les barres
            for bar, count in zip(bars, counts_list):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
            
            ax3.set_xlabel('Étage')
            ax3.set_ylabel('Nombre de Points d\'Accès')
            ax3.set_xticks(floors_list)
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Graphique 4: Informations du modèle
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        ax4.set_title("Informations du Modèle GMM", fontsize=14, fontweight='bold')
        
        # Récupération des métriques du meilleur modèle
        if 'gmm_metrics' in config:
            gmm_metrics = config['gmm_metrics']
            
            info_text = f"Algorithme: GMM + EM\n"
            info_text += f"Composantes: {config['n_components']}\n"
            info_text += f"Points d'accès: {len(access_points)}\n"
            info_text += f"Couverture finale: {config['stats']['coverage_percent']:.1f}%\n"
            info_text += f"Score total: {config['score']:.3f}\n\n"
            
            info_text += f"Métriques GMM:\n"
            info_text += f"• AIC: {gmm_metrics['aic']:.1f}\n"
            info_text += f"• BIC: {gmm_metrics['bic']:.1f}\n"
            info_text += f"• Log-vraisemblance: {gmm_metrics['log_likelihood']:.1f}\n"
            info_text += f"• Convergé: {'Oui' if gmm_metrics['converged'] else 'Non'}\n"
            info_text += f"• Itérations: {gmm_metrics['n_iter']}\n\n"
            
            info_text += f"Poids des composantes:\n"
            for i, weight in enumerate(gmm_metrics['weights']):
                info_text += f"• Composante {i+1}: {weight:.3f}\n"
            
            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def compare_with_other_algorithms_3d(self, coverage_points: List[Tuple[float, float, float]], 
                                        grid_info: Dict, longueur: float, largeur: float, hauteur_totale: float,
                                        target_coverage_db: float, min_coverage_percent: float,
                                        power_tx: float, max_access_points: int) -> Dict:
        """
        Compare l'algorithme GMM avec d'autres approches 3D.
        
        Args:
            coverage_points, grid_info, longueur, largeur, hauteur_totale: Configuration de base
            target_coverage_db, min_coverage_percent, power_tx, max_access_points: Paramètres d'optimisation
            
        Returns:
            Dictionnaire avec les résultats comparatifs
        """
        
        results = {}
        
        # Résultat GMM
        gmm_result = self.optimize_clustering_gmm_3d(
            coverage_points, grid_info, longueur, largeur, hauteur_totale,
            target_coverage_db, min_coverage_percent, power_tx, max_access_points
        )
        
        if gmm_result[0]:
            config, analysis = gmm_result
            results['gmm'] = {
                'config': config,
                'analysis': analysis,
                'algorithm_name': 'GMM + EM'
            }
        
        # Comparaison avec K-means simple
        try:
            points_array = np.array(coverage_points)
            
            # Test K-means avec même nombre de clusters que le meilleur GMM
            best_n_components = results['gmm']['config']['n_components'] if 'gmm' in results else 3
            
            kmeans = KMeans(n_clusters=best_n_components, random_state=42, n_init=10)
            kmeans.fit(points_array)
            
            kmeans_centers = [(center[0], center[1], center[2], power_tx) for center in kmeans.cluster_centers_]
            kmeans_score, kmeans_stats = self._evaluate_configuration_3d(
                kmeans_centers, coverage_points, grid_info, target_coverage_db, min_coverage_percent
            )
            
            results['kmeans'] = {
                'config': {
                    'access_points': kmeans_centers,
                    'score': kmeans_score,
                    'stats': kmeans_stats,
                    'n_clusters': best_n_components
                },
                'analysis': {'type': 'kmeans'},
                'algorithm_name': 'K-means'
            }
            
        except Exception as e:
            print(f"⚠️  Erreur comparaison K-means: {e}")
        
        return results
