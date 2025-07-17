import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class GMMOptimizer:
    """
    Optimiseur utilisant Gaussian Mixture Model (GMM) avec algorithme EM
    pour le placement de points d'accès WiFi.
    
    Avantages par rapport à K-means:
    - Peut capturer des clusters de formes ellipsoïdales
    - Modélise la variance de chaque cluster
    - Donne des probabilités d'appartenance
    - Meilleure adaptation aux zones de formes irrégulières
    """
    
    def __init__(self, covariance_type='diag', max_iter=100, random_state=42):
        """
        Initialise l'optimiseur GMM.
        
        Args:
            covariance_type: Type de covariance ('diag', 'full', 'tied', 'spherical')
                           'diag' recommandé pour efficacité
            max_iter: Nombre maximal d'itérations EM
            random_state: Graine aléatoire pour reproductibilité
        """
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.random_state = random_state
        
    def optimize_clustering_gmm(self, coverage_points, grid_info, longueur, largeur,
                               target_coverage_db=-70.0, min_coverage_percent=90.0,
                               power_tx=20.0, max_access_points=6):
        """
        Optimise le placement des points d'accès avec GMM + EM.
        
        Args:
            coverage_points: Points à couvrir [(x, y), ...]
            grid_info: Informations sur la grille
            longueur, largeur: Dimensions
            target_coverage_db: Signal minimal requis
            min_coverage_percent: Couverture minimale
            power_tx: Puissance de transmission
            max_access_points: Nombre maximal de points d'accès
            
        Returns:
            best_config: Meilleure configuration trouvée
            gmm_analysis: Analyse des modèles GMM
        """
        if len(coverage_points) == 0:
            return {'access_points': [], 'score': 0.0, 'stats': {}}, {}
        
        # Conversion en array numpy
        points_array = np.array(coverage_points)
        
        best_config = None
        best_score = -1.0
        gmm_analysis = {}
        
        # Test différents nombres de composantes (points d'accès)
        max_components = min(max_access_points, len(coverage_points), 8)
        print(f"GMM+EM: test de 1 à {max_components} composantes (objectif {min_coverage_percent}%)")
        
        for n_components in range(1, max_components + 1):
            try:
                # Création et ajustement du modèle GMM
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type=self.covariance_type,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    init_params='kmeans'  # Initialisation avec K-means pour stabilité
                )
                
                # Ajustement du modèle (algorithme EM)
                gmm.fit(points_array)
                
                # Prédiction des labels et probabilités
                labels = gmm.predict(points_array)
                probabilities = gmm.predict_proba(points_array)
                
                # Extraction des centres (moyennes des gaussiennes)
                centers = gmm.means_
                
                # Ajustement des centres pour éviter les murs
                adjusted_centers = []
                for i, center in enumerate(centers):
                    x, y = center
                    
                    # Vérification si dans un mur
                    x_pixel = int(np.clip(x / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                    y_pixel = int(np.clip(y / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                    
                    # Si dans un mur, utiliser le point pondéré le plus proche
                    if grid_info['walls_detected'][y_pixel, x_pixel] > 0:
                        # Trouver les points du cluster avec leurs probabilités
                        cluster_points = points_array[labels == i]
                        cluster_probs = probabilities[labels == i, i]
                        
                        if len(cluster_points) > 0:
                            # Moyenne pondérée par les probabilités
                            weights = cluster_probs / np.sum(cluster_probs)
                            x = np.average(cluster_points[:, 0], weights=weights)
                            y = np.average(cluster_points[:, 1], weights=weights)
                            
                            # Re-vérification si toujours dans un mur
                            x_pixel = int(np.clip(x / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                            y_pixel = int(np.clip(y / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                            
                            if grid_info['walls_detected'][y_pixel, x_pixel] > 0:
                                # Utiliser le centroïde simple si la moyenne pondérée échoue
                                x, y = np.mean(cluster_points, axis=0)
                    
                    adjusted_centers.append((x, y, power_tx))
                
                # Calcul des métriques de qualité GMM
                aic = gmm.aic(points_array)
                bic = gmm.bic(points_array)
                log_likelihood = gmm.score(points_array)
                
                # Évaluation de la configuration avec le calculateur de couverture
                # (Cette partie dépend de votre calculate_coverage_quality_2d existant)
                score, stats = self._evaluate_configuration(
                    adjusted_centers, coverage_points, grid_info,
                    target_coverage_db, min_coverage_percent
                )
                
                # Stockage des informations d'analyse
                gmm_analysis[n_components] = {
                    'centers': adjusted_centers,
                    'score': score,
                    'stats': stats,
                    'labels': labels,
                    'probabilities': probabilities,
                    'covariances': gmm.covariances_,
                    'weights': gmm.weights_,
                    'aic': aic,
                    'bic': bic,
                    'log_likelihood': log_likelihood,
                    'converged': gmm.converged_,
                    'n_iter': gmm.n_iter_
                }
                
                # Mise à jour du meilleur score
                if score > best_score:
                    best_score = score
                    best_config = {
                        'access_points': adjusted_centers,
                        'score': score,
                        'stats': stats,
                        'n_components': n_components,
                        'algorithm': 'GMM+EM',
                        'gmm_metrics': {
                            'aic': aic,
                            'bic': bic,
                            'log_likelihood': log_likelihood,
                            'converged': gmm.converged_
                        }
                    }
                
                # Arrêt anticipé si objectif atteint
                current_coverage = stats.get('coverage_percent', 0.0)
                if current_coverage >= min_coverage_percent:
                    print(f"✅ GMM: Objectif {min_coverage_percent}% atteint avec {n_components} composantes ({current_coverage:.1f}%)")
                    break
                else:
                    print(f"📊 GMM {n_components} composantes: {current_coverage:.1f}% de couverture")
                    
            except Exception as e:
                print(f"⚠️ Erreur GMM avec {n_components} composantes: {e}")
                continue
        
        return best_config, gmm_analysis
    
    def _evaluate_configuration(self, access_points, coverage_points, grid_info,
                               target_coverage_db, min_coverage_percent):
        """
        Évalue une configuration de points d'accès.
        Cette méthode doit être adaptée selon votre calculateur existant.
        """
        # Calcul simplifié pour l'exemple - à remplacer par votre méthode réelle
        covered_points = 0
        total_points = len(coverage_points)
        
        for point in coverage_points:
            x_rx, y_rx = point
            best_signal = -200.0
            
            for ap in access_points:
                x_tx, y_tx, power_tx = ap
                
                # Distance simple
                distance = np.sqrt((x_rx - x_tx)**2 + (y_rx - y_tx)**2)
                
                # Estimation simple du pathloss (à remplacer par votre modèle ML)
                pathloss = 40 + 20 * np.log10(max(distance, 0.1))
                received_power = power_tx - pathloss
                
                if received_power > best_signal:
                    best_signal = received_power
            
            if best_signal >= target_coverage_db:
                covered_points += 1
        
        coverage_percent = (covered_points / total_points) * 100 if total_points > 0 else 0.0
        
        # Score avec pénalité pour trop de points d'accès
        num_aps = len(access_points)
        coverage_score = coverage_percent / 100.0
        efficiency_penalty = num_aps * 0.05
        score = coverage_score - efficiency_penalty
        
        if coverage_percent >= min_coverage_percent:
            score += 0.5
        
        stats = {
            'covered_points': covered_points,
            'total_points': total_points,
            'coverage_percent': coverage_percent,
            'num_access_points': num_aps
        }
        
        return max(score, 0.0), stats
    
    def visualize_gmm_clusters(self, points_array, gmm_analysis, best_n_components, 
                              longueur, largeur, image_array=None):
        """
        Visualise les clusters GMM et leurs ellipses de confiance.
        
        Args:
            points_array: Points de couverture
            gmm_analysis: Analyse des modèles GMM
            best_n_components: Nombre optimal de composantes
            longueur, largeur: Dimensions
            image_array: Image de fond optionnelle
            
        Returns:
            fig: Figure matplotlib
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if best_n_components in gmm_analysis:
            analysis = gmm_analysis[best_n_components]
            centers = analysis['centers']
            labels = analysis['labels']
            covariances = analysis['covariances']
            
            # Graphique 1: Clusters avec ellipses de confiance
            if image_array is not None:
                ax1.imshow(image_array, extent=[0, longueur, largeur, 0], cmap='gray', alpha=0.7)
            
            # Points colorés par cluster
            scatter = ax1.scatter(points_array[:, 0], points_array[:, 1], 
                                c=labels, cmap='tab10', alpha=0.6, s=20)
            
            # Centres des clusters
            for i, (x, y, power) in enumerate(centers):
                ax1.scatter(x, y, c='red', s=200, marker='*', 
                          edgecolors='black', linewidth=2)
                ax1.annotate(f'AP{i+1}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontweight='bold')
                
                # Ellipse de confiance (approximation pour covariance diagonale)
                if self.covariance_type == 'diag' and i < len(covariances):
                    std_x = np.sqrt(covariances[i][0])
                    std_y = np.sqrt(covariances[i][1])
                    
                    ellipse = plt.Circle((x, y), 2*max(std_x, std_y), 
                                       fill=False, color='red', alpha=0.5, linestyle='--')
                    ax1.add_patch(ellipse)
            
            ax1.set_xlim(0, longueur)
            ax1.set_ylim(largeur, 0)
            ax1.set_xlabel('Longueur (m)')
            ax1.set_ylabel('Largeur (m)')
            ax1.set_title(f'Clusters GMM - {best_n_components} composantes')
            ax1.grid(True, alpha=0.3)
            
            # Graphique 2: Métriques de sélection de modèle
            n_components_list = list(gmm_analysis.keys())
            aic_values = [gmm_analysis[n]['aic'] for n in n_components_list]
            bic_values = [gmm_analysis[n]['bic'] for n in n_components_list]
            scores = [gmm_analysis[n]['score'] for n in n_components_list]
            
            ax2_twin = ax2.twinx()
            
            line1 = ax2.plot(n_components_list, aic_values, 'b-o', label='AIC', linewidth=2)
            line2 = ax2.plot(n_components_list, bic_values, 'g-s', label='BIC', linewidth=2)
            line3 = ax2_twin.plot(n_components_list, scores, 'r-^', label='Score Couverture', linewidth=2, color='red')
            
            ax2.axvline(x=best_n_components, color='black', linestyle='--', alpha=0.7, label='Optimal')
            
            ax2.set_xlabel('Nombre de Composantes')
            ax2.set_ylabel('AIC / BIC', color='blue')
            ax2_twin.set_ylabel('Score de Couverture', color='red')
            ax2.set_title('Sélection du Nombre Optimal de Composantes')
            ax2.grid(True, alpha=0.3)
            
            # Légende combinée
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.tight_layout()
        return fig
    
    def compare_with_kmeans(self, points_array, n_components):
        """
        Compare GMM avec K-means sur les mêmes données.
        
        Args:
            points_array: Points à clusterer
            n_components: Nombre de clusters/composantes
            
        Returns:
            comparison: Dictionnaire de comparaison
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        # GMM
        gmm = GaussianMixture(n_components=n_components, 
                            covariance_type=self.covariance_type,
                            random_state=self.random_state)
        gmm_labels = gmm.fit_predict(points_array)
        
        # K-means
        kmeans = KMeans(n_clusters=n_components, random_state=self.random_state, n_init=10)
        kmeans_labels = kmeans.fit_predict(points_array)
        
        # Métriques de comparaison
        comparison = {
            'gmm': {
                'silhouette_score': silhouette_score(points_array, gmm_labels),
                'calinski_harabasz_score': calinski_harabasz_score(points_array, gmm_labels),
                'aic': gmm.aic(points_array),
                'bic': gmm.bic(points_array),
                'log_likelihood': gmm.score(points_array)
            },
            'kmeans': {
                'silhouette_score': silhouette_score(points_array, kmeans_labels),
                'calinski_harabasz_score': calinski_harabasz_score(points_array, kmeans_labels),
                'inertia': kmeans.inertia_
            }
        }
        
        return comparison

    def get_cluster_statistics(self, gmm_analysis, n_components):
        """
        Obtient des statistiques détaillées sur un modèle GMM spécifique.
        
        Args:
            gmm_analysis: Analyse des modèles GMM
            n_components: Nombre de composantes à analyser
            
        Returns:
            statistics: Statistiques détaillées
        """
        if n_components not in gmm_analysis:
            return None
        
        analysis = gmm_analysis[n_components]
        
        statistics = {
            'model_selection': {
                'aic': analysis['aic'],
                'bic': analysis['bic'],
                'log_likelihood': analysis['log_likelihood'],
                'converged': analysis['converged'],
                'iterations': analysis['n_iter']
            },
            'clusters': []
        }
        
        # Statistiques par cluster
        for i in range(n_components):
            cluster_mask = analysis['labels'] == i
            cluster_points = np.sum(cluster_mask)
            cluster_weight = analysis['weights'][i]
            
            cluster_stats = {
                'cluster_id': i + 1,
                'n_points': int(cluster_points),
                'weight': float(cluster_weight),
                'center': analysis['centers'][i][:2],  # x, y seulement
                'coverage_contribution': float(cluster_points * cluster_weight)
            }
            
            if self.covariance_type == 'diag':
                cluster_stats['std_x'] = float(np.sqrt(analysis['covariances'][i][0]))
                cluster_stats['std_y'] = float(np.sqrt(analysis['covariances'][i][1]))
            
            statistics['clusters'].append(cluster_stats)
        
        return statistics
