import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import differential_evolution
from sklearn.cluster import KMeans
import pandas as pd
import io
from pathloss_calculator import PathlossCalculator
from image_processor import ImageProcessor
from gmm_optimizer import GMMOptimizer
from greedy_optimizer import GreedyOptimizer

class AccessPointOptimizer2D:
    def __init__(self, frequency_mhz):
        """
        Optimiseur pour la placement automatique des points d'accès 2D.
        
        Args:
            frequency_mhz: Fréquence en MHz
        """
        self.frequency_mhz = frequency_mhz
        self.calculator = PathlossCalculator(frequency_mhz)
        self.processor = ImageProcessor()
        self.gmm_optimizer = GMMOptimizer()
        self.greedy_optimizer = GreedyOptimizer(frequency_mhz * 1e6)  # Conversion MHz vers Hz
        
    def generate_coverage_grid_2d(self, walls_detected, longueur, largeur, resolution=25):
        """
        Génère une grille de points à couvrir dans l'espace 2D.
        
        Args:
            walls_detected: Masque binaire des murs
            longueur, largeur: Dimensions en mètres
            resolution: Résolution de la grille
            
        Returns:
            coverage_points: Liste des points à couvrir [(x, y), ...]
            grid_info: Informations sur la grille
        """
        # Création des grilles de coordonnées
        x_coords = np.linspace(0.5, longueur - 0.5, resolution)
        y_coords = np.linspace(0.5, largeur - 0.5, resolution)
        
        # Échelles de conversion pour les murs 2D
        height_2d, width_2d = walls_detected.shape
        scale_x = longueur / width_2d
        scale_y = largeur / height_2d
        
        coverage_points = []
        
        for y in y_coords:
            for x in x_coords:
                # Vérification si le point n'est pas dans un mur
                x_pixel = int(np.clip(x / scale_x, 0, width_2d - 1))
                y_pixel = int(np.clip(y / scale_y, 0, height_2d - 1))
                
                # Si pas dans un mur, ajouter à la liste des points à couvrir
                if walls_detected[y_pixel, x_pixel] == 0:
                    coverage_points.append((x, y))
        
        grid_info = {
            'x_coords': x_coords,
            'y_coords': y_coords,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'walls_detected': walls_detected
        }
        
        return coverage_points, grid_info
    
    def calculate_coverage_quality_2d(self, access_points, coverage_points, grid_info, 
                                    target_coverage_db=-70.0, min_coverage_percent=90.0):
        """
        Calcule la qualité de couverture pour une configuration de points d'accès 2D.
        
        Args:
            access_points: Liste des points d'accès [(x, y, power), ...]
            coverage_points: Points à couvrir
            grid_info: Informations sur la grille
            target_coverage_db: Niveau de signal minimal requis
            min_coverage_percent: Pourcentage de couverture minimal
            
        Returns:
            score: Score de qualité (plus élevé = meilleur)
            coverage_stats: Statistiques de couverture
        """
        if len(access_points) == 0:
            return 0.0, {'covered_points': 0, 'total_points': len(coverage_points), 'coverage_percent': 0.0}
        
        covered_points = 0
        signal_levels = []
        
        for point in coverage_points:
            x_rx, y_rx = point
            best_signal = -200.0  # Très faible
            
            for ap in access_points:
                x_tx, y_tx, power_tx = ap
                
                # Distance 2D
                distance_2d = np.sqrt((x_rx - x_tx)**2 + (y_rx - y_tx)**2)
                
                if distance_2d < 0.1:  # Très proche
                    received_power = power_tx - 10
                else:
                    # Conversion en pixels pour comptage des murs
                    x_tx_pixel = int(np.clip(x_tx / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                    y_tx_pixel = int(np.clip(y_tx / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                    x_rx_pixel = int(np.clip(x_rx / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                    y_rx_pixel = int(np.clip(y_rx / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                    
                    # Comptage des murs
                    wall_count = self.processor.count_walls_between_points(
                        grid_info['walls_detected'],
                        (x_tx_pixel, y_tx_pixel),
                        (x_rx_pixel, y_rx_pixel)
                    )
                    
                    # Calcul du pathloss 2D
                    pathloss = self.calculator.calculate_pathloss(distance_2d, wall_count)
                    
                    received_power = power_tx - pathloss
                
                # Garder le meilleur signal
                if received_power > best_signal:
                    best_signal = received_power
            
            signal_levels.append(best_signal)
            
            # Vérifier si le point est couvert
            if best_signal >= target_coverage_db:
                covered_points += 1
        
        # Calcul des statistiques
        total_points = len(coverage_points)
        coverage_percent = (covered_points / total_points) * 100 if total_points > 0 else 0.0
        
        # Score de qualité (pénalise le nombre d'AP tout en favorisant la couverture)
        num_aps = len(access_points)
        coverage_score = coverage_percent / 100.0
        efficiency_penalty = num_aps * 0.05  # Pénalité pour trop d'AP
        
        # Score final
        score = coverage_score - efficiency_penalty
        
        # Bonus si on atteint l'objectif minimal
        if coverage_percent >= min_coverage_percent:
            score += 0.5
        
        coverage_stats = {
            'covered_points': covered_points,
            'total_points': total_points,
            'coverage_percent': coverage_percent,
            'signal_levels': signal_levels,
            'num_access_points': num_aps
        }
        
        return max(score, 0.0), coverage_stats
    
    def optimize_access_points_genetic_2d(self, coverage_points, grid_info, longueur, largeur, 
                                        target_coverage_db=-70.0, min_coverage_percent=90.0, 
                                        max_access_points=6, power_tx=20.0):
        """
        Optimise le placement des points d'accès 2D avec un algorithme génétique.
        
        Args:
            coverage_points: Points à couvrir
            grid_info: Informations sur la grille
            longueur, largeur: Dimensions
            target_coverage_db: Signal minimal requis
            min_coverage_percent: Couverture minimale
            max_access_points: Nombre maximal d'AP
            power_tx: Puissance de transmission
            
        Returns:
            best_config: Meilleure configuration trouvée
            optimization_history: Historique de l'optimisation
        """
        def objective_function(x):
            """Fonction objectif pour l'optimisation"""
            # Décodage des paramètres
            num_aps = int(x[0])
            
            # VÉRIFICATION: Respect de la contrainte max_access_points
            if num_aps == 0 or num_aps > max_access_points:
                return 1000.0  # Pénalité pour violation des contraintes
            
            access_points = []
            for i in range(num_aps):
                if i * 2 + 2 < len(x):
                    ap_x = x[i * 2 + 1] * longueur
                    ap_y = x[i * 2 + 2] * largeur
                    access_points.append((ap_x, ap_y, power_tx))
            
            if len(access_points) == 0:
                return 1000.0
            
            score, stats = self.calculate_coverage_quality_2d(
                access_points, coverage_points, grid_info,
                target_coverage_db, min_coverage_percent
            )
            
            # NOUVELLES CONTRAINTES: Arrêt anticipé si objectif atteint avec moins d'AP
            current_coverage = stats.get('coverage_percent', 0.0)
            
            # Si on dépasse largement l'objectif, favoriser moins d'AP
            if current_coverage >= min_coverage_percent:
                # Bonus pour atteindre l'objectif avec le minimum d'AP
                efficiency_bonus = (max_access_points - num_aps) * 0.1
                score += efficiency_bonus
                
                # Pénalité légère pour dépassement excessif de l'objectif
                if current_coverage > min_coverage_percent + 20:
                    excess_penalty = (current_coverage - min_coverage_percent - 20) * 0.01
                    score -= excess_penalty
            
            return -score  # Minimisation (négatif du score)
        
        # Limites pour l'optimisation
        # [num_aps, x1, y1, x2, y2, ...]
        bounds = [(1, max_access_points)]  # Nombre d'AP
        for i in range(max_access_points):
            bounds.extend([(0.1, 0.9), (0.1, 0.9)])  # x, y normalisés
        
        # Optimisation
        print(f"Début de l'optimisation génétique 2D (max {max_access_points} AP, objectif {min_coverage_percent}%)...")
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=30,
            popsize=15,
            seed=42,
            atol=1e-3,
            tol=1e-3
        )
        
        # Décodage du résultat
        x_opt = result.x
        num_aps_opt = int(x_opt[0])
        
        # VÉRIFICATION FINALE: Respect des contraintes
        if num_aps_opt > max_access_points:
            num_aps_opt = max_access_points
            print(f"⚠️ Contrainte appliquée: limitation à {max_access_points} AP")
        
        optimized_access_points = []
        for i in range(num_aps_opt):
            if i * 2 + 2 < len(x_opt):
                ap_x = x_opt[i * 2 + 1] * longueur
                ap_y = x_opt[i * 2 + 2] * largeur
                optimized_access_points.append((ap_x, ap_y, power_tx))
        
        # Calcul des statistiques finales
        final_score, final_stats = self.calculate_coverage_quality_2d(
            optimized_access_points, coverage_points, grid_info,
            target_coverage_db, min_coverage_percent
        )
        
        best_config = {
            'access_points': optimized_access_points,
            'score': final_score,
            'stats': final_stats,
            'optimization_result': result
        }
        
        optimization_history = {
            'function_evaluations': result.nfev,
            'success': result.success,
            'final_score': final_score
        }
        
        return best_config, optimization_history
    
    def optimize_with_clustering_2d(self, coverage_points, grid_info, longueur, largeur, 
                                  target_coverage_db=-70.0, min_coverage_percent=90.0, 
                                  power_tx=20.0, max_access_points=6):
        """
        Optimise en utilisant le clustering pour placer les AP près des centres de zones 2D.
        
        Args:
            coverage_points: Points à couvrir
            grid_info: Informations sur la grille
            longueur, largeur: Dimensions
            target_coverage_db: Signal minimal requis
            min_coverage_percent: Couverture minimale
            power_tx: Puissance de transmission
            max_access_points: Nombre maximal de points d'accès
            
        Returns:
            best_config: Meilleure configuration trouvée
            cluster_analysis: Analyse des clusters
        """
        if len(coverage_points) == 0:
            return {'access_points': [], 'score': 0.0, 'stats': {}}, {}
        
        # Conversion en array numpy
        points_array = np.array(coverage_points)
        
        best_config = None
        best_score = -1.0
        cluster_analysis = {}
        
        # Test différents nombres de clusters (AP) - RESPECT DE LA CONTRAINTE MAX
        max_clusters_to_test = min(max_access_points, 6)  # Respect de la contrainte utilisateur
        print(f"Clustering 2D: test de 1 à {max_clusters_to_test} AP (objectif {min_coverage_percent}%)")
        
        for num_clusters in range(1, max_clusters_to_test + 1):
            # Clustering K-means
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(points_array)
            cluster_centers = kmeans.cluster_centers_
            
            # Ajustement des centres pour éviter les murs
            adjusted_centers = []
            for center in cluster_centers:
                x, y = center
                
                # Vérification si dans un mur
                x_pixel = int(np.clip(x / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                y_pixel = int(np.clip(y / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                
                # Si dans un mur, déplacer vers le point le plus proche qui n'est pas dans un mur
                if grid_info['walls_detected'][y_pixel, x_pixel] > 0:
                    # Trouver le point du cluster le plus proche qui n'est pas dans un mur
                    cluster_points = points_array[cluster_labels == len(adjusted_centers)]
                    if len(cluster_points) > 0:
                        # Prendre le centroïde des points valides
                        x, y = np.mean(cluster_points, axis=0)
                
                adjusted_centers.append((x, y, power_tx))
            
            # Évaluation de cette configuration
            score, stats = self.calculate_coverage_quality_2d(
                adjusted_centers, coverage_points, grid_info,
                target_coverage_db, min_coverage_percent
            )
            
            cluster_analysis[num_clusters] = {
                'centers': adjusted_centers,
                'score': score,
                'stats': stats,
                'cluster_labels': cluster_labels
            }
            
            # Mise à jour du meilleur score
            if score > best_score:
                best_score = score
                best_config = {
                    'access_points': adjusted_centers,
                    'score': score,
                    'stats': stats,
                    'num_clusters': num_clusters
                }
            
            # ARRÊT ANTICIPÉ: Si l'objectif est atteint avec ce nombre d'AP
            current_coverage = stats.get('coverage_percent', 0.0)
            if current_coverage >= min_coverage_percent:
                print(f"✅ Objectif de couverture {min_coverage_percent}% atteint avec {num_clusters} AP ({current_coverage:.1f}%)")
                break  # Arrêt pour éviter d'ajouter plus d'AP inutilement
            else:
                print(f"📊 {num_clusters} AP: {current_coverage:.1f}% de couverture")
        
        return best_config, cluster_analysis
    
    def visualize_optimization_result_2d(self, best_config, coverage_points, grid_info, 
                                       longueur, largeur, image_array):
        """
        Visualise le résultat de l'optimisation en 2D avec matplotlib.
        
        Args:
            best_config: Configuration optimale
            coverage_points: Points à couvrir
            grid_info: Informations sur la grille
            longueur, largeur: Dimensions
            image_array: Image du plan original
            
        Returns:
            fig: Figure matplotlib
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        access_points = best_config['access_points']
        stats = best_config['stats']
        
        # === GRAPHIQUE 1: Plan avec points d'accès optimisés ===
        
        # Affichage du plan de base
        ax1.imshow(image_array, extent=[0, longueur, largeur, 0], cmap='gray', alpha=0.7)
        ax1.set_xlim(0, longueur)
        ax1.set_ylim(largeur, 0)
        ax1.set_xlabel('Longueur (m)')
        ax1.set_ylabel('Largeur (m)')
        ax1.set_title(f'Points d\'Accès Optimisés\n{len(access_points)} AP - {stats["coverage_percent"]:.1f}% couverture')
        
        # Affichage des points d'accès
        for i, (x, y, power) in enumerate(access_points):
            # Point d'accès
            ax1.scatter(x, y, c='red', s=200, marker='*', edgecolors='black', linewidth=2, 
                       label=f'AP{i+1}' if i == 0 else '', zorder=5)
            
            # Rayon de couverture approximatif
            estimated_range = 15.0 - len(access_points) * 2  # Approximation simple
            circle = plt.Circle((x, y), estimated_range, fill=False, color='red', alpha=0.6, linestyle='--')
            ax1.add_patch(circle)
            
            # Étiquette
            ax1.annotate(f'AP{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, fontweight='bold', color='red')
        
        # Points de couverture
        if len(coverage_points) < 500:  # Éviter la surcharge visuelle
            coverage_x = [p[0] for p in coverage_points]
            coverage_y = [p[1] for p in coverage_points]
            ax1.scatter(coverage_x, coverage_y, c='lightblue', s=10, alpha=0.5, label='Points à couvrir')
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # === GRAPHIQUE 2: Heatmap de qualité de signal ===
        
        # Génération d'une grille pour la heatmap
        resolution_heatmap = 30
        x_heat = np.linspace(0, longueur, resolution_heatmap)
        y_heat = np.linspace(0, largeur, resolution_heatmap)
        X_heat, Y_heat = np.meshgrid(x_heat, y_heat)
        
        signal_strength = np.zeros_like(X_heat)
        
        for i in range(resolution_heatmap):
            for j in range(resolution_heatmap):
                x_pos, y_pos = X_heat[i, j], Y_heat[i, j]
                
                # Vérifier si dans un mur
                x_pixel = int(np.clip(x_pos / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                y_pixel = int(np.clip(y_pos / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                
                if grid_info['walls_detected'][y_pixel, x_pixel] > 0:
                    signal_strength[i, j] = -120  # Mur
                else:
                    best_signal = -200
                    for ap_x, ap_y, power in access_points:
                        distance = np.sqrt((x_pos - ap_x)**2 + (y_pos - ap_y)**2)
                        if distance < 0.1:
                            received = power - 10
                        else:
                            # Calcul simplifié pour la visualisation
                            wall_count = int(distance * 0.3)  # Approximation
                            pathloss = self.calculator.calculate_pathloss(distance, wall_count)
                            received = power - pathloss
                        
                        if received > best_signal:
                            best_signal = received
                    
                    signal_strength[i, j] = best_signal
        
        # Affichage de la heatmap
        im = ax2.imshow(signal_strength, extent=[0, longueur, largeur, 0], 
                       cmap='RdYlGn', vmin=-100, vmax=-30, alpha=0.8)
        
        # Contours de qualité
        levels = [-90, -70, -50]
        contours = ax2.contour(X_heat, Y_heat, signal_strength, levels=levels, 
                              colors=['orange', 'yellow', 'green'], linewidths=2)
        ax2.clabel(contours, inline=True, fontsize=8, fmt='%d dB')
        
        # Points d'accès sur la heatmap
        for i, (x, y, power) in enumerate(access_points):
            ax2.scatter(x, y, c='black', s=150, marker='*', edgecolors='white', linewidth=2)
            ax2.annotate(f'AP{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, fontweight='bold', color='white')
        
        ax2.set_xlim(0, longueur)
        ax2.set_ylim(largeur, 0)
        ax2.set_xlabel('Longueur (m)')
        ax2.set_ylabel('Largeur (m)')
        ax2.set_title('Heatmap de Qualité du Signal (dBm)')
        
        # Barre de couleur
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Puissance du signal (dBm)')
        
        plt.tight_layout()
        return fig
    
    def generate_optimization_report_2d(self, best_config, cluster_analysis, optimization_history):
        """
        Génère un rapport détaillé de l'optimisation 2D.
        
        Args:
            best_config: Configuration optimale
            cluster_analysis: Analyse des clusters
            optimization_history: Historique de l'optimisation
            
        Returns:
            report: Dictionnaire avec le rapport détaillé
        """
        stats = best_config['stats']
        access_points = best_config['access_points']
        
        # Analyse de la qualité de couverture
        signal_levels = stats.get('signal_levels', [])
        if signal_levels:
            excellent_coverage = sum(1 for s in signal_levels if s >= -50)
            good_coverage = sum(1 for s in signal_levels if -70 <= s < -50)
            poor_coverage = sum(1 for s in signal_levels if -85 <= s < -70)
            no_coverage = sum(1 for s in signal_levels if s < -85)
        else:
            excellent_coverage = good_coverage = poor_coverage = no_coverage = 0
        
        # Génération des recommandations
        recommendations = []
        
        if stats['coverage_percent'] < 80:
            recommendations.append("La couverture est insuffisante. Considérez l'ajout de points d'accès supplémentaires.")
        elif stats['coverage_percent'] > 95:
            recommendations.append("Excellente couverture atteinte. La configuration est optimale.")
        
        if len(access_points) > 4:
            recommendations.append("Nombre élevé de points d'accès. Vérifiez s'il est possible de réduire en augmentant la puissance.")
        
        if poor_coverage > len(signal_levels) * 0.2:
            recommendations.append("Zones avec signal faible détectées. Repositionnez les points d'accès vers ces zones.")
        
        report = {
            'summary': {
                'num_access_points': len(access_points),
                'coverage_percent': stats['coverage_percent'],
                'covered_points': stats['covered_points'],
                'total_points': stats['total_points'],
                'optimization_score': best_config['score']
            },
            'access_points_config': [
                {
                    'id': i+1,
                    'position_x': round(ap[0], 2),
                    'position_y': round(ap[1], 2),
                    'power_dbm': round(ap[2], 1)
                }
                for i, ap in enumerate(access_points)
            ],
            'coverage_analysis': {
                'excellent_coverage': excellent_coverage,
                'good_coverage': good_coverage,
                'poor_coverage': poor_coverage,
                'no_coverage': no_coverage
            },
            'optimization_details': optimization_history,
            'cluster_analysis': cluster_analysis,
            'recommendations': recommendations
        }
        
        return report
    
    def export_optimization_csv_2d(self, best_config, report):
        """
        Exporte la configuration optimisée en format CSV.
        
        Args:
            best_config: Configuration optimale
            report: Rapport d'optimisation
            
        Returns:
            csv_data: Données CSV en format string
        """
        output = io.StringIO()
        
        # En-tête du rapport
        output.write("RAPPORT D'OPTIMISATION DES POINTS D'ACCES 2D\n")
        output.write("=" * 50 + "\n\n")
        
        # Résumé
        summary = report['summary']
        output.write("RESUME DE LA CONFIGURATION\n")
        output.write(f"Nombre de points d'accès,{summary['num_access_points']}\n")
        output.write(f"Couverture (%),{summary['coverage_percent']:.1f}\n")
        output.write(f"Points couverts,{summary['covered_points']}/{summary['total_points']}\n")
        output.write(f"Score d'optimisation,{summary['optimization_score']:.3f}\n\n")
        
        # Configuration des points d'accès
        output.write("CONFIGURATION DES POINTS D'ACCES\n")
        output.write("ID,Position X (m),Position Y (m),Puissance (dBm)\n")
        
        for ap_config in report['access_points_config']:
            output.write(f"AP{ap_config['id']},{ap_config['position_x']},{ap_config['position_y']},{ap_config['power_dbm']}\n")
        
        output.write("\n")
        
        # Analyse de couverture
        coverage = report['coverage_analysis']
        output.write("ANALYSE DE LA QUALITE DE COUVERTURE\n")
        output.write("Qualité,Nombre de points\n")
        output.write(f"Excellent (≥-50dB),{coverage['excellent_coverage']}\n")
        output.write(f"Bon (-50 à -70dB),{coverage['good_coverage']}\n")
        output.write(f"Faible (-70 à -85dB),{coverage['poor_coverage']}\n")
        output.write(f"Pas de couverture (<-85dB),{coverage['no_coverage']}\n\n")
        
        # Recommandations
        output.write("RECOMMANDATIONS\n")
        for i, rec in enumerate(report['recommendations'], 1):
            output.write(f"{i}. {rec}\n")
        
        csv_data = output.getvalue()
        output.close()
        
        return csv_data
    
    def optimize_with_algorithm_choice_2d(self, coverage_points, grid_info, longueur, largeur,
                                         target_coverage_db=-70.0, min_coverage_percent=90.0,
                                         power_tx=20.0, max_access_points=6, algorithm='kmeans'):
        """
        Optimise le placement des points d'accès 2D avec choix d'algorithme.
        
        Args:
            coverage_points: Points à couvrir
            grid_info: Informations sur la grille
            longueur, largeur: Dimensions
            target_coverage_db: Signal minimal requis
            min_coverage_percent: Couverture minimale
            power_tx: Puissance de transmission
            max_access_points: Nombre maximal de points d'accès
            algorithm: 'kmeans', 'gmm' ou 'greedy' - algorithme d'optimisation à utiliser
            
        Returns:
            best_config: Meilleure configuration trouvée
            analysis: Analyse de l'algorithme utilisé
        """
        if algorithm.lower() == 'gmm':
            return self._optimize_with_gmm_2d(
                coverage_points, grid_info, longueur, largeur,
                target_coverage_db, min_coverage_percent, power_tx, max_access_points
            )
        elif algorithm.lower() == 'greedy':
            return self._optimize_with_greedy_2d(
                coverage_points, grid_info, longueur, largeur,
                target_coverage_db, min_coverage_percent, power_tx, max_access_points
            )
        elif algorithm.lower() == 'kmeans':
            return self.optimize_with_clustering_2d(
                coverage_points, grid_info, longueur, largeur,
                target_coverage_db, min_coverage_percent, power_tx, max_access_points
            )
        else:
            raise ValueError(f"Algorithme non supporté: {algorithm}. Utilisez 'kmeans', 'gmm' ou 'greedy'.")
    
    def _optimize_with_gmm_2d(self, coverage_points, grid_info, longueur, largeur,
                             target_coverage_db=-70.0, min_coverage_percent=90.0,
                             power_tx=20.0, max_access_points=6):
        """
        Optimise avec GMM + EM en utilisant le module gmm_optimizer.
        """
        if len(coverage_points) == 0:
            return {'access_points': [], 'score': 0.0, 'stats': {}}, {}
        
        # Adapter la méthode d'évaluation du GMM optimizer pour utiliser notre calculateur
        original_evaluate = self.gmm_optimizer._evaluate_configuration
        
        def adapted_evaluate(access_points, coverage_points, grid_info, target_coverage_db, min_coverage_percent):
            # Utiliser notre méthode de calcul de qualité existante
            return self.calculate_coverage_quality_2d(
                access_points, coverage_points, grid_info, target_coverage_db, min_coverage_percent
            )
        
        # Remplacer temporairement la méthode d'évaluation
        self.gmm_optimizer._evaluate_configuration = adapted_evaluate
        
        try:
            # Utiliser l'optimiseur GMM
            best_config, gmm_analysis = self.gmm_optimizer.optimize_clustering_gmm(
                coverage_points, grid_info, longueur, largeur,
                target_coverage_db, min_coverage_percent, power_tx, max_access_points
            )
            
            # Ajouter des informations spécifiques à notre contexte
            if best_config:
                best_config['algorithm_used'] = 'GMM+EM'
                best_config['frequency_mhz'] = self.frequency_mhz
            
            return best_config, gmm_analysis
            
        finally:
            # Restaurer la méthode d'évaluation originale
            self.gmm_optimizer._evaluate_configuration = original_evaluate
    
    def _optimize_with_greedy_2d(self, coverage_points, grid_info, longueur, largeur,
                                target_coverage_db=-70.0, min_coverage_percent=90.0,
                                power_tx=20.0, max_access_points=6):
        """
        Optimise avec l'algorithme Greedy en utilisant le module greedy_optimizer.
        """
        if len(coverage_points) == 0:
            return {'access_points': [], 'score': 0.0, 'stats': {}}, {}
        
        print("🎯 Optimisation avec algorithme Greedy...")
        
        # Utiliser l'optimiseur Greedy avec les bons paramètres
        result = self.greedy_optimizer.optimize_greedy_placement(
            coverage_points, grid_info, longueur, largeur,
            target_coverage_db, min_coverage_percent, power_tx, max_access_points
        )
        
        if result:
            best_config, greedy_analysis = result
            
            # Ajouter des informations spécifiques à notre contexte
            if best_config:
                best_config['algorithm_used'] = 'Greedy'
                best_config['frequency_mhz'] = self.frequency_mhz
            
            return best_config, greedy_analysis
        else:
            return {'access_points': [], 'score': 0.0, 'stats': {}}, {}

    def compare_algorithms_2d(self, coverage_points, grid_info, longueur, largeur,
                             target_coverage_db=-70.0, min_coverage_percent=90.0,
                             power_tx=20.0, max_access_points=6):
        """
        Compare les performances de K-means vs GMM vs Greedy sur le même jeu de données.
        
        Returns:
            comparison_results: Résultats de comparaison
        """
        print("🔬 Comparaison K-means vs GMM vs Greedy...")
        
        # Test avec K-means
        print("📊 Test K-means...")
        kmeans_config, kmeans_analysis = self.optimize_with_algorithm_choice_2d(
            coverage_points, grid_info, longueur, largeur,
            target_coverage_db, min_coverage_percent, power_tx, max_access_points,
            algorithm='kmeans'
        )
        
        # Test avec GMM
        print("📊 Test GMM...")
        gmm_config, gmm_analysis = self.optimize_with_algorithm_choice_2d(
            coverage_points, grid_info, longueur, largeur,
            target_coverage_db, min_coverage_percent, power_tx, max_access_points,
            algorithm='gmm'
        )
        
        # Test avec Greedy
        print("📊 Test Greedy...")
        greedy_config, greedy_analysis = self.optimize_with_algorithm_choice_2d(
            coverage_points, grid_info, longueur, largeur,
            target_coverage_db, min_coverage_percent, power_tx, max_access_points,
            algorithm='greedy'
        )
        
        # Comparaison des résultats
        comparison = {
            'kmeans': {
                'config': kmeans_config,
                'analysis': kmeans_analysis,
                'coverage_percent': kmeans_config['stats']['coverage_percent'] if kmeans_config else 0,
                'num_access_points': len(kmeans_config['access_points']) if kmeans_config else 0,
                'score': kmeans_config['score'] if kmeans_config else 0
            },
            'gmm': {
                'config': gmm_config,
                'analysis': gmm_analysis,
                'coverage_percent': gmm_config['stats']['coverage_percent'] if gmm_config else 0,
                'num_access_points': len(gmm_config['access_points']) if gmm_config else 0,
                'score': gmm_config['score'] if gmm_config else 0
            },
            'greedy': {
                'config': greedy_config,
                'analysis': greedy_analysis,
                'coverage_percent': greedy_config['stats']['coverage_percent'] if greedy_config else 0,
                'num_access_points': len(greedy_config['access_points']) if greedy_config else 0,
                'score': greedy_config['score'] if greedy_config else 0
            }
        }
        
        # Déterminer le meilleur algorithme
        best_algorithm = None
        best_score = -1
        
        for algo in ['kmeans', 'gmm', 'greedy']:
            if comparison[algo]['config'] and comparison[algo]['score'] > best_score:
                best_score = comparison[algo]['score']
                best_algorithm = algo
        
        comparison['recommended'] = best_algorithm
        
        if best_algorithm:
            # Calculer l'amélioration par rapport aux autres
            other_scores = [comparison[algo]['score'] for algo in ['kmeans', 'gmm', 'greedy'] 
                          if algo != best_algorithm and comparison[algo]['config']]
            if other_scores:
                comparison['improvement'] = best_score - max(other_scores)
            else:
                comparison['improvement'] = 0
        
        return comparison
    
    def visualize_algorithm_comparison_2d(self, comparison_results, coverage_points, 
                                          grid_info, longueur, largeur, image_array):
        """
        Visualise la comparaison entre K-means, GMM et Greedy.
        
        Args:
            comparison_results: Résultats de la comparaison
            coverage_points: Points à couvrir
            grid_info: Informations sur la grille
            longueur, largeur: Dimensions
            image_array: Image de fond
            
        Returns:
            fig: Figure matplotlib avec comparaison
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comparaison K-means vs GMM vs Greedy pour Optimisation WiFi 2D', 
                    fontsize=16, fontweight='bold')
        
        # Configuration des sous-graphiques
        algorithms = ['kmeans', 'gmm', 'greedy']
        algorithm_names = ['K-means', 'GMM + EM', 'Greedy']
        colors = ['blue', 'green', 'red']
        
        for idx, (algo, name, color) in enumerate(zip(algorithms, algorithm_names, colors)):
            if algo in comparison_results and comparison_results[algo]['config']:
                config = comparison_results[algo]['config']
                access_points = config['access_points']
                stats = config['stats']
                
                # Graphique des positions (ligne du haut)
                ax_pos = axes[0, idx]
                
                # Image de fond
                if image_array is not None:
                    ax_pos.imshow(image_array, extent=[0, longueur, largeur, 0], cmap='gray', alpha=0.7)
                
                # Points de couverture
                if len(coverage_points) < 300:  # Éviter la surcharge
                    coverage_x = [p[0] for p in coverage_points]
                    coverage_y = [p[1] for p in coverage_points]
                    ax_pos.scatter(coverage_x, coverage_y, c='lightblue', s=8, alpha=0.4, label='Points à couvrir')
                
                # Points d'accès
                for i, (x, y, power) in enumerate(access_points):
                    ax_pos.scatter(x, y, c=color, s=200, marker='*', 
                                 edgecolors='black', linewidth=2, zorder=5)
                    
                    # Rayon de couverture estimé
                    estimated_range = max(3.0, min(12.0, power / 3.0))
                    circle = plt.Circle((x, y), estimated_range, fill=False, 
                                      color=color, alpha=0.6, linestyle='--')
                    ax_pos.add_patch(circle)
                    
                    # Étiquette
                    ax_pos.annotate(f'AP{i+1}', (x, y), xytext=(5, 5), 
                                  textcoords='offset points', fontsize=9, 
                                  fontweight='bold', color='white')
                
                ax_pos.set_xlim(0, longueur)
                ax_pos.set_ylim(largeur, 0)
                ax_pos.set_xlabel('Longueur (m)')
                ax_pos.set_ylabel('Largeur (m)')
                ax_pos.set_title(f'{name}\n{len(access_points)} AP - {stats["coverage_percent"]:.1f}% couverture')
                ax_pos.grid(True, alpha=0.3)
                if idx == 0 and len(coverage_points) < 300:
                    ax_pos.legend(fontsize=8)
                
                # Métriques (ligne du bas)
                ax_metrics = axes[1, idx]
                
                # Données pour le graphique en barres
                metrics_names = ['Couverture\n(%)', 'Nb AP', 'Score\n(*10)', 'Points\nCouverts']
                metrics_values = [
                    stats['coverage_percent'],
                    len(access_points),
                    config['score'] * 10,  # Multiplié pour visibilité
                    stats['covered_points']
                ]
                
                bars = ax_metrics.bar(metrics_names, metrics_values, color=color, alpha=0.7)
                
                # Ajout des valeurs sur les barres
                for bar, value, metric in zip(bars, metrics_values, metrics_names):
                    height = bar.get_height()
                    if 'Score' in metric:
                        display_value = f'{value/10:.3f}'  # Valeur réelle du score
                    elif 'Couverture' in metric:
                        display_value = f'{value:.1f}%'
                    else:
                        display_value = f'{int(value)}'
                    
                    ax_metrics.text(bar.get_x() + bar.get_width()/2., height + max(metrics_values)*0.01,
                                  display_value, ha='center', va='bottom', fontweight='bold')
                
                ax_metrics.set_title(f'Métriques {name}')
                ax_metrics.set_ylabel('Valeur')
                ax_metrics.grid(True, alpha=0.3, axis='y')
                
                # Informations supplémentaires spécifiques à chaque algorithme
                info_text = ""
                if algo == 'gmm' and 'gmm_metrics' in config:
                    gmm_metrics = config['gmm_metrics']
                    info_text = f"AIC: {gmm_metrics['aic']:.1f}\n"
                    info_text += f"BIC: {gmm_metrics['bic']:.1f}\n"
                    info_text += f"Convergé: {'Oui' if gmm_metrics['converged'] else 'Non'}"
                elif algo == 'greedy' and 'steps' in comparison_results[algo]['analysis']:
                    steps = comparison_results[algo]['analysis']['steps']
                    info_text = f"Étapes: {len(steps)}\n"
                    info_text += f"Itérations: {comparison_results[algo]['analysis']['total_iterations']}\n"
                    info_text += f"Convergence: {comparison_results[algo]['analysis']['convergence_reason'][:20]}..."
                elif algo == 'kmeans':
                    info_text = f"Clusters: {len(access_points)}\n"
                    info_text += f"Clustering rapide\n"
                    info_text += f"Stable et efficace"
                
                if info_text:
                    ax_metrics.text(0.02, 0.98, info_text, transform=ax_metrics.transAxes,
                                  fontsize=8, verticalalignment='top',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Résumé de comparaison
        if 'recommended' in comparison_results:
            recommended = comparison_results['recommended']
            if recommended:
                improvement = comparison_results.get('improvement', 0)
                algo_names = {'kmeans': 'K-means', 'gmm': 'GMM + EM', 'greedy': 'Greedy'}
                
                summary_text = f"🏆 Algorithme recommandé: {algo_names[recommended]}\n"
                summary_text += f"📈 Amélioration du score: +{improvement:.3f}"
                
                fig.text(0.5, 0.02, summary_text, ha='center', fontsize=12, 
                        fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1, top=0.93)
        return fig
