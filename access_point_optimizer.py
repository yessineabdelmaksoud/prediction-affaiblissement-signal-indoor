import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import differential_evolution, minimize
from sklearn.cluster import KMeans
import pandas as pd
from pathloss_calculator_3d import PathlossCalculator3D
from image_processor import ImageProcessor
from gmm_optimizer_3d import GMMOptimizer3D
from greedy_optimizer_3d import GreedyOptimizer3D

class AccessPointOptimizer:
    def __init__(self, frequency_mhz):
        """
        Optimiseur pour la placement automatique des points d'accès 3D.
        
        Args:
            frequency_mhz: Fréquence en MHz
        """
        self.frequency_mhz = frequency_mhz
        self.calculator_3d = PathlossCalculator3D(frequency_mhz)
        self.processor = ImageProcessor()
        # Initialisation des optimiseurs spécialisés
        self.gmm_optimizer = GMMOptimizer3D(frequency_mhz)
        self.greedy_optimizer = GreedyOptimizer3D(frequency_mhz)
        
    def generate_coverage_zones(self, walls_detected, longueur, largeur, hauteur_totale, 
                               resolution_xy=20, resolution_z=8):
        """
        Génère une grille de points à couvrir dans l'espace 3D.
        
        Args:
            walls_detected: Masque binaire des murs
            longueur, largeur, hauteur_totale: Dimensions en mètres
            resolution_xy: Résolution dans le plan XY
            resolution_z: Résolution en Z
            
        Returns:
            coverage_points: Liste des points à couvrir [(x, y, z), ...]
            grid_info: Informations sur la grille
        """
        # Création des grilles de coordonnées
        x_coords = np.linspace(0.5, longueur - 0.5, resolution_xy)
        y_coords = np.linspace(0.5, largeur - 0.5, resolution_xy)
        z_coords = np.linspace(0.5, hauteur_totale - 0.5, resolution_z)
        
        # Échelles de conversion pour les murs 2D
        height_2d, width_2d = walls_detected.shape
        scale_x = longueur / width_2d
        scale_y = largeur / height_2d
        
        coverage_points = []
        
        for z in z_coords:
            for y in y_coords:
                for x in x_coords:
                    # Vérification si le point n'est pas dans un mur
                    x_pixel = int(np.clip(x / scale_x, 0, width_2d - 1))
                    y_pixel = int(np.clip(y / scale_y, 0, height_2d - 1))
                    
                    # Si pas dans un mur, ajouter à la liste des points à couvrir
                    if walls_detected[y_pixel, x_pixel] == 0:
                        coverage_points.append((x, y, z))
        
        grid_info = {
            'x_coords': x_coords,
            'y_coords': y_coords,
            'z_coords': z_coords,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'walls_detected': walls_detected
        }
        
        return coverage_points, grid_info
    
    def calculate_coverage_quality(self, access_points, coverage_points, grid_info, 
                                 target_coverage_db=-70.0, min_coverage_percent=90.0):
        """
        Calcule la qualité de couverture pour une configuration de points d'accès.
        
        Args:
            access_points: Liste des points d'accès [(x, y, z, power), ...]
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
            x_rx, y_rx, z_rx = point
            best_signal = -200.0  # Très faible
            
            for ap in access_points:
                x_tx, y_tx, z_tx, power_tx = ap
                
                # Distance 3D
                distance_3d = np.sqrt((x_rx - x_tx)**2 + (y_rx - y_tx)**2 + (z_rx - z_tx)**2)
                
                if distance_3d < 0.1:  # Très proche
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
                    
                    # Différence d'étages (estimation)
                    floor_tx = int(z_tx // 2.7)
                    floor_rx = int(z_rx // 2.7)
                    floor_difference = abs(floor_rx - floor_tx)
                    
                    # Calcul du pathloss
                    pathloss = self.calculator_3d.calculate_pathloss_3d(
                        distance_3d, wall_count, floor_difference
                    )
                    
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
    
    def optimize_access_points_genetic(self, coverage_points, grid_info, longueur, largeur, 
                                     hauteur_totale, target_coverage_db=-70.0, 
                                     min_coverage_percent=90.0, max_access_points=8,
                                     power_tx=20.0):
        """
        Optimise le placement des points d'accès avec un algorithme génétique.
        
        Args:
            coverage_points: Points à couvrir
            grid_info: Informations sur la grille
            longueur, largeur, hauteur_totale: Dimensions
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
            if num_aps == 0:
                return 1000.0  # Pénalité pour 0 AP
            
            access_points = []
            for i in range(num_aps):
                if i * 4 + 3 < len(x):
                    ap_x = x[i * 4 + 1] * longueur
                    ap_y = x[i * 4 + 2] * largeur
                    ap_z = x[i * 4 + 3] * hauteur_totale
                    access_points.append((ap_x, ap_y, ap_z, power_tx))
            
            if len(access_points) == 0:
                return 1000.0
            
            score, _ = self.calculate_coverage_quality(
                access_points, coverage_points, grid_info,
                target_coverage_db, min_coverage_percent
            )
            
            return -score  # Minimisation (négatif du score)
        
        # Limites pour l'optimisation
        # [num_aps, x1, y1, z1, x2, y2, z2, ...]
        bounds = [(1, max_access_points)]  # Nombre d'AP
        for i in range(max_access_points):
            bounds.extend([(0.1, 0.9), (0.1, 0.9), (0.1, 0.9)])  # x, y, z normalisés
        
        # Optimisation
        print("Début de l'optimisation génétique...")
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=50,
            popsize=15,
            seed=42,
            atol=1e-3,
            tol=1e-3
        )
        
        # Décodage du résultat
        x_opt = result.x
        num_aps_opt = int(x_opt[0])
        
        optimized_access_points = []
        for i in range(num_aps_opt):
            if i * 4 + 3 < len(x_opt):
                ap_x = x_opt[i * 4 + 1] * longueur
                ap_y = x_opt[i * 4 + 2] * largeur
                ap_z = x_opt[i * 4 + 3] * hauteur_totale
                optimized_access_points.append((ap_x, ap_y, ap_z, power_tx))
        
        # Calcul des statistiques finales
        final_score, final_stats = self.calculate_coverage_quality(
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
    
    def optimize_with_clustering(self, coverage_points, grid_info, longueur, largeur, 
                                hauteur_totale, target_coverage_db=-70.0, 
                                min_coverage_percent=90.0, power_tx=20.0):
        """
        Optimise en utilisant le clustering pour placer les AP près des centres de zones.
        
        Args:
            coverage_points: Points à couvrir
            grid_info: Informations sur la grille
            longueur, largeur, hauteur_totale: Dimensions
            target_coverage_db: Signal minimal requis
            min_coverage_percent: Couverture minimale
            power_tx: Puissance de transmission
            
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
        
        # Test différents nombres de clusters (AP)
        for num_clusters in range(1, 9):
            # Clustering K-means
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(points_array)
            cluster_centers = kmeans.cluster_centers_
            
            # Ajustement des centres pour éviter les murs
            adjusted_centers = []
            for center in cluster_centers:
                x, y, z = center
                
                # Vérification si dans un mur
                x_pixel = int(np.clip(x / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                y_pixel = int(np.clip(y / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                
                # Si dans un mur, déplacer vers le point le plus proche qui n'est pas dans un mur
                if grid_info['walls_detected'][y_pixel, x_pixel] > 0:
                    # Trouver le point du cluster le plus proche qui n'est pas dans un mur
                    cluster_points = points_array[cluster_labels == len(adjusted_centers)]
                    if len(cluster_points) > 0:
                        # Prendre le centroïde des points valides
                        x, y, z = np.mean(cluster_points, axis=0)
                
                adjusted_centers.append((x, y, z, power_tx))
            
            # Évaluation de cette configuration
            score, stats = self.calculate_coverage_quality(
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
        
        return best_config, cluster_analysis
    
    def visualize_optimization_result(self, best_config, coverage_points, grid_info, 
                                    longueur, largeur, hauteur_totale):
        """
        Visualise le résultat de l'optimisation en 3D.
        
        Args:
            best_config: Configuration optimale
            coverage_points: Points à couvrir
            grid_info: Informations sur la grille
            longueur, largeur, hauteur_totale: Dimensions
            
        Returns:
            fig: Figure Plotly 3D
        """
        fig = go.Figure()
        
        access_points = best_config['access_points']
        stats = best_config['stats']
        
        if 'signal_levels' in stats:
            signal_levels = stats['signal_levels']
            
            # Séparation des points par qualité de signal
            excellent_points = []
            good_points = []
            poor_points = []
            no_coverage_points = []
            
            for i, point in enumerate(coverage_points):
                signal = signal_levels[i] if i < len(signal_levels) else -200
                
                if signal >= -50:
                    excellent_points.append(point)
                elif signal >= -70:
                    good_points.append(point)
                elif signal >= -85:
                    poor_points.append(point)
                else:
                    no_coverage_points.append(point)
            
            # Affichage des points par qualité
            coverage_layers = [
                ("Excellent (>-50dB)", excellent_points, 'green', 8),
                ("Bon (-50 à -70dB)", good_points, 'yellow', 6),
                ("Faible (-70 à -85dB)", poor_points, 'orange', 4),
                ("Pas de couverture (<-85dB)", no_coverage_points, 'red', 3)
            ]
            
            for layer_name, points, color, size in coverage_layers:
                if points:
                    points_array = np.array(points)
                    fig.add_trace(go.Scatter3d(
                        x=points_array[:, 0],
                        y=points_array[:, 1],
                        z=points_array[:, 2],
                        mode='markers',
                        marker=dict(
                            size=size,
                            color=color,
                            opacity=0.6,
                            line=dict(width=0)
                        ),
                        name=layer_name
                    ))
        
        # Affichage des points d'accès optimisés
        if access_points:
            ap_coords = np.array([(ap[0], ap[1], ap[2]) for ap in access_points])
            
            fig.add_trace(go.Scatter3d(
                x=ap_coords[:, 0],
                y=ap_coords[:, 1],
                z=ap_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=20,
                    color='red',
                    symbol='diamond',
                    line=dict(width=3, color='black')
                ),
                name=f"Points d'accès optimisés ({len(access_points)})",
                text=[f"AP{i+1}: {ap[3]:.1f}dBm" for i, ap in enumerate(access_points)]
            ))
            
            # Zones de couverture (sphères semi-transparentes)
            for i, ap in enumerate(access_points):
                x_ap, y_ap, z_ap, power = ap
                
                # Estimation du rayon de couverture
                estimated_range = min(15.0, max(5.0, power / 4.0))  # Estimation simple
                
                # Création d'une sphère
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x_sphere = x_ap + estimated_range * np.outer(np.cos(u), np.sin(v))
                y_sphere = y_ap + estimated_range * np.outer(np.sin(u), np.sin(v))
                z_sphere = z_ap + estimated_range * np.outer(np.ones(np.size(u)), np.cos(v))
                
                fig.add_trace(go.Surface(
                    x=x_sphere,
                    y=y_sphere,
                    z=z_sphere,
                    opacity=0.2,
                    colorscale=[[0, 'blue'], [1, 'blue']],
                    showscale=False,
                    name=f"Zone AP{i+1}"
                ))
        
        # Configuration de la mise en page
        fig.update_layout(
            title=f"Optimisation Points d'Accès 3D - {self.frequency_mhz} MHz<br>" +
                  f"Couverture: {stats.get('coverage_percent', 0):.1f}% " +
                  f"({stats.get('covered_points', 0)}/{stats.get('total_points', 0)} points)",
            scene=dict(
                xaxis_title="Longueur (m)",
                yaxis_title="Largeur (m)",
                zaxis_title="Hauteur (m)",
                aspectmode="data",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    def generate_optimization_report(self, best_config, cluster_analysis, optimization_history):
        """
        Génère un rapport d'optimisation détaillé.
        
        Args:
            best_config: Configuration optimale
            cluster_analysis: Analyse des clusters
            optimization_history: Historique d'optimisation
            
        Returns:
            report: Rapport d'optimisation
        """
        report = {
            'summary': {},
            'access_points': [],
            'coverage_analysis': {},
            'recommendations': []
        }
        
        # Résumé
        stats = best_config['stats']
        report['summary'] = {
            'num_access_points': stats.get('num_access_points', 0),
            'coverage_percent': stats.get('coverage_percent', 0),
            'covered_points': stats.get('covered_points', 0),
            'total_points': stats.get('total_points', 0),
            'optimization_score': best_config['score']
        }
        
        # Détails des points d'accès
        for i, ap in enumerate(best_config['access_points']):
            x, y, z, power = ap
            report['access_points'].append({
                'id': i + 1,
                'position_x': round(x, 2),
                'position_y': round(y, 2),
                'position_z': round(z, 2),
                'power_dbm': round(power, 1),
                'floor': int(z // 2.7) + 1
            })
        
        # Analyse de couverture
        if 'signal_levels' in stats:
            signal_levels = stats['signal_levels']
            report['coverage_analysis'] = {
                'excellent_coverage': len([s for s in signal_levels if s >= -50]),
                'good_coverage': len([s for s in signal_levels if -70 <= s < -50]),
                'poor_coverage': len([s for s in signal_levels if -85 <= s < -70]),
                'no_coverage': len([s for s in signal_levels if s < -85]),
                'average_signal': round(np.mean(signal_levels), 1),
                'min_signal': round(np.min(signal_levels), 1),
                'max_signal': round(np.max(signal_levels), 1)
            }
        
        # Recommandations
        recommendations = []
        
        if stats.get('coverage_percent', 0) < 90:
            recommendations.append("Couverture insuffisante. Considérez augmenter la puissance ou ajouter des points d'accès.")
        
        if stats.get('num_access_points', 0) > 6:
            recommendations.append("Nombre élevé de points d'accès. Vérifiez si une puissance plus élevée pourrait réduire ce nombre.")
        
        # Analyse par étage
        access_points = best_config['access_points']
        floors = [int(ap[2] // 2.7) for ap in access_points]
        unique_floors = set(floors)
        
        if len(unique_floors) == 1 and len(access_points) > 1:
            recommendations.append("Tous les points d'accès sont au même étage. Considérez une distribution verticale.")
        
        if stats.get('coverage_percent', 0) >= 95:
            recommendations.append("Excellente couverture atteinte. Configuration optimale.")
        
        if len(recommendations) == 0:
            recommendations.append("Configuration acceptable. Validez avec des mesures terrain.")
        
        report['recommendations'] = recommendations
        
        return report
    
    def export_optimization_csv(self, best_config, report):
        """
        Exporte les résultats d'optimisation en CSV.
        
        Args:
            best_config: Configuration optimale
            report: Rapport d'optimisation
            
        Returns:
            csv_string: Données CSV
        """
        # DataFrame pour les points d'accès
        ap_data = []
        for ap_info in report['access_points']:
            ap_data.append({
                'AP_ID': ap_info['id'],
                'Position_X_m': ap_info['position_x'],
                'Position_Y_m': ap_info['position_y'],
                'Position_Z_m': ap_info['position_z'],
                'Etage': ap_info['floor'],
                'Puissance_dBm': ap_info['power_dbm'],
                'Couverture_Percent': report['summary']['coverage_percent'],
                'Points_Couverts': report['summary']['covered_points'],
                'Points_Total': report['summary']['total_points']
            })
        
        df_ap = pd.DataFrame(ap_data)
        
        # Ajout des statistiques de couverture
        if report['coverage_analysis']:
            coverage_stats = pd.DataFrame([{
                'Metric': 'Couverture_Excellente',
                'Value': report['coverage_analysis']['excellent_coverage']
            }, {
                'Metric': 'Couverture_Bonne',
                'Value': report['coverage_analysis']['good_coverage']
            }, {
                'Metric': 'Couverture_Faible',
                'Value': report['coverage_analysis']['poor_coverage']
            }, {
                'Metric': 'Sans_Couverture',
                'Value': report['coverage_analysis']['no_coverage']
            }, {
                'Metric': 'Signal_Moyen_dB',
                'Value': report['coverage_analysis']['average_signal']
            }])
            
            # Combinaison des données
            csv_content = "=== POINTS D'ACCES OPTIMISES ===\n"
            csv_content += df_ap.to_csv(index=False)
            csv_content += "\n=== STATISTIQUES DE COUVERTURE ===\n"
            csv_content += coverage_stats.to_csv(index=False)
            
            return csv_content
        else:
            return df_ap.to_csv(index=False)
    
    def optimize_with_algorithm_choice_3d(self, algorithm_choice, coverage_points, grid_info, 
                                         longueur, largeur, hauteur_totale, target_coverage_db=-70.0, 
                                         min_coverage_percent=90.0, max_access_points=8, power_tx=20.0):
        """
        Optimise le placement des points d'accès 3D avec choix d'algorithme.
        
        Args:
            algorithm_choice: 'genetic', 'kmeans', 'gmm', ou 'greedy'
            coverage_points: Points à couvrir
            grid_info: Informations sur la grille
            longueur, largeur, hauteur_totale: Dimensions
            target_coverage_db: Signal minimal requis
            min_coverage_percent: Couverture minimale
            max_access_points: Nombre maximal d'AP
            power_tx: Puissance de transmission
            
        Returns:
            best_config: Configuration optimale
            algorithm_analysis: Analyse spécifique à l'algorithme
        """
        
        print(f"🚀 Optimisation 3D avec algorithme: {algorithm_choice.upper()}")
        
        if algorithm_choice == 'genetic':
            return self.optimize_access_points_genetic(
                coverage_points, grid_info, longueur, largeur, hauteur_totale,
                target_coverage_db, min_coverage_percent, max_access_points, power_tx
            )
        
        elif algorithm_choice == 'kmeans':
            return self.optimize_with_clustering(
                coverage_points, grid_info, longueur, largeur, hauteur_totale,
                target_coverage_db, min_coverage_percent, power_tx
            )
        
        elif algorithm_choice == 'gmm':
            return self._optimize_with_gmm_3d(
                coverage_points, grid_info, longueur, largeur, hauteur_totale,
                target_coverage_db, min_coverage_percent, max_access_points, power_tx
            )
        
        elif algorithm_choice == 'greedy':
            return self._optimize_with_greedy_3d(
                coverage_points, grid_info, longueur, largeur, hauteur_totale,
                target_coverage_db, min_coverage_percent, max_access_points, power_tx
            )
        
        else:
            raise ValueError(f"Algorithme non supporté: {algorithm_choice}")
    
    def _optimize_with_gmm_3d(self, coverage_points, grid_info, longueur, largeur, hauteur_totale,
                             target_coverage_db, min_coverage_percent, max_access_points, power_tx):
        """
        Optimisation avec algorithme GMM 3D.
        """
        # Injection de la méthode d'évaluation dans l'optimiseur GMM
        self.gmm_optimizer._evaluate_configuration_3d = lambda aps, cps, gi, tcdb, mcp: \
            self.calculate_coverage_quality(aps, cps, gi, tcdb, mcp)
        
        return self.gmm_optimizer.optimize_clustering_gmm_3d(
            coverage_points, grid_info, longueur, largeur, hauteur_totale,
            target_coverage_db, min_coverage_percent, power_tx, max_access_points
        )
    
    def _optimize_with_greedy_3d(self, coverage_points, grid_info, longueur, largeur, hauteur_totale,
                                target_coverage_db, min_coverage_percent, max_access_points, power_tx):
        """
        Optimisation avec algorithme Greedy 3D.
        """
        # Injection de la méthode d'évaluation dans l'optimiseur Greedy
        self.greedy_optimizer._evaluate_configuration_3d = lambda aps, cps, gi, tcdb, mcp: \
            self.calculate_coverage_quality(aps, cps, gi, tcdb, mcp)
        
        return self.greedy_optimizer.optimize_greedy_placement_3d(
            coverage_points, grid_info, longueur, largeur, hauteur_totale,
            target_coverage_db, min_coverage_percent, power_tx, max_access_points
        )
    
    def compare_algorithms_3d(self, coverage_points, grid_info, longueur, largeur, hauteur_totale,
                             target_coverage_db=-70.0, min_coverage_percent=90.0, 
                             max_access_points=8, power_tx=20.0):
        """
        Compare tous les algorithmes d'optimisation 3D disponibles.
        
        Args:
            coverage_points: Points à couvrir
            grid_info: Informations sur la grille
            longueur, largeur, hauteur_totale: Dimensions
            target_coverage_db: Signal minimal requis
            min_coverage_percent: Couverture minimale
            max_access_points: Nombre maximal d'AP
            power_tx: Puissance de transmission
            
        Returns:
            comparison_results: Résultats comparatifs
        """
        
        algorithms = ['genetic', 'kmeans', 'gmm', 'greedy']
        results = {}
        
        print("🔬 Comparaison des algorithmes d'optimisation 3D...")
        
        for algorithm in algorithms:
            try:
                print(f"\n📊 Test algorithme: {algorithm.upper()}")
                
                config, analysis = self.optimize_with_algorithm_choice_3d(
                    algorithm, coverage_points, grid_info, longueur, largeur, hauteur_totale,
                    target_coverage_db, min_coverage_percent, max_access_points, power_tx
                )
                
                if config:
                    results[algorithm] = {
                        'config': config,
                        'analysis': analysis,
                        'algorithm_name': algorithm.upper(),
                        'success': True
                    }
                    
                    stats = config['stats']
                    print(f"✅ {algorithm.upper()}: {stats['coverage_percent']:.1f}% couverture, "
                          f"{stats['num_access_points']} APs, score {config['score']:.3f}")
                else:
                    results[algorithm] = {
                        'config': None,
                        'analysis': {},
                        'algorithm_name': algorithm.upper(),
                        'success': False
                    }
                    print(f"❌ {algorithm.upper()}: Échec de l'optimisation")
                    
            except Exception as e:
                print(f"⚠️  Erreur {algorithm.upper()}: {e}")
                results[algorithm] = {
                    'config': None,
                    'analysis': {'error': str(e)},
                    'algorithm_name': algorithm.upper(),
                    'success': False
                }
        
        # Détermination du meilleur algorithme
        best_algorithm = None
        best_score = -1.0
        
        for algo, result in results.items():
            if result['success'] and result['config']:
                score = result['config']['score']
                if score > best_score:
                    best_score = score
                    best_algorithm = algo
        
        comparison_results = {
            'algorithms': results,
            'best_algorithm': best_algorithm,
            'best_score': best_score,
            'summary': self._generate_comparison_summary_3d(results)
        }
        
        print(f"\n🏆 Meilleur algorithme: {best_algorithm.upper() if best_algorithm else 'Aucun'}")
        
        return comparison_results
    
    def _generate_comparison_summary_3d(self, results):
        """
        Génère un résumé de la comparaison des algorithmes 3D.
        """
        summary = {
            'total_algorithms': len(results),
            'successful_algorithms': sum(1 for r in results.values() if r['success']),
            'failed_algorithms': sum(1 for r in results.values() if not r['success']),
            'performance_ranking': []
        }
        
        # Classement par performance
        successful_results = [(algo, result) for algo, result in results.items() if result['success']]
        successful_results.sort(key=lambda x: x[1]['config']['score'] if x[1]['config'] else 0, reverse=True)
        
        for i, (algo, result) in enumerate(successful_results):
            stats = result['config']['stats']
            summary['performance_ranking'].append({
                'rank': i + 1,
                'algorithm': algo.upper(),
                'coverage_percent': stats['coverage_percent'],
                'num_access_points': stats['num_access_points'],
                'score': result['config']['score']
            })
        
        return summary
