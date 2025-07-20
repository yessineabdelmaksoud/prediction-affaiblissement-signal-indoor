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
from heatmap_generator import HeatmapGenerator

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
        self.gmm_optimizer = GMMOptimizer(frequency_mhz * 1e6)  # Conversion MHz vers Hz
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
        
        # Test différents nombres de clusters (AP)
        max_clusters_to_test = max_access_points  # Utiliser directement la contrainte utilisateur
        print(f"Clustering 2D: test de 1 à {max_clusters_to_test} AP (objectif {min_coverage_percent}%)")
        
        for num_clusters in range(1, max_clusters_to_test + 1):
            # Clustering K-means
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(points_array)
            cluster_centers = kmeans.cluster_centers_
            
            # Ajustement des centres pour éviter les murs
            adjusted_centers = []
            for cluster_idx, center in enumerate(cluster_centers):
                x, y = center
                
                # Vérification si dans un mur
                x_pixel = int(np.clip(x / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                y_pixel = int(np.clip(y / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                
                # Si dans un mur, déplacer vers un point valide
                if grid_info['walls_detected'][y_pixel, x_pixel] > 0:
                    # Trouver les points de ce cluster spécifique
                    cluster_points = points_array[cluster_labels == cluster_idx]
                    if len(cluster_points) > 0:
                        # Prendre le centroïde des points valides du cluster
                        x, y = np.mean(cluster_points, axis=0)
                        
                        # Re-vérifier si le nouveau centre est encore dans un mur
                        x_pixel_new = int(np.clip(x / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                        y_pixel_new = int(np.clip(y / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                        
                        if grid_info['walls_detected'][y_pixel_new, x_pixel_new] > 0:
                            # Si encore dans un mur, chercher le point le plus proche hors mur
                            best_point = None
                            min_distance = float('inf')
                            
                            for point in cluster_points:
                                px, py = point
                                px_pixel = int(np.clip(px / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                                py_pixel = int(np.clip(py / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                                
                                if grid_info['walls_detected'][py_pixel, px_pixel] == 0:  # Pas dans un mur
                                    distance = np.sqrt((px - center[0])**2 + (py - center[1])**2)
                                    if distance < min_distance:
                                        min_distance = distance
                                        best_point = (px, py)
                            
                            if best_point is not None:
                                x, y = best_point
                
                adjusted_centers.append((x, y, power_tx))
            
            # Validation finale : s'assurer qu'aucun AP n'est dans un mur
            validated_centers = []
            for x, y, power in adjusted_centers:
                x_pixel = int(np.clip(x / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                y_pixel = int(np.clip(y / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                
                if grid_info['walls_detected'][y_pixel, x_pixel] == 0:  # Position valide
                    validated_centers.append((x, y, power))
                else:
                    # Dernière tentative : déplacer légèrement la position
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            new_x_pixel = np.clip(x_pixel + dx, 0, grid_info['walls_detected'].shape[1] - 1)
                            new_y_pixel = np.clip(y_pixel + dy, 0, grid_info['walls_detected'].shape[0] - 1)
                            
                            if grid_info['walls_detected'][new_y_pixel, new_x_pixel] == 0:
                                new_x = new_x_pixel * grid_info['scale_x']
                                new_y = new_y_pixel * grid_info['scale_y']
                                validated_centers.append((new_x, new_y, power))
                                break
                        else:
                            continue
                        break
                    else:
                        # Si aucune position valide trouvée, garder la position originale avec avertissement
                        print(f"⚠️ AP à ({x:.1f}, {y:.1f}) pourrait être dans un mur")
                        validated_centers.append((x, y, power))
            
            adjusted_centers = validated_centers
            
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
            
            # ARRÊT ANTICIPÉ INTELLIGENT: Vérifier si l'objectif est atteint
            current_coverage = stats.get('coverage_percent', 0.0)
            covered_points = stats.get('covered_points', 0)
            total_points = stats.get('total_points', len(coverage_points))
            
            print(f"📊 K-means {num_clusters} AP: {current_coverage:.1f}% de couverture ({covered_points}/{total_points} points)")
            
            # Si objectif atteint, s'arrêter immédiatement avec cette configuration
            if current_coverage >= min_coverage_percent:
                print(f"✅ Objectif {min_coverage_percent}% atteint avec {num_clusters} AP - ARRÊT OPTIMISATION")
                
                # Cette configuration respecte l'objectif, on s'arrête ici
                best_config = {
                    'access_points': adjusted_centers,
                    'score': score,
                    'stats': stats,
                    'num_clusters': num_clusters,
                    'early_stop': True,
                    'early_stop_reason': f"Objectif {min_coverage_percent}% atteint"
                }
                break  # Sortir de la boucle immédiatement
        
        # Affichage final uniforme
        if best_config:
            print(f"✅ Optimisation K-means terminée:")
            print(f"   - Algorithme: K-means Clustering")
            print(f"   - {len(best_config['access_points'])} points d'accès placés")
            final_coverage = best_config['stats']['coverage_percent']
            covered = best_config['stats']['covered_points']
            total = best_config['stats']['total_points']
            print(f"   - {final_coverage:.1f}% de couverture ({covered}/{total} points)")
            print(f"   - Score: {best_config['score']:.3f}")
            
            # Indiquer si arrêt anticipé
            if best_config.get('early_stop', False):
                reason = best_config.get('early_stop_reason', 'Objectif atteint')
                print(f"   - Arrêt anticipé: {reason}")
                print(f"   - Optimisation efficace: minimum d'AP pour l'objectif")
            else:
                print(f"   - Optimisation complète: meilleur score global")
        else:
            print("❌ Aucune configuration K-means trouvée")
        
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
        
        # === GRAPHIQUE 2: Heatmap de qualité de signal avec HeatmapGenerator ===
        
        # Préparation des données émetteur pour le HeatmapGenerator
        emetteurs = []
        for i, (x, y, power) in enumerate(access_points):
            # Conversion des coordonnées en pixels pour compatibilité
            x_pixel = int(x / grid_info['scale_x'])
            y_pixel = int(y / grid_info['scale_y'])
            
            emetteur_data = {
                'position_meter': (x, y),
                'position_pixel': (x_pixel, y_pixel),
                'puissance_totale': power
            }
            emetteurs.append(emetteur_data)
        
        # Création du générateur de heatmap avec la même fréquence
        heatmap_generator = HeatmapGenerator(self.frequency_mhz)
        
        # Génération de la heatmap avec palette coolwarm et résolution 100
        try:
            heatmap_data, extent, heatmap_fig = heatmap_generator.generate_heatmap_2d(
                image_array=image_array,
                walls_detected=grid_info['walls_detected'], 
                emetteurs=emetteurs,
                longueur=longueur,
                largeur=largeur,
                resolution=100,
                colormap='coolwarm'
            )
            
            # Récupération des données de la figure générée pour l'intégrer dans notre subplot
            heatmap_ax = heatmap_fig.gca()
            
            # Copie des éléments de la heatmap vers notre ax2
            for image in heatmap_ax.get_images():
                # Copier l'image de la heatmap
                ax2.imshow(image.get_array(), extent=image.get_extent(), 
                          cmap=image.get_cmap(), alpha=0.8, 
                          vmin=image.get_clim()[0], vmax=image.get_clim()[1])
            
            # Copie des cercles des émetteurs si présents
            for patch in heatmap_ax.patches:
                if isinstance(patch, plt.Circle):
                    new_circle = plt.Circle(patch.center, patch.radius, 
                                          color=patch.get_facecolor(), 
                                          ec=patch.get_edgecolor(),
                                          linewidth=patch.get_linewidth(),
                                          zorder=10)
                    ax2.add_patch(new_circle)
            
            # Copie des annotations
            for text in heatmap_ax.texts:
                ax2.annotate(text.get_text(), text.get_position(),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold', color='white',
                           zorder=11)
            
            # Fermer la figure temporaire
            plt.close(heatmap_fig)
            
        except Exception as e:
            print(f"⚠️ Erreur lors de la génération de la heatmap: {e}")
            # Fallback vers l'ancienne méthode en cas d'erreur
            # Génération d'une grille pour la heatmap
            resolution_heatmap = 50
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
            
            # Affichage de la heatmap de fallback avec coolwarm
            im = ax2.imshow(signal_strength, extent=[0, longueur, largeur, 0], 
                           cmap='coolwarm', vmin=-100, vmax=-30, alpha=0.8)
            
            # Points d'accès sur la heatmap de fallback
            for i, (x, y, power) in enumerate(access_points):
                ax2.scatter(x, y, c='black', s=150, marker='*', edgecolors='white', linewidth=2)
                ax2.annotate(f'AP{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', 
                            fontsize=10, fontweight='bold', color='white')
        
        ax2.set_xlim(0, longueur)
        ax2.set_ylim(largeur, 0)
        ax2.set_xlabel('Longueur (m)')
        ax2.set_ylabel('Largeur (m)')
        ax2.set_title('Heatmap de Qualité du Signal (dBm)')
        
        # Barre de couleur (essayer de la récupérer de la heatmap générée)
        try:
            # Récupérer l'image pour la barre de couleur
            images = ax2.get_images()
            if images:
                cbar = plt.colorbar(images[-1], ax=ax2)
                cbar.set_label('Puissance du signal (dBm)')
        except:
            # Fallback sans barre de couleur si erreur
            pass
        
        plt.tight_layout()
        return fig
    
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
