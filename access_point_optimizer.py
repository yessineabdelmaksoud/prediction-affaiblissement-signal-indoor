import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import differential_evolution, minimize
from sklearn.cluster import KMeans
import pandas as pd 
from typing import Dict, Tuple, List
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
        # Initialisation des optimiseurs spécialisés (correction de fréquence)
        self.gmm_optimizer = GMMOptimizer3D(frequency_mhz)  # MHz au lieu de Hz
        self.greedy_optimizer = GreedyOptimizer3D(frequency_mhz)  # MHz au lieu de Hz
        
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
                
                if distance_3d < 0.1:  # Très proche - éviter division par zéro
                    # 🔧 HARMONISATION: Même logique que GMM pour comparaison équitable
                    received_power = power_tx - 10  # Aligné sur GMM
                else:
                    # Conversion en pixels pour comptage des murs (avec validation)
                    x_tx_pixel = int(np.clip(x_tx / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                    y_tx_pixel = int(np.clip(y_tx / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                    x_rx_pixel = int(np.clip(x_rx / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                    y_rx_pixel = int(np.clip(y_rx / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                    
                    # Validation des coordonnées pixel
                    if (x_tx_pixel == x_rx_pixel and y_tx_pixel == y_rx_pixel):
                        # Même position en pixels, pas de murs à compter
                        wall_count = 0
                    else:
                        # Comptage des murs avec gestion d'erreur
                        try:
                            wall_count = self.processor.count_walls_between_points(
                                grid_info['walls_detected'],
                                (x_tx_pixel, y_tx_pixel),
                                (x_rx_pixel, y_rx_pixel)
                            )
                        except:
                            wall_count = 0  # Fallback en cas d'erreur
                    
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
        
        # Score de qualité amélioré - favorise d'abord l'atteinte de l'objectif de couverture
        num_aps = len(access_points)
        coverage_score = coverage_percent / 100.0
        
        # Système de scoring inspiré du GMM - privilégier la couverture avant l'efficacité
        if coverage_percent < min_coverage_percent:
            # Si objectif non atteint, pénaliser fortement et favoriser plus d'AP
            score = coverage_score * 2.0  # Doubler l'importance de la couverture
            efficiency_penalty = num_aps * 0.01  # Pénalité très faible pour encourager plus d'AP
            score -= efficiency_penalty
        else:
            # Si objectif atteint, alors optimiser l'efficacité
            score = 1.0 + coverage_score  # Bonus de base pour avoir atteint l'objectif
            # 🔧 HARMONISATION: Même pénalité que GMM pour comparaison équitable
            efficiency_penalty = (num_aps - 1) * 0.05  # Aligné sur GMM (était 0.03)
            score -= efficiency_penalty
            
            # Bonus supplémentaire pour dépassement significatif de l'objectif
            if coverage_percent > min_coverage_percent + 5:
                score += 0.2
        
        coverage_stats = {
            'covered_points': covered_points,
            'total_points': total_points,
            'coverage_percent': coverage_percent,
            'signal_levels': signal_levels,
            'num_access_points': num_aps
        }
        
        return max(score, 0.0), coverage_stats
    
    def optimize_with_clustering(self, coverage_points, grid_info, longueur, largeur, 
                                hauteur_totale, target_coverage_db=-70.0, 
                                min_coverage_percent=90.0, power_tx=20.0, max_access_points=8):
        """
        Optimise le placement des points d'accès 3D avec K-means clustering.
        
        Cette méthode suit le même flux de travail que l'algorithme GMM :
        1. Test de différents nombres de clusters
        2. Application du K-means pour chaque nombre
        3. Ajustement des positions pour éviter les murs
        4. Évaluation de chaque configuration
        5. Arrêt anticipé si l'objectif est atteint
        
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
        
        # Import du module sklearn (avec gestion d'erreur)
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            print("❌ Erreur: sklearn n'est pas installé. Installation nécessaire pour K-means.")
            return None, {}
        
        # Conversion en array numpy
        points_array = np.array(coverage_points)
        
        print(f"🔄 Optimisation K-means 3D: {len(coverage_points)} points à couvrir")
        print(f"📦 Volume: {longueur}m x {largeur}m x {hauteur_totale}m")
        
        best_config = None
        best_score = -1.0
        cluster_analysis = {}
        
        # Test différents nombres de clusters (points d'accès) - même logique que GMM
        max_clusters = min(max_access_points, 15)
        print(f"🔬 Test K-means: 1 à {max_clusters} clusters (objectif {min_coverage_percent}%)")
        
        for num_clusters in range(1, max_clusters + 1):
            try:
                # Configuration K-means optimisée pour WiFi 3D
                # Utilisation d'une initialisation plus sophistiquée
                kmeans = KMeans(
                    n_clusters=num_clusters,
                    init='k-means++',  # Initialisation intelligente
                    max_iter=500,      # Plus d'itérations pour convergence
                    n_init=20,         # Plus d'essais pour meilleur optimum
                    random_state=42,   # Reproductibilité
                    tol=1e-6,          # Tolérance plus stricte
                    algorithm='elkan'  # Algorithme optimisé pour données denses
                )
                
                # Ajustement du modèle
                kmeans.fit(points_array)
                
                # Extraction des centres des clusters
                centers = kmeans.cluster_centers_
                cluster_labels = kmeans.labels_
                
                # Ajustement des centres pour éviter les murs et contraintes 3D
                # Méthode améliorée inspirée de GMM avec pondération par densité
                adjusted_centers = []
                for i, center in enumerate(centers):
                    x, y, z = center
                    
                    # Contraintes de positionnement (identiques au GMM)
                    x = np.clip(x, 1.0, longueur - 1.0)
                    y = np.clip(y, 1.0, largeur - 1.0)
                    z = np.clip(z, 0.5, hauteur_totale - 0.5)
                    
                    # Vérification et ajustement pour éviter les murs (même méthode que GMM)
                    x, y = self._adjust_position_to_avoid_walls(x, y, grid_info, longueur, largeur)
                    
                    adjusted_centers.append((x, y, z, power_tx))
                
                # Évaluation de cette configuration (même méthode que GMM)
                score, stats = self.calculate_coverage_quality(
                    adjusted_centers, coverage_points, grid_info, 
                    target_coverage_db, min_coverage_percent
                )
                
                # Métriques K-means (adaptation des métriques GMM)
                kmeans_metrics = {
                    'inertia': kmeans.inertia_,              # SSE (Sum of Squared Errors)
                    'n_iter': kmeans.n_iter_,                # Nombre d'itérations
                    'cluster_centers': kmeans.cluster_centers_.tolist(),
                    'cluster_sizes': [np.sum(cluster_labels == i) for i in range(num_clusters)],
                    'converged': kmeans.n_iter_ < kmeans.max_iter  # Test de convergence
                }
                
                cluster_analysis[num_clusters] = {
                    'centers': adjusted_centers,
                    'score': score,
                    'stats': stats,
                    'kmeans_metrics': kmeans_metrics,
                    'cluster_labels': cluster_labels.tolist(),
                    'original_centers': kmeans.cluster_centers_.tolist()
                }
                
                # Mise à jour du meilleur score (même logique que GMM)
                if score > best_score:
                    best_score = score
                    best_config = {
                        'access_points': adjusted_centers,
                        'score': score,
                        'stats': stats,
                        'kmeans_metrics': kmeans_metrics,
                        'n_clusters': num_clusters
                    }
                
                # Affichage du progrès (format similaire au GMM)
                current_coverage = stats.get('coverage_percent', 0.0)
                inertia = kmeans_metrics['inertia']
                print(f"🔄 {num_clusters} clusters: {current_coverage:.1f}% couverture (Inertie: {inertia:.1f}, Conv: {'Oui' if kmeans_metrics['converged'] else 'Non'})")
                
                # Arrêt anticipé si objectif atteint (comme GMM)
                # GMM s'arrête immédiatement quand l'objectif est atteint
                if current_coverage >= min_coverage_percent:
                    print(f"✅ K-means: Objectif {min_coverage_percent}% atteint avec {num_clusters} clusters ({current_coverage:.1f}%)")
                    print(f"🎯 Arrêt anticipé - Retour immédiat de la configuration optimale")
                    break
                    
            except Exception as e:
                print(f"⚠️  Erreur K-means avec {num_clusters} clusters: {e}")
                cluster_analysis[num_clusters] = {
                    'error': str(e),
                    'score': 0.0,
                    'stats': {'coverage_percent': 0.0}
                }
        
        # Validation finale (même logique que GMM)
        if best_config and best_config['stats']['coverage_percent'] < min_coverage_percent:
            coverage_achieved = best_config['stats']['coverage_percent']
            print(f"⚠️  K-means 3D: Objectif {min_coverage_percent}% non atteint. Meilleur: {coverage_achieved:.1f}%")
            print(f"💡 Recommandation: Augmenter la puissance TX ou ajouter plus de points d'accès")
        elif best_config:
            coverage_achieved = best_config['stats']['coverage_percent']
            n_aps = len(best_config['access_points'])
            print(f"🎉 K-means 3D: Succès! {coverage_achieved:.1f}% de couverture avec {n_aps} points d'accès")
        
        return best_config, cluster_analysis
    
    def _adjust_position_to_avoid_walls(self, x: float, y: float, grid_info: Dict, 
                                       longueur: float, largeur: float) -> Tuple[float, float]:
        """
        Ajuste une position pour éviter les murs en trouvant la position libre la plus proche.
        IDENTIQUE à la méthode GMM pour harmonisation complète.
        
        Args:
            x, y: Position initiale
            grid_info: Informations sur la grille
            longueur, largeur: Dimensions
            
        Returns:
            Tuple (x_ajusté, y_ajusté)
        """
        # Vérification si la position actuelle est dans un mur
        x_pixel = int(np.clip(x / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
        y_pixel = int(np.clip(y / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
        
        # Si pas dans un mur, retourner la position originale
        if grid_info['walls_detected'][y_pixel, x_pixel] == 0:
            return x, y
        
        # Recherche en spirale pour trouver une position libre
        max_radius = 5  # Recherche dans un rayon de 5 mètres
        for radius in range(1, max_radius + 1):
            # Points sur le cercle de rayon 'radius'
            for angle in np.arange(0, 2 * np.pi, np.pi / 8):  # 16 directions
                test_x = x + radius * np.cos(angle)
                test_y = y + radius * np.sin(angle)
                
                # Vérifier les contraintes de l'environnement
                if test_x < 1.0 or test_x > longueur - 1.0:
                    continue
                if test_y < 1.0 or test_y > largeur - 1.0:
                    continue
                
                # Convertir en pixels pour vérifier les murs
                test_x_pixel = int(np.clip(test_x / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                test_y_pixel = int(np.clip(test_y / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                
                # Si position libre trouvée
                if grid_info['walls_detected'][test_y_pixel, test_x_pixel] == 0:
                    return test_x, test_y
        
        # Si aucune position libre trouvée, retourner le centre de l'environnement
        center_x = longueur / 2
        center_y = largeur / 2
        print(f"⚠️  Aucune position libre trouvée près de ({x:.1f}, {y:.1f}), utilisation du centre ({center_x:.1f}, {center_y:.1f})")
        return center_x, center_y

    def _adjust_position_to_avoid_walls_kmeans_improved(self, x: float, y: float, cluster_idx: int, 
                                                       cluster_labels: np.ndarray, points_array: np.ndarray,
                                                       grid_info: dict, longueur: float, largeur: float) -> tuple:
        """
        Ajuste une position pour éviter les murs en K-means (version améliorée inspirée du GMM).
        
        Args:
            x, y: Position initiale
            cluster_idx: Index du cluster
            cluster_labels: Labels des clusters
            points_array: Array des points
            grid_info: Informations sur la grille
            longueur, largeur: Dimensions
            
        Returns:
            Tuple (x_ajusté, y_ajusté)
        """
        # Vérification si la position actuelle est dans un mur
        x_pixel = int(np.clip(x / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
        y_pixel = int(np.clip(y / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
        
        # Si pas dans un mur, retourner la position originale
        if grid_info['walls_detected'][y_pixel, x_pixel] == 0:
            return x, y
        
        # Stratégie 1: Chercher dans les points du cluster (comme GMM mais amélioré)
        cluster_points = points_array[cluster_labels == cluster_idx]
        
        if len(cluster_points) > 0:
            # Trier par distance pour garder la cohérence du clustering
            center_3d = np.array([x, y, cluster_points[:, 2].mean()])  # Centre 3D approximatif
            distances = np.sqrt(np.sum((cluster_points - center_3d)**2, axis=1))
            sorted_indices = np.argsort(distances)
            
            # Essayer les points les plus proches du centre original
            for idx in sorted_indices[:min(10, len(sorted_indices))]:  # Limiter à 10 points max
                px, py, pz = cluster_points[idx]
                px_pixel = int(np.clip(px / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                py_pixel = int(np.clip(py / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                
                # Si ce point n'est pas dans un mur, l'utiliser
                if grid_info['walls_detected'][py_pixel, px_pixel] == 0:
                    return px, py
        
        # Stratégie 2: Recherche en spirale (méthode GMM simplifiée)
        max_radius = 5  # Comme dans GMM
        for radius in range(1, max_radius + 1):
            # 16 directions comme dans GMM
            for angle in np.arange(0, 2 * np.pi, np.pi / 8):
                test_x = x + radius * np.cos(angle)
                test_y = y + radius * np.sin(angle)
                
                # Vérifier les contraintes de l'environnement
                if test_x < 1.0 or test_x > longueur - 1.0:
                    continue
                if test_y < 1.0 or test_y > largeur - 1.0:
                    continue
                
                # Convertir en pixels pour vérifier les murs
                test_x_pixel = int(np.clip(test_x / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                test_y_pixel = int(np.clip(test_y / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                
                # Si position libre trouvée
                if grid_info['walls_detected'][test_y_pixel, test_x_pixel] == 0:
                    return test_x, test_y
        
        # Stratégie 3: Centre de l'environnement (comme GMM)
        center_x = longueur / 2
        center_y = largeur / 2
        print(f"⚠️  K-means cluster {cluster_idx}: Utilisation du centre ({center_x:.1f}, {center_y:.1f})")
        return center_x, center_y
    
    def _find_any_valid_position(self, grid_info: dict, longueur: float, largeur: float) -> tuple:
        """
        Trouve n'importe quelle position valide dans l'environnement comme fallback.
        
        Args:
            grid_info: Informations sur la grille
            longueur, largeur: Dimensions
            
        Returns:
            Tuple (x, y) d'une position valide
        """
        # Recherche systématique dans une grille
        step = min(longueur, largeur) / 20  # 20 points de test par dimension
        
        for x in np.arange(1.0, longueur - 1.0, step):
            for y in np.arange(1.0, largeur - 1.0, step):
                x_pixel = int(np.clip(x / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                y_pixel = int(np.clip(y / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                
                if grid_info['walls_detected'][y_pixel, x_pixel] == 0:
                    return x, y
        
        # Si vraiment rien trouvé, retourner le centre
        return longueur / 2, largeur / 2

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

    def optimize_with_algorithm_choice_3d(self, algorithm_choice, coverage_points, grid_info, 
                                         longueur, largeur, hauteur_totale, target_coverage_db=-70.0, 
                                         min_coverage_percent=90.0, max_access_points=8, power_tx=20.0):
        
        print(f"🚀 Optimisation 3D avec algorithme: {algorithm_choice.upper()}")
        
        if algorithm_choice == 'kmeans':
            return self.optimize_with_clustering(
                coverage_points, grid_info, longueur, largeur, hauteur_totale,
                target_coverage_db, min_coverage_percent, power_tx, max_access_points
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
        return self.gmm_optimizer.optimize_clustering_gmm_3d(
            coverage_points, grid_info, longueur, largeur, hauteur_totale,
            target_coverage_db, min_coverage_percent, power_tx, max_access_points
        )
    
    def _optimize_with_greedy_3d(self, coverage_points, grid_info, longueur, largeur, hauteur_totale,
                                target_coverage_db, min_coverage_percent, max_access_points, power_tx):
        """
        Optimisation avec algorithme Greedy 3D.
        """
        return self.greedy_optimizer.optimize_greedy_placement_3d(
            coverage_points, grid_info, longueur, largeur, hauteur_totale,
            target_coverage_db, min_coverage_percent, power_tx, max_access_points
        )