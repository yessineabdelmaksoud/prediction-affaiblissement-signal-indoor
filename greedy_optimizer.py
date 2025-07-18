import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
from pathloss_calculator import PathlossCalculator
from image_processor import ImageProcessor


class GreedyOptimizer:
    """
    Optimiseur utilisant un algorithme glouton pour le placement de points d'accès WiFi.
    
    L'algorithme place séquentiellement les points d'accès en choisissant à chaque
    étape la position qui maximise la couverture additionnelle.
    """
    
    def __init__(self, frequency: float): 
        """
        Initialise l'optimiseur Greedy.
        
        Args:
            frequency: Fréquence de transmission en Hz
        """
        self.frequency = frequency
        self.frequency_mhz = frequency / 1e6  # Conversion Hz vers MHz
        self.pathloss_calculator = PathlossCalculator(self.frequency_mhz)
        self.processor = ImageProcessor()
    
    def optimize_greedy_placement(self, coverage_points: List[Tuple[float, float]], 
                                 grid_info: Dict, longueur: float, largeur: float,
                                 target_coverage_db: float, min_coverage_percent: float,
                                 power_tx: float, max_access_points: int) -> Optional[Tuple[Dict, Dict]]:
        """
        Optimise le placement des points d'accès avec l'algorithme Greedy amélioré.
        
        Args:
            coverage_points: Liste des points à couvrir [(x, y), ...]
            grid_info: Informations sur la grille
            longueur, largeur: Dimensions de la zone
            target_coverage_db: Niveau de signal minimum requis (dBm)
            min_coverage_percent: Pourcentage minimum de couverture requis
            power_tx: Puissance de transmission (dBm)
            max_access_points: Nombre maximum de points d'accès
            
        Returns:
            Tuple (configuration, analyse) ou None si échec
        """
        
        if not coverage_points:
            return None
        
        # Générer les positions candidates optimisées
        candidate_positions = self._generate_smart_candidate_positions(
            coverage_points, longueur, largeur, grid_info
        )
        
        # Initialisation
        access_points = []
        covered_mask = np.zeros(len(coverage_points), dtype=bool)
        steps = []
        total_iterations = 0
        
        print(f"🎯 Optimisation Greedy améliorée: {len(coverage_points)} points à couvrir")
        print(f"📍 {len(candidate_positions)} positions candidates intelligentes")
        print(f"🎯 Objectif: {min_coverage_percent}% de couverture")
        
        # Algorithme Greedy - placement séquentiel avec arrêt anticipé
        for ap_count in range(max_access_points):
            best_position = None
            best_gain = 0
            best_new_coverage = None
            best_efficiency = 0
            
            # Tester chaque position candidate
            for pos_x, pos_y in candidate_positions:
                total_iterations += 1
                
                # Vérifier que la position n'est pas dans un mur
                if not self._is_valid_ap_position(pos_x, pos_y, grid_info):
                    continue
                
                # Calculer la nouvelle couverture pour cette position
                new_coverage_mask = self._calculate_new_coverage(
                    pos_x, pos_y, coverage_points, target_coverage_db, power_tx, grid_info
                )
                
                # Calculer le gain (nouveaux points couverts)
                additional_coverage = new_coverage_mask & ~covered_mask
                gain = np.sum(additional_coverage)
                
                # Calculer l'efficacité (gain par rapport à la distance aux autres AP)
                efficiency = self._calculate_position_efficiency(
                    pos_x, pos_y, access_points, gain, coverage_points
                )
                
                # Mettre à jour le meilleur choix (prioriser gain puis efficacité)
                if gain > best_gain or (gain == best_gain and efficiency > best_efficiency):
                    best_gain = gain
                    best_efficiency = efficiency
                    best_position = (pos_x, pos_y)
                    best_new_coverage = new_coverage_mask
            
            # Vérifier si on a trouvé une position utile
            if best_position is None or best_gain == 0:
                print(f"🛑 Arrêt Greedy: Aucun gain supplémentaire possible")
                break
            
            # Ajouter le meilleur point d'accès
            access_points.append((best_position[0], best_position[1], power_tx))
            covered_mask |= best_new_coverage
            
            # Calculer le pourcentage de couverture actuel
            coverage_percent = (np.sum(covered_mask) / len(coverage_points)) * 100
            
            # Enregistrer l'étape
            step_info = {
                'position': best_position,
                'coverage_gain': best_gain,
                'total_coverage': np.sum(covered_mask),
                'coverage_percent': coverage_percent,
                'efficiency': best_efficiency
            }
            steps.append(step_info)
            
            print(f"📍 AP {ap_count + 1}: Position ({best_position[0]:.1f}, {best_position[1]:.1f}) -> {coverage_percent:.1f}% couverture (+{best_gain} points)")
            
            # ARRÊT ANTICIPÉ: Vérifier si l'objectif de couverture est atteint
            if coverage_percent >= min_coverage_percent:
                print(f"✅ Objectif {min_coverage_percent}% atteint avec {ap_count + 1} AP - ARRÊT OPTIMISATION")
                
                # Calculer le score pour cette configuration optimale
                final_coverage_percent = coverage_percent
                covered_points = np.sum(covered_mask)
                coverage_score = final_coverage_percent / 100.0
                efficiency_score = 1.0 - (len(access_points) / max_access_points)
                total_score = 0.7 * coverage_score + 0.3 * efficiency_score
                
                # Configuration avec arrêt anticipé
                config = {
                    'access_points': access_points,
                    'score': total_score,
                    'early_stop': True,
                    'early_stop_reason': f"Objectif {min_coverage_percent}% atteint",
                    'stats': {
                        'coverage_percent': final_coverage_percent,
                        'covered_points': covered_points,
                        'total_points': len(coverage_points)
                    }
                }
                
                # Analyse avec arrêt anticipé
                analysis = {
                    'type': 'greedy',
                    'steps': steps,
                    'total_iterations': total_iterations,
                    'convergence_reason': "target_coverage_reached",
                    'final_coverage_mask': covered_mask,
                    'early_stop': True
                }
                
                print(f"✅ Optimisation Greedy terminée (arrêt anticipé):")
                print(f"   - Algorithme: Greedy Séquentiel Amélioré")
                print(f"   - {len(access_points)} points d'accès placés")
                print(f"   - {final_coverage_percent:.1f}% de couverture ({covered_points}/{len(coverage_points)} points)")
                print(f"   - Score: {total_score:.3f}")
                print(f"   - {total_iterations} itérations totales")
                print(f"   - Arrêt anticipé: Objectif atteint efficacement")
                
                return config, analysis
            
            # Optimisation: retirer les positions proches de celle choisie pour éviter la redondance
            candidate_positions = self._filter_candidates_around_position(
                candidate_positions, best_position, min_distance=3.0
            )
        
        # Si on arrive ici, l'objectif n'a pas été atteint dans la limite d'AP
        # Calculer les statistiques finales
        final_coverage_percent = (np.sum(covered_mask) / len(coverage_points)) * 100
        covered_points = np.sum(covered_mask)
        
        # Calculer le score (balance entre couverture et nombre d'APs)
        coverage_score = final_coverage_percent / 100.0
        efficiency_score = 1.0 - (len(access_points) / max_access_points)
        total_score = 0.7 * coverage_score + 0.3 * efficiency_score
        
        # Déterminer la raison d'arrêt
        if len(access_points) >= max_access_points:
            convergence_reason = "max_access_points_reached"
        else:
            convergence_reason = "no_additional_gain"
        
        # Configuration finale
        config = {
            'access_points': access_points,
            'score': total_score,
            'early_stop': False,
            'stats': {
                'coverage_percent': final_coverage_percent,
                'covered_points': covered_points,
                'total_points': len(coverage_points)
            }
        }
        
        # Analyse détaillée
        analysis = {
            'type': 'greedy',
            'steps': steps,
            'total_iterations': total_iterations,
            'convergence_reason': convergence_reason,
            'final_coverage_mask': covered_mask,
            'early_stop': False
        }
        
        print(f"✅ Optimisation Greedy terminée:")
        print(f"   - Algorithme: Greedy Séquentiel Amélioré")
        print(f"   - {len(access_points)} points d'accès placés")
        print(f"   - {final_coverage_percent:.1f}% de couverture ({covered_points}/{len(coverage_points)} points)")
        print(f"   - Score: {total_score:.3f}")
        print(f"   - {total_iterations} itérations totales")
        print(f"   - Convergence: {convergence_reason}")
        
        return config, analysis
    
    def _generate_candidate_positions(self, longueur: float, largeur: float, 
                                    resolution: int = 15) -> List[Tuple[float, float]]:
        """
        Génère les positions candidates pour les points d'accès (méthode basique).
        
        Args:
            longueur, largeur: Dimensions de la zone
            resolution: Résolution de la grille de candidats
            
        Returns:
            Liste des positions candidates [(x, y), ...]
        """
        positions = []
        
        # Éviter les bords (marge de 1 mètre)
        margin = 1.0
        x_min, x_max = margin, longueur - margin
        y_min, y_max = margin, largeur - margin
        
        # Créer une grille de positions candidates
        x_step = (x_max - x_min) / resolution
        y_step = (y_max - y_min) / resolution
        
        for i in range(resolution):
            for j in range(resolution):
                x = x_min + i * x_step
                y = y_min + j * y_step
                positions.append((x, y))
        
        return positions
    
    def _generate_smart_candidate_positions(self, coverage_points: List[Tuple[float, float]], 
                                          longueur: float, largeur: float, 
                                          grid_info: Dict) -> List[Tuple[float, float]]:
        """
        Génère des positions candidates intelligentes basées sur la distribution des points de couverture.
        
        Args:
            coverage_points: Points à couvrir
            longueur, largeur: Dimensions de la zone
            grid_info: Informations sur la grille (pour éviter les murs)
            
        Returns:
            Liste des positions candidates optimisées
        """
        positions = []
        
        # Convertir les points de couverture en array pour le calcul
        points_array = np.array(coverage_points)
        
        # 1. Positions basées sur les centroides de zones denses
        density_positions = self._get_density_based_positions(points_array, longueur, largeur)
        positions.extend(density_positions)
        
        # 2. Positions basées sur une grille stratégique (plus sparse que l'ancienne)
        strategic_positions = self._get_strategic_grid_positions(longueur, largeur)
        positions.extend(strategic_positions)
        
        # 3. Filtrer les positions dans les murs
        valid_positions = []
        for x, y in positions:
            if self._is_valid_ap_position(x, y, grid_info):
                valid_positions.append((x, y))
        
        # 4. Si trop peu de positions valides, ajouter des positions de secours
        if len(valid_positions) < 20:
            backup_positions = self._get_backup_positions(longueur, largeur, grid_info)
            valid_positions.extend(backup_positions)
        
        return valid_positions
    
    def _get_density_based_positions(self, points_array: np.ndarray, 
                                   longueur: float, largeur: float) -> List[Tuple[float, float]]:
        """Génère des positions basées sur la densité des points de couverture."""
        positions = []
        
        # Diviser l'espace en zones et calculer les centroïdes des zones denses
        n_zones_x, n_zones_y = 6, 5  # Grille 6x5 pour l'analyse de densité
        
        for i in range(n_zones_x):
            for j in range(n_zones_y):
                # Définir les limites de la zone
                x_min = (i * longueur) / n_zones_x
                x_max = ((i + 1) * longueur) / n_zones_x
                y_min = (j * largeur) / n_zones_y
                y_max = ((j + 1) * largeur) / n_zones_y
                
                # Trouver les points dans cette zone
                zone_mask = (
                    (points_array[:, 0] >= x_min) & (points_array[:, 0] < x_max) &
                    (points_array[:, 1] >= y_min) & (points_array[:, 1] < y_max)
                )
                zone_points = points_array[zone_mask]
                
                # Si la zone a suffisamment de points, calculer le centroïde
                if len(zone_points) >= 3:  # Seuil de densité
                    centroid_x = np.mean(zone_points[:, 0])
                    centroid_y = np.mean(zone_points[:, 1])
                    
                    # Ajouter quelques positions autour du centroïde
                    positions.append((centroid_x, centroid_y))
                    
                    # Positions décalées pour plus de diversité
                    offset = min(2.0, (x_max - x_min) / 4)
                    positions.extend([
                        (centroid_x + offset, centroid_y),
                        (centroid_x - offset, centroid_y),
                        (centroid_x, centroid_y + offset),
                        (centroid_x, centroid_y - offset)
                    ])
        
        return positions
    
    def _get_strategic_grid_positions(self, longueur: float, largeur: float) -> List[Tuple[float, float]]:
        """Génère des positions sur une grille stratégique optimisée."""
        positions = []
        
        # Grille plus sparse mais mieux répartie
        margin = 2.0  # Marge plus importante pour éviter les bords
        n_x, n_y = 8, 6  # Grille plus petite pour réduire les calculs
        
        x_positions = np.linspace(margin, longueur - margin, n_x)
        y_positions = np.linspace(margin, largeur - margin, n_y)
        
        for x in x_positions:
            for y in y_positions:
                positions.append((x, y))
        
        return positions
    
    def _get_backup_positions(self, longueur: float, largeur: float, 
                            grid_info: Dict) -> List[Tuple[float, float]]:
        """Génère des positions de secours si pas assez de positions valides."""
        positions = []
        
        # Positions stratégiques de base (centres, coins sûrs)
        margin = 3.0
        backup_candidates = [
            (longueur / 2, largeur / 2),  # Centre
            (longueur / 3, largeur / 3),  # Tiers
            (2 * longueur / 3, largeur / 3),
            (longueur / 3, 2 * largeur / 3),
            (2 * longueur / 3, 2 * largeur / 3),
            (longueur / 4, largeur / 2),  # Quarts
            (3 * longueur / 4, largeur / 2),
            (longueur / 2, largeur / 4),
            (longueur / 2, 3 * largeur / 4)
        ]
        
        for x, y in backup_candidates:
            if (margin <= x <= longueur - margin and 
                margin <= y <= largeur - margin and
                self._is_valid_ap_position(x, y, grid_info)):
                positions.append((x, y))
        
        return positions
    
    def _is_valid_ap_position(self, x: float, y: float, grid_info: Dict) -> bool:
        """Vérifie si une position est valide pour un point d'accès."""
        if grid_info is None or 'walls_detected' not in grid_info:
            return True
        
        try:
            # Conversion en pixels
            x_pixel = int(np.clip(x / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
            y_pixel = int(np.clip(y / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
            
            # Vérifier que ce n'est pas dans un mur
            if grid_info['walls_detected'][y_pixel, x_pixel] > 0:
                return False
            
            # Vérifier un périmètre de sécurité (pas trop près des murs)
            safety_radius = 2
            for dx in range(-safety_radius, safety_radius + 1):
                for dy in range(-safety_radius, safety_radius + 1):
                    check_x = np.clip(x_pixel + dx, 0, grid_info['walls_detected'].shape[1] - 1)
                    check_y = np.clip(y_pixel + dy, 0, grid_info['walls_detected'].shape[0] - 1)
                    if grid_info['walls_detected'][check_y, check_x] > 0:
                        # Trop près d'un mur
                        return False
            
            return True
            
        except Exception:
            # En cas d'erreur, être conservateur
            return True
    
    def _calculate_position_efficiency(self, x: float, y: float, 
                                     existing_aps: List[Tuple[float, float, float]], 
                                     gain: int, coverage_points: List) -> float:
        """
        Calcule l'efficacité d'une position en considérant la distance aux autres AP.
        
        Args:
            x, y: Position candidate
            existing_aps: Points d'accès déjà placés
            gain: Gain en couverture de cette position
            coverage_points: Points de couverture totaux
            
        Returns:
            Score d'efficacité (plus élevé = mieux)
        """
        if gain == 0:
            return 0.0
        
        # Score de base proportionnel au gain
        efficiency = gain
        
        # Pénalité pour proximité avec d'autres AP (éviter le clustering)
        min_distance_to_existing = float('inf')
        for ap_x, ap_y, _ in existing_aps:
            distance = np.sqrt((x - ap_x)**2 + (y - ap_y)**2)
            min_distance_to_existing = min(min_distance_to_existing, distance)
        
        if min_distance_to_existing != float('inf'):
            # Bonus pour distance optimale (ni trop près, ni trop loin)
            optimal_distance = 8.0  # Distance optimale entre AP
            distance_penalty = abs(min_distance_to_existing - optimal_distance) / optimal_distance
            efficiency *= (1.0 - 0.3 * distance_penalty)  # Pénalité jusqu'à 30%
        
        # Bonus pour position centrale (éviter les positions extrêmes)
        # On ne veut pas d'AP en (0,0) ou aux coins
        center_bonus = 1.0 - (abs(x) + abs(y)) / 1000.0  # Petit bonus pour éviter (0,0)
        efficiency *= (1.0 + 0.1 * center_bonus)
        
        return max(0.0, efficiency)
    
    def _filter_candidates_around_position(self, candidates: List[Tuple[float, float]], 
                                         placed_position: Tuple[float, float], 
                                         min_distance: float = 3.0) -> List[Tuple[float, float]]:
        """
        Filtre les positions candidates trop proches d'une position déjà choisie.
        
        Args:
            candidates: Liste des positions candidates
            placed_position: Position déjà placée
            min_distance: Distance minimale à respecter
            
        Returns:
            Liste filtrée des candidates
        """
        px, py = placed_position
        filtered = []
        
        for x, y in candidates:
            distance = np.sqrt((x - px)**2 + (y - py)**2)
            if distance >= min_distance:
                filtered.append((x, y))
        
        return filtered
    
    def _calculate_new_coverage(self, ap_x: float, ap_y: float, 
                               coverage_points: List[Tuple[float, float]],
                               target_coverage_db: float, power_tx: float,
                               grid_info: Dict = None) -> np.ndarray:
        """
        Calcule le masque de couverture pour un point d'accès à une position donnée.
        
        Args:
            ap_x, ap_y: Position du point d'accès
            coverage_points: Points à vérifier
            target_coverage_db: Niveau minimum requis
            power_tx: Puissance de transmission
            grid_info: Informations sur la grille pour détection des murs
            
        Returns:
            Masque booléen indiquant les points couverts
        """
        coverage_mask = np.zeros(len(coverage_points), dtype=bool)
        
        for i, (point_x, point_y) in enumerate(coverage_points):
            # Calculer la distance 2D
            distance = np.sqrt((ap_x - point_x)**2 + (ap_y - point_y)**2)
            
            if distance < 0.1:  # Très proche
                received_power = power_tx - 10
            else:
                # Si grid_info est disponible, utiliser la détection de murs
                if grid_info is not None and 'walls_detected' in grid_info:
                    # Conversion en pixels pour comptage des murs
                    ap_x_pixel = int(np.clip(ap_x / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                    ap_y_pixel = int(np.clip(ap_y / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                    point_x_pixel = int(np.clip(point_x / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
                    point_y_pixel = int(np.clip(point_y / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
                    
                    # Comptage des murs entre les points
                    wall_count = self.processor.count_walls_between_points(
                        grid_info['walls_detected'],
                        (ap_x_pixel, ap_y_pixel),
                        (point_x_pixel, point_y_pixel)
                    )
                    
                    # Calcul du pathloss avec le calculateur unifié
                    pathloss = self.pathloss_calculator.calculate_pathloss(distance, wall_count)
                else:
                    # Version simplifiée sans détection de murs
                    wall_count = int(distance * 0.2)  # Estimation approximative
                    pathloss = self.pathloss_calculator.calculate_pathloss(distance, wall_count)
                
                received_power = power_tx - pathloss
            
            # Vérifier si le point est couvert
            coverage_mask[i] = received_power >= target_coverage_db
        
        return coverage_mask
    
    def visualize_greedy_process(self, config: Dict, analysis: Dict, 
                               coverage_points: List[Tuple[float, float]],
                               longueur: float, largeur: float) -> plt.Figure:
        """
        Visualise le processus d'optimisation Greedy.
        
        Args:
            config: Configuration des points d'accès
            analysis: Analyse du processus Greedy
            coverage_points: Points de couverture
            longueur, largeur: Dimensions
            
        Returns:
            Figure matplotlib
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Graphique 1: Positions finales
        ax1.set_title("Résultat Final - Placement Greedy", fontsize=14, fontweight='bold')
        
        # Points de couverture
        if len(coverage_points) < 200:  # Éviter la surcharge
            coverage_x = [p[0] for p in coverage_points]
            coverage_y = [p[1] for p in coverage_points]
            ax1.scatter(coverage_x, coverage_y, c='lightblue', s=20, alpha=0.6, label='Points à couvrir')
        
        # Points d'accès avec ordre de placement
        access_points = config['access_points']
        colors = plt.cm.viridis(np.linspace(0, 1, len(access_points)))
        
        for i, (x, y, power) in enumerate(access_points):
            ax1.scatter(x, y, c=[colors[i]], s=300, marker='*', 
                       edgecolors='black', linewidth=2, zorder=5)
            
            # Rayon de couverture estimé
            estimated_range = max(3.0, min(12.0, power / 3.0))
            circle = plt.Circle((x, y), estimated_range, fill=False, 
                              color=colors[i], alpha=0.7, linestyle='--')
            ax1.add_patch(circle)
            
            # Numéro d'ordre
            ax1.annotate(f'{i+1}', (x, y), xytext=(0, 0), 
                        textcoords='offset points', fontsize=12, 
                        fontweight='bold', color='white', ha='center', va='center')
        
        ax1.set_xlim(0, longueur)
        ax1.set_ylim(0, largeur)
        ax1.set_xlabel('Longueur (m)')
        ax1.set_ylabel('Largeur (m)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Graphique 2: Évolution de la couverture
        ax2.set_title("Évolution de la Couverture - Algorithme Greedy", fontsize=14, fontweight='bold')
        
        steps = analysis['steps']
        if steps:
            step_numbers = list(range(1, len(steps) + 1))
            coverage_percentages = [step['coverage_percent'] for step in steps]
            coverage_gains = [step['coverage_gain'] for step in steps]
            
            # Courbe de couverture
            ax2_twin = ax2.twinx()
            
            line1 = ax2.plot(step_numbers, coverage_percentages, 'b-o', linewidth=2, 
                           markersize=8, label='Couverture Totale (%)')
            bars = ax2_twin.bar([x - 0.3 for x in step_numbers], coverage_gains, 
                              width=0.6, alpha=0.6, color='green', label='Gain par Étape')
            
            ax2.set_xlabel('Étape (Numéro AP)')
            ax2.set_ylabel('Couverture Totale (%)', color='blue')
            ax2_twin.set_ylabel('Points Ajoutés', color='green')
            
            ax2.tick_params(axis='y', labelcolor='blue')
            ax2_twin.tick_params(axis='y', labelcolor='green')
            
            # Ligne d'objectif
            if 'min_coverage_percent' in analysis:
                ax2.axhline(y=analysis['min_coverage_percent'], color='red', 
                          linestyle='--', alpha=0.7, label='Objectif')
            
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)
            
            # Légendes combinées
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        # Informations sur le processus
        info_text = f"Algorithme: Greedy Séquentiel\n"
        info_text += f"Points d'accès: {len(access_points)}\n"
        info_text += f"Couverture finale: {config['stats']['coverage_percent']:.1f}%\n"
        info_text += f"Score total: {config['score']:.3f}\n"
        info_text += f"Itérations: {analysis['total_iterations']}\n"
        info_text += f"Convergence: {analysis['convergence_reason']}"
        
        fig.text(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        return fig
    
    def compare_with_other_algorithms(self, coverage_points: List[Tuple[float, float]], 
                                    grid_info: Dict, longueur: float, largeur: float,
                                    target_coverage_db: float, min_coverage_percent: float,
                                    power_tx: float, max_access_points: int) -> Dict:
        """
        Compare l'algorithme Greedy avec d'autres approches.
        
        Args:
            coverage_points, grid_info, longueur, largeur: Configuration de base
            target_coverage_db, min_coverage_percent, power_tx, max_access_points: Paramètres d'optimisation
            
        Returns:
            Dictionnaire avec les résultats comparatifs
        """
        
        results = {}
        
        # Résultat Greedy
        greedy_result = self.optimize_greedy_placement(
            coverage_points, grid_info, longueur, largeur,
            target_coverage_db, min_coverage_percent, power_tx, max_access_points
        )
        
        if greedy_result:
            config, analysis = greedy_result
            results['greedy'] = {
                'config': config,
                'analysis': analysis,
                'algorithm_name': 'Greedy'
            }
        
        return results
