#!/usr/bin/env python3
"""
Optimiseur Greedy pour placement de points d'accès WiFi 3D.

Ce module implémente un algorithme glouton (greedy) pour optimiser 
le placement des points d'accès WiFi dans un environnement 3D.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, Optional, Any
from pathloss_calculator_3d import PathlossCalculator3D
from image_processor import ImageProcessor


class GreedyOptimizer3D:
    """
    Optimiseur utilisant un algorithme glouton pour le placement de points d'accès WiFi 3D.
    
    L'algorithme place séquentiellement les points d'accès là où ils apportent
    la plus grande amélioration de couverture.
    """
    
    def __init__(self, frequency: float):
        """
        Initialise l'optimiseur Greedy 3D.
        
        Args:
            frequency: Fréquence de transmission en MHz
        """
        self.frequency_mhz = frequency
        self.calculator_3d = PathlossCalculator3D(frequency)
        self.processor = ImageProcessor()
    
    def optimize_greedy_placement_3d(self, coverage_points: List[Tuple[float, float, float]], 
                                    grid_info: Dict, longueur: float, largeur: float, hauteur_totale: float,
                                    target_coverage_db: float, min_coverage_percent: float, 
                                    power_tx: float, max_access_points: int) -> Tuple[Optional[Dict], Dict]:
        """
        Optimise le placement des points d'accès 3D avec l'algorithme glouton amélioré.
        
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
        
        print(f"🎯 Optimisation Greedy 3D améliorée: {len(coverage_points)} points à couvrir")
        print(f"📦 Volume: {longueur}m x {largeur}m x {hauteur_totale}m")
        print(f"🎯 Objectif: {min_coverage_percent}% de couverture")
        
        # Initialisation
        access_points = []
        placement_history = []
        current_coverage = 0.0
        
        # Génération intelligente de positions candidates
        candidate_positions = self._generate_smart_candidate_positions_3d(
            coverage_points, longueur, largeur, hauteur_totale, grid_info
        )
        
        print(f"🔍 {len(candidate_positions)} positions candidates générées intelligemment")
        
        # Placement glouton avec arrêt anticipé
        for ap_index in range(max_access_points):
            print(f"\n🔍 Recherche position optimale pour AP{ap_index + 1}")
            
            # Recherche de la meilleure position
            best_position = None
            best_improvement = 0.0
            best_stats = None
            best_efficiency = 0.0
            
            # Évaluation de chaque position candidate
            for candidate_pos in candidate_positions:
                # Test de cette position
                test_access_points = access_points + [(candidate_pos[0], candidate_pos[1], candidate_pos[2], power_tx)]
                
                test_score, test_stats = self._evaluate_configuration_3d(
                    test_access_points, coverage_points, grid_info,
                    target_coverage_db, min_coverage_percent
                )
                
                improvement = test_stats['coverage_percent'] - current_coverage
                
                # Calcul d'efficacité (amélioration par rapport à la distance aux autres APs)
                efficiency = self._calculate_position_efficiency_3d(candidate_pos, access_points, improvement)
                
                # Sélection basée sur l'amélioration ET l'efficacité
                if improvement > best_improvement or (improvement == best_improvement and efficiency > best_efficiency):
                    best_improvement = improvement
                    best_position = candidate_pos
                    best_stats = test_stats
                    best_efficiency = efficiency
            
            # Vérification si un placement est bénéfique
            if best_position and best_improvement > 0.05:  # Seuil d'amélioration minimale
                new_ap = (best_position[0], best_position[1], best_position[2], power_tx)
                access_points.append(new_ap)
                
                placement_history.append({
                    'ap_index': ap_index + 1,
                    'position': best_position,
                    'coverage_before': current_coverage,
                    'coverage_after': best_stats['coverage_percent'],
                    'improvement': best_improvement,
                    'efficiency': best_efficiency,
                    'covered_points': best_stats['covered_points'],
                    'total_points': best_stats['total_points']
                })
                
                current_coverage = best_stats['coverage_percent']
                
                print(f"✅ AP{ap_index + 1} placé à ({best_position[0]:.1f}, {best_position[1]:.1f}, {best_position[2]:.1f})")
                print(f"   ➜ Couverture: {current_coverage:.1f}% (+{best_improvement:.1f}%)")
                print(f"   ➜ Efficacité: {best_efficiency:.3f}")
                
                # Arrêt anticipé si objectif atteint
                if current_coverage >= min_coverage_percent:
                    print(f"🎯 Objectif {min_coverage_percent}% atteint avec {len(access_points)} APs!")
                    print(f"✅ Arrêt anticipé - Retour immédiat de la configuration optimale")
                    
                    # Configuration optimale trouvée - retour immédiat
                    config = {
                        'access_points': access_points,
                        'score': best_stats.get('score', current_coverage / 100.0),
                        'stats': best_stats,
                        'algorithm': 'greedy_3d',
                        'placement_steps': len(access_points)
                    }
                    
                    analysis = {
                        'placement_history': placement_history,
                        'initial_candidates': len(candidate_positions),
                        'final_candidates': len(candidate_positions),
                        'convergence_step': len(access_points),
                        'early_stopping': True
                    }
                    
                    return config, analysis
                
                # Filtrage intelligent des positions candidates pour éviter la proximité
                candidate_positions = self._filter_candidates_around_position_3d(
                    candidate_positions, best_position, min_distance=3.0
                )
                
            else:
                print(f"🛑 Arrêt: Aucune amélioration significative (< 0.05%) pour AP{ap_index + 1}")
                break
        
        # Configuration finale
        if access_points:
            final_score, final_stats = self._evaluate_configuration_3d(
                access_points, coverage_points, grid_info,
                target_coverage_db, min_coverage_percent
            )
            
            config = {
                'access_points': access_points,
                'score': final_score,
                'stats': final_stats,
                'algorithm': 'greedy_3d',
                'placement_steps': len(access_points)
            }
            
            analysis = {
                'placement_history': placement_history,
                'initial_candidates': len(candidate_positions),
                'final_candidates': len(candidate_positions),
                'convergence_step': len(access_points),
                'early_stopping': final_stats['coverage_percent'] >= min_coverage_percent
            }
            
            final_coverage = final_stats['coverage_percent']
            if final_coverage >= min_coverage_percent:
                print(f"✅ Greedy 3D terminé: {final_coverage:.1f}% couverture avec {len(access_points)} APs")
            else:
                print(f"⚠️  Greedy 3D terminé: {final_coverage:.1f}% < {min_coverage_percent}% (limite atteinte)")
            
            return config, analysis
        
        else:
            print("❌ Aucun point d'accès n'a pu être placé")
            return None, {}
    
    def _generate_smart_candidate_positions_3d(self, coverage_points: List[Tuple[float, float, float]], 
                                              longueur: float, largeur: float, hauteur_totale: float, 
                                              grid_info: Dict) -> List[Tuple[float, float, float]]:
        """
        Génère des positions candidates intelligentes basées sur la densité des points de couverture.
        
        Args:
            coverage_points: Points à couvrir
            longueur, largeur, hauteur_totale: Dimensions
            grid_info: Informations sur la grille
            
        Returns:
            Liste optimisée des positions candidates
        """
        print("🧠 Génération intelligente des positions candidates 3D...")
        
        candidate_positions = []
        
        # Stratégie 1: Positions basées sur la densité des points de couverture
        density_positions = self._get_density_based_positions_3d(coverage_points, longueur, largeur, hauteur_totale)
        for pos in density_positions:
            if self._is_valid_ap_position_3d(pos[0], pos[1], pos[2], grid_info, longueur, largeur, hauteur_totale):
                # Ajustement pour éviter les murs si nécessaire
                adjusted_pos = self._adjust_position_to_avoid_walls(pos[0], pos[1], grid_info, longueur, largeur)
                final_pos = (adjusted_pos[0], adjusted_pos[1], pos[2])
                candidate_positions.append(final_pos)
        
        # Stratégie 2: Grille stratégique 3D
        strategic_positions = self._get_strategic_grid_positions_3d(longueur, largeur, hauteur_totale)
        for pos in strategic_positions:
            if self._is_valid_ap_position_3d(pos[0], pos[1], pos[2], grid_info, longueur, largeur, hauteur_totale):
                # Ajustement pour éviter les murs si nécessaire
                adjusted_pos = self._adjust_position_to_avoid_walls(pos[0], pos[1], grid_info, longueur, largeur)
                final_pos = (adjusted_pos[0], adjusted_pos[1], pos[2])
                candidate_positions.append(final_pos)
        
        # Stratégie 3: Positions centrales par étage optimisées
        floor_positions = self._get_floor_centered_positions_3d(longueur, largeur, hauteur_totale)
        for pos in floor_positions:
            if self._is_valid_ap_position_3d(pos[0], pos[1], pos[2], grid_info, longueur, largeur, hauteur_totale):
                # Ajustement pour éviter les murs si nécessaire
                adjusted_pos = self._adjust_position_to_avoid_walls(pos[0], pos[1], grid_info, longueur, largeur)
                final_pos = (adjusted_pos[0], adjusted_pos[1], pos[2])
                candidate_positions.append(final_pos)
        
        # Suppression des doublons et tri par qualité
        unique_positions = self._remove_duplicate_positions_3d(candidate_positions)
        
        print(f"✨ {len(unique_positions)} positions candidates intelligentes générées")
        return unique_positions
    
    def _get_density_based_positions_3d(self, coverage_points: List[Tuple[float, float, float]], 
                                       longueur: float, largeur: float, hauteur_totale: float) -> List[Tuple[float, float, float]]:
        """
        Génère des positions basées sur la densité des points de couverture en 3D.
        """
        positions = []
        
        if not coverage_points:
            return positions
        
        # Division en cubes 3D
        num_cubes_x = max(3, int(longueur // 4))
        num_cubes_y = max(3, int(largeur // 4))
        num_cubes_z = max(2, int(hauteur_totale // 3))
        
        cube_size_x = longueur / num_cubes_x
        cube_size_y = largeur / num_cubes_y
        cube_size_z = hauteur_totale / num_cubes_z
        
        # Comptage des points par cube
        cube_counts = {}
        for point in coverage_points:
            x, y, z = point
            cube_i = min(int(x // cube_size_x), num_cubes_x - 1)
            cube_j = min(int(y // cube_size_y), num_cubes_y - 1)
            cube_k = min(int(z // cube_size_z), num_cubes_z - 1)
            
            cube_key = (cube_i, cube_j, cube_k)
            cube_counts[cube_key] = cube_counts.get(cube_key, 0) + 1
        
        # Sélection des cubes les plus denses
        sorted_cubes = sorted(cube_counts.items(), key=lambda x: x[1], reverse=True)
        top_cubes = sorted_cubes[:min(12, len(sorted_cubes))]  # Top 12 cubes
        
        # Placement au centre des cubes denses
        for (cube_i, cube_j, cube_k), count in top_cubes:
            center_x = (cube_i + 0.5) * cube_size_x
            center_y = (cube_j + 0.5) * cube_size_y
            center_z = (cube_k + 0.5) * cube_size_z
            
            # Ajustement pour éviter les bords
            center_x = np.clip(center_x, 1.0, longueur - 1.0)
            center_y = np.clip(center_y, 1.0, largeur - 1.0)
            center_z = np.clip(center_z, 0.5, hauteur_totale - 0.5)
            
            positions.append((center_x, center_y, center_z))
        
        return positions
    
    def _get_strategic_grid_positions_3d(self, longueur: float, largeur: float, hauteur_totale: float) -> List[Tuple[float, float, float]]:
        """
        Génère une grille stratégique de positions 3D.
        """
        positions = []
        
        # Grille adaptative basée sur la taille
        if longueur * largeur * hauteur_totale < 200:  # Petit volume
            x_positions = [longueur * 0.5]
            y_positions = [largeur * 0.5]
        elif longueur * largeur * hauteur_totale < 1000:  # Volume moyen
            x_positions = [longueur * 0.3, longueur * 0.7]
            y_positions = [largeur * 0.3, largeur * 0.7]
        else:  # Grand volume
            x_positions = [longueur * 0.2, longueur * 0.5, longueur * 0.8]
            y_positions = [largeur * 0.2, largeur * 0.5, largeur * 0.8]
        
        # Positions Z par étage
        num_floors = max(1, int(hauteur_totale // 2.7))
        z_positions = []
        for floor in range(num_floors):
            z = (floor + 0.4) * (hauteur_totale / num_floors)  # Légèrement au-dessus du centre de l'étage
            z_positions.append(min(z, hauteur_totale - 0.5))
        
        # Combinaisons stratégiques
        for x in x_positions:
            for y in y_positions:
                for z in z_positions:
                    positions.append((x, y, z))
        
        return positions
    
    def _get_floor_centered_positions_3d(self, longueur: float, largeur: float, hauteur_totale: float) -> List[Tuple[float, float, float]]:
        """
        Génère des positions centrées optimisées par étage.
        """
        positions = []
        
        num_floors = max(1, int(hauteur_totale // 2.7))
        
        for floor in range(num_floors):
            z = (floor + 0.6) * 2.7  # Position légèrement élevée dans l'étage
            if z >= hauteur_totale:
                z = hauteur_totale - 0.5
            
            # Centre principal
            center_x, center_y = longueur * 0.5, largeur * 0.5
            positions.append((center_x, center_y, z))
            
            # Positions secondaires pour grands espaces
            if longueur > 15 or largeur > 15:
                positions.extend([
                    (longueur * 0.25, largeur * 0.5, z),
                    (longueur * 0.75, largeur * 0.5, z),
                    (longueur * 0.5, largeur * 0.25, z),
                    (longueur * 0.5, largeur * 0.75, z)
                ])
        
        return positions
    
    def _remove_duplicate_positions_3d(self, positions: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Supprime les positions dupliquées avec une tolérance 3D.
        """
        unique_positions = []
        tolerance = 0.8  # Distance minimale entre positions
        
        for pos in positions:
            is_duplicate = False
            for existing_pos in unique_positions:
                distance = np.sqrt(sum((pos[i] - existing_pos[i])**2 for i in range(3)))
                if distance < tolerance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_positions.append(pos)
        
        return unique_positions
    
    def _is_valid_ap_position_3d(self, x: float, y: float, z: float, grid_info: Dict, 
                                longueur: float, largeur: float, hauteur_totale: float) -> bool:
        """
        Vérifie si une position 3D est valide pour placer un point d'accès avec marges de sécurité.
        """
        # Marges de sécurité plus larges
        margin_xy = 1.5
        margin_z = 0.5
        
        # Vérification des limites avec marges
        if x < margin_xy or x > longueur - margin_xy:
            return False
        if y < margin_xy or y > largeur - margin_xy:
            return False
        if z < margin_z or z > hauteur_totale - margin_z:
            return False
        
        # Vérification des murs (projection 2D) avec zone de sécurité
        try:
            # Vérification du point central
            x_pixel = int(np.clip(x / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
            y_pixel = int(np.clip(y / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
            
            if grid_info['walls_detected'][y_pixel, x_pixel] > 0:
                return False
            
            # Vérification de la zone autour (3x3 pixels)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    check_x = int(np.clip((x_pixel + dx), 0, grid_info['walls_detected'].shape[1] - 1))
                    check_y = int(np.clip((y_pixel + dy), 0, grid_info['walls_detected'].shape[0] - 1))
                    
                    if grid_info['walls_detected'][check_y, check_x] > 0:
                        return False
            
        except Exception:
            return False
        
        return True
    
    def _calculate_position_efficiency_3d(self, candidate_pos: Tuple[float, float, float], 
                                        existing_aps: List[Tuple[float, float, float, float]], 
                                        improvement: float) -> float:
        """
        Calcule l'efficacité d'une position candidate en 3D.
        """
        if not existing_aps:
            return improvement
        
        # Distance minimale aux APs existants
        min_distance = float('inf')
        for ap in existing_aps:
            distance = np.sqrt(sum((candidate_pos[i] - ap[i])**2 for i in range(3)))
            min_distance = min(min_distance, distance)
        
        # Calcul du facteur de distance par rapport à la distance optimale
        optimal_distance = 8.0  # Distance optimale en mètres
        distance_factor = 1.0
        
        if min_distance < 3.0:  # Trop proche
            distance_factor = 0.3
        elif min_distance < 6.0:  # Proche mais acceptable
            distance_factor = 0.7
        elif min_distance <= optimal_distance + 4.0:  # Distance optimale (8-12m)
            distance_factor = 1.0
        else:  # Trop loin
            distance_factor = 0.8
        
        return improvement * distance_factor
    
    def _filter_candidates_around_position_3d(self, candidates: List[Tuple[float, float, float]], 
                                            placed_position: Tuple[float, float, float], 
                                            min_distance: float) -> List[Tuple[float, float, float]]:
        """
        Filtre les candidats trop proches d'une position placée en 3D.
        """
        filtered = []
        
        for candidate in candidates:
            distance = np.sqrt(sum((candidate[i] - placed_position[i])**2 for i in range(3)))
            if distance >= min_distance:
                filtered.append(candidate)
        
        return filtered
    
    def _adjust_position_to_avoid_walls(self, x: float, y: float, grid_info: Dict, 
                                       longueur: float, largeur: float) -> Tuple[float, float]:
        """
        Ajuste une position pour éviter les murs en trouvant la position libre la plus proche.
        
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

    
    def _evaluate_configuration_3d(self, access_points: List[Tuple[float, float, float, float]], 
                                  coverage_points: List[Tuple[float, float, float]], 
                                  grid_info: Dict, target_coverage_db: float, 
                                  min_coverage_percent: float) -> Tuple[float, Dict]:
        """
        Évalue la qualité d'une configuration de points d'accès 3D avec PathlossCalculator3D.
        
        Args:
            access_points: Liste des points d'accès [(x, y, z, power), ...]
            coverage_points: Points à couvrir
            grid_info: Informations sur la grille
            target_coverage_db: Signal minimal requis
            min_coverage_percent: Couverture minimale requise
            
        Returns:
            Tuple (score, statistiques)
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
                    
                    # Validation des coordonnées pixel (amélioration de robustesse)
                    if (x_tx_pixel == x_rx_pixel and y_tx_pixel == y_rx_pixel):
                        # Même position en pixels, pas de murs à compter
                        wall_count = 0
                    else:
                        # Comptage des murs avec gestion d'erreur robuste
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
                    
                    # Calcul du pathloss avec PathlossCalculator3D
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
        
        # Statistiques
        total_points = len(coverage_points)
        coverage_percent = (covered_points / total_points) * 100 if total_points > 0 else 0.0
        
        # Score favorisant d'abord l'atteinte de l'objectif (harmonisation avec GMM et K-means)
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
            
            # Bonus supplémentaire pour dépassement significatif (harmonisation avec K-means)
            if coverage_percent > min_coverage_percent + 5:
                score += 0.2
        
        stats = {
            'covered_points': covered_points,
            'total_points': total_points,
            'coverage_percent': coverage_percent,
            'signal_levels': signal_levels,
            'num_access_points': num_aps
        }
        
        return max(score, 0.0), stats
    
    def visualize_greedy_process_3d(self, config: Dict, analysis: Dict,
                                   coverage_points: List[Tuple[float, float, float]],
                                   longueur: float, largeur: float, hauteur_totale: float) -> plt.Figure:
        """
        Visualise le processus d'optimisation Greedy 3D.
        
        Args:
            config: Configuration des points d'accès
            analysis: Analyse du processus Greedy
            coverage_points: Points de couverture
            longueur, largeur, hauteur_totale: Dimensions
            
        Returns:
            Figure matplotlib 3D
        """
        fig = plt.figure(figsize=(18, 12))
        
        # Graphique 1: Placement séquentiel 3D
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.set_title("Placement Séquentiel Greedy 3D", fontsize=14, fontweight='bold')
        
        # Points de couverture (échantillonnés pour performance)
        if len(coverage_points) > 1000:
            sample_indices = np.random.choice(len(coverage_points), 1000, replace=False)
            sample_points = [coverage_points[i] for i in sample_indices]
        else:
            sample_points = coverage_points
        
        if sample_points:
            points_array = np.array(sample_points)
            ax1.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], 
                       c='lightblue', s=6, alpha=0.3, label='Points à couvrir')
        
        # Points d'accès avec ordre de placement
        access_points = config['access_points']
        colors = plt.cm.plasma(np.linspace(0, 1, len(access_points)))
        
        for i, (x, y, z, power) in enumerate(access_points):
            ax1.scatter(x, y, z, c=[colors[i]], s=300, marker='*', 
                       edgecolors='white', linewidth=2, zorder=5)
            ax1.text(x, y, z + 0.5, f'{i+1}', fontsize=12, fontweight='bold', 
                    color='black', ha='center')
        
        ax1.set_xlim(0, longueur)
        ax1.set_ylim(0, largeur)
        ax1.set_zlim(0, hauteur_totale)
        ax1.set_xlabel('Longueur (m)')
        ax1.set_ylabel('Largeur (m)')
        ax1.set_zlabel('Hauteur (m)')
        ax1.legend()
        
        # Graphique 2: Évolution de la couverture
        ax2 = fig.add_subplot(222)
        ax2.set_title("Évolution de la Couverture par Étape", fontsize=14, fontweight='bold')
        
        if 'placement_history' in analysis:
            history = analysis['placement_history']
            steps = [0] + [h['ap_index'] for h in history]
            coverage = [0] + [h['coverage_after'] for h in history]
            improvements = [h['improvement'] for h in history]
            
            # Courbe de couverture
            line1 = ax2.plot(steps, coverage, 'b-o', linewidth=3, markersize=8, 
                           label='Couverture Cumulative')
            ax2.fill_between(steps, coverage, alpha=0.3, color='blue')
            
            # Barres d'amélioration
            ax2_twin = ax2.twinx()
            bars = ax2_twin.bar(steps[1:], improvements, alpha=0.6, color='orange', 
                               width=0.6, label='Amélioration par Étape')
            
            ax2.set_xlabel('Étape (AP ajouté)')
            ax2.set_ylabel('Couverture Cumulative (%)', color='blue')
            ax2_twin.set_ylabel('Amélioration (%)', color='orange')
            
            ax2.tick_params(axis='y', labelcolor='blue')
            ax2_twin.tick_params(axis='y', labelcolor='orange')
            
            # Légendes combinées
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(steps)
        
        # Graphique 3: Distribution verticale des APs
        ax3 = fig.add_subplot(223)
        ax3.set_title("Distribution Verticale des Points d'Accès", fontsize=14, fontweight='bold')
        
        if access_points:
            z_coords = [ap[2] for ap in access_points]
            floors = [int(z // 2.7) + 1 for z in z_coords]
            
            # Histogramme par étage
            floor_counts = {}
            for floor in floors:
                floor_counts[floor] = floor_counts.get(floor, 0) + 1
            
            if floor_counts:
                floors_list = sorted(floor_counts.keys())
                counts_list = [floor_counts[f] for f in floors_list]
                
                bars = ax3.bar(floors_list, counts_list, color='lightgreen', 
                              alpha=0.7, edgecolor='black')
                
                # Ajout des valeurs sur les barres
                for bar, count in zip(bars, counts_list):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                            f'{count}', ha='center', va='bottom', fontweight='bold')
                
                ax3.set_xlabel('Étage')
                ax3.set_ylabel('Nombre de Points d\'Accès')
                ax3.set_xticks(floors_list)
                ax3.grid(True, alpha=0.3, axis='y')
            
            # Graphique en violon pour distribution continue en Z
            ax3_twin = ax3.twinx()
            parts = ax3_twin.violinplot([z_coords], positions=[max(floors_list) + 1] if floors_list else [1], 
                                       widths=0.5, showmeans=True, showmedians=True)
            
            for pc in parts['bodies']:
                pc.set_facecolor('lightcoral')
                pc.set_alpha(0.6)
            
            ax3_twin.set_ylabel('Hauteur (m)', color='red')
            ax3_twin.tick_params(axis='y', labelcolor='red')
        
        # Graphique 4: Informations sur l'algorithme
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        ax4.set_title("Informations Algorithme Greedy", fontsize=14, fontweight='bold')
        
        info_text = f"Algorithme: Greedy (Glouton)\n"
        info_text += f"Points d'accès placés: {len(access_points)}\n"
        info_text += f"Couverture finale: {config['stats']['coverage_percent']:.1f}%\n"
        info_text += f"Score total: {config['score']:.3f}\n\n"
        
        if 'placement_history' in analysis:
            info_text += f"Historique de placement:\n"
            for i, step in enumerate(analysis['placement_history']):
                info_text += f"• AP{step['ap_index']}: +{step['improvement']:.1f}% "
                info_text += f"({step['coverage_after']:.1f}% total)\n"
        
        if analysis.get('total_evaluations'):
            info_text += f"\nÉvaluations totales: {analysis['total_evaluations']}\n"
            info_text += f"Positions candidates: {analysis['candidate_positions_used']}\n"
        
        info_text += f"\nCaractéristiques Greedy:\n"
        info_text += f"• Placement séquentiel optimal\n"
        info_text += f"• Maximise l'amélioration locale\n"
        info_text += f"• Convergence rapide\n"
        info_text += f"• Adaptatif au terrain"
        
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def get_optimization_summary_3d(self, config: Dict, analysis: Dict) -> Dict:
        """
        Génère un résumé de l'optimisation Greedy 3D.
        
        Args:
            config: Configuration optimale
            analysis: Analyse du processus
            
        Returns:
            Dictionnaire avec le résumé
        """
        summary = {
            'algorithm': 'Greedy 3D',
            'num_access_points': len(config['access_points']),
            'final_coverage': config['stats']['coverage_percent'],
            'final_score': config['score'],
            'convergence_steps': config.get('placement_steps', 0)
        }
        
        if 'placement_history' in analysis:
            history = analysis['placement_history']
            summary['total_improvement'] = sum(h['improvement'] for h in history)
            summary['average_improvement_per_step'] = summary['total_improvement'] / len(history) if history else 0
            summary['placement_efficiency'] = len(history) / len(config['access_points']) if config['access_points'] else 0
        
        # Analyse des positions
        if config['access_points']:
            z_coords = [ap[2] for ap in config['access_points']]
            summary['height_distribution'] = {
                'min_height': min(z_coords),
                'max_height': max(z_coords),
                'avg_height': sum(z_coords) / len(z_coords),
                'height_spread': max(z_coords) - min(z_coords)
            }
            
            # Distribution par étage
            floors = [int(z // 2.7) + 1 for z in z_coords]
            unique_floors = len(set(floors))
            summary['floor_distribution'] = {
                'floors_used': unique_floors,
                'total_floors': max(floors),
                'floor_utilization': unique_floors / max(floors) if max(floors) > 0 else 0
            }
        
        return summary
