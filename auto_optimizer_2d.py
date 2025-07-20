#!/usr/bin/env python3
"""
Optimiseur automatique 2D pour minimiser le pathloss.

Ce module implémente des algorithmes d'optimisation pour placer automatiquement
les points d'accès WiFi de manière à minimiser le pathloss vers des récepteurs donnés.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict, Optional
from pathloss_calculator import PathlossCalculator
from image_processor import ImageProcessor


class AutoOptimizer2D:
    """
    Optimiseur automatique pour placement de points d'accès 2D avec minimisation du pathloss.
    """
    
    def __init__(self, frequency_mhz: float):
        """
        Initialise l'optimiseur automatique 2D.
        
        Args:
            frequency_mhz: Fréquence en MHz
        """
        self.frequency_mhz = frequency_mhz
        self.calculator = PathlossCalculator(frequency_mhz)
        self.processor = ImageProcessor()
    
    def optimize_access_points(self, walls_detected: np.ndarray, receivers: List[Tuple[float, float]], 
                             longueur: float, largeur: float, max_access_points: int = 5,
                             power_tx: float = 20.0) -> Dict:
        """
        Optimise automatiquement le placement des points d'accès pour minimiser le pathloss.
        
        Args:
            walls_detected: Masque binaire des murs
            receivers: Liste des positions des récepteurs [(x_meter, y_meter), ...]
            longueur, largeur: Dimensions en mètres
            max_access_points: Nombre maximum de points d'accès
            power_tx: Puissance de transmission en dBm
            
        Returns:
            Dictionnaire avec la configuration optimale
        """
        print(f"🎯 Optimisation automatique 2D: {len(receivers)} récepteurs")
        print(f"📦 Zone: {longueur}m x {largeur}m, Fréquence: {self.frequency_mhz} MHz")
        
        return self._optimize_gradient_descent(
            walls_detected, receivers, longueur, largeur, max_access_points, power_tx
        )
    
    def _optimize_gradient_descent(self, walls_detected: np.ndarray, receivers: List[Tuple[float, float]],
                                  longueur: float, largeur: float, max_access_points: int,
                                  power_tx: float) -> Dict:
        """
        Optimisation par descente de gradient avec adaptation intelligente du nombre d'APs.
        """
        print("⬇️ Optimisation par descente de gradient...")
        
        receivers_array = np.array(receivers)
        height, width = walls_detected.shape
        scale_x = longueur / width
        scale_y = largeur / height
        
        # Analyse de la géométrie pour déterminer le nombre optimal d'APs
        num_aps_needed = self._determine_optimal_num_aps(receivers_array, longueur, largeur)
        num_aps_needed = min(num_aps_needed, max_access_points)
        
        print(f"📊 Analyse géométrique: {num_aps_needed} point(s) d'accès recommandé(s)")
        
        if num_aps_needed == 1:
            # Un seul AP suffit - placer au centroïde optimisé
            center_x = np.mean(receivers_array[:, 0])
            center_y = np.mean(receivers_array[:, 1])
            
            # Contraintes
            center_x = np.clip(center_x, 1.0, longueur - 1.0)
            center_y = np.clip(center_y, 1.0, largeur - 1.0)
            
            # Ajustement pour éviter les murs
            center_x, center_y = self._find_nearest_valid_position(
                center_x, center_y, walls_detected, scale_x, scale_y, longueur, largeur
            )
            
            optimal_aps = [(center_x, center_y, power_tx)]
            
        else:
            # Plusieurs APs nécessaires - utiliser clustering intelligent
            optimal_aps = self._place_multiple_aps_gradient(
                receivers_array, walls_detected, scale_x, scale_y, 
                longueur, largeur, num_aps_needed, power_tx
            )
        
        # Calcul des statistiques
        stats = self._calculate_detailed_stats(
            optimal_aps, receivers, walls_detected, 
            longueur / walls_detected.shape[1], largeur / walls_detected.shape[0]
        )
        
        config = {
            'access_points': optimal_aps,
            'num_access_points': len(optimal_aps),
            'total_score': stats['avg_pathloss'],
            'avg_pathloss': stats['avg_pathloss'],
            'max_pathloss': stats['max_pathloss'],
            'min_pathloss': stats['min_pathloss'],
            'stats': stats,
            'optimization_success': True
        }
        
        return {
            'best_config': config,
            'all_results': {len(optimal_aps): config},
            'algorithm': 'gradient',
            'receivers': receivers
        }
    
    def _determine_optimal_num_aps(self, receivers_array: np.ndarray, longueur: float, largeur: float) -> int:
        """
        Détermine le nombre optimal de points d'accès basé sur la géométrie des récepteurs.
        
        Args:
            receivers_array: Array des positions des récepteurs
            longueur, largeur: Dimensions de l'environnement
            
        Returns:
            Nombre optimal de points d'accès
        """
        num_receivers = len(receivers_array)
        
        if num_receivers == 1:
            return 1
        
        # Calcul de la dispersion géométrique
        center = np.mean(receivers_array, axis=0)
        distances_from_center = np.sqrt(np.sum((receivers_array - center)**2, axis=1))
        max_distance = np.max(distances_from_center)
        avg_distance = np.mean(distances_from_center)
        
        # Calcul de la distance maximale entre tous les récepteurs
        max_pairwise_distance = 0
        for i in range(num_receivers):
            for j in range(i + 1, num_receivers):
                dist = np.sqrt(np.sum((receivers_array[i] - receivers_array[j])**2))
                max_pairwise_distance = max(max_pairwise_distance, dist)
        
        # Règles heuristiques pour déterminer le nombre d'APs
        if max_pairwise_distance <= 10:
            # Récepteurs proches - 1 AP suffit
            return 1
        elif max_pairwise_distance <= 20:
            # Distance moyenne - 2 APs recommandés
            return 2
        elif max_pairwise_distance <= 35:
            # Grande distance - 3 APs
            return 3
        else:
            # Très grande distance - estimer basé sur la densité
            area_per_ap = 150  # m² par AP (estimation conservative)
            total_area = longueur * largeur
            estimated_aps = max(2, min(5, int(total_area / area_per_ap)))
            
            # Limiter par le nombre de récepteurs
            return min(estimated_aps, max(2, num_receivers // 2))
    
    def _place_multiple_aps_gradient(self, receivers_array: np.ndarray, walls_detected: np.ndarray,
                                    scale_x: float, scale_y: float, longueur: float, largeur: float,
                                    num_aps: int, power_tx: float) -> List[Tuple[float, float, float]]:
        """
        Place intelligemment plusieurs points d'accès en utilisant K-means + optimisation locale.
        
        Args:
            receivers_array: Positions des récepteurs
            walls_detected: Masque des murs
            scale_x, scale_y: Échelles de conversion
            longueur, largeur: Dimensions
            num_aps: Nombre de points d'accès à placer
            power_tx: Puissance de transmission
            
        Returns:
            Liste des positions optimisées des APs
        """
        try:
            from sklearn.cluster import KMeans
            
            # Clustering des récepteurs
            kmeans = KMeans(n_clusters=num_aps, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(receivers_array)
            centers = kmeans.cluster_centers_
            
            optimal_aps = []
            
            for i, center in enumerate(centers):
                x, y = center
                
                # Contraintes de l'environnement
                x = np.clip(x, 1.0, longueur - 1.0)
                y = np.clip(y, 1.0, largeur - 1.0)
                
                # Optimisation locale pour ce cluster
                cluster_receivers = receivers_array[cluster_labels == i]
                
                if len(cluster_receivers) > 0:
                    # Optimiser la position pour minimiser le pathloss vers ce groupe
                    x, y = self._optimize_local_position(
                        x, y, cluster_receivers, walls_detected, 
                        scale_x, scale_y, longueur, largeur, power_tx
                    )
                
                # Ajustement final pour éviter les murs
                x, y = self._find_nearest_valid_position(
                    x, y, walls_detected, scale_x, scale_y, longueur, largeur
                )
                
                optimal_aps.append((x, y, power_tx))
            
            return optimal_aps
            
        except ImportError:
            # Fallback sans sklearn - placement géométrique simple
            return self._fallback_geometric_placement(
                receivers_array, walls_detected, scale_x, scale_y, 
                longueur, largeur, num_aps, power_tx
            )
    
    def _optimize_local_position(self, initial_x: float, initial_y: float, 
                                cluster_receivers: np.ndarray, walls_detected: np.ndarray,
                                scale_x: float, scale_y: float, longueur: float, largeur: float,
                                power_tx: float) -> Tuple[float, float]:
        """
        Optimise localement la position d'un AP pour un groupe de récepteurs.
        """
        best_x, best_y = initial_x, initial_y
        best_score = self._evaluate_ap_position(
            initial_x, initial_y, cluster_receivers, walls_detected, scale_x, scale_y, power_tx
        )
        
        # Recherche par grille locale (±3m autour de la position initiale)
        search_radius = 3.0
        step_size = 0.5
        
        for dx in np.arange(-search_radius, search_radius + step_size, step_size):
            for dy in np.arange(-search_radius, search_radius + step_size, step_size):
                test_x = initial_x + dx
                test_y = initial_y + dy
                
                # Vérifier les contraintes
                if test_x < 1.0 or test_x > longueur - 1.0:
                    continue
                if test_y < 1.0 or test_y > largeur - 1.0:
                    continue
                
                # Évaluer cette position
                score = self._evaluate_ap_position(
                    test_x, test_y, cluster_receivers, walls_detected, scale_x, scale_y, power_tx
                )
                
                if score < best_score:
                    best_score = score
                    best_x, best_y = test_x, test_y
        
        return best_x, best_y
    
    def _evaluate_ap_position(self, ap_x: float, ap_y: float, receivers: np.ndarray,
                             walls_detected: np.ndarray, scale_x: float, scale_y: float,
                             power_tx: float) -> float:
        """
        Évalue une position d'AP en calculant le pathloss total vers un groupe de récepteurs.
        """
        total_pathloss = 0.0
        height, width = walls_detected.shape
        
        for rx_x, rx_y in receivers:
            # Distance 2D
            distance_2d = np.sqrt((rx_x - ap_x)**2 + (rx_y - ap_y)**2)
            
            # Coordonnées pixel
            ap_x_pixel = int(np.clip(ap_x / scale_x, 0, width - 1))
            ap_y_pixel = int(np.clip(ap_y / scale_y, 0, height - 1))
            rx_x_pixel = int(np.clip(rx_x / scale_x, 0, width - 1))
            rx_y_pixel = int(np.clip(rx_y / scale_y, 0, height - 1))
            
            # Comptage des murs
            try:
                wall_count = self.processor.count_walls_between_points(
                    walls_detected,
                    (ap_x_pixel, ap_y_pixel),
                    (rx_x_pixel, rx_y_pixel)
                )
            except:
                wall_count = 0
            
            # Calcul du pathloss
            pathloss = self.calculator.calculate_pathloss(distance_2d, wall_count)
            total_pathloss += pathloss
        
        return total_pathloss / len(receivers)  # Pathloss moyen
    
    def _fallback_geometric_placement(self, receivers_array: np.ndarray, walls_detected: np.ndarray,
                                     scale_x: float, scale_y: float, longueur: float, largeur: float,
                                     num_aps: int, power_tx: float) -> List[Tuple[float, float, float]]:
        """
        Placement géométrique fallback sans sklearn.
        """
        optimal_aps = []
        
        if num_aps == 2:
            # Diviser les récepteurs en deux groupes par position X
            sorted_by_x = receivers_array[receivers_array[:, 0].argsort()]
            mid_point = len(sorted_by_x) // 2
            
            group1 = sorted_by_x[:mid_point]
            group2 = sorted_by_x[mid_point:]
            
            for group in [group1, group2]:
                if len(group) > 0:
                    center_x = np.mean(group[:, 0])
                    center_y = np.mean(group[:, 1])
                    
                    center_x = np.clip(center_x, 1.0, longueur - 1.0)
                    center_y = np.clip(center_y, 1.0, largeur - 1.0)
                    
                    center_x, center_y = self._find_nearest_valid_position(
                        center_x, center_y, walls_detected, scale_x, scale_y, longueur, largeur
                    )
                    
                    optimal_aps.append((center_x, center_y, power_tx))
        
        else:
            # Pour 3+ APs, placer aux coins et au centre
            positions = [
                (longueur * 0.25, largeur * 0.25),
                (longueur * 0.75, largeur * 0.25),
                (longueur * 0.5, largeur * 0.75)
            ]
            
            for i in range(min(num_aps, len(positions))):
                x, y = positions[i]
                x, y = self._find_nearest_valid_position(
                    x, y, walls_detected, scale_x, scale_y, longueur, largeur
                )
                optimal_aps.append((x, y, power_tx))
        
        return optimal_aps
    
    def _find_nearest_valid_position(self, x: float, y: float, walls_detected: np.ndarray,
                                    scale_x: float, scale_y: float, longueur: float, largeur: float) -> Tuple[float, float]:
        """
        Trouve la position valide la plus proche (hors des murs).
        """
        height, width = walls_detected.shape
        x_pixel = int(np.clip(x / scale_x, 0, width - 1))
        y_pixel = int(np.clip(y / scale_y, 0, height - 1))
        
        # Si déjà valide, retourner
        if walls_detected[y_pixel, x_pixel] == 0:
            return x, y
        
        # Recherche en spirale
        max_radius = 10
        for radius in range(1, max_radius + 1):
            for angle in np.arange(0, 2 * np.pi, np.pi / 8):
                test_x = x + radius * 0.5 * np.cos(angle)  # 0.5m par step
                test_y = y + radius * 0.5 * np.sin(angle)
                
                if test_x < 1.0 or test_x > longueur - 1.0:
                    continue
                if test_y < 1.0 or test_y > largeur - 1.0:
                    continue
                
                test_x_pixel = int(np.clip(test_x / scale_x, 0, width - 1))
                test_y_pixel = int(np.clip(test_y / scale_y, 0, height - 1))
                
                if walls_detected[test_y_pixel, test_x_pixel] == 0:
                    return test_x, test_y
        
        # Fallback: centre de l'environnement
        return longueur / 2, largeur / 2
    
    def _calculate_detailed_stats(self, access_points: List[Tuple[float, float, float]], 
                                 receivers: List[Tuple[float, float]], walls_detected: np.ndarray,
                                 scale_x: float, scale_y: float) -> Dict:
        """
        Calcule les statistiques détaillées de la configuration.
        """
        height, width = walls_detected.shape
        pathloss_values = []
        receiver_stats = []
        
        for rx_x, rx_y in receivers:
            min_pathloss = float('inf')
            best_ap_idx = -1
            
            for i, (ap_x, ap_y, ap_power) in enumerate(access_points):
                # Distance 2D
                distance_2d = np.sqrt((rx_x - ap_x)**2 + (rx_y - ap_y)**2)
                
                # Coordonnées pixel
                ap_x_pixel = int(np.clip(ap_x / scale_x, 0, width - 1))
                ap_y_pixel = int(np.clip(ap_y / scale_y, 0, height - 1))
                rx_x_pixel = int(np.clip(rx_x / scale_x, 0, width - 1))
                rx_y_pixel = int(np.clip(rx_y / scale_y, 0, height - 1))
                
                # Comptage des murs
                try:
                    wall_count = self.processor.count_walls_between_points(
                        walls_detected,
                        (ap_x_pixel, ap_y_pixel),
                        (rx_x_pixel, rx_y_pixel)
                    )
                except:
                    wall_count = 0
                
                # Pathloss
                pathloss = self.calculator.calculate_pathloss(distance_2d, wall_count)
                
                if pathloss < min_pathloss:
                    min_pathloss = pathloss
                    best_ap_idx = i
            
            pathloss_values.append(min_pathloss)
            
            # Puissance reçue
            received_power = access_points[best_ap_idx][2] - min_pathloss
            
            receiver_stats.append({
                'position': (rx_x, rx_y),
                'pathloss': min_pathloss,
                'received_power': received_power,
                'best_ap': best_ap_idx,
                'distance_to_best_ap': np.sqrt(
                    (rx_x - access_points[best_ap_idx][0])**2 + 
                    (rx_y - access_points[best_ap_idx][1])**2
                )
            })
        
        return {
            'avg_pathloss': np.mean(pathloss_values),
            'max_pathloss': np.max(pathloss_values),
            'min_pathloss': np.min(pathloss_values),
            'std_pathloss': np.std(pathloss_values),
            'pathloss_values': pathloss_values,
            'receiver_stats': receiver_stats,
            'coverage_quality': self._assess_coverage_quality(pathloss_values)
        }
    
    def _assess_coverage_quality(self, pathloss_values: List[float]) -> Dict:
        """
        Évalue la qualité de la couverture basée sur les valeurs de pathloss.
        """
        excellent_count = sum(1 for pl in pathloss_values if pl <= 50)
        good_count = sum(1 for pl in pathloss_values if 50 < pl <= 70)
        fair_count = sum(1 for pl in pathloss_values if 70 < pl <= 90)
        poor_count = sum(1 for pl in pathloss_values if pl > 90)
        
        total = len(pathloss_values)
        
        return {
            'excellent_percent': (excellent_count / total) * 100 if total > 0 else 0,
            'good_percent': (good_count / total) * 100 if total > 0 else 0,
            'fair_percent': (fair_count / total) * 100 if total > 0 else 0,
            'poor_percent': (poor_count / total) * 100 if total > 0 else 0,
            'total_receivers': total
        }
    
    def visualize_optimization_result(self, result: Dict, walls_detected: np.ndarray,
                                    longueur: float, largeur: float) -> plt.Figure:
        """
        Visualise le résultat de l'optimisation.
        """
        config = result['best_config']
        access_points = config['access_points']
        receivers = result['receivers']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Graphique 1: Plan avec positions optimisées
        ax1.imshow(walls_detected, cmap='gray_r', extent=[0, longueur, largeur, 0])
        ax1.set_title(f"Positions Optimisées - {result['algorithm'].upper()}\n"
                     f"Score: {config['total_score']:.1f}, Avg PL: {config['avg_pathloss']:.1f}dB")
        
        # Points d'accès
        for i, (ap_x, ap_y, ap_power) in enumerate(access_points):
            ax1.scatter(ap_x, ap_y, c='red', s=200, marker='*', 
                       edgecolors='black', linewidth=2, zorder=5)
            ax1.annotate(f'AP{i+1}\n{ap_power}dBm', (ap_x, ap_y), 
                        xytext=(5, 5), textcoords='offset points',
                        fontweight='bold', color='red')
        
        # Récepteurs avec qualité de signal
        for i, (rx_x, rx_y) in enumerate(receivers):
            stats = config['stats']['receiver_stats'][i]
            pathloss = stats['pathloss']
            
            if pathloss <= 50:
                color = 'green'
            elif pathloss <= 70:
                color = 'orange'
            else:
                color = 'red'
            
            ax1.scatter(rx_x, rx_y, c=color, s=100, marker='o',
                       edgecolors='black', linewidth=1, zorder=4)
            ax1.annotate(f'R{i+1}\n{pathloss:.0f}dB', (rx_x, rx_y),
                        xytext=(5, -15), textcoords='offset points',
                        fontsize=8)
        
        ax1.set_xlabel('Longueur (m)')
        ax1.set_ylabel('Largeur (m)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(['Murs', 'Points d\'accès', 'Récepteurs'], loc='upper right')
        
        # Graphique 2: Statistiques
        ax2.set_title("Analyse des Performances")
        
        # Histogramme des pathloss
        pathloss_values = config['stats']['pathloss_values']
        bins = np.arange(0, max(pathloss_values) + 10, 10)
        ax2.hist(pathloss_values, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(config['avg_pathloss'], color='red', linestyle='--', linewidth=2, 
                   label=f'Moyenne: {config["avg_pathloss"]:.1f}dB')
        ax2.set_xlabel('Pathloss (dB)')
        ax2.set_ylabel('Nombre de récepteurs')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Texte avec statistiques
        coverage_quality = config['stats']['coverage_quality']
        stats_text = f"""Résumé de l'optimisation:
• Algorithme: {result['algorithm'].upper()}
• Points d'accès: {config['num_access_points']}
• Récepteurs: {len(receivers)}
• Pathloss moyen: {config['avg_pathloss']:.1f} dB
• Pathloss max: {config['max_pathloss']:.1f} dB
• Pathloss min: {config['min_pathloss']:.1f} dB

Qualité de couverture:
• Excellente: {coverage_quality['excellent_percent']:.0f}%
• Bonne: {coverage_quality['good_percent']:.0f}%
• Correcte: {coverage_quality['fair_percent']:.0f}%
• Faible: {coverage_quality['poor_percent']:.0f}%"""
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9)
        
        plt.tight_layout()
        return fig
