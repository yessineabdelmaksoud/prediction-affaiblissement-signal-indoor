#!/usr/bin/env python3
"""
Optimiseur automatique 3D pour minimiser le pathloss.

Ce module implémente des algorithmes d'optimisation pour placer automatiquement
les points d'accès WiFi de manière à minimiser le pathloss vers des récepteurs donnés en 3D.
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict, Optional
from pathloss_calculator_3d import PathlossCalculator3D
from image_processor import ImageProcessor

class AutoOptimizer3D:
    """
    Optimiseur automatique pour placement de points d'accès 3D avec minimisation du pathloss.
    """
    
    def __init__(self, frequency_mhz: float):
        self.frequency_mhz = frequency_mhz
        self.calculator = PathlossCalculator3D(frequency_mhz)
        self.processor = ImageProcessor()

    def optimize_access_points(self, walls_detected: np.ndarray, receivers: List[Tuple[float, float, float]],
                              longueur: float, largeur: float, hauteur: float, max_access_points: int = 5,
                              power_tx: float = 20.0) -> Dict:
        """
        Optimise automatiquement le placement des points d'accès pour minimiser le pathloss en 3D.
        """
        print(f"🎯 Optimisation automatique 3D: {len(receivers)} récepteurs")
        print(f"📦 Volume: {longueur}m x {largeur}m x {hauteur}m, Fréquence: {self.frequency_mhz} MHz")
        return self._optimize_gradient_descent(
            walls_detected, receivers, longueur, largeur, hauteur, max_access_points, power_tx
        )

    def _optimize_gradient_descent(self, walls_detected: np.ndarray, receivers: List[Tuple[float, float, float]],
                                   longueur: float, largeur: float, hauteur: float, max_access_points: int,
                                   power_tx: float) -> Dict:
        print("⬇️ Optimisation par descente de gradient (3D)...")
        receivers_array = np.array(receivers)
        height2d, width2d = walls_detected.shape
        scale_x = longueur / width2d
        scale_y = largeur / height2d
        
        num_aps_needed = self._determine_optimal_num_aps(receivers_array, longueur, largeur, hauteur)
        num_aps_needed = min(num_aps_needed, max_access_points)
        print(f"📊 Analyse géométrique: {num_aps_needed} point(s) d'accès recommandé(s)")
        
        if num_aps_needed == 1:
            center = np.mean(receivers_array, axis=0)
            x, y, z = center
            x = np.clip(x, 1.0, longueur - 1.0)
            y = np.clip(y, 1.0, largeur - 1.0)
            z = np.clip(z, 0.5, hauteur - 0.5)
            x, y = self._find_nearest_valid_position(x, y, walls_detected, scale_x, scale_y, longueur, largeur)
            optimal_aps = [(x, y, z, power_tx)]
        else:
            optimal_aps = self._place_multiple_aps_gradient(
                receivers_array, walls_detected, scale_x, scale_y, longueur, largeur, hauteur, num_aps_needed, power_tx
            )
        stats = self._calculate_detailed_stats(
            optimal_aps, receivers, walls_detected, scale_x, scale_y
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

    def _determine_optimal_num_aps(self, receivers_array: np.ndarray, longueur: float, largeur: float, hauteur: float) -> int:
        num_receivers = len(receivers_array)
        if num_receivers == 1:
            return 1
        center = np.mean(receivers_array, axis=0)
        distances_from_center = np.linalg.norm(receivers_array - center, axis=1)
        max_distance = np.max(distances_from_center)
        max_pairwise_distance = 0
        for i in range(num_receivers):
            for j in range(i + 1, num_receivers):
                dist = np.linalg.norm(receivers_array[i] - receivers_array[j])
                max_pairwise_distance = max(max_pairwise_distance, dist)
        if max_pairwise_distance <= 10:
            return 1
        elif max_pairwise_distance <= 20:
            return 2
        elif max_pairwise_distance <= 35:
            return 3
        else:
            volume_per_ap = 300  # m³ par AP (estimation)
            total_volume = longueur * largeur * hauteur
            estimated_aps = max(2, min(8, int(total_volume / volume_per_ap)))
            return min(estimated_aps, max(2, num_receivers // 2))

    def _place_multiple_aps_gradient(self, receivers_array: np.ndarray, walls_detected: np.ndarray,
                                     scale_x: float, scale_y: float, longueur: float, largeur: float, hauteur: float,
                                     num_aps: int, power_tx: float) -> List[Tuple[float, float, float, float]]:
        try:
            kmeans = KMeans(n_clusters=num_aps, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(receivers_array)
            centers = kmeans.cluster_centers_
            optimal_aps = []
            for i, center in enumerate(centers):
                x, y, z = center
                x = np.clip(x, 1.0, longueur - 1.0)
                y = np.clip(y, 1.0, largeur - 1.0)
                z = np.clip(z, 0.5, hauteur - 0.5)
                cluster_receivers = receivers_array[cluster_labels == i]
                if len(cluster_receivers) > 0:
                    x, y = self._find_nearest_valid_position(x, y, walls_detected, scale_x, scale_y, longueur, largeur)
                optimal_aps.append((x, y, z, power_tx))
            return optimal_aps
        except ImportError:
            return self._fallback_geometric_placement(
                receivers_array, walls_detected, scale_x, scale_y, longueur, largeur, hauteur, num_aps, power_tx
            )

    def _find_nearest_valid_position(self, x: float, y: float, walls_detected: np.ndarray,
                                     scale_x: float, scale_y: float, longueur: float, largeur: float) -> Tuple[float, float]:
        height2d, width2d = walls_detected.shape
        x_pixel = int(np.clip(x / scale_x, 0, width2d - 1))
        y_pixel = int(np.clip(y / scale_y, 0, height2d - 1))
        if walls_detected[y_pixel, x_pixel] == 0:
            return x, y
        max_radius = 10
        for radius in range(1, max_radius + 1):
            for angle in np.arange(0, 2 * np.pi, np.pi / 8):
                test_x = x + radius * 0.5 * np.cos(angle)
                test_y = y + radius * 0.5 * np.sin(angle)
                if test_x < 1.0 or test_x > longueur - 1.0:
                    continue
                if test_y < 1.0 or test_y > largeur - 1.0:
                    continue
                test_x_pixel = int(np.clip(test_x / scale_x, 0, width2d - 1))
                test_y_pixel = int(np.clip(test_y / scale_y, 0, height2d - 1))
                if walls_detected[test_y_pixel, test_x_pixel] == 0:
                    return test_x, test_y
        return longueur / 2, largeur / 2

    def _calculate_detailed_stats(self, access_points: List[Tuple[float, float, float, float]],
                                  receivers: List[Tuple[float, float, float]], walls_detected: np.ndarray,
                                  scale_x: float, scale_y: float) -> Dict:
        height2d, width2d = walls_detected.shape
        pathloss_values = []
        receiver_stats = []
        for rx_x, rx_y, rx_z in receivers:
            min_pathloss = float('inf')
            best_ap_idx = -1
            for i, (ap_x, ap_y, ap_z, ap_power) in enumerate(access_points):
                distance_3d = np.sqrt((rx_x - ap_x)**2 + (rx_y - ap_y)**2 + (rx_z - ap_z)**2)
                ap_x_pixel = int(np.clip(ap_x / scale_x, 0, width2d - 1))
                ap_y_pixel = int(np.clip(ap_y / scale_y, 0, height2d - 1))
                rx_x_pixel = int(np.clip(rx_x / scale_x, 0, width2d - 1))
                rx_y_pixel = int(np.clip(rx_y / scale_y, 0, height2d - 1))
                try:
                    wall_count = ImageProcessor().count_walls_between_points(
                        walls_detected,
                        (ap_x_pixel, ap_y_pixel),
                        (rx_x_pixel, rx_y_pixel)
                    )
                except:
                    wall_count = 0
                floor_tx = int(ap_z // 2.7)
                floor_rx = int(rx_z // 2.7)
                floor_difference = abs(floor_rx - floor_tx)
                pathloss = self.calculator.calculate_pathloss_3d(distance_3d, wall_count, floor_difference)
                if pathloss < min_pathloss:
                    min_pathloss = pathloss
                    best_ap_idx = i
            pathloss_values.append(min_pathloss)
            received_power = access_points[best_ap_idx][3] - min_pathloss
            receiver_stats.append({
                'position': (rx_x, rx_y, rx_z),
                'pathloss': min_pathloss,
                'received_power': received_power,
                'best_ap': best_ap_idx,
                'distance_to_best_ap': np.sqrt(
                    (rx_x - access_points[best_ap_idx][0])**2 +
                    (rx_y - access_points[best_ap_idx][1])**2 +
                    (rx_z - access_points[best_ap_idx][2])**2
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
