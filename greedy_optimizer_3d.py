#!/usr/bin/env python3
"""
Optimiseur Greedy pour placement de points d'accès WiFi 3D.

Ce module implémente un algorithme glouton (greedy) pour optimiser 
le placement des points d'accès WiFi dans un environnement 3D.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, Optional, Any
import random


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
            frequency: Fréquence de transmission en Hz
        """
        self.frequency = frequency
    
    def optimize_greedy_placement_3d(self, coverage_points: List[Tuple[float, float, float]], 
                                    grid_info: Dict, longueur: float, largeur: float, hauteur_totale: float,
                                    target_coverage_db: float, min_coverage_percent: float, 
                                    power_tx: float, max_access_points: int) -> Tuple[Optional[Dict], Dict]:
        """
        Optimise le placement des points d'accès 3D avec l'algorithme glouton.
        
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
        
        print(f"🎯 Optimisation Greedy 3D: {len(coverage_points)} points à couvrir")
        print(f"📦 Volume: {longueur}m x {largeur}m x {hauteur_totale}m")
        
        # Initialisation
        access_points = []
        uncovered_points = set(range(len(coverage_points)))
        placement_history = []
        
        # Génération de positions candidates
        candidate_positions = self._generate_candidate_positions_3d(
            longueur, largeur, hauteur_totale, grid_info
        )
        
        print(f"🔍 {len(candidate_positions)} positions candidates générées")
        print(f"🎯 Objectif: {min_coverage_percent}% de couverture")
        
        # Placement glouton
        for ap_index in range(max_access_points):
            # Évaluation initiale
            if ap_index == 0:
                current_score, current_stats = self._evaluate_configuration_3d(
                    access_points, coverage_points, grid_info, 
                    target_coverage_db, min_coverage_percent
                )
                print(f"🎯 État initial: {current_stats['coverage_percent']:.1f}% couverture")
            
            # Recherche de la meilleure position pour le prochain AP
            best_position = None
            best_improvement = 0.0
            best_stats = None
            
            for candidate_pos in candidate_positions:
                # Test de cette position
                test_access_points = access_points + [(candidate_pos[0], candidate_pos[1], candidate_pos[2], power_tx)]
                
                test_score, test_stats = self._evaluate_configuration_3d(
                    test_access_points, coverage_points, grid_info,
                    target_coverage_db, min_coverage_percent
                )
                
                improvement = test_stats['coverage_percent'] - current_stats['coverage_percent']
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_position = candidate_pos
                    best_stats = test_stats
            
            # Ajout du meilleur point d'accès trouvé
            if best_position and best_improvement > 0.1:  # Amélioration minimale de 0.1%
                new_ap = (best_position[0], best_position[1], best_position[2], power_tx)
                access_points.append(new_ap)
                
                placement_history.append({
                    'ap_index': ap_index + 1,
                    'position': best_position,
                    'coverage_before': current_stats['coverage_percent'],
                    'coverage_after': best_stats['coverage_percent'],
                    'improvement': best_improvement,
                    'covered_points': best_stats['covered_points'],
                    'total_points': best_stats['total_points']
                })
                
                current_stats = best_stats
                
                print(f"📍 AP{ap_index + 1} placé à ({best_position[0]:.1f}, {best_position[1]:.1f}, {best_position[2]:.1f})")
                print(f"   ➜ Couverture: {current_stats['coverage_percent']:.1f}% (+{best_improvement:.1f}%)")
                
                # Vérification si objectif atteint
                if current_stats['coverage_percent'] >= min_coverage_percent:
                    print(f"✅ Objectif Greedy {min_coverage_percent}% atteint avec {len(access_points)} points d'accès!")
                    break
                
                # Mise à jour des positions candidates (éviter trop proche)
                candidate_positions = self._filter_candidate_positions(
                    candidate_positions, best_position, min_distance=3.0
                )
                
            else:
                print(f"⚠️  Aucune amélioration significative trouvée pour AP{ap_index + 1}")
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
                'algorithm': 'greedy',
                'placement_steps': len(access_points)
            }
            
            analysis = {
                'placement_history': placement_history,
                'candidate_positions_used': len(candidate_positions),
                'total_evaluations': len(candidate_positions) * len(access_points),
                'convergence_step': len(access_points)
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
    
    def _generate_candidate_positions_3d(self, longueur: float, largeur: float, hauteur_totale: float, 
                                        grid_info: Dict) -> List[Tuple[float, float, float]]:
        """
        Génère des positions candidates pour les points d'accès 3D.
        
        Args:
            longueur, largeur, hauteur_totale: Dimensions
            grid_info: Informations sur la grille
            
        Returns:
            Liste des positions candidates
        """
        candidate_positions = []
        
        # Stratégie 1: Grille régulière 3D
        x_step = max(2.0, longueur / 10)
        y_step = max(2.0, largeur / 10)
        z_step = max(1.0, hauteur_totale / 8)
        
        for x in np.arange(x_step, longueur - x_step, x_step):
            for y in np.arange(y_step, largeur - y_step, y_step):
                for z in np.arange(z_step, hauteur_totale - z_step, z_step):
                    # Vérification que la position n'est pas dans un mur (projection 2D)
                    if self._is_valid_position(x, y, z, grid_info, longueur, largeur, hauteur_totale):
                        candidate_positions.append((x, y, z))
        
        # Stratégie 2: Positions aléatoires pour diversité
        for _ in range(200):
            x = random.uniform(1.0, longueur - 1.0)
            y = random.uniform(1.0, largeur - 1.0)
            z = random.uniform(0.5, hauteur_totale - 0.5)
            
            if self._is_valid_position(x, y, z, grid_info, longueur, largeur, hauteur_totale):
                candidate_positions.append((x, y, z))
        
        # Stratégie 3: Positions centrales par étage
        num_floors = max(1, int(hauteur_totale // 2.7))
        for floor in range(num_floors):
            z = (floor + 0.5) * 2.7
            if z < hauteur_totale:
                # Centre de l'étage
                center_x, center_y = longueur / 2, largeur / 2
                if self._is_valid_position(center_x, center_y, z, grid_info, longueur, largeur, hauteur_totale):
                    candidate_positions.append((center_x, center_y, z))
                
                # Coins de l'étage
                corners = [
                    (longueur * 0.25, largeur * 0.25, z),
                    (longueur * 0.75, largeur * 0.25, z),
                    (longueur * 0.25, largeur * 0.75, z),
                    (longueur * 0.75, largeur * 0.75, z)
                ]
                
                for corner_x, corner_y, corner_z in corners:
                    if self._is_valid_position(corner_x, corner_y, corner_z, grid_info, longueur, largeur, hauteur_totale):
                        candidate_positions.append((corner_x, corner_y, corner_z))
        
        # Suppression des doublons
        unique_positions = []
        for pos in candidate_positions:
            is_duplicate = False
            for existing_pos in unique_positions:
                distance = np.sqrt(sum((pos[i] - existing_pos[i])**2 for i in range(3)))
                if distance < 0.5:  # Seuil de proximité
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_positions.append(pos)
        
        return unique_positions
    
    def _is_valid_position(self, x: float, y: float, z: float, grid_info: Dict, 
                          longueur: float, largeur: float, hauteur_totale: float) -> bool:
        """
        Vérifie si une position 3D est valide pour placer un point d'accès.
        
        Args:
            x, y, z: Coordonnées de la position
            grid_info: Informations sur la grille
            longueur, largeur, hauteur_totale: Dimensions
            
        Returns:
            True si la position est valide
        """
        # Vérification des limites
        if x < 0.5 or x > longueur - 0.5:
            return False
        if y < 0.5 or y > largeur - 0.5:
            return False
        if z < 0.3 or z > hauteur_totale - 0.3:
            return False
        
        # Vérification des murs (projection 2D)
        try:
            x_pixel = int(np.clip(x / grid_info['scale_x'], 0, grid_info['walls_detected'].shape[1] - 1))
            y_pixel = int(np.clip(y / grid_info['scale_y'], 0, grid_info['walls_detected'].shape[0] - 1))
            
            # Si dans un mur, position invalide
            if grid_info['walls_detected'][y_pixel, x_pixel] > 0:
                return False
        except:
            return False
        
        return True
    
    def _filter_candidate_positions(self, candidate_positions: List[Tuple[float, float, float]], 
                                   placed_position: Tuple[float, float, float], 
                                   min_distance: float) -> List[Tuple[float, float, float]]:
        """
        Filtre les positions candidates pour éviter la proximité avec un AP déjà placé.
        
        Args:
            candidate_positions: Positions candidates
            placed_position: Position d'un AP déjà placé
            min_distance: Distance minimale requise
            
        Returns:
            Liste filtrée des positions candidates
        """
        filtered_positions = []
        
        for pos in candidate_positions:
            distance = np.sqrt(sum((pos[i] - placed_position[i])**2 for i in range(3)))
            if distance >= min_distance:
                filtered_positions.append(pos)
        
        return filtered_positions
    
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
            coverage_threshold = 12.0  # mètres (plus conservateur que GMM)
            if best_distance < coverage_threshold:
                covered_points += 1
        
        # Statistiques de base
        total_points = len(coverage_points)
        coverage_percent = (covered_points / total_points) * 100 if total_points > 0 else 0.0
        
        # Score favorisant d'abord l'atteinte de l'objectif
        num_aps = len(access_points)
        coverage_score = coverage_percent / 100.0
        
        # Score greedy: privilégie fortement la couverture
        if coverage_percent < min_coverage_percent:
            score = coverage_score * 3.0  # Bonus important pour coverage
            efficiency_penalty = num_aps * 0.005  # Pénalité très faible
            score -= efficiency_penalty
        else:
            score = 2.0 + coverage_score  # Bonus énorme si objectif atteint
            efficiency_penalty = (num_aps - 1) * 0.02  # Pénalité modérée pour efficacité
            score -= efficiency_penalty
        
        stats = {
            'covered_points': covered_points,
            'total_points': total_points,
            'coverage_percent': coverage_percent,
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
