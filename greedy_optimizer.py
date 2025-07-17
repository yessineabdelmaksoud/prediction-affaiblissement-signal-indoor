import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
from pathloss_calculator import PathlossCalculator


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
        self.pathloss_calculator = PathlossCalculator(frequency)
    
    def optimize_greedy_placement(self, coverage_points: List[Tuple[float, float]], 
                                 grid_info: Dict, longueur: float, largeur: float,
                                 target_coverage_db: float, min_coverage_percent: float,
                                 power_tx: float, max_access_points: int) -> Optional[Tuple[Dict, Dict]]:
        """
        Optimise le placement des points d'accès avec l'algorithme Greedy.
        
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
        
        # Générer les positions candidates pour les points d'accès
        candidate_positions = self._generate_candidate_positions(longueur, largeur)
        
        # Initialisation
        access_points = []
        covered_mask = np.zeros(len(coverage_points), dtype=bool)
        steps = []
        total_iterations = 0
        
        print(f"🎯 Optimisation Greedy: {len(coverage_points)} points à couvrir")
        print(f"📍 {len(candidate_positions)} positions candidates")
        
        # Algorithme Greedy - placement séquentiel
        for ap_count in range(max_access_points):
            best_position = None
            best_gain = 0
            best_new_coverage = None
            
            # Tester chaque position candidate
            for pos_x, pos_y in candidate_positions:
                total_iterations += 1
                
                # Calculer la nouvelle couverture pour cette position
                new_coverage_mask = self._calculate_new_coverage(
                    pos_x, pos_y, coverage_points, target_coverage_db, power_tx
                )
                
                # Calculer le gain (nouveaux points couverts)
                additional_coverage = new_coverage_mask & ~covered_mask
                gain = np.sum(additional_coverage)
                
                # Mettre à jour le meilleur choix
                if gain > best_gain:
                    best_gain = gain
                    best_position = (pos_x, pos_y)
                    best_new_coverage = new_coverage_mask
            
            # Vérifier si on a trouvé une position utile
            if best_position is None or best_gain == 0:
                print(f"🛑 Arrêt Greedy: Aucun gain supplémentaire possible")
                break
            
            # Ajouter le meilleur point d'accès
            access_points.append((best_position[0], best_position[1], power_tx))
            covered_mask |= best_new_coverage
            
            # Enregistrer l'étape
            coverage_percent = (np.sum(covered_mask) / len(coverage_points)) * 100
            step_info = {
                'position': best_position,
                'coverage_gain': best_gain,
                'total_coverage': np.sum(covered_mask),
                'coverage_percent': coverage_percent
            }
            steps.append(step_info)
            
            print(f"📍 AP {ap_count + 1}: Position {best_position} -> {coverage_percent:.1f}% couverture (+{best_gain} points)")
            
            # Vérifier si l'objectif de couverture est atteint
            if coverage_percent >= min_coverage_percent:
                print(f"🎯 Objectif atteint: {coverage_percent:.1f}% >= {min_coverage_percent}%")
                break
            
            # Retirer la position utilisée des candidats pour éviter les doublons
            candidate_positions.remove(best_position)
        
        # Calculer les statistiques finales
        final_coverage_percent = (np.sum(covered_mask) / len(coverage_points)) * 100
        covered_points = np.sum(covered_mask)
        
        # Calculer le score (balance entre couverture et nombre d'APs)
        coverage_score = final_coverage_percent / 100.0
        efficiency_score = 1.0 - (len(access_points) / max_access_points)
        total_score = 0.7 * coverage_score + 0.3 * efficiency_score
        
        # Déterminer la raison d'arrêt
        if final_coverage_percent >= min_coverage_percent:
            convergence_reason = "target_coverage_reached"
        elif len(access_points) >= max_access_points:
            convergence_reason = "max_access_points_reached"
        else:
            convergence_reason = "no_additional_gain"
        
        # Configuration finale
        config = {
            'access_points': access_points,
            'score': total_score,
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
            'final_coverage_mask': covered_mask
        }
        
        print(f"✅ Optimisation Greedy terminée:")
        print(f"   - {len(access_points)} points d'accès")
        print(f"   - {final_coverage_percent:.1f}% de couverture")
        print(f"   - Score: {total_score:.3f}")
        print(f"   - {total_iterations} itérations")
        
        return config, analysis
    
    def _generate_candidate_positions(self, longueur: float, largeur: float, 
                                    resolution: int = 15) -> List[Tuple[float, float]]:
        """
        Génère les positions candidates pour les points d'accès.
        
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
    
    def _calculate_new_coverage(self, ap_x: float, ap_y: float, 
                               coverage_points: List[Tuple[float, float]],
                               target_coverage_db: float, power_tx: float) -> np.ndarray:
        """
        Calcule le masque de couverture pour un point d'accès à une position donnée.
        
        Args:
            ap_x, ap_y: Position du point d'accès
            coverage_points: Points à vérifier
            target_coverage_db: Niveau minimum requis
            power_tx: Puissance de transmission
            
        Returns:
            Masque booléen indiquant les points couverts
        """
        coverage_mask = np.zeros(len(coverage_points), dtype=bool)
        
        for i, (point_x, point_y) in enumerate(coverage_points):
            # Calculer la distance
            distance = np.sqrt((ap_x - point_x)**2 + (ap_y - point_y)**2)
            
            # Calculer le pathloss (version simplifiée)
            if distance < 0.1:  # Éviter division par zéro
                distance = 0.1
            
            # Formule de Friis simplifiée avec atténuation
            pathloss_db = 20 * np.log10(distance) + 20 * np.log10(self.frequency) - 147.55
            
            # Ajouter de l'atténuation pour les murs (estimation simple)
            wall_attenuation = min(10.0, distance * 0.5)  # 0.5 dB par mètre
            pathloss_db += wall_attenuation
            
            # Signal reçu
            received_power = power_tx - pathloss_db
            
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
