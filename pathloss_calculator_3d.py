import numpy as np
import math
from ml_pathloss_predictor_3d import ml_predictor_3d

class PathlossCalculator3D:
    def __init__(self, frequency_mhz):
        """
        Initialise le calculateur de pathloss 3D avec prédiction ML.
        
        Args:
            frequency_mhz: Fréquence en MHz
        """
        self.frequency_mhz = frequency_mhz
        self.frequency_hz = frequency_mhz * 1e6
        
        # Constantes
        self.c = 3e8  # Vitesse de la lumière (m/s)
        
        # Paramètres d'atténuation pour différents matériaux (fallback)
        self.wall_attenuation_db = {
            'concrete': 12,      # Béton
            'brick': 8,          # Brique
            'drywall': 3,        # Cloison sèche
            'wood': 4,           # Bois
            'glass': 2,          # Verre
            'default': 6         # Valeur par défaut
        }
        
        # Atténuation par étage (plancher/plafond)
        self.floor_attenuation_db = 15  # dB par étage traversé
        
        # Facteurs d'environnement 3D
        self.environment_factors = {
            'residential': 1.0,   # Résidentiel
            'office': 1.2,       # Bureau
            'industrial': 1.5,   # Industriel
            'outdoor': 0.8       # Extérieur
        }
        
        # Indicateur pour utiliser ML ou calcul théorique
        self.use_ml_prediction = True
    
    def calculate_free_space_pathloss_3d(self, distance_3d):
        """
        Calcule le pathloss en espace libre pour une distance 3D.
        
        Args:
            distance_3d: Distance 3D en mètres
            
        Returns:
            pathloss_db: Pathloss en dB
        """
        if distance_3d <= 0:
            return 0
        
        # Formule de Friis pour l'espace libre
        wavelength = self.c / self.frequency_hz
        pathloss_db = 20 * math.log10(4 * math.pi * distance_3d / wavelength)
        
        return pathloss_db
    
    def calculate_pathloss_3d(self, distance_3d, wall_count_2d, floor_difference, 
                             wall_type='default', environment='residential'):
        """
        Calcule le pathloss total en 3D en utilisant la prédiction ML (priorité) ou calcul théorique.
        
        Args:
            distance_3d: Distance 3D en mètres
            wall_count_2d: Nombre de murs traversés dans le plan 2D
            floor_difference: Différence d'étages entre émetteur et récepteur
            wall_type: Type de mur (utilisé en fallback)
            environment: Type d'environnement (utilisé en fallback)
            
        Returns:
            total_pathloss_db: Pathloss total en dB
        """
        # Prioriser la prédiction ML si disponible
        if self.use_ml_prediction and ml_predictor_3d.model_loaded:
            try:
                # Utiliser le modèle ML 3D
                ml_pathloss = ml_predictor_3d.predict_pathloss_3d(
                    distance=distance_3d,
                    num_walls=wall_count_2d,
                    floor_difference=floor_difference,
                    frequency=self.frequency_mhz
                )
                return ml_pathloss
            except Exception as e:
                print(f"⚠️ Erreur prédiction ML 3D, fallback théorique: {e}")
        
        # Fallback vers calcul théorique
        return self._calculate_theoretical_pathloss_3d(distance_3d, wall_count_2d, floor_difference, wall_type, environment)
    
    def _calculate_theoretical_pathloss_3d(self, distance_3d, wall_count_2d, floor_difference, 
                                          wall_type='default', environment='residential'):
        """
        Calcul théorique du pathloss 3D (méthode originale).
        """
        # Pathloss en espace libre
        free_space_loss = self.calculate_free_space_pathloss_3d(distance_3d)
        
        # Atténuation due aux murs
        wall_loss = wall_count_2d * self.wall_attenuation_db.get(wall_type, 
                                                                 self.wall_attenuation_db['default'])
        
        # Atténuation due à la traversée d'étages
        floor_loss = floor_difference * self.floor_attenuation_db
        
        # Facteur d'environnement
        env_factor = self.environment_factors.get(environment, 1.0)
        
        # Pathloss total avec facteur d'environnement
        total_pathloss = (free_space_loss + wall_loss + floor_loss) * env_factor
        
        return total_pathloss
    
    def calculate_path_components_3d(self, point1_3d, point2_3d, walls_2d, 
                                   longueur, largeur, hauteur_etage):
        """
        Calcule les composantes détaillées du trajet 3D.
        
        Args:
            point1_3d: (x1, y1, z1) point émetteur
            point2_3d: (x2, y2, z2) point récepteur
            walls_2d: Masque binaire des murs
            longueur, largeur: Dimensions du bâtiment
            hauteur_etage: Hauteur d'un étage
            
        Returns:
            components: Dictionnaire avec les composantes du trajet
        """
        x1, y1, z1 = point1_3d
        x2, y2, z2 = point2_3d
        
        # Distances
        distance_2d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distance_3d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        distance_vertical = abs(z2 - z1)
        
        # Angles
        elevation_angle = math.degrees(math.atan2(distance_vertical, distance_2d))
        azimuth_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        
        # Étages
        etage1 = int(z1 // hauteur_etage)
        etage2 = int(z2 // hauteur_etage)
        floor_difference = abs(etage2 - etage1)
        
        # Analyse du trajet
        components = {
            'distance_2d': distance_2d,
            'distance_3d': distance_3d,
            'distance_vertical': distance_vertical,
            'elevation_angle': elevation_angle,
            'azimuth_angle': azimuth_angle,
            'etage_emetteur': etage1 + 1,
            'etage_recepteur': etage2 + 1,
            'difference_etages': floor_difference,
            'type_propagation': self._determine_propagation_type(elevation_angle, floor_difference)
        }
        
        return components
    
    def _determine_propagation_type(self, elevation_angle, floor_difference):
        """
        Détermine le type de propagation basé sur l'angle et les étages.
        """
        if floor_difference == 0:
            if abs(elevation_angle) < 5:
                return "Horizontale"
            else:
                return "Inclinée intra-étage"
        else:
            if abs(elevation_angle) > 60:
                return "Principalement verticale"
            elif abs(elevation_angle) > 30:
                return "Oblique inter-étages"
            else:
                return "Horizontale inter-étages"
    
    def calculate_multipath_effects_3d(self, point1_3d, point2_3d, walls_2d, 
                                     longueur, largeur, nb_etages, hauteur_etage):
        """
        Calcule les effets de propagation multi-trajets en 3D.
        
        Args:
            point1_3d, point2_3d: Points émetteur et récepteur
            walls_2d: Masque des murs
            longueur, largeur: Dimensions du bâtiment
            nb_etages: Nombre d'étages
            hauteur_etage: Hauteur d'un étage
            
        Returns:
            multipath_analysis: Analyse des trajets multiples
        """
        x1, y1, z1 = point1_3d
        x2, y2, z2 = point2_3d
        
        # Trajet direct
        direct_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        direct_pathloss = self.calculate_free_space_pathloss_3d(direct_distance)
        
        # Trajets réfléchis possibles
        reflection_paths = []
        
        # Réflexion sur les murs (simplifiée)
        wall_positions = [0, longueur]  # Murs gauche et droit
        for wall_x in wall_positions:
            # Point de réflexion sur le mur
            x_refl = wall_x
            y_refl = (y1 + y2) / 2  # Approximation
            z_refl = (z1 + z2) / 2
            
            # Distance du trajet réfléchi
            dist1 = math.sqrt((x_refl - x1)**2 + (y_refl - y1)**2 + (z_refl - z1)**2)
            dist2 = math.sqrt((x2 - x_refl)**2 + (y2 - y_refl)**2 + (z2 - z_refl)**2)
            total_refl_dist = dist1 + dist2
            
            reflection_paths.append({
                'type': 'mur_vertical',
                'distance': total_refl_dist,
                'pathloss': self.calculate_free_space_pathloss_3d(total_refl_dist),
                'delay': (total_refl_dist - direct_distance) / self.c * 1e9  # ns
            })
        
        # Réflexion sur les planchers/plafonds
        for etage in range(nb_etages + 1):
            z_refl = etage * hauteur_etage
            
            # Distance du trajet réfléchi sur le plancher/plafond
            dist1 = math.sqrt((x1 - x1)**2 + (y1 - y1)**2 + (z_refl - z1)**2)
            dist2 = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z_refl)**2)
            total_refl_dist = dist1 + dist2
            
            if total_refl_dist > direct_distance:  # Éviter les trajets impossibles
                reflection_paths.append({
                    'type': f'plancher_etage_{etage}',
                    'distance': total_refl_dist,
                    'pathloss': self.calculate_free_space_pathloss_3d(total_refl_dist),
                    'delay': (total_refl_dist - direct_distance) / self.c * 1e9
                })
        
        multipath_analysis = {
            'direct_path': {
                'distance': direct_distance,
                'pathloss': direct_pathloss,
                'delay': 0
            },
            'reflection_paths': sorted(reflection_paths, key=lambda x: x['distance'])[:5],  # Top 5
            'rms_delay_spread': self._calculate_rms_delay_spread(reflection_paths, direct_pathloss)
        }
        
        return multipath_analysis
    
    def _calculate_rms_delay_spread(self, reflection_paths, direct_pathloss):
        """
        Calcule l'étalement temporel RMS des trajets multiples.
        """
        if not reflection_paths:
            return 0
        
        # Calcul simplifié de l'étalement RMS
        delays = [0] + [path['delay'] for path in reflection_paths[:3]]  # Direct + 3 réflexions
        powers = [direct_pathloss] + [path['pathloss'] for path in reflection_paths[:3]]
        
        # Conversion en puissances linéaires
        linear_powers = [10**(-p/10) for p in powers]
        total_power = sum(linear_powers)
        
        # Moyenne pondérée des délais
        mean_delay = sum(d * p for d, p in zip(delays, linear_powers)) / total_power
        
        # Variance des délais
        delay_variance = sum(p * (d - mean_delay)**2 for d, p in zip(delays, linear_powers)) / total_power
        
        return math.sqrt(delay_variance)
    
    def estimate_channel_quality_3d(self, pathloss_db, multipath_analysis=None):
        """
        Estime la qualité du canal 3D.
        
        Args:
            pathloss_db: Pathloss en dB
            multipath_analysis: Analyse des trajets multiples (optionnel)
            
        Returns:
            quality_metrics: Métriques de qualité du canal
        """
        # Classification basique du signal
        if pathloss_db < 50:
            signal_strength = "Excellent"
            quality_score = 5
        elif pathloss_db < 70:
            signal_strength = "Très bon"
            quality_score = 4
        elif pathloss_db < 90:
            signal_strength = "Bon"
            quality_score = 3
        elif pathloss_db < 110:
            signal_strength = "Faible"
            quality_score = 2
        else:
            signal_strength = "Très faible"
            quality_score = 1
        
        # Facteurs de dégradation
        degradation_factors = []
        
        if multipath_analysis:
            rms_delay = multipath_analysis['rms_delay_spread']
            if rms_delay > 50:  # ns
                degradation_factors.append("Étalement temporel élevé")
                quality_score = max(1, quality_score - 1)
            
            if len(multipath_analysis['reflection_paths']) > 3:
                degradation_factors.append("Environnement très réfléchissant")
        
        # Recommandations basées sur la fréquence
        recommendations = []
        if self.frequency_mhz > 5000:  # 5 GHz+
            recommendations.append("Fréquence élevée : portée limitée mais moins d'interférences")
        elif self.frequency_mhz < 1000:  # < 1 GHz
            recommendations.append("Fréquence basse : bonne pénétration mais risque d'interférences")
        
        quality_metrics = {
            'pathloss_db': pathloss_db,
            'signal_strength': signal_strength,
            'quality_score': quality_score,
            'degradation_factors': degradation_factors,
            'recommendations': recommendations,
            'frequency_band': self._get_frequency_band()
        }
        
        return quality_metrics
    
    def _get_frequency_band(self):
        """
        Retourne la bande de fréquence.
        """
        if self.frequency_mhz < 300:
            return "HF/VHF"
        elif self.frequency_mhz < 1000:
            return "UHF"
        elif self.frequency_mhz < 3000:
            return "SHF (2.4 GHz)"
        elif self.frequency_mhz < 6000:
            return "SHF (5 GHz)"
        else:
            return "SHF/EHF"
