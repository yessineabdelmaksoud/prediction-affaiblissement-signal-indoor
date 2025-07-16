import numpy as np
import math
from ml_pathloss_predictor_2d import ml_predictor_2d

class PathlossCalculator:
    def __init__(self, frequency_mhz):
        """
        Initialise le calculateur de pathloss avec prédiction ML.
        
        Args:
            frequency_mhz: Fréquence en MHz
        """
        self.frequency_mhz = frequency_mhz
        self.frequency_hz = frequency_mhz * 1e6
        
        # Constantes pour le calcul de fallback
        self.c = 3e8  # Vitesse de la lumière (m/s)
        
        # Paramètres typiques pour les matériaux de construction (fallback)
        self.wall_attenuation_db = {
            'concrete': 12,      # Béton
            'brick': 8,          # Brique
            'drywall': 3,        # Cloison sèche
            'wood': 4,           # Bois
            'glass': 2,          # Verre
            'default': 6         # Valeur par défaut pour mur standard
        }
        
        self.floor_attenuation_db = 15  # Atténuation par étage
        
        # Indicateur pour utiliser ML ou calcul théorique
        self.use_ml_prediction = True
    
    def calculate_free_space_pathloss(self, distance_m):
        """
        Calcule le pathloss en espace libre.
        
        Formula: PL = 20*log10(4*π*d*f/c)
        
        Args:
            distance_m: Distance en mètres
            
        Returns:
            pathloss_db: Pathloss en dB
        """
        if distance_m <= 0:
            return 0
        
        # Formule de Friis
        wavelength = self.c / self.frequency_hz
        pathloss_db = 20 * math.log10(4 * math.pi * distance_m / wavelength)
        
        return pathloss_db
    
    def calculate_pathloss(self, distance_2d, wall_count, floor_count=0, wall_type='default'):
        """
        Calcule le pathloss total en utilisant la prédiction ML (priorité) ou calcul théorique.
        
        Args:
            distance_2d: Distance 2D en mètres
            wall_count: Nombre de murs traversés
            floor_count: Nombre d'étages traversés (pour compatibilité, mais utilisé par 3D)
            wall_type: Type de mur pour l'atténuation (utilisé en fallback)
            
        Returns:
            total_pathloss_db: Pathloss total en dB
        """
        # Prioriser la prédiction ML si disponible
        if self.use_ml_prediction and ml_predictor_2d.model_loaded:
            try:
                # Utiliser le modèle ML 2D
                ml_pathloss = ml_predictor_2d.predict_pathloss(
                    distance=distance_2d,
                    num_walls=wall_count,
                    frequency=self.frequency_mhz
                )
                return ml_pathloss
            except Exception as e:
                print(f"⚠️ Erreur prédiction ML 2D, fallback théorique: {e}")
        
        # Fallback vers calcul théorique
        return self._calculate_theoretical_pathloss(distance_2d, wall_count, floor_count, wall_type)
    
    def _calculate_theoretical_pathloss(self, distance_2d, wall_count, floor_count=0, wall_type='default'):
        """
        Calcul théorique du pathloss (méthode originale).
        """
        # Pathloss en espace libre
        free_space_loss = self.calculate_free_space_pathloss(distance_2d)
        
        # Atténuation due aux murs
        wall_loss = wall_count * self.wall_attenuation_db.get(wall_type, 
                                                              self.wall_attenuation_db['default'])
        
        # Atténuation due aux étages
        floor_loss = floor_count * self.floor_attenuation_db
        
        # Pathloss total
        total_pathloss = free_space_loss + wall_loss + floor_loss
        
        return total_pathloss
    
    def calculate_received_power(self, transmitted_power_dbm, pathloss_db):
        """
        Calcule la puissance reçue.
        
        Args:
            transmitted_power_dbm: Puissance transmise en dBm
            pathloss_db: Pathloss en dB
            
        Returns:
            received_power_dbm: Puissance reçue en dBm
        """
        return transmitted_power_dbm - pathloss_db
    
    def estimate_signal_quality(self, received_power_dbm):
        """
        Estime la qualité du signal basée sur la puissance reçue.
        
        Args:
            received_power_dbm: Puissance reçue en dBm
            
        Returns:
            quality: Dictionnaire avec qualité et description
        """
        if received_power_dbm >= -30:
            return {'level': 'Excellent', 'description': 'Signal très fort', 'color': 'green'}
        elif received_power_dbm >= -50:
            return {'level': 'Très bon', 'description': 'Signal fort', 'color': 'lightgreen'}
        elif received_power_dbm >= -70:
            return {'level': 'Bon', 'description': 'Signal correct', 'color': 'yellow'}
        elif received_power_dbm >= -85:
            return {'level': 'Faible', 'description': 'Signal faible mais utilisable', 'color': 'orange'}
        else:
            return {'level': 'Très faible', 'description': 'Signal très faible ou inutilisable', 'color': 'red'}
    
    def calculate_multiple_models(self, distance_2d, wall_count):
        """
        Calcule le pathloss avec différents modèles pour comparaison.
        
        Args:
            distance_2d: Distance 2D en mètres
            wall_count: Nombre de murs
            
        Returns:
            results: Dictionnaire avec différents modèles
        """
        results = {}
        
        # Modèle espace libre
        results['free_space'] = self.calculate_free_space_pathloss(distance_2d)
        
        # Modèle avec murs standards
        results['with_walls'] = self.calculate_pathloss(distance_2d, wall_count)
        
        # Modèle pessimiste (murs en béton)
        results['pessimistic'] = self.calculate_pathloss(distance_2d, wall_count, wall_type='concrete')
        
        # Modèle optimiste (cloisons sèches)
        results['optimistic'] = self.calculate_pathloss(distance_2d, wall_count, wall_type='drywall')
        
        return results
    
    def calculate_coverage_radius(self, max_acceptable_pathloss_db, wall_density=0):
        """
        Calcule le rayon de couverture approximatif.
        
        Args:
            max_acceptable_pathloss_db: Pathloss maximum acceptable
            wall_density: Densité moyenne de murs par mètre
            
        Returns:
            radius_m: Rayon de couverture en mètres
        """
        # Estimation basée sur l'espace libre ajustée pour les murs
        wavelength = self.c / self.frequency_hz
        
        # Pathloss disponible pour la distance (en retirant l'effet des murs)
        distance_pathloss = max_acceptable_pathloss_db - (wall_density * self.wall_attenuation_db['default'])
        
        if distance_pathloss <= 0:
            return 0
        
        # Calcul de la distance maximale
        distance = wavelength * (10 ** (distance_pathloss / 20)) / (4 * math.pi)
        
        return distance
    
    def get_frequency_characteristics(self):
        """
        Retourne les caractéristiques de la fréquence utilisée.
        """
        wavelength = self.c / self.frequency_hz
        
        if self.frequency_mhz < 1000:
            band = "UHF"
        elif self.frequency_mhz < 3000:
            band = "SHF (2.4 GHz)"
        elif self.frequency_mhz < 6000:
            band = "SHF (5 GHz)"
        else:
            band = "SHF/EHF"
        
        return {
            'frequency_mhz': self.frequency_mhz,
            'wavelength_m': wavelength,
            'wavelength_cm': wavelength * 100,
            'band': band
        }
