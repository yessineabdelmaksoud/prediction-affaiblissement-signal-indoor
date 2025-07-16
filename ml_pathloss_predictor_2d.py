"""
Module de prédiction de pathloss 2D utilisant le modèle de machine learning pré-entraîné.
Remplace les calculs théoriques par des prédictions ML.
"""

import joblib
import pandas as pd
import numpy as np
import os

class MLPathlossPredictor2D:
    """
    Prédicteur de pathloss 2D utilisant un modèle ML pré-entraîné.
    """
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.model_path = 'pathloss_predictor.pkl'
        
        # Essayer de charger le modèle automatiquement
        self.load_model()
    
    def load_model(self):
        """
        Charge le modèle de prédiction 2D.
        """
        try:
            # Chercher le modèle dans le répertoire courant et test-model
            possible_paths = [
                self.model_path,
                os.path.join('test-model', self.model_path),
                os.path.join('..', 'test-model', self.model_path)
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.model = joblib.load(path)
                    self.model_loaded = True
                    print(f"✓ Modèle 2D chargé depuis: {path}")
                    return True
            
            print(f"❌ Modèle 2D non trouvé dans: {possible_paths}")
            return False
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle 2D: {e}")
            self.model_loaded = False
            return False
    
    def predict_pathloss(self, distance, num_walls, frequency):
        """
        Prédit le pathloss en 2D en utilisant le modèle ML.
        
        Args:
            distance (float): Distance en mètres
            num_walls (int): Nombre de murs traversés
            frequency (float): Fréquence en MHz
            
        Returns:
            float: Pathloss prédit en dB
        """
        if not self.model_loaded:
            # Fallback vers calcul théorique si le modèle n'est pas disponible
            return self._calculate_theoretical_pathloss_2d(distance, num_walls, frequency)
        
        try:
            # Préparer les données d'entrée
            features = pd.DataFrame({
                'num_walls': [num_walls],
                'distance': [distance],
                'frequency': [frequency]
            })
            
            # Prédiction
            pathloss = self.model.predict(features)[0]
            
            # Valider la prédiction
            if np.isnan(pathloss) or pathloss < 0:
                print(f"⚠️ Prédiction invalide: {pathloss}, utilisation du calcul théorique")
                return self._calculate_theoretical_pathloss_2d(distance, num_walls, frequency)
            
            return float(pathloss)
            
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction 2D: {e}")
            # Fallback vers calcul théorique
            return self._calculate_theoretical_pathloss_2d(distance, num_walls, frequency)
    
    def _calculate_theoretical_pathloss_2d(self, distance, num_walls, frequency):
        """
        Calcul théorique de fallback si le modèle ML n'est pas disponible.
        """
        # Éviter division par zéro
        if distance <= 0:
            distance = 0.1
        
        # Formule de Friis modifiée avec pertes dues aux murs
        frequency_ghz = frequency / 1000.0
        
        # Pathloss en espace libre
        fspl = 20 * np.log10(distance) + 20 * np.log10(frequency_ghz) + 32.44
        
        # Ajout des pertes dues aux murs (3-8 dB par mur selon la fréquence)
        wall_loss_per_wall = 3.0 + (frequency_ghz - 1.0) * 1.5
        wall_loss = num_walls * wall_loss_per_wall
        
        total_pathloss = fspl + wall_loss
        
        return total_pathloss
    
    def predict_multiple(self, distances, num_walls_list, frequencies):
        """
        Prédit le pathloss pour plusieurs points simultanément.
        
        Args:
            distances (list): Liste des distances
            num_walls_list (list): Liste du nombre de murs
            frequencies (list): Liste des fréquences
            
        Returns:
            list: Liste des pathloss prédits
        """
        if not self.model_loaded:
            return [self._calculate_theoretical_pathloss_2d(d, w, f) 
                   for d, w, f in zip(distances, num_walls_list, frequencies)]
        
        try:
            # Préparer les données
            features = pd.DataFrame({
                'num_walls': num_walls_list,
                'distance': distances,
                'frequency': frequencies
            })
            
            # Prédictions
            predictions = self.model.predict(features)
            
            # Valider et nettoyer les prédictions
            cleaned_predictions = []
            for i, pred in enumerate(predictions):
                if np.isnan(pred) or pred < 0:
                    # Fallback pour cette prédiction
                    cleaned_predictions.append(
                        self._calculate_theoretical_pathloss_2d(
                            distances[i], num_walls_list[i], frequencies[i]
                        )
                    )
                else:
                    cleaned_predictions.append(float(pred))
            
            return cleaned_predictions
            
        except Exception as e:
            print(f"❌ Erreur lors des prédictions multiples 2D: {e}")
            # Fallback complet
            return [self._calculate_theoretical_pathloss_2d(d, w, f) 
                   for d, w, f in zip(distances, num_walls_list, frequencies)]
    
    def get_model_info(self):
        """
        Retourne les informations sur le modèle chargé.
        """
        if not self.model_loaded:
            return {"status": "non_chargé", "model_type": "fallback_théorique"}
        
        try:
            info = {
                "status": "chargé",
                "model_type": type(self.model).__name__,
                "features_required": ["num_walls", "distance", "frequency"]
            }
            
            # Essayer d'obtenir les paramètres
            if hasattr(self.model, 'get_params'):
                params = self.model.get_params()
                important_params = ['n_estimators', 'max_depth', 'learning_rate']
                info['params'] = {k: v for k, v in params.items() if k in important_params}
            
            return info
        except:
            return {"status": "chargé", "model_type": "inconnu"}

# Instance globale pour utilisation dans l'application
ml_predictor_2d = MLPathlossPredictor2D()
