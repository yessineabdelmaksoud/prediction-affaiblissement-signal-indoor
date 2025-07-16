"""
Module de prédiction de pathloss 3D utilisant le modèle de machine learning pré-entraîné.
Remplace les calculs théoriques par des prédictions ML.
"""

import pickle
import pandas as pd
import numpy as np
import os

class MLPathlossPredictor3D:
    """
    Prédicteur de pathloss 3D utilisant un modèle ML pré-entraîné.
    """
    
    def __init__(self):
        self.model = None
        self.model_info = None
        self.model_loaded = False
        self.model_path = 'xgboost_radio_propagation_model.pkl'
        
        # Essayer de charger le modèle automatiquement
        self.load_model()
    
    def load_model(self):
        """
        Charge le modèle de prédiction 3D.
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
                    with open(path, 'rb') as f:
                        loaded_data = pickle.load(f)
                    
                    # Vérifier si c'est un dictionnaire model_info ou juste le modèle
                    if isinstance(loaded_data, dict) and 'model' in loaded_data:
                        self.model_info = loaded_data
                        self.model = loaded_data['model']
                        print(f"✓ Modèle 3D chargé (structure complète) depuis: {path}")
                    else:
                        self.model = loaded_data
                        print(f"✓ Modèle 3D chargé (modèle seul) depuis: {path}")
                    
                    self.model_loaded = True
                    return True
            
            print(f"❌ Modèle 3D non trouvé dans: {possible_paths}")
            return False
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle 3D: {e}")
            self.model_loaded = False
            return False
    
    def predict_pathloss_3d(self, distance, num_walls, floor_difference, frequency):
        """
        Prédit le pathloss en 3D en utilisant le modèle ML.
        
        Args:
            distance (float): Distance 3D en mètres
            num_walls (int): Nombre de murs traversés
            floor_difference (int): Différence d'étages
            frequency (float): Fréquence en MHz
            
        Returns:
            float: Pathloss prédit en dB
        """
        if not self.model_loaded:
            # Fallback vers calcul théorique si le modèle n'est pas disponible
            return self._calculate_theoretical_pathloss_3d(distance, num_walls, floor_difference, frequency)
        
        try:
            # Préparer les données d'entrée (ordre selon l'entraînement)
            features = pd.DataFrame({
                'distance': [distance],
                'numwall': [num_walls],
                'etage': [floor_difference],
                'frequence': [frequency]
            })
            
            # Prédiction
            pathloss = self.model.predict(features)[0]
            
            # Valider la prédiction
            if np.isnan(pathloss) or pathloss < 0:
                print(f"⚠️ Prédiction 3D invalide: {pathloss}, utilisation du calcul théorique")
                return self._calculate_theoretical_pathloss_3d(distance, num_walls, floor_difference, frequency)
            
            return float(pathloss)
            
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction 3D: {e}")
            # Fallback vers calcul théorique
            return self._calculate_theoretical_pathloss_3d(distance, num_walls, floor_difference, frequency)
    
    def _calculate_theoretical_pathloss_3d(self, distance, num_walls, floor_difference, frequency):
        """
        Calcul théorique de fallback si le modèle ML n'est pas disponible.
        """
        # Éviter division par zéro
        if distance <= 0:
            distance = 0.1
        
        # Formule de Friis modifiée avec pertes 3D
        frequency_ghz = frequency / 1000.0
        
        # Pathloss en espace libre
        fspl = 20 * np.log10(distance) + 20 * np.log10(frequency_ghz) + 32.44
        
        # Ajout des pertes dues aux murs
        wall_loss_per_wall = 3.0 + (frequency_ghz - 1.0) * 1.5
        wall_loss = num_walls * wall_loss_per_wall
        
        # Ajout des pertes dues aux étages (plus importantes)
        floor_loss_per_floor = 15.0 + (frequency_ghz - 1.0) * 3.0
        floor_loss = floor_difference * floor_loss_per_floor
        
        total_pathloss = fspl + wall_loss + floor_loss
        
        return total_pathloss
    
    def predict_multiple_3d(self, distances, num_walls_list, floor_differences, frequencies):
        """
        Prédit le pathloss 3D pour plusieurs points simultanément.
        
        Args:
            distances (list): Liste des distances 3D
            num_walls_list (list): Liste du nombre de murs
            floor_differences (list): Liste des différences d'étages
            frequencies (list): Liste des fréquences
            
        Returns:
            list: Liste des pathloss prédits
        """
        if not self.model_loaded:
            return [self._calculate_theoretical_pathloss_3d(d, w, f, freq) 
                   for d, w, f, freq in zip(distances, num_walls_list, floor_differences, frequencies)]
        
        try:
            # Préparer les données
            features = pd.DataFrame({
                'distance': distances,
                'numwall': num_walls_list,
                'etage': floor_differences,
                'frequence': frequencies
            })
            
            # Prédictions
            predictions = self.model.predict(features)
            
            # Valider et nettoyer les prédictions
            cleaned_predictions = []
            for i, pred in enumerate(predictions):
                if np.isnan(pred) or pred < 0:
                    # Fallback pour cette prédiction
                    cleaned_predictions.append(
                        self._calculate_theoretical_pathloss_3d(
                            distances[i], num_walls_list[i], floor_differences[i], frequencies[i]
                        )
                    )
                else:
                    cleaned_predictions.append(float(pred))
            
            return cleaned_predictions
            
        except Exception as e:
            print(f"❌ Erreur lors des prédictions multiples 3D: {e}")
            # Fallback complet
            return [self._calculate_theoretical_pathloss_3d(d, w, f, freq) 
                   for d, w, f, freq in zip(distances, num_walls_list, floor_differences, frequencies)]
    
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
                "features_required": ["distance", "numwall", "etage", "frequence"]
            }
            
            # Si nous avons les métadonnées complètes
            if self.model_info and isinstance(self.model_info, dict):
                if 'metrics' in self.model_info:
                    info['metrics'] = self.model_info['metrics']
                if 'feature_names' in self.model_info:
                    info['feature_names'] = self.model_info['feature_names']
            
            # Essayer d'obtenir les paramètres
            if hasattr(self.model, 'get_params'):
                params = self.model.get_params()
                important_params = ['n_estimators', 'max_depth', 'learning_rate', 'subsample']
                info['params'] = {k: v for k, v in params.items() if k in important_params}
            
            return info
        except:
            return {"status": "chargé", "model_type": "inconnu"}

# Instance globale pour utilisation dans l'application
ml_predictor_3d = MLPathlossPredictor3D()
