import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import measure
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self):
        self.wall_thickness_threshold = 5  # pixels
    
    def process_image(self, image_array):
        """
        Traite l'image pour extraire les murs.
        
        Args:
            image_array: Image numpy array
            
        Returns:
            processed_image: Image avec murs détectés
            walls_binary: Masque binaire des murs
        """
        # Conversion en niveaux de gris si nécessaire
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array.copy()
        
        # Binarisation: seuillage pour séparer murs (noir) et espaces libres (blanc)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Inversion pour avoir les murs en blanc et l'espace libre en noir
        walls_binary = cv2.bitwise_not(binary)
        
        # Nettoyage du bruit avec des opérations morphologiques
        kernel = np.ones((3,3), np.uint8)
        walls_cleaned = cv2.morphologyEx(walls_binary, cv2.MORPH_CLOSE, kernel)
        walls_cleaned = cv2.morphologyEx(walls_cleaned, cv2.MORPH_OPEN, kernel)
        
        # Détection intelligente des murs
        walls_detected = self._detect_walls_intelligent(walls_cleaned)
        
        # Création de l'image de visualisation
        processed_image = self._create_visualization(image_array, walls_detected)
        
        return processed_image, walls_detected
    
    def _detect_walls_intelligent(self, binary_image):
        """
        Détection intelligente des murs basée sur la connectivité et l'épaisseur.
        """
        # Copie de l'image binaire
        walls = binary_image.copy()
        
        # Détection des contours pour analyser les composants connectés
        contours, _ = cv2.findContours(walls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Création d'un masque pour les murs valides
        valid_walls = np.zeros_like(walls)
        
        for contour in contours:
            # Calcul de l'aire et du périmètre
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Filtrage basé sur la taille (éliminer les petits artefacts)
            if area > 50:  # Seuil minimum pour considérer comme un mur
                # Calcul du ratio aire/périmètre pour identifier les structures linéaires
                if perimeter > 0:
                    compactness = (perimeter * perimeter) / (4 * np.pi * area)
                    
                    # Les murs ont généralement une compacité élevée (forme allongée)
                    if compactness > 2:  # Seuil pour structures allongées
                        cv2.fillPoly(valid_walls, [contour], 255)
                    else:
                        # Pour les structures plus compactes, vérifier l'épaisseur
                        thickness = self._estimate_thickness(contour)
                        if thickness < 50:  # Seuil d'épaisseur maximum pour un mur
                            cv2.fillPoly(valid_walls, [contour], 255)
        
        # Application d'un filtre médian pour lisser
        valid_walls = cv2.medianBlur(valid_walls, 3)
        
        return valid_walls
    
    def _estimate_thickness(self, contour):
        """
        Estime l'épaisseur d'un contour en calculant la distance minimale
        entre les points opposés.
        """
        if len(contour) < 4:
            return 0
        
        # Calcul de la boîte englobante
        x, y, w, h = cv2.boundingRect(contour)
        
        # L'épaisseur est approximativement le minimum entre largeur et hauteur
        return min(w, h)
    
    def _create_visualization(self, original_image, walls_detected):
        """
        Crée une image de visualisation avec les murs surlignés.
        """
        # Copie de l'image originale
        if len(original_image.shape) == 3:
            viz_image = original_image.copy()
        else:
            viz_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Surlignage des murs détectés en rouge
        red_overlay = np.zeros_like(viz_image)
        red_overlay[:,:,0] = walls_detected  # Canal rouge
        
        # Mélange avec l'image originale
        alpha = 0.3
        result = cv2.addWeighted(viz_image, 1-alpha, red_overlay, alpha, 0)
        
        return result
    
    def count_walls_between_points(self, walls_binary, point1, point2):
        """
        Compte le nombre de murs entre deux points en traçant une ligne droite.
        
        Args:
            walls_binary: Masque binaire des murs
            point1: (x1, y1) point de départ
            point2: (x2, y2) point d'arrivée
            
        Returns:
            wall_count: Nombre de murs traversés
        """
        x1, y1 = point1
        x2, y2 = point2
        
        # Création d'une ligne entre les deux points
        line_points = self._get_line_points(x1, y1, x2, y2)
        
        # Vérification de chaque point sur la ligne
        wall_crossings = []
        in_wall = False
        
        for x, y in line_points:
            if 0 <= x < walls_binary.shape[1] and 0 <= y < walls_binary.shape[0]:
                pixel_value = walls_binary[y, x]
                
                if pixel_value > 0 and not in_wall:
                    # Entrée dans un mur
                    wall_crossings.append(1)
                    in_wall = True
                elif pixel_value == 0 and in_wall:
                    # Sortie d'un mur
                    in_wall = False
        
        return len(wall_crossings)
    
    def _get_line_points(self, x1, y1, x2, y2):
        """
        Génère les points d'une ligne entre deux points (algorithme de Bresenham).
        """
        points = []
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        
        err = dx - dy
        
        x, y = x1, y1
        
        while True:
            points.append((int(x), int(y)))
            
            if x == x2 and y == y2:
                break
                
            e2 = 2 * err
            
            if e2 > -dy:
                err -= dy
                x += sx
                
            if e2 < dx:
                err += dx
                y += sy
        
        return points
    
    def analyze_wall_structure(self, walls_binary):
        """
        Analyse la structure des murs pour des statistiques détaillées.
        """
        # Calcul des statistiques
        total_wall_pixels = np.sum(walls_binary > 0)
        total_pixels = walls_binary.shape[0] * walls_binary.shape[1]
        wall_ratio = total_wall_pixels / total_pixels
        
        # Détection des composants connectés
        num_labels, labels = cv2.connectedComponents(walls_binary)
        
        # Calcul des tailles des composants
        component_sizes = []
        for i in range(1, num_labels):
            size = np.sum(labels == i)
            component_sizes.append(size)
        
        stats = {
            'total_wall_pixels': total_wall_pixels,
            'wall_ratio': wall_ratio,
            'num_components': num_labels - 1,
            'component_sizes': component_sizes,
            'avg_component_size': np.mean(component_sizes) if component_sizes else 0
        }
        
        return stats
