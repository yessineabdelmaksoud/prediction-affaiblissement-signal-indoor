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
        
        # Amélioration du preprocessing pour capturer tous les murs
        # Application d'un filtre pour réduire le bruit tout en préservant les structures
        kernel_small = np.ones((2,2), np.uint8)
        walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kernel_small)
        
        # Détection des contours pour analyser les composants connectés
        contours, _ = cv2.findContours(walls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Création d'un masque pour les murs valides
        valid_walls = np.zeros_like(walls)
        
        # Paramètres plus permissifs pour capturer tous les murs
        min_area = 20  # Seuil minimum réduit
        min_perimeter = 10  # Seuil minimum de périmètre
        
        for contour in contours:
            # Calcul de l'aire et du périmètre
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Filtrage basé sur la taille (éliminer seulement les très petits artefacts)
            if area > min_area and perimeter > min_perimeter:
                # Calcul du ratio aire/périmètre pour identifier les structures
                if perimeter > 0:
                    compactness = (perimeter * perimeter) / (4 * np.pi * area)
                    
                    # Critères plus permissifs pour inclure plus de structures
                    if compactness > 1.5:  # Seuil réduit pour structures allongées
                        cv2.fillPoly(valid_walls, [contour], 255)
                    else:
                        # Pour les structures plus compactes, vérifier l'épaisseur
                        thickness = self._estimate_thickness(contour)
                        if thickness < 100:  # Seuil d'épaisseur augmenté
                            cv2.fillPoly(valid_walls, [contour], 255)
                else:
                    # Si le périmètre est 0, inclure quand même si l'aire est suffisante
                    cv2.fillPoly(valid_walls, [contour], 255)
        
        # Détection des contours internes aussi (pour les murs avec des trous)
        contours_internal, _ = cv2.findContours(walls, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_internal:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.fillPoly(valid_walls, [contour], 255)
        
        # Application d'un filtre médian pour lisser mais préserver les structures
        valid_walls = cv2.medianBlur(valid_walls, 3)
        
        # Si très peu de murs détectés, utiliser l'image originale avec moins de filtrage
        total_detected = np.sum(valid_walls > 0)
        total_original = np.sum(walls > 0)
        
        if total_detected < total_original * 0.7:  # Si moins de 70% des murs détectés
            # Utiliser une approche plus conservative
            valid_walls = walls.copy()
            # Juste un léger nettoyage
            kernel_tiny = np.ones((2,2), np.uint8)
            valid_walls = cv2.morphologyEx(valid_walls, cv2.MORPH_OPEN, kernel_tiny)
        
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
        Crée une image de visualisation avec les murs surlignés et leurs contours.
        """
        # Copie de l'image originale
        if len(original_image.shape) == 3:
            viz_image = original_image.copy()
        else:
            viz_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Méthode 1: Surlignage des murs détectés en rouge (zones pleines)
        red_overlay = np.zeros_like(viz_image)
        red_overlay[:,:,0] = walls_detected  # Canal rouge
        
        # Mélange avec l'image originale
        alpha = 0.3
        result = cv2.addWeighted(viz_image, 1-alpha, red_overlay, alpha, 0)
        
        # Méthode 2: Dessiner tous les contours des murs en vert
        # Trouvez tous les contours dans l'image des murs détectés
        contours, _ = cv2.findContours(walls_detected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Dessiner tous les contours en vert épais
        cv2.drawContours(result, contours, -1, (0, 255, 0), thickness=2)
        
        # Méthode 3: Dessiner les contours internes aussi
        contours_internal, _ = cv2.findContours(walls_detected, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours_internal, -1, (0, 255, 255), thickness=1)  # Cyan pour contours internes
        
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
    
    def diagnostic_wall_detection(self, image_array):
        """
        Méthode de diagnostic pour analyser la détection des murs étape par étape.
        Retourne des images intermédiaires pour le débogage.
        """
        # Conversion en niveaux de gris
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array.copy()
        
        # Étape 1: Binarisation
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        walls_binary = cv2.bitwise_not(binary)
        
        # Étape 2: Nettoyage initial
        kernel = np.ones((3,3), np.uint8)
        walls_cleaned = cv2.morphologyEx(walls_binary, cv2.MORPH_CLOSE, kernel)
        walls_cleaned = cv2.morphologyEx(walls_cleaned, cv2.MORPH_OPEN, kernel)
        
        # Étape 3: Détection intelligente
        walls_detected = self._detect_walls_intelligent(walls_cleaned)
        
        # Statistiques
        original_wall_pixels = np.sum(walls_binary > 0)
        cleaned_wall_pixels = np.sum(walls_cleaned > 0)
        detected_wall_pixels = np.sum(walls_detected > 0)
        
        diagnostic_info = {
            'original_binary': walls_binary,
            'cleaned': walls_cleaned,
            'detected': walls_detected,
            'stats': {
                'original_pixels': original_wall_pixels,
                'cleaned_pixels': cleaned_wall_pixels,
                'detected_pixels': detected_wall_pixels,
                'retention_rate': detected_wall_pixels / original_wall_pixels if original_wall_pixels > 0 else 0
            }
        }
        
        return diagnostic_info

    def create_enhanced_visualization(self, original_image, walls_detected):
        """
        Crée une visualisation améliorée avec plusieurs types de marqueurs.
        """
        # Image de base
        if len(original_image.shape) == 3:
            viz_image = original_image.copy()
        else:
            viz_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # 1. Fond rouge transparent pour toutes les zones de murs
        red_overlay = np.zeros_like(viz_image)
        red_overlay[:,:,0] = walls_detected
        result = cv2.addWeighted(viz_image, 0.7, red_overlay, 0.3, 0)
        
        # 2. Contours externes épais en vert
        contours_ext, _ = cv2.findContours(walls_detected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours_ext, -1, (0, 255, 0), thickness=3)
        
        # 3. Tous les contours fins en cyan
        contours_all, _ = cv2.findContours(walls_detected, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours_all, -1, (0, 255, 255), thickness=1)
        
        # 4. Points de contour en bleu
        for contour in contours_ext:
            for point in contour:
                cv2.circle(result, tuple(point[0]), 2, (255, 0, 0), -1)
        
        return result
