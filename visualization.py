import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

class Visualizer:
    def __init__(self):
        self.colors = {
            'transmitter': (0, 255, 0),    # Vert
            'receiver': (255, 0, 0),       # Rouge
            'path': (0, 0, 255),           # Bleu
            'wall': (128, 128, 128),       # Gris
            'background': (255, 255, 255)   # Blanc
        }
    
    def visualize_path_and_points(self, original_image, point1, point2, walls_binary):
        """
        Visualise le trajet entre deux points avec les murs.
        
        Args:
            original_image: Image originale
            point1: (x1, y1) point émetteur
            point2: (x2, y2) point récepteur
            walls_binary: Masque binaire des murs
            
        Returns:
            result_image: Image avec visualisation
        """
        # Copie de l'image originale
        if len(original_image.shape) == 3:
            result = original_image.copy()
        else:
            result = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Dessiner la ligne de trajet
        cv2.line(result, point1, point2, self.colors['path'], 2)
        
        # Dessiner les points
        self._draw_point(result, point1, self.colors['transmitter'], 'TX')
        self._draw_point(result, point2, self.colors['receiver'], 'RX')
        
        # Surligner les intersections avec les murs
        intersections = self._find_wall_intersections(point1, point2, walls_binary)
        for intersection in intersections:
            cv2.circle(result, intersection, 3, (255, 255, 0), -1)  # Jaune
        
        return result
    
    def _draw_point(self, image, point, color, label):
        """
        Dessine un point avec un label.
        """
        x, y = point
        
        # Cercle principal
        cv2.circle(image, (x, y), 8, color, -1)
        cv2.circle(image, (x, y), 10, (0, 0, 0), 2)  # Contour noir
        
        # Label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Calculer la taille du texte pour le centrer
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = x - text_size[0] // 2
        text_y = y - 15
        
        # Fond blanc pour le texte
        cv2.rectangle(image, (text_x - 2, text_y - text_size[1] - 2), 
                     (text_x + text_size[0] + 2, text_y + 2), (255, 255, 255), -1)
        
        # Texte
        cv2.putText(image, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    
    def _find_wall_intersections(self, point1, point2, walls_binary):
        """
        Trouve les intersections entre la ligne et les murs.
        """
        x1, y1 = point1
        x2, y2 = point2
        
        intersections = []
        line_points = self._get_line_points(x1, y1, x2, y2)
        
        in_wall = False
        for i, (x, y) in enumerate(line_points):
            if 0 <= x < walls_binary.shape[1] and 0 <= y < walls_binary.shape[0]:
                pixel_value = walls_binary[y, x]
                
                if pixel_value > 0 and not in_wall:
                    # Entrée dans un mur
                    intersections.append((x, y))
                    in_wall = True
                elif pixel_value == 0 and in_wall:
                    # Sortie d'un mur
                    intersections.append((x, y))
                    in_wall = False
        
        return intersections
    
    def _get_line_points(self, x1, y1, x2, y2):
        """
        Génère les points d'une ligne (algorithme de Bresenham).
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
    
    def create_heatmap(self, image_shape, transmitter_pos, frequency_mhz, 
                      walls_binary, longueur, largeur, resolution=50):
        """
        Crée une heatmap de pathloss sur tout le plan.
        
        Args:
            image_shape: (height, width) de l'image
            transmitter_pos: (x, y) position de l'émetteur en pixels
            frequency_mhz: Fréquence en MHz
            walls_binary: Masque binaire des murs
            longueur: Longueur réelle en mètres
            largeur: Largeur réelle en mètres
            resolution: Résolution de la grille (points par dimension)
            
        Returns:
            heatmap: Carte de chaleur du pathloss
            extent: Limites pour l'affichage matplotlib
        """
        from pathloss_calculator import PathlossCalculator
        from image_processor import ImageProcessor
        
        height, width = image_shape
        calculator = PathlossCalculator(frequency_mhz)
        processor = ImageProcessor()
        
        # Création de la grille
        x_grid = np.linspace(0, width-1, resolution)
        y_grid = np.linspace(0, height-1, resolution)
        
        # Échelles de conversion
        scale_x = longueur / width
        scale_y = largeur / height
        
        # Initialisation de la heatmap
        heatmap = np.zeros((resolution, resolution))
        
        tx_x, tx_y = transmitter_pos
        
        for i, y_pixel in enumerate(y_grid):
            for j, x_pixel in enumerate(x_grid):
                # Conversion en coordonnées métriques
                x_meter = x_pixel * scale_x
                y_meter = y_pixel * scale_y
                tx_x_meter = tx_x * scale_x
                tx_y_meter = tx_y * scale_y
                
                # Distance 2D
                distance_2d = np.sqrt((x_meter - tx_x_meter)**2 + (y_meter - tx_y_meter)**2)
                
                # Comptage des murs
                wall_count = processor.count_walls_between_points(
                    walls_binary, (tx_x, tx_y), (int(x_pixel), int(y_pixel))
                )
                
                # Calcul du pathloss
                if distance_2d > 0:
                    pathloss = calculator.calculate_pathloss(distance_2d, wall_count)
                else:
                    pathloss = 0
                
                heatmap[i, j] = pathloss
        
        # Extent pour matplotlib (en mètres)
        extent = [0, longueur, largeur, 0]
        
        return heatmap, extent
    
    def plot_heatmap_matplotlib(self, heatmap, extent, transmitter_pos, receiver_pos, 
                               longueur, largeur, image_shape):
        """
        Affiche la heatmap avec matplotlib.
        """
        # Conversion des positions en coordonnées métriques
        height, width = image_shape
        scale_x = longueur / width
        scale_y = largeur / height
        
        tx_x_meter = transmitter_pos[0] * scale_x
        tx_y_meter = transmitter_pos[1] * scale_y
        rx_x_meter = receiver_pos[0] * scale_x
        rx_y_meter = receiver_pos[1] * scale_y
        
        # Création de la figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Affichage de la heatmap
        im = ax.imshow(heatmap, extent=extent, cmap='plasma', aspect='equal', 
                      vmin=heatmap.min(), vmax=heatmap.max())
        
        # Points émetteur et récepteur
        ax.plot(tx_x_meter, tx_y_meter, 'go', markersize=12, markeredgecolor='black', 
                markeredgewidth=2, label='Émetteur')
        ax.plot(rx_x_meter, rx_y_meter, 'ro', markersize=12, markeredgecolor='black', 
                markeredgewidth=2, label='Récepteur')
        
        # Ligne de trajet
        ax.plot([tx_x_meter, rx_x_meter], [tx_y_meter, rx_y_meter], 'w--', 
                linewidth=2, alpha=0.7, label='Trajet direct')
        
        # Configuration des axes
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Distance (m)')
        ax.set_title('Heatmap du Pathloss')
        ax.legend()
        
        # Barre de couleur
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Pathloss (dB)')
        
        # Grille
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_coverage_analysis(self, pathloss_results, signal_thresholds):
        """
        Crée une analyse de couverture basée sur les seuils de signal.
        
        Args:
            pathloss_results: Résultats de pathloss en différents points
            signal_thresholds: Seuils de qualité du signal
            
        Returns:
            coverage_map: Carte de couverture par zones
        """
        coverage_zones = {
            'excellent': {'threshold': -50, 'color': 'green', 'label': 'Excellent'},
            'good': {'threshold': -70, 'color': 'yellow', 'label': 'Bon'},
            'fair': {'threshold': -85, 'color': 'orange', 'label': 'Correct'},
            'poor': {'threshold': float('-inf'), 'color': 'red', 'label': 'Faible'}
        }
        
        return coverage_zones
    
    def add_measurement_annotations(self, image, measurements):
        """
        Ajoute des annotations de mesures sur l'image.
        
        Args:
            image: Image à annoter
            measurements: Dictionnaire avec les mesures
            
        Returns:
            annotated_image: Image avec annotations
        """
        result = image.copy()
        
        # Position pour les annotations
        y_start = 30
        line_height = 25
        
        # Fond semi-transparent pour les annotations
        overlay = result.copy()
        cv2.rectangle(overlay, (10, 10), (300, y_start + len(measurements) * line_height + 10), 
                     (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.8, result, 0.2, 0, result)
        
        # Ajout des mesures
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 0, 0)
        thickness = 1
        
        for i, (key, value) in enumerate(measurements.items()):
            text = f"{key}: {value}"
            y_pos = y_start + i * line_height
            cv2.putText(result, text, (15, y_pos), font, font_scale, color, thickness)
        
        return result
