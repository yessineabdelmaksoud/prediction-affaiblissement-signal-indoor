import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import cv2
from pathloss_calculator import PathlossCalculator
from image_processor import ImageProcessor
import pandas as pd

class HeatmapGenerator:
    def __init__(self, frequency_mhz):
        """
        Initialise le générateur de heatmap.
        
        Args:
            frequency_mhz: Fréquence en MHz
        """
        self.frequency_mhz = frequency_mhz
        self.calculator = PathlossCalculator(frequency_mhz)
        self.processor = ImageProcessor()
        
        # Palettes de couleurs disponibles
        self.colormaps = {
            'plasma': plt.cm.plasma,
            'viridis': plt.cm.viridis,
            'hot': plt.cm.hot,
            'coolwarm': plt.cm.coolwarm,
            'RdYlGn_r': plt.cm.RdYlGn_r
        }
    
    def generate_heatmap_2d(self, image_array, walls_detected, emetteurs, 
                           longueur, largeur, resolution=50, colormap='plasma'):
        """
        Génère une heatmap 2D du pathloss.
        
        Args:
            image_array: Image originale du plan
            walls_detected: Masque binaire des murs
            emetteurs: Liste des émetteurs avec leurs positions et puissances
            longueur, largeur: Dimensions réelles en mètres
            resolution: Résolution de la grille
            colormap: Palette de couleurs
            
        Returns:
            heatmap_data: Données de la heatmap
            extent: Limites pour l'affichage
            fig: Figure matplotlib
        """
        # Création de la grille de calcul
        x_grid = np.linspace(0, longueur, resolution)
        y_grid = np.linspace(0, largeur, resolution)
        
        # Initialisation de la heatmap avec des valeurs très élevées
        heatmap_data = np.full((resolution, resolution), 200.0)  # 200 dB = signal très faible
        
        # Échelles de conversion
        height, width = image_array.shape[:2]
        scale_x = longueur / width
        scale_y = largeur / height
        
        # Calcul pour chaque point de la grille
        for i, y_meter in enumerate(y_grid):
            for j, x_meter in enumerate(x_grid):
                # Vérification si le point est dans un mur
                x_pixel = int(x_meter / scale_x)
                y_pixel = int(y_meter / scale_y)
                
                if (0 <= x_pixel < width and 0 <= y_pixel < height and 
                    walls_detected[y_pixel, x_pixel] > 0):
                    # Point dans un mur = signal bloqué
                    heatmap_data[i, j] = 200.0
                    continue
                
                # Calcul du pathloss pour chaque émetteur
                pathloss_values = []
                
                for emetteur in emetteurs:
                    x_tx, y_tx = emetteur['position_meter']
                    x_tx_pixel, y_tx_pixel = emetteur['position_pixel']
                    puissance_totale = emetteur['puissance_totale']
                    
                    # Distance 2D
                    distance_2d = np.sqrt((x_meter - x_tx)**2 + (y_meter - y_tx)**2)
                    
                    if distance_2d < 0.1:  # Très proche de l'émetteur
                        received_power = puissance_totale - 20  # Pathloss minimal
                    else:
                        # Comptage des murs entre émetteur et récepteur
                        wall_count = self.processor.count_walls_between_points(
                            walls_detected, 
                            (x_tx_pixel, y_tx_pixel), 
                            (x_pixel, y_pixel)
                        )
                        
                        # Calcul du pathloss
                        pathloss = self.calculator.calculate_pathloss(distance_2d, wall_count)
                        
                        # Puissance reçue = Puissance émise - Pathloss
                        received_power = puissance_totale - pathloss
                    
                    pathloss_values.append(received_power)
                
                # Combinaison des signaux (prise du meilleur signal)
                if pathloss_values:
                    best_received_power = max(pathloss_values)
                    # Conversion en pathloss pour l'affichage (référence à 0 dBm)
                    heatmap_data[i, j] = -best_received_power  # Pathloss négatif = signal fort
        
        # Création de la figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Extent pour l'affichage (coordonnées en mètres)
        extent = [0, longueur, largeur, 0]
        
        # Affichage de l'image de fond (plan)
        if len(image_array.shape) == 3:
            ax.imshow(image_array, extent=extent, alpha=0.3, cmap='gray')
        else:
            ax.imshow(image_array, extent=extent, alpha=0.3, cmap='gray')
        
        # Masquage des zones de murs dans la heatmap
        heatmap_masked = np.ma.masked_where(heatmap_data >= 150, heatmap_data)
        
        # Affichage de la heatmap
        im = ax.imshow(heatmap_masked, extent=extent, cmap=self.colormaps[colormap], 
                      alpha=0.7, vmin=np.min(heatmap_masked), vmax=np.max(heatmap_masked))
        
        # Ajout des émetteurs
        for i, emetteur in enumerate(emetteurs):
            x_tx, y_tx = emetteur['position_meter']
            circle = Circle((x_tx, y_tx), 0.3, color='red', zorder=10)
            ax.add_patch(circle)
            ax.annotate(f'TX{i+1}', (x_tx, y_tx), xytext=(5, 5), 
                       textcoords='offset points', color='white', fontweight='bold')
        
        # Configuration des axes
        ax.set_xlabel('Longueur (m)')
        ax.set_ylabel('Largeur (m)')
        ax.set_title(f'Heatmap du Pathloss 2D - {self.frequency_mhz} MHz')
        
        # Barre de couleur
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Pathloss (dB)')
        
        # Grille
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return heatmap_data, extent, fig
    
    def generate_coverage_zones(self, heatmap_data, extent, emetteurs, seuils, longueur, largeur):
        """
        Génère une carte de couverture par zones de qualité.
        
        Args:
            heatmap_data: Données de pathloss
            extent: Limites d'affichage
            emetteurs: Liste des émetteurs
            seuils: Dictionnaire des seuils de qualité
            longueur, largeur: Dimensions
            
        Returns:
            coverage_map: Carte de couverture
            fig: Figure matplotlib
        """
        # Conversion des seuils de pathloss en zones
        coverage_map = np.zeros_like(heatmap_data)
        
        # Classification par zones (plus le pathloss est faible, meilleure la couverture)
        coverage_map[heatmap_data <= -seuils['excellent']] = 4  # Excellent (vert foncé)
        coverage_map[(heatmap_data > -seuils['excellent']) & 
                    (heatmap_data <= -seuils['bon'])] = 3  # Bon (vert clair)
        coverage_map[(heatmap_data > -seuils['bon']) & 
                    (heatmap_data <= -seuils['faible'])] = 2  # Faible (jaune)
        coverage_map[heatmap_data > -seuils['faible']] = 1  # Mauvais (rouge)
        coverage_map[heatmap_data >= 150] = 0  # Murs (noir)
        
        # Création de la palette de couleurs personnalisée
        colors = ['black', 'red', 'yellow', 'lightgreen', 'darkgreen']
        cmap = mcolors.ListedColormap(colors)
        bounds = [0, 1, 2, 3, 4, 5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Création de la figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Affichage de la carte de couverture
        im = ax.imshow(coverage_map, extent=extent, cmap=cmap, norm=norm)
        
        # Ajout des émetteurs
        for i, emetteur in enumerate(emetteurs):
            x_tx, y_tx = emetteur['position_meter']
            circle = Circle((x_tx, y_tx), 0.3, color='white', ec='black', linewidth=2, zorder=10)
            ax.add_patch(circle)
            ax.annotate(f'TX{i+1}', (x_tx, y_tx), xytext=(5, 5), 
                       textcoords='offset points', color='black', fontweight='bold')
        
        # Configuration des axes
        ax.set_xlabel('Longueur (m)')
        ax.set_ylabel('Largeur (m)')
        ax.set_title(f'Carte de Couverture par Zones - {self.frequency_mhz} MHz')
        
        # Légende personnalisée
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='darkgreen', label='Excellent'),
            plt.Rectangle((0,0),1,1, facecolor='lightgreen', label='Bon'),
            plt.Rectangle((0,0),1,1, facecolor='yellow', label='Faible'),
            plt.Rectangle((0,0),1,1, facecolor='red', label='Mauvais'),
            plt.Rectangle((0,0),1,1, facecolor='black', label='Murs')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Grille
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return coverage_map, fig
    
    def calculate_coverage_statistics(self, heatmap_data, seuils):
        """
        Calcule les statistiques de couverture.
        
        Args:
            heatmap_data: Données de pathloss
            seuils: Seuils de qualité
            
        Returns:
            stats: Dictionnaire des statistiques
        """
        # Masquer les murs pour le calcul
        valid_data = heatmap_data[heatmap_data < 150]
        total_points = len(valid_data)
        
        if total_points == 0:
            return {'excellent': 0, 'bon': 0, 'faible': 0, 'mauvaise': 0}
        
        # Comptage par zones
        excellent = np.sum(valid_data <= -seuils['excellent'])
        bon = np.sum((valid_data > -seuils['excellent']) & (valid_data <= -seuils['bon']))
        faible = np.sum((valid_data > -seuils['bon']) & (valid_data <= -seuils['faible']))
        mauvaise = np.sum(valid_data > -seuils['faible'])
        
        # Conversion en pourcentages
        stats = {
            'excellent': (excellent / total_points) * 100,
            'bon': (bon / total_points) * 100,
            'faible': (faible / total_points) * 100,
            'mauvaise': (mauvaise / total_points) * 100
        }
        
        return stats
    
    
    def export_data_csv(self, heatmap_data, extent):
        """
        Exporte les données de heatmap en format CSV.
        
        Args:
            heatmap_data: Données de pathloss
            extent: Limites d'affichage
            
        Returns:
            csv_string: Données au format CSV
        """
        # Création des coordonnées
        resolution = heatmap_data.shape[0]
        x_coords = np.linspace(extent[0], extent[1], resolution)
        y_coords = np.linspace(extent[2], extent[3], resolution)
        
        # Création du DataFrame
        data_rows = []
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                data_rows.append({
                    'x_meter': x,
                    'y_meter': y,
                    'pathloss_db': heatmap_data[i, j] if heatmap_data[i, j] < 150 else 'WALL'
                })
        
        df = pd.DataFrame(data_rows)
        return df.to_csv(index=False)
    
    def generate_3d_coverage_visualization(self, heatmap_data, extent, emetteurs, hauteur_etage=2.7):
        """
        Génère une visualisation 3D de la couverture (projection en hauteur).
        
        Args:
            heatmap_data: Données de pathloss 2D
            extent: Limites d'affichage
            emetteurs: Liste des émetteurs
            hauteur_etage: Hauteur pour la projection 3D
            
        Returns:
            fig: Figure matplotlib 3D
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Création des coordonnées 3D
        resolution = heatmap_data.shape[0]
        x_coords = np.linspace(extent[0], extent[1], resolution)
        y_coords = np.linspace(extent[2], extent[3], resolution)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Les données de pathloss deviennent la hauteur (inversée pour que bon signal = haut)
        Z = np.where(heatmap_data < 150, -heatmap_data / 10, 0)  # Division par 10 pour ajuster l'échelle
        
        # Surface 3D
        surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8)
        
        # Ajout des émetteurs en 3D
        for i, emetteur in enumerate(emetteurs):
            x_tx, y_tx = emetteur['position_meter']
            z_tx = hauteur_etage  # Hauteur de l'émetteur
            ax.scatter([x_tx], [y_tx], [z_tx], color='red', s=100, label=f'TX{i+1}' if i == 0 else "")
        
        ax.set_xlabel('Longueur (m)')
        ax.set_ylabel('Largeur (m)')
        ax.set_zlabel('Qualité du signal (normalisée)')
        ax.set_title(f'Visualisation 3D de la Couverture - {self.frequency_mhz} MHz')
        
        # Barre de couleur
        fig.colorbar(surf, ax=ax, shrink=0.5)
        
        if len(emetteurs) > 0:
            ax.legend()
        
        return fig
