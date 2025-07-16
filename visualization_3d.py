import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2

class Visualizer3D:
    def __init__(self):
        self.colors = {
            'walls': 'gray',
            'floor': 'lightblue',
            'ceiling': 'lightgray',
            'transmitter': 'green',
            'receiver': 'red',
            'path': 'blue'
        }
    
    def create_3d_building(self, walls_2d, longueur, largeur, nb_etages, hauteur_etage):
        """
        Crée un modèle 3D du bâtiment à partir du plan 2D.
        
        Args:
            walls_2d: Masque binaire des murs en 2D
            longueur: Longueur du bâtiment en mètres
            largeur: Largeur du bâtiment en mètres
            nb_etages: Nombre d'étages
            hauteur_etage: Hauteur d'un étage en mètres
            
        Returns:
            fig: Figure Plotly 3D
        """
        fig = go.Figure()
        
        # Échelles de conversion pixel vers mètre
        height, width = walls_2d.shape
        scale_x = longueur / width
        scale_y = largeur / height
        
        # Génération des murs pour chaque étage
        for etage in range(nb_etages):
            z_base = etage * hauteur_etage
            z_top = (etage + 1) * hauteur_etage
            
            # Extraction des contours des murs
            contours, _ = cv2.findContours(walls_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 50:  # Filtrer les petits contours
                    # Conversion des coordonnées pixel vers mètres
                    wall_points = []
                    for point in contour:
                        x_pixel, y_pixel = point[0]
                        x_meter = x_pixel * scale_x
                        y_meter = y_pixel * scale_y
                        wall_points.append([x_meter, y_meter])
                    
                    # Création des murs verticaux
                    self._add_wall_3d(fig, wall_points, z_base, z_top, f"Mur_etage_{etage+1}")
        
        # Ajout des planchers et plafonds
        self._add_floors_ceilings(fig, longueur, largeur, nb_etages, hauteur_etage)
        
        # Configuration de la mise en page
        fig.update_layout(
            title="Modèle 3D du Bâtiment",
            scene=dict(
                xaxis_title="Longueur (m)",
                yaxis_title="Largeur (m)",
                zaxis_title="Hauteur (m)",
                aspectmode="data",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def _add_wall_3d(self, fig, wall_points, z_base, z_top, name):
        """
        Ajoute un mur 3D à la figure.
        """
        if len(wall_points) < 3:
            return
        
        # Fermeture du contour si nécessaire
        if wall_points[0] != wall_points[-1]:
            wall_points.append(wall_points[0])
        
        # Extraction des coordonnées
        x_coords = [point[0] for point in wall_points]
        y_coords = [point[1] for point in wall_points]
        
        # Création des faces du mur (extrusion verticale)
        for i in range(len(x_coords) - 1):
            # Points du segment de mur
            x1, y1 = x_coords[i], y_coords[i]
            x2, y2 = x_coords[i + 1], y_coords[i + 1]
            
            # Face rectangulaire du mur
            x_wall = [x1, x2, x2, x1, x1]
            y_wall = [y1, y2, y2, y1, y1]
            z_wall = [z_base, z_base, z_top, z_top, z_base]
            
            fig.add_trace(go.Scatter3d(
                x=x_wall,
                y=y_wall,
                z=z_wall,
                mode='lines',
                line=dict(color=self.colors['walls'], width=3),
                name=name,
                showlegend=(i == 0)  # Afficher la légende seulement pour le premier segment
            ))
            
            # Surface du mur (optionnel, pour un rendu plus réaliste)
            fig.add_trace(go.Mesh3d(
                x=[x1, x2, x2, x1],
                y=[y1, y2, y2, y1],
                z=[z_base, z_base, z_top, z_top],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color=self.colors['walls'],
                opacity=0.3,
                showlegend=False
            ))
    
    def _add_floors_ceilings(self, fig, longueur, largeur, nb_etages, hauteur_etage):
        """
        Ajoute les planchers et plafonds.
        """
        # Coordonnées des coins du bâtiment
        x_corners = [0, longueur, longueur, 0, 0]
        y_corners = [0, 0, largeur, largeur, 0]
        
        for etage in range(nb_etages + 1):  # +1 pour inclure le toit
            z_level = etage * hauteur_etage
            
            if etage == 0:
                # Sol
                fig.add_trace(go.Scatter3d(
                    x=x_corners,
                    y=y_corners,
                    z=[z_level] * len(x_corners),
                    mode='lines',
                    line=dict(color=self.colors['floor'], width=2),
                    name="Sol"
                ))
            elif etage == nb_etages:
                # Toit
                fig.add_trace(go.Scatter3d(
                    x=x_corners,
                    y=y_corners,
                    z=[z_level] * len(x_corners),
                    mode='lines',
                    line=dict(color=self.colors['ceiling'], width=2),
                    name="Toit"
                ))
            else:
                # Plancher intermédiaire
                fig.add_trace(go.Scatter3d(
                    x=x_corners,
                    y=y_corners,
                    z=[z_level] * len(x_corners),
                    mode='lines',
                    line=dict(color=self.colors['floor'], width=1, dash='dash'),
                    name=f"Plancher étage {etage+1}",
                    showlegend=(etage == 1)
                ))
    
    def visualize_3d_path(self, walls_2d, point1_3d, point2_3d, longueur, largeur, nb_etages, hauteur_etage):
        """
        Visualise le trajet 3D entre deux points dans le bâtiment.
        
        Args:
            walls_2d: Masque binaire des murs
            point1_3d: (x1, y1, z1) point émetteur en mètres
            point2_3d: (x2, y2, z2) point récepteur en mètres
            longueur, largeur: Dimensions du bâtiment
            nb_etages: Nombre d'étages
            hauteur_etage: Hauteur d'un étage
            
        Returns:
            fig: Figure Plotly 3D avec trajet
        """
        # Création du modèle 3D de base
        fig = self.create_3d_building(walls_2d, longueur, largeur, nb_etages, hauteur_etage)
        
        x1, y1, z1 = point1_3d
        x2, y2, z2 = point2_3d
        
        # Ajout du trajet direct
        fig.add_trace(go.Scatter3d(
            x=[x1, x2],
            y=[y1, y2],
            z=[z1, z2],
            mode='lines+markers',
            line=dict(color=self.colors['path'], width=4),
            marker=dict(size=8),
            name="Trajet direct 3D"
        ))
        
        # Ajout du point émetteur
        fig.add_trace(go.Scatter3d(
            x=[x1],
            y=[y1],
            z=[z1],
            mode='markers',
            marker=dict(
                size=12,
                color=self.colors['transmitter'],
                symbol='circle',
                line=dict(width=2, color='black')
            ),
            name="Émetteur",
            text=f"TX: ({x1:.1f}, {y1:.1f}, {z1:.1f})",
            textposition="top center"
        ))
        
        # Ajout du point récepteur
        fig.add_trace(go.Scatter3d(
            x=[x2],
            y=[y2],
            z=[z2],
            mode='markers',
            marker=dict(
                size=12,
                color=self.colors['receiver'],
                symbol='circle',
                line=dict(width=2, color='black')
            ),
            name="Récepteur",
            text=f"RX: ({x2:.1f}, {y2:.1f}, {z2:.1f})",
            textposition="top center"
        ))
        
        # Ajout des projections sur les plans pour aider la visualisation
        self._add_projections(fig, point1_3d, point2_3d, longueur, largeur, nb_etages * hauteur_etage)
        
        # Mise à jour du titre
        fig.update_layout(
            title="Visualisation 3D du Trajet RF",
            scene=dict(
                xaxis_title="Longueur (m)",
                yaxis_title="Largeur (m)",
                zaxis_title="Hauteur (m)",
                aspectmode="data",
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.2)
                )
            )
        )
        
        return fig
    
    def _add_projections(self, fig, point1_3d, point2_3d, longueur, largeur, hauteur_totale):
        """
        Ajoute les projections du trajet sur les plans principaux.
        """
        x1, y1, z1 = point1_3d
        x2, y2, z2 = point2_3d
        
        # Projection sur le plan XY (vue du dessus)
        fig.add_trace(go.Scatter3d(
            x=[x1, x2],
            y=[y1, y2],
            z=[0, 0],  # Projection sur z=0
            mode='lines',
            line=dict(color='orange', width=2, dash='dot'),
            name="Projection XY",
            opacity=0.6
        ))
        
        # Projection sur le plan XZ (vue de côté)
        fig.add_trace(go.Scatter3d(
            x=[x1, x2],
            y=[0, 0],  # Projection sur y=0
            z=[z1, z2],
            mode='lines',
            line=dict(color='purple', width=2, dash='dot'),
            name="Projection XZ",
            opacity=0.6
        ))
        
        # Projection sur le plan YZ (vue de face)
        fig.add_trace(go.Scatter3d(
            x=[0, 0],  # Projection sur x=0
            y=[y1, y2],
            z=[z1, z2],
            mode='lines',
            line=dict(color='brown', width=2, dash='dot'),
            name="Projection YZ",
            opacity=0.6
        ))
    
    def create_coverage_heatmap_3d(self, walls_2d, transmitter_pos_3d, frequence, 
                                  longueur, largeur, nb_etages, hauteur_etage, resolution=20):
        """
        Crée une heatmap 3D de la couverture.
        
        Args:
            walls_2d: Masque binaire des murs
            transmitter_pos_3d: Position 3D de l'émetteur
            frequence: Fréquence en MHz
            longueur, largeur: Dimensions du bâtiment
            nb_etages: Nombre d'étages
            hauteur_etage: Hauteur d'un étage
            resolution: Résolution de la grille
            
        Returns:
            fig: Figure Plotly 3D avec heatmap
        """
        from pathloss_calculator_3d import PathlossCalculator3D
        from image_processor import ImageProcessor
        
        calculator_3d = PathlossCalculator3D(frequence)
        processor = ImageProcessor()
        
        x_tx, y_tx, z_tx = transmitter_pos_3d
        
        # Grilles pour chaque étage
        x_grid = np.linspace(0, longueur, resolution)
        y_grid = np.linspace(0, largeur, resolution)
        
        fig = go.Figure()
        
        # Calcul pour chaque étage
        for etage in range(nb_etages):
            z_etage = etage * hauteur_etage + hauteur_etage / 2
            
            pathloss_grid = np.zeros((resolution, resolution))
            
            for i, y_meter in enumerate(y_grid):
                for j, x_meter in enumerate(x_grid):
                    # Distance 3D
                    distance_3d = np.sqrt((x_meter - x_tx)**2 + (y_meter - y_tx)**2 + (z_etage - z_tx)**2)
                    
                    # Conversion en coordonnées pixel pour compter les murs
                    height, width = walls_2d.shape
                    scale_x = longueur / width
                    scale_y = largeur / height
                    
                    x_pixel = int(x_meter / scale_x)
                    y_pixel = int(y_meter / scale_y)
                    x_tx_pixel = int(x_tx / scale_x)
                    y_tx_pixel = int(y_tx / scale_y)
                    
                    # Comptage des murs en 2D
                    wall_count = processor.count_walls_between_points(
                        walls_2d, (x_tx_pixel, y_tx_pixel), (x_pixel, y_pixel)
                    )
                    
                    # Différence d'étages
                    etage_tx = int(z_tx // hauteur_etage)
                    floor_diff = abs(etage - etage_tx)
                    
                    # Calcul du pathloss
                    pathloss = calculator_3d.calculate_pathloss_3d(distance_3d, wall_count, floor_diff)
                    pathloss_grid[i, j] = pathloss
            
            # Ajout de la surface de l'étage
            fig.add_trace(go.Surface(
                x=x_grid,
                y=y_grid,
                z=np.full_like(pathloss_grid, z_etage),
                surfacecolor=pathloss_grid,
                colorscale='Viridis',
                name=f"Étage {etage + 1}",
                opacity=0.7,
                showscale=(etage == 0)
            ))
        
        # Ajout de l'émetteur
        fig.add_trace(go.Scatter3d(
            x=[x_tx],
            y=[y_tx],
            z=[z_tx],
            mode='markers',
            marker=dict(size=15, color='red', symbol='diamond'),
            name="Émetteur"
        ))
        
        fig.update_layout(
            title="Heatmap 3D de Couverture",
            scene=dict(
                xaxis_title="Longueur (m)",
                yaxis_title="Largeur (m)",
                zaxis_title="Hauteur (m)",
                aspectmode="data"
            )
        )
        
        return fig
