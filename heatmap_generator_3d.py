import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
from pathloss_calculator_3d import PathlossCalculator3D
from image_processor import ImageProcessor
import pandas as pd
from scipy.ndimage import zoom

class HeatmapGenerator3D:
    def __init__(self, frequency_mhz):
        """
        Initialise le générateur de heatmap 3D.
        
        Args:
            frequency_mhz: Fréquence en MHz
        """
        self.frequency_mhz = frequency_mhz
        self.calculator_3d = PathlossCalculator3D(frequency_mhz)
        self.processor = ImageProcessor()
        
        # Palettes de couleurs 3D
        self.colormaps_3d = {
            'plasma': px.colors.sequential.Plasma,
            'viridis': px.colors.sequential.Viridis,
            'hot': px.colors.sequential.Hot,
            'coolwarm': px.colors.diverging.RdBu,
            'jet': px.colors.sequential.Turbo
        }
    
    def generate_voxel_grid(self, walls_detected, emetteurs_3d, longueur, largeur, 
                           hauteur_totale, resolution_xy=25, resolution_z=10):
        """
        Génère une grille de voxels 3D avec calcul du pathloss pour chaque voxel.
        
        Args:
            walls_detected: Masque binaire des murs en 2D
            emetteurs_3d: Liste des émetteurs 3D
            longueur, largeur, hauteur_totale: Dimensions en mètres
            resolution_xy: Résolution dans le plan XY
            resolution_z: Résolution en Z
            
        Returns:
            voxel_data: Array 3D des valeurs de pathloss
            coordinates: Coordonnées des voxels
        """
        # Création des grilles de coordonnées
        x_coords = np.linspace(0, longueur, resolution_xy)
        y_coords = np.linspace(0, largeur, resolution_xy)
        z_coords = np.linspace(0.1, hauteur_totale, resolution_z)
        
        # Initialisation de la grille de voxels
        voxel_data = np.full((resolution_z, resolution_xy, resolution_xy), 200.0)
        
        # Échelles de conversion pour les murs 2D
        height_2d, width_2d = walls_detected.shape
        scale_x = longueur / width_2d
        scale_y = largeur / height_2d
        
        # Calcul pour chaque voxel
        total_voxels = resolution_xy * resolution_xy * resolution_z
        processed_voxels = 0
        
        for k, z_meter in enumerate(z_coords):
            for i, y_meter in enumerate(y_coords):
                for j, x_meter in enumerate(x_coords):
                    processed_voxels += 1
                    
                    # Vérification si le voxel est dans un mur (projection 2D)
                    x_pixel = int(np.clip(x_meter / scale_x, 0, width_2d - 1))
                    y_pixel = int(np.clip(y_meter / scale_y, 0, height_2d - 1))
                    
                    # Si dans un mur, marquer comme bloqué
                    if walls_detected[y_pixel, x_pixel] > 0:
                        voxel_data[k, i, j] = 250.0  # Valeur spéciale pour les murs
                        continue
                    
                    # Calcul du meilleur signal reçu de tous les émetteurs
                    best_received_power = -200.0  # Très faible
                    
                    for emetteur in emetteurs_3d:
                        x_tx, y_tx, z_tx = emetteur['position_meter']
                        x_tx_pixel, y_tx_pixel = emetteur['position_pixel']
                        puissance_totale = emetteur['puissance_totale']
                        directivite = emetteur['directivite']
                        
                        # Distance 3D
                        distance_3d = np.sqrt((x_meter - x_tx)**2 + (y_meter - y_tx)**2 + (z_meter - z_tx)**2)
                        
                        if distance_3d < 0.1:  # Très proche de l'émetteur
                            received_power = puissance_totale - 10
                        else:
                            # Comptage des murs en 2D
                            wall_count_2d = self.processor.count_walls_between_points(
                                walls_detected, 
                                (x_tx_pixel, y_tx_pixel), 
                                (x_pixel, y_pixel)
                            )
                            
                            # Calcul de la différence d'étages
                            etage_tx = int(z_tx // (hauteur_totale / resolution_z * (resolution_z / 10)))
                            etage_rx = int(z_meter // (hauteur_totale / resolution_z * (resolution_z / 10)))
                            floor_difference = abs(etage_rx - etage_tx)
                            
                            # Calcul du pathloss 3D
                            pathloss = self.calculator_3d.calculate_pathloss_3d(
                                distance_3d, wall_count_2d, floor_difference
                            )
                            
                            # Facteur de directivité
                            if directivite == "Directif":
                                # Simplification: réduction de 3dB si dans la direction principale
                                angle_factor = self._calculate_directivity_factor(
                                    (x_tx, y_tx, z_tx), (x_meter, y_meter, z_meter)
                                )
                                pathloss *= angle_factor
                            
                            # Puissance reçue
                            received_power = puissance_totale - pathloss
                        
                        # Garder le meilleur signal
                        if received_power > best_received_power:
                            best_received_power = received_power
                    
                    # Stockage du pathloss (négatif de la puissance reçue)
                    voxel_data[k, i, j] = -best_received_power
        
        # Coordonnées pour l'affichage
        coordinates = {
            'x': x_coords,
            'y': y_coords,
            'z': z_coords
        }
        
        return voxel_data, coordinates
    
    def _calculate_directivity_factor(self, pos_tx, pos_rx):
        """
        Calcule un facteur de directivité simplifié.
        """
        # Simplification: facteur basé sur la distance horizontale vs verticale
        dx = abs(pos_rx[0] - pos_tx[0])
        dy = abs(pos_rx[1] - pos_tx[1])
        dz = abs(pos_rx[2] - pos_tx[2])
        
        horizontal_dist = np.sqrt(dx**2 + dy**2)
        
        if horizontal_dist > dz:
            return 0.7  # Pénalité pour direction non optimale
        else:
            return 1.0  # Direction optimale
    
    def visualize_voxel_heatmap(self, voxel_data, coordinates, emetteurs_3d, colormap='plasma'):
        """
        Visualise la heatmap 3D avec des voxels colorés.
        
        Args:
            voxel_data: Données des voxels
            coordinates: Coordonnées
            emetteurs_3d: Liste des émetteurs
            colormap: Palette de couleurs
            
        Returns:
            fig: Figure Plotly 3D
        """
        fig = go.Figure()
        
        # Masquer les murs et créer des indices pour les voxels valides
        valid_mask = voxel_data < 200
        
        # Création des coordonnées pour chaque voxel valide
        z_indices, y_indices, x_indices = np.where(valid_mask)
        
        if len(z_indices) == 0:
            # Aucun voxel valide
            fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', 
                                     marker=dict(size=1), name="Aucune donnée"))
        else:
            # Coordonnées réelles des voxels
            x_coords = coordinates['x'][x_indices]
            y_coords = coordinates['y'][y_indices]
            z_coords = coordinates['z'][z_indices]
            
            # Valeurs de pathloss correspondantes
            pathloss_values = voxel_data[z_indices, y_indices, x_indices]
            
            # Normalisation pour la couleur
            vmin, vmax = np.min(pathloss_values), np.max(pathloss_values)
            
            # Taille des voxels basée sur la qualité du signal
            sizes = np.clip(15 - (pathloss_values - vmin) / (vmax - vmin) * 10, 3, 15)
            
            # Transparence fixe pour éviter les erreurs Plotly
            opacity_fixed = 0.7
            
            # Ajout des voxels
            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='markers',
                marker=dict(
                    size=sizes.tolist(),  # Conversion en liste
                    color=pathloss_values.tolist(),  # Conversion en liste
                    colorscale=colormap,
                    opacity=opacity_fixed,  # Valeur fixe
                    colorbar=dict(title="Pathloss (dB)"),
                    showscale=True
                ),
                text=[f"Pathloss: {val:.1f} dB" for val in pathloss_values],
                name="Voxels RF"
            ))
        
        # Ajout des émetteurs
        for i, emetteur in enumerate(emetteurs_3d):
            x_tx, y_tx, z_tx = emetteur['position_meter']
            fig.add_trace(go.Scatter3d(
                x=[x_tx],
                y=[y_tx],
                z=[z_tx],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='diamond',
                    line=dict(width=2, color='black')
                ),
                name=f"Émetteur {i+1}",
                text=f"TX{i+1}: {emetteur['puissance_totale']} dBm"
            ))
        
        # Configuration de la mise en page
        fig.update_layout(
            title=f"Heatmap 3D Voxels - {self.frequency_mhz} MHz",
            scene=dict(
                xaxis_title="Longueur (m)",
                yaxis_title="Largeur (m)",
                zaxis_title="Hauteur (m)",
                aspectmode="data",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=900,
            height=700
        )
        
        return fig
    
    def visualize_voxel_heatmap_layered(self, voxel_data, coordinates, emetteurs_3d, colormap='plasma'):
        """
        Visualise la heatmap 3D avec des voxels colorés en couches pour permettre les transparences variables.
        
        Args:
            voxel_data: Données des voxels
            coordinates: Coordonnées
            emetteurs_3d: Liste des émetteurs
            colormap: Palette de couleurs
            
        Returns:
            fig: Figure Plotly 3D
        """
        fig = go.Figure()
        
        # Masquer les murs et créer des indices pour les voxels valides
        valid_mask = voxel_data < 200
        
        # Création des coordonnées pour chaque voxel valide
        z_indices, y_indices, x_indices = np.where(valid_mask)
        
        if len(z_indices) == 0:
            # Aucun voxel valide
            fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', 
                                     marker=dict(size=1), name="Aucune donnée"))
        else:
            # Coordonnées réelles des voxels
            x_coords = coordinates['x'][x_indices]
            y_coords = coordinates['y'][y_indices]
            z_coords = coordinates['z'][z_indices]
            
            # Valeurs de pathloss correspondantes
            pathloss_values = voxel_data[z_indices, y_indices, x_indices]
            
            # Normalisation pour la couleur
            vmin, vmax = np.min(pathloss_values), np.max(pathloss_values)
            
            # Séparation en couches de qualité pour transparences différentes
            quality_layers = [
                ("Excellent", pathloss_values <= 40, 'green', 0.9, 12),
                ("Bon", (pathloss_values > 40) & (pathloss_values <= 70), 'yellow', 0.7, 10),
                ("Moyen", (pathloss_values > 70) & (pathloss_values <= 100), 'orange', 0.5, 8),
                ("Faible", pathloss_values > 100, 'red', 0.3, 6)
            ]
            
            for layer_name, mask, color, opacity, size in quality_layers:
                if np.any(mask):
                    layer_x = x_coords[mask]
                    layer_y = y_coords[mask]
                    layer_z = z_coords[mask]
                    layer_values = pathloss_values[mask]
                    
                    fig.add_trace(go.Scatter3d(
                        x=layer_x,
                        y=layer_y,
                        z=layer_z,
                        mode='markers',
                        marker=dict(
                            size=size,
                            color=color,
                            opacity=opacity,
                            line=dict(width=0)
                        ),
                        text=[f"Pathloss: {val:.1f} dB" for val in layer_values],
                        name=f"Signal {layer_name}"
                    ))
        
        # Ajout des émetteurs
        for i, emetteur in enumerate(emetteurs_3d):
            x_tx, y_tx, z_tx = emetteur['position_meter']
            fig.add_trace(go.Scatter3d(
                x=[x_tx],
                y=[y_tx],
                z=[z_tx],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='diamond',
                    line=dict(width=2, color='black')
                ),
                name=f"Émetteur {i+1}",
                text=f"TX{i+1}: {emetteur['puissance_totale']} dBm"
            ))
        
        # Configuration de la mise en page
        fig.update_layout(
            title=f"Heatmap 3D Voxels (Couches) - {self.frequency_mhz} MHz",
            scene=dict(
                xaxis_title="Longueur (m)",
                yaxis_title="Largeur (m)",
                zaxis_title="Hauteur (m)",
                aspectmode="data",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=900,
            height=700
        )
        
        return fig
    
    def visualize_cross_sections(self, voxel_data, coordinates, colormap, nb_etages):
        """
        Crée des coupes transversales par étage.
        
        Args:
            voxel_data: Données des voxels
            coordinates: Coordonnées
            colormap: Palette de couleurs
            nb_etages: Nombre d'étages
            
        Returns:
            figs: Liste de figures pour chaque étage
        """
        figs = []
        resolution_z = len(coordinates['z'])
        voxels_par_etage = resolution_z // nb_etages
        
        for etage in range(nb_etages):
            # Indices pour cet étage
            z_start = etage * voxels_par_etage
            z_end = min((etage + 1) * voxels_par_etage, resolution_z)
            
            # Moyenne des valeurs sur la hauteur de l'étage
            etage_data = np.mean(voxel_data[z_start:z_end, :, :], axis=0)
            
            # Masquer les murs
            etage_data_clean = np.where(etage_data < 200, etage_data, np.nan)
            
            # Création de la heatmap 2D
            fig = go.Figure(data=go.Heatmap(
                x=coordinates['x'],
                y=coordinates['y'],
                z=etage_data_clean,
                colorscale=colormap,
                showscale=True,
                colorbar=dict(title="Pathloss (dB)")
            ))
            
            fig.update_layout(
                title=f"Coupe Étage {etage + 1} - {self.frequency_mhz} MHz",
                xaxis_title="Longueur (m)",
                yaxis_title="Largeur (m)",
                width=600,
                height=500
            )
            
            figs.append(fig)
        
        return figs
    
    def calculate_3d_coverage_statistics(self, voxel_data, seuils):
        """
        Calcule les statistiques de couverture 3D.
        
        Args:
            voxel_data: Données des voxels
            seuils: Seuils de qualité
            
        Returns:
            stats: Statistiques de couverture 3D
        """
        # Masquer les murs
        valid_voxels = voxel_data[voxel_data < 200]
        total_voxels = len(valid_voxels)
        
        if total_voxels == 0:
            return {
                'excellent': 0, 'bon': 0, 'faible': 0, 'mauvaise': 0,
                'total_voxels': 0
            }
        
        # Comptage par zones (rappel: pathloss négatif = bon signal)
        excellent = np.sum(valid_voxels <= -seuils['excellent'])
        bon = np.sum((valid_voxels > -seuils['excellent']) & (valid_voxels <= -seuils['bon']))
        faible = np.sum((valid_voxels > -seuils['bon']) & (valid_voxels <= -seuils['faible']))
        mauvaise = np.sum(valid_voxels > -seuils['faible'])
        
        # Conversion en pourcentages
        stats = {
            'excellent': (excellent / total_voxels) * 100,
            'bon': (bon / total_voxels) * 100,
            'faible': (faible / total_voxels) * 100,
            'mauvaise': (mauvaise / total_voxels) * 100,
            'total_voxels': total_voxels
        }
        
        return stats
    
    def analyze_coverage_by_floor(self, voxel_data, coordinates, nb_etages, hauteur_etage, seuils):
        """
        Analyse la couverture par étage.
        
        Args:
            voxel_data: Données des voxels
            coordinates: Coordonnées
            nb_etages: Nombre d'étages
            hauteur_etage: Hauteur d'un étage
            seuils: Seuils de qualité
            
        Returns:
            stats_par_etage: Statistiques pour chaque étage
        """
        stats_par_etage = []
        resolution_z = len(coordinates['z'])
        voxels_par_etage = resolution_z // nb_etages
        
        for etage in range(nb_etages):
            z_start = etage * voxels_par_etage
            z_end = min((etage + 1) * voxels_par_etage, resolution_z)
            
            # Données pour cet étage
            etage_data = voxel_data[z_start:z_end, :, :]
            valid_voxels_etage = etage_data[etage_data < 200]
            
            if len(valid_voxels_etage) == 0:
                stats_etage = {'excellent': 0, 'bon': 0, 'faible': 0, 'mauvaise': 0}
            else:
                total_etage = len(valid_voxels_etage)
                
                excellent = np.sum(valid_voxels_etage <= -seuils['excellent'])
                bon = np.sum((valid_voxels_etage > -seuils['excellent']) & 
                           (valid_voxels_etage <= -seuils['bon']))
                faible = np.sum((valid_voxels_etage > -seuils['bon']) & 
                              (valid_voxels_etage <= -seuils['faible']))
                mauvaise = np.sum(valid_voxels_etage > -seuils['faible'])
                
                stats_etage = {
                    'excellent': (excellent / total_etage) * 100,
                    'bon': (bon / total_etage) * 100,
                    'faible': (faible / total_etage) * 100,
                    'mauvaise': (mauvaise / total_etage) * 100
                }
            
            stats_par_etage.append(stats_etage)
        
        return stats_par_etage
    
    def export_voxel_data_csv(self, voxel_data, coordinates):
        """
        Exporte les données des voxels en format CSV.
        
        Args:
            voxel_data: Données des voxels
            coordinates: Coordonnées
            
        Returns:
            csv_string: Données au format CSV
        """
        data_rows = []
        
        for k, z in enumerate(coordinates['z']):
            for i, y in enumerate(coordinates['y']):
                for j, x in enumerate(coordinates['x']):
                    pathloss_value = voxel_data[k, i, j]
                    
                    data_rows.append({
                        'x_meter': x,
                        'y_meter': y,
                        'z_meter': z,
                        'pathloss_db': pathloss_value if pathloss_value < 200 else 'WALL',
                        'etage': int(z // 2.7) + 1
                    })
        
        df = pd.DataFrame(data_rows)
        return df.to_csv(index=False)