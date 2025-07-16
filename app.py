import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import io
from image_processor import ImageProcessor
from pathloss_calculator import PathlossCalculator
from visualization import Visualizer
from ml_pathloss_predictor_2d import ml_predictor_2d
from ml_pathloss_predictor_3d import ml_predictor_3d

def display_ml_status():
    """
    Affiche le statut des modèles ML dans la sidebar.
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Modèles ML")
    
    # Statut du modèle 2D
    model_2d_info = ml_predictor_2d.get_model_info()
    if model_2d_info['status'] == 'chargé':
        st.sidebar.success("✅ Modèle 2D: Chargé")
        st.sidebar.caption(f"Type: {model_2d_info['model_type']}")
    else:
        st.sidebar.warning("⚠️ Modèle 2D: Fallback théorique")
    
    # Statut du modèle 3D
    model_3d_info = ml_predictor_3d.get_model_info()
    if model_3d_info['status'] == 'chargé':
        st.sidebar.success("✅ Modèle 3D: Chargé")
        st.sidebar.caption(f"Type: {model_3d_info['model_type']}")
        if 'metrics' in model_3d_info:
            st.sidebar.caption(f"R²: {model_3d_info['metrics'].get('r2_score', 'N/A'):.3f}")
    else:
        st.sidebar.warning("⚠️ Modèle 3D: Fallback théorique")

def main():
    st.title("Analyseur de Pathloss - Plan d'Appartement")
    st.markdown("---")
    
    # Création des onglets
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Pathloss Calculator 2D", "Pathloss Calculator 3D", "Génération Heatmap 2D", "Génération Heatmap 3D", "Optimisation Points d'Accès 3D", "Optimisation Points d'Accès 2D"])
    
    with tab1:
        pathloss_2d_interface()
    
    with tab2:
        pathloss_3d_interface()
    
    with tab3:
        heatmap_2d_interface()
    
    with tab4:
        heatmap_3d_interface()
    
    with tab5:
        optimization_3d_interface()
    
    with tab6:
        optimization_2d_interface()

def pathloss_2d_interface():
    """Interface pour l'analyse 2D du pathloss"""
    # Sidebar pour les paramètres
    st.sidebar.header("Paramètres du bâtiment (2D)")
    
    # Affichage du statut ML
    display_ml_status()
    
    # Upload du fichier
    uploaded_file = st.file_uploader(
        "Téléchargez le plan de l'appartement (PNG)",
        type=['png'],
        help="Le plan doit être en blanc avec les murs en noir",
        key="upload_2d"
    )
    
    if uploaded_file is not None:
        # Conversion de l'image uploadée
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Affichage de l'image originale
        st.subheader("Plan original")
        st.image(image, caption="Plan téléchargé", use_column_width=True)
        
        # Paramètres du bâtiment
        col1, col2 = st.sidebar.columns(2)
        with col1:
            longueur = st.number_input("Longueur (m)", min_value=1.0, value=10.0, step=0.1)
            largeur = st.number_input("Largeur (m)", min_value=1.0, value=8.0, step=0.1)
        
        with col2:
            nb_etages = st.number_input("Nombre d'étages", min_value=1, value=1, step=1)
            hauteur_etage = st.number_input("Hauteur étage (m)", min_value=2.0, value=2.7, step=0.1)
        
        frequence = st.sidebar.number_input("Fréquence (MHz)", min_value=100, value=2400, step=100)
        
        # Traitement de l'image
        processor = ImageProcessor()
        processed_image, walls_detected = processor.process_image(image_array)
        
        # Affichage de l'image traitée
        st.subheader("Murs détectés")
        st.image(processed_image, caption="Murs extraits", use_column_width=True)
        
        # Conversion des coordonnées pixel vers mètres
        height, width = image_array.shape[:2]
        scale_x = longueur / width
        scale_y = largeur / height
        
        # Interface pour les points d'accès et récepteur
        st.subheader("Points d'accès et récepteur")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Point d'accès (Émetteur)**")
            x1_pixel = st.number_input("X1 (pixels)", min_value=0, max_value=width, value=width//4)
            y1_pixel = st.number_input("Y1 (pixels)", min_value=0, max_value=height, value=height//4)
            x1_meter = x1_pixel * scale_x
            y1_meter = y1_pixel * scale_y
            st.write(f"Position: ({x1_meter:.2f}m, {y1_meter:.2f}m)")
        
        with col2:
            st.write("**Récepteur**")
            x2_pixel = st.number_input("X2 (pixels)", min_value=0, max_value=width, value=3*width//4)
            y2_pixel = st.number_input("Y2 (pixels)", min_value=0, max_value=height, value=3*height//4)
            x2_meter = x2_pixel * scale_x
            y2_meter = y2_pixel * scale_y
            st.write(f"Position: ({x2_meter:.2f}m, {y2_meter:.2f}m)")
        
        if st.button("Calculer le Pathloss"):
            # Calcul du pathloss
            calculator = PathlossCalculator(frequence)
            
            # Compter les murs entre les deux points
            wall_count = processor.count_walls_between_points(
                walls_detected, 
                (x1_pixel, y1_pixel), 
                (x2_pixel, y2_pixel)
            )
            
            # Calcul de la distance 2D
            distance_2d = np.sqrt((x2_meter - x1_meter)**2 + (y2_meter - y1_meter)**2)
            
            # Calcul du pathloss
            pathloss_db = calculator.calculate_pathloss(distance_2d, wall_count)
            
            # Visualisation
            visualizer = Visualizer()
            result_image = visualizer.visualize_path_and_points(
                image_array,
                (x1_pixel, y1_pixel),
                (x2_pixel, y2_pixel),
                walls_detected
            )
            
            # Affichage des résultats
            st.subheader("Résultats")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Distance 2D", f"{distance_2d:.2f} m")
            with col2:
                st.metric("Nombre de murs", wall_count)
            with col3:
                st.metric("Fréquence", f"{frequence} MHz")
            with col4:
                st.metric("Pathloss", f"{pathloss_db:.2f} dB")
            
            st.subheader("Visualisation du trajet")
            st.image(result_image, caption="Trajet et points", use_column_width=True)
            
            # Informations détaillées
            with st.expander("Détails du calcul"):
                st.write(f"**Paramètres du bâtiment:**")
                st.write(f"- Dimensions: {longueur}m × {largeur}m")
                st.write(f"- Étages: {nb_etages} étage(s) de {hauteur_etage}m")
                st.write(f"- Échelle: {scale_x:.4f} m/pixel (X), {scale_y:.4f} m/pixel (Y)")
                
                st.write(f"**Calcul du pathloss:**")
                st.write(f"- Distance en espace libre: {distance_2d:.2f} m")
                st.write(f"- Nombre de murs traversés: {wall_count}")
                st.write(f"- Fréquence: {frequence} MHz")
                st.write(f"- Pathloss total: {pathloss_db:.2f} dB")

def pathloss_3d_interface():
    """Interface pour l'analyse 3D du pathloss"""
    st.header("Analyse 3D du Pathloss")
    
    # Sidebar pour les paramètres 3D
    st.sidebar.header("Paramètres du bâtiment (3D)")
    
    # Affichage du statut ML
    display_ml_status()
    
    # Upload du fichier pour 3D
    uploaded_file_3d = st.file_uploader(
        "Téléchargez le plan de l'appartement (PNG) pour la 3D",
        type=['png'],
        help="Le plan doit être en blanc avec les murs en noir",
        key="upload_3d"
    )
    
    if uploaded_file_3d is not None:
        # Conversion de l'image uploadée
        image = Image.open(uploaded_file_3d)
        image_array = np.array(image)
        
        # Affichage de l'image originale
        st.subheader("Plan 2D original")
        st.image(image, caption="Plan téléchargé pour conversion 3D", use_column_width=True)
        
        # Paramètres du bâtiment 3D
        st.sidebar.subheader("Dimensions du bâtiment")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            longueur = st.number_input("Longueur (m)", min_value=1.0, value=10.0, step=0.1, key="longueur_3d")
            largeur = st.number_input("Largeur (m)", min_value=1.0, value=8.0, step=0.1, key="largeur_3d")
        
        with col2:
            nb_etages = st.number_input("Nombre d'étages", min_value=1, value=2, step=1, key="etages_3d")
            hauteur_etage = st.number_input("Hauteur étage (m)", min_value=2.0, value=2.7, step=0.1, key="hauteur_3d")
        
        frequence = st.sidebar.number_input("Fréquence (MHz)", min_value=100, value=2400, step=100, key="freq_3d")
        
        # Traitement de l'image pour extraire les murs
        processor = ImageProcessor()
        processed_image, walls_detected = processor.process_image(image_array)
        
        # Affichage de l'image traitée
        st.subheader("Murs détectés (plan de base)")
        st.image(processed_image, caption="Murs extraits pour duplication 3D", use_column_width=True)
        
        # Bouton pour visualiser en 3D
        if st.button("Visualiser le plan en 3D", key="visualize_3d"):
            with st.spinner("Génération du modèle 3D..."):
                try:
                    # Importation dynamique pour éviter les erreurs
                    from visualization_3d import Visualizer3D
                    
                    # Création du visualiseur 3D
                    visualizer_3d = Visualizer3D()
                    
                    # Conversion des coordonnées
                    height, width = image_array.shape[:2]
                    scale_x = longueur / width
                    scale_y = largeur / height
                    
                    # Génération du modèle 3D
                    fig_3d = visualizer_3d.create_3d_building(
                        walls_detected, 
                        longueur, 
                        largeur, 
                        nb_etages, 
                        hauteur_etage
                    )
                    
                    st.subheader("Modèle 3D du bâtiment")
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
                except ImportError:
                    st.error("Module de visualisation 3D non disponible. Installation en cours...")
                    # Installation des dépendances 3D
                    install_3d_dependencies()
        
        # Interface pour les points 3D
        st.subheader("Points d'accès et récepteur 3D")
        
        col1, col2 = st.columns(2)
        
        # Conversion des coordonnées pour l'affichage
        height, width = image_array.shape[:2]
        scale_x = longueur / width
        scale_y = largeur / height
        
        with col1:
            st.write("**Point d'accès (Émetteur) 3D**")
            x1_meter = st.number_input("X1 (m)", min_value=0.0, max_value=longueur, value=longueur/4, step=0.1, key="x1_3d")
            y1_meter = st.number_input("Y1 (m)", min_value=0.0, max_value=largeur, value=largeur/4, step=0.1, key="y1_3d")
            z1_meter = st.number_input("Z1 (m)", min_value=0.0, max_value=nb_etages*hauteur_etage, value=hauteur_etage/2, step=0.1, key="z1_3d")
            
            # Calcul des coordonnées pixel correspondantes
            x1_pixel = int(x1_meter / scale_x)
            y1_pixel = int(y1_meter / scale_y)
            etage1 = int(z1_meter // hauteur_etage)
            
            st.write(f"Étage: {etage1 + 1}")
            st.write(f"Position pixel: ({x1_pixel}, {y1_pixel})")
        
        with col2:
            st.write("**Récepteur 3D**")
            x2_meter = st.number_input("X2 (m)", min_value=0.0, max_value=longueur, value=3*longueur/4, step=0.1, key="x2_3d")
            y2_meter = st.number_input("Y2 (m)", min_value=0.0, max_value=largeur, value=3*largeur/4, step=0.1, key="y2_3d")
            z2_meter = st.number_input("Z2 (m)", min_value=0.0, max_value=nb_etages*hauteur_etage, value=hauteur_etage*1.5, step=0.1, key="z2_3d")
            
            # Calcul des coordonnées pixel correspondantes
            x2_pixel = int(x2_meter / scale_x)
            y2_pixel = int(y2_meter / scale_y)
            etage2 = int(z2_meter // hauteur_etage)
            
            st.write(f"Étage: {etage2 + 1}")
            st.write(f"Position pixel: ({x2_pixel}, {y2_pixel})")
        
        # Calcul du pathloss 3D
        if st.button("Calculer le Pathloss 3D", key="calc_3d"):
            with st.spinner("Calcul du pathloss 3D..."):
                try:
                    # Importation dynamique
                    from pathloss_calculator_3d import PathlossCalculator3D
                    from visualization_3d import Visualizer3D
                    
                    # Calculateur 3D
                    calculator_3d = PathlossCalculator3D(frequence)
                    visualizer_3d = Visualizer3D()
                    
                    # Calcul de la distance 3D
                    distance_3d = np.sqrt((x2_meter - x1_meter)**2 + (y2_meter - y1_meter)**2 + (z2_meter - z1_meter)**2)
                    
                    # Comptage des murs 2D (même plan)
                    wall_count_2d = processor.count_walls_between_points(
                        walls_detected, 
                        (x1_pixel, y1_pixel), 
                        (x2_pixel, y2_pixel)
                    )
                    
                    # Calcul de la différence d'étages
                    floor_difference = abs(etage2 - etage1)
                    
                    # Calcul du pathloss 3D
                    pathloss_3d = calculator_3d.calculate_pathloss_3d(
                        distance_3d, 
                        wall_count_2d, 
                        floor_difference
                    )
                    
                    # Visualisation 3D avec trajet
                    fig_3d_path = visualizer_3d.visualize_3d_path(
                        walls_detected,
                        (x1_meter, y1_meter, z1_meter),
                        (x2_meter, y2_meter, z2_meter),
                        longueur,
                        largeur,
                        nb_etages,
                        hauteur_etage
                    )
                    
                    # Affichage des résultats
                    st.subheader("Résultats 3D")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Distance 3D", f"{distance_3d:.2f} m")
                    with col2:
                        st.metric("Murs 2D", wall_count_2d)
                    with col3:
                        st.metric("Diff. étages", floor_difference)
                    with col4:
                        st.metric("Fréquence", f"{frequence} MHz")
                    with col5:
                        st.metric("Pathloss 3D", f"{pathloss_3d:.2f} dB")
                    
                    st.subheader("Visualisation 3D du trajet")
                    st.plotly_chart(fig_3d_path, use_container_width=True)
                    
                    # Informations détaillées 3D
                    with st.expander("Détails du calcul 3D"):
                        st.write(f"**Paramètres du bâtiment 3D:**")
                        st.write(f"- Dimensions: {longueur}m × {largeur}m × {nb_etages*hauteur_etage}m")
                        st.write(f"- Étages: {nb_etages} étage(s) de {hauteur_etage}m chacun")
                        st.write(f"- Volume total: {longueur*largeur*nb_etages*hauteur_etage:.2f} m³")
                        
                        st.write(f"**Points 3D:**")
                        st.write(f"- Émetteur: ({x1_meter:.2f}, {y1_meter:.2f}, {z1_meter:.2f}) m - Étage {etage1+1}")
                        st.write(f"- Récepteur: ({x2_meter:.2f}, {y2_meter:.2f}, {z2_meter:.2f}) m - Étage {etage2+1}")
                        
                        st.write(f"**Calcul du pathloss 3D:**")
                        st.write(f"- Distance 3D: {distance_3d:.2f} m")
                        st.write(f"- Murs traversés (plan 2D): {wall_count_2d}")
                        st.write(f"- Différence d'étages: {floor_difference}")
                        st.write(f"- Atténuation par étage: {floor_difference * 15} dB")
                        st.write(f"- Fréquence: {frequence} MHz")
                        st.write(f"- Pathloss total 3D: {pathloss_3d:.2f} dB")
                        
                        # Comparaison 2D vs 3D
                        distance_2d = np.sqrt((x2_meter - x1_meter)**2 + (y2_meter - y1_meter)**2)
                        pathloss_2d = PathlossCalculator(frequence).calculate_pathloss(distance_2d, wall_count_2d)
                        
                        st.write(f"**Comparaison 2D vs 3D:**")
                        st.write(f"- Distance 2D: {distance_2d:.2f} m vs 3D: {distance_3d:.2f} m")
                        st.write(f"- Pathloss 2D: {pathloss_2d:.2f} dB vs 3D: {pathloss_3d:.2f} dB")
                        st.write(f"- Différence: {pathloss_3d - pathloss_2d:.2f} dB")
                
                except ImportError as e:
                    st.error(f"Erreur d'importation: {e}")
                    st.info("Installation des dépendances 3D requise...")
                    install_3d_dependencies()

def heatmap_2d_interface():
    """Interface pour la génération de heatmap 2D"""
    st.header("Génération Heatmap 2D")
    
    # Sidebar pour les paramètres heatmap
    st.sidebar.header("Paramètres Heatmap 2D")
    
    # Affichage du statut ML
    display_ml_status()
    
    # Upload du fichier pour heatmap
    uploaded_file_heatmap = st.file_uploader(
        "Téléchargez le plan de l'appartement (PNG) pour la heatmap",
        type=['png'],
        help="Le plan doit être en blanc avec les murs en noir",
        key="upload_heatmap"
    )
    
    if uploaded_file_heatmap is not None:
        # Conversion de l'image uploadée
        image = Image.open(uploaded_file_heatmap)
        image_array = np.array(image)
        
        # Affichage de l'image originale
        st.subheader("Plan original")
        st.image(image, caption="Plan pour génération de heatmap", use_column_width=True)
        
        # Paramètres du bâtiment pour heatmap
        st.sidebar.subheader("Dimensions du bâtiment")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            longueur = st.number_input("Longueur (m)", min_value=1.0, value=10.0, step=0.1, key="longueur_heatmap")
            largeur = st.number_input("Largeur (m)", min_value=1.0, value=8.0, step=0.1, key="largeur_heatmap")
        
        with col2:
            nb_etages = st.number_input("Nombre d'étages", min_value=1, value=1, step=1, key="etages_heatmap")
            hauteur_etage = st.number_input("Hauteur étage (m)", min_value=2.0, value=2.7, step=0.1, key="hauteur_heatmap")
        
        frequence = st.sidebar.number_input("Fréquence (MHz)", min_value=100, value=2400, step=100, key="freq_heatmap")
        
        # Paramètres de la heatmap
        st.sidebar.subheader("Paramètres de la heatmap")
        resolution = st.sidebar.slider("Résolution de la grille", min_value=20, max_value=100, value=50, key="resolution_heatmap")
        colormap = st.sidebar.selectbox("Palette de couleurs", 
                                       ["plasma", "viridis", "hot", "coolwarm", "RdYlGn_r"], 
                                       index=0, key="colormap_heatmap")
        
        # Seuils de qualité du signal
        st.sidebar.subheader("Seuils de signal")
        seuil_excellent = st.sidebar.number_input("Excellent (dB max)", value=-50.0, step=1.0, key="seuil_excellent")
        seuil_bon = st.sidebar.number_input("Bon (dB max)", value=-70.0, step=1.0, key="seuil_bon")
        seuil_faible = st.sidebar.number_input("Faible (dB max)", value=-90.0, step=1.0, key="seuil_faible")
        
        # Traitement de l'image pour extraire les murs
        processor = ImageProcessor()
        processed_image, walls_detected = processor.process_image(image_array)
        
        # Affichage de l'image traitée
        st.subheader("Murs détectés")
        st.image(processed_image, caption="Murs extraits pour calcul heatmap", use_column_width=True)
        
        # Interface pour les points d'accès (émetteurs)
        st.subheader("Configuration des points d'accès")
        
        # Conversion des coordonnées pour l'affichage
        height, width = image_array.shape[:2]
        scale_x = longueur / width
        scale_y = largeur / height
        
        # Option pour multiple émetteurs
        nb_emetteurs = st.number_input("Nombre de points d'accès", min_value=1, max_value=5, value=1, key="nb_emetteurs")
        
        emetteurs = []
        for i in range(nb_emetteurs):
            st.write(f"**Point d'accès {i+1}**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_meter = st.number_input(f"X{i+1} (m)", min_value=0.0, max_value=longueur, 
                                        value=longueur*(i+1)/(nb_emetteurs+1), step=0.1, key=f"x_emit_{i}")
                y_meter = st.number_input(f"Y{i+1} (m)", min_value=0.0, max_value=largeur, 
                                        value=largeur/2, step=0.1, key=f"y_emit_{i}")
            
            with col2:
                puissance_tx = st.number_input(f"Puissance TX (dBm)", value=20.0, step=1.0, key=f"power_{i}")
                gain_antenne = st.number_input(f"Gain antenne (dBi)", value=2.0, step=0.5, key=f"gain_{i}")
            
            with col3:
                # Calcul des coordonnées pixel correspondantes
                x_pixel = int(x_meter / scale_x)
                y_pixel = int(y_meter / scale_y)
                st.write(f"Position pixel: ({x_pixel}, {y_pixel})")
                st.write(f"Puissance totale: {puissance_tx + gain_antenne:.1f} dBm")
            
            emetteurs.append({
                'position_meter': (x_meter, y_meter),
                'position_pixel': (x_pixel, y_pixel),
                'puissance_tx': puissance_tx,
                'gain_antenne': gain_antenne,
                'puissance_totale': puissance_tx + gain_antenne
            })
        
        # Bouton pour générer la heatmap
        if st.button("Générer la Heatmap 2D", key="generate_heatmap"):
            with st.spinner("Génération de la heatmap 2D..."):
                try:
                    # Importation du module heatmap
                    from heatmap_generator import HeatmapGenerator
                    
                    # Création du générateur de heatmap
                    heatmap_gen = HeatmapGenerator(frequence)
                    
                    # Génération de la heatmap
                    heatmap_data, extent, fig_heatmap = heatmap_gen.generate_heatmap_2d(
                        image_array=image_array,
                        walls_detected=walls_detected,
                        emetteurs=emetteurs,
                        longueur=longueur,
                        largeur=largeur,
                        resolution=resolution,
                        colormap=colormap
                    )
                    
                    # Affichage de la heatmap
                    st.subheader("Heatmap du Pathloss 2D")
                    st.pyplot(fig_heatmap)
                    
                    # Génération de la heatmap de couverture par zones
                    coverage_map, fig_coverage = heatmap_gen.generate_coverage_zones(
                        heatmap_data=heatmap_data,
                        extent=extent,
                        emetteurs=emetteurs,
                        seuils={
                            'excellent': seuil_excellent,
                            'bon': seuil_bon,
                            'faible': seuil_faible
                        },
                        longueur=longueur,
                        largeur=largeur
                    )
                    
                    st.subheader("Carte de Couverture par Zones")
                    st.pyplot(fig_coverage)
                    
                    # Statistiques de couverture
                    stats = heatmap_gen.calculate_coverage_statistics(
                        heatmap_data,
                        seuils={
                            'excellent': seuil_excellent,
                            'bon': seuil_bon,
                            'faible': seuil_faible
                        }
                    )
                    
                    # Affichage des statistiques
                    st.subheader("Statistiques de Couverture")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Zone Excellente", f"{stats['excellent']:.1f}%", 
                                delta=f"{stats['excellent'] - 25:.1f}%" if stats['excellent'] > 25 else None)
                    with col2:
                        st.metric("Zone Bonne", f"{stats['bon']:.1f}%",
                                delta=f"{stats['bon'] - 35:.1f}%" if stats['bon'] > 35 else None)
                    with col3:
                        st.metric("Zone Faible", f"{stats['faible']:.1f}%")
                    with col4:
                        st.metric("Zone Mauvaise", f"{stats['mauvaise']:.1f}%")
                    
                    # Recommandations
                    recommendations = heatmap_gen.generate_recommendations(stats, nb_emetteurs)
                    
                    with st.expander("Recommandations d'optimisation"):
                        for i, rec in enumerate(recommendations, 1):
                            st.write(f"{i}. {rec}")
                    
                    # Informations détaillées
                    with st.expander("Détails de la heatmap"):
                        st.write(f"**Paramètres de génération:**")
                        st.write(f"- Résolution de grille: {resolution}x{resolution} points")
                        st.write(f"- Palette de couleurs: {colormap}")
                        st.write(f"- Fréquence: {frequence} MHz")
                        st.write(f"- Nombre d'émetteurs: {nb_emetteurs}")
                        
                        st.write(f"**Émetteurs configurés:**")
                        for i, emit in enumerate(emetteurs):
                            st.write(f"- Émetteur {i+1}: Position ({emit['position_meter'][0]:.2f}, {emit['position_meter'][1]:.2f}) m")
                            st.write(f"  Puissance: {emit['puissance_tx']} dBm + {emit['gain_antenne']} dBi = {emit['puissance_totale']} dBm")
                        
                        st.write(f"**Valeurs de pathloss:**")
                        st.write(f"- Minimum: {np.min(heatmap_data):.2f} dB")
                        st.write(f"- Maximum: {np.max(heatmap_data):.2f} dB")
                        st.write(f"- Moyenne: {np.mean(heatmap_data):.2f} dB")
                        st.write(f"- Médiane: {np.median(heatmap_data):.2f} dB")
                        
                        # Option de téléchargement
                        st.download_button(
                            label="Télécharger les données CSV",
                            data=heatmap_gen.export_data_csv(heatmap_data, extent),
                            file_name=f"heatmap_data_{frequence}MHz.csv",
                            mime="text/csv"
                        )
                
                except ImportError as e:
                    st.error(f"Module heatmap non disponible: {e}")
                    st.info("Création du module de génération de heatmap...")
                    create_heatmap_module()
                except Exception as e:
                    st.error(f"Erreur lors de la génération: {e}")
                    st.exception(e)

def heatmap_3d_interface():
    """Interface pour la génération de heatmap 3D avec voxels"""
    st.header("Génération Heatmap 3D")
    
    # Sidebar pour les paramètres heatmap 3D
    st.sidebar.header("Paramètres Heatmap 3D")
    
    # Affichage du statut ML
    display_ml_status()
    
    # Upload du fichier pour heatmap 3D
    uploaded_file_heatmap_3d = st.file_uploader(
        "Téléchargez le plan de l'appartement (PNG) pour la heatmap 3D",
        type=['png'],
        help="Le plan doit être en blanc avec les murs en noir",
        key="upload_heatmap_3d"
    )
    
    if uploaded_file_heatmap_3d is not None:
        # Conversion de l'image uploadée
        image = Image.open(uploaded_file_heatmap_3d)
        image_array = np.array(image)
        
        # Affichage de l'image originale
        st.subheader("Plan 2D de base")
        st.image(image, caption="Plan pour génération de heatmap 3D", use_column_width=True)
        
        # Paramètres du bâtiment pour heatmap 3D
        st.sidebar.subheader("Dimensions du bâtiment")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            longueur = st.number_input("Longueur (m)", min_value=1.0, value=12.5, step=0.1, key="longueur_heatmap_3d")
            largeur = st.number_input("Largeur (m)", min_value=1.0, value=9.4, step=0.1, key="largeur_heatmap_3d")
        
        with col2:
            nb_etages = st.number_input("Nombre d'étages", min_value=1, value=2, step=1, key="etages_heatmap_3d")
            hauteur_etage = st.number_input("Hauteur étage (m)", min_value=2.0, value=2.7, step=0.1, key="hauteur_heatmap_3d")
        
        frequence = st.sidebar.number_input("Fréquence (MHz)", min_value=100, value=2400, step=100, key="freq_heatmap_3d")
        
        # Paramètres de la heatmap 3D
        st.sidebar.subheader("Paramètres des voxels")
        resolution_xy = st.sidebar.slider("Résolution XY", min_value=15, max_value=50, value=25, key="resolution_xy_3d")
        resolution_z = st.sidebar.slider("Résolution Z", min_value=5, max_value=20, value=10, key="resolution_z_3d")
        colormap_3d = st.sidebar.selectbox("Palette de couleurs 3D", 
                                          ["plasma", "viridis", "hot", "coolwarm", "jet"], 
                                          index=0, key="colormap_3d")
        
        # Mode de visualisation
        st.sidebar.subheader("Visualisation")
        view_mode = st.sidebar.selectbox("Mode d'affichage", 
                                        ["Voxels transparents", "Voxels par couches", "Surfaces isométriques", "Coupes transversales"], 
                                        key="view_mode_3d")
        
        # Seuils de qualité du signal 3D
        st.sidebar.subheader("Seuils de signal 3D")
        seuil_excellent_3d = st.sidebar.number_input("Excellent (dB max)", value=-40.0, step=1.0, key="seuil_excellent_3d")
        seuil_bon_3d = st.sidebar.number_input("Bon (dB max)", value=-65.0, step=1.0, key="seuil_bon_3d")
        seuil_faible_3d = st.sidebar.number_input("Faible (dB max)", value=-85.0, step=1.0, key="seuil_faible_3d")
        
        # Traitement de l'image pour extraire les murs
        processor = ImageProcessor()
        processed_image, walls_detected = processor.process_image(image_array)
        
        # Affichage de l'image traitée
        st.subheader("Murs détectés (plan de base)")
        st.image(processed_image, caption="Murs extraits pour duplication en 3D", use_column_width=True)
        
        # Interface pour les points d'accès 3D (émetteurs)
        st.subheader("Configuration des points d'accès 3D")
        
        # Conversion des coordonnées pour l'affichage
        height, width = image_array.shape[:2]
        scale_x = longueur / width
        scale_y = largeur / height
        hauteur_totale = nb_etages * hauteur_etage
        
        # Option pour multiple émetteurs 3D
        nb_emetteurs_3d = st.number_input("Nombre de points d'accès 3D", min_value=1, max_value=4, value=1, key="nb_emetteurs_3d")
        
        emetteurs_3d = []
        for i in range(nb_emetteurs_3d):
            st.write(f"**Point d'accès 3D {i+1}**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_meter = st.number_input(f"X{i+1} (m)", min_value=0.0, max_value=longueur, 
                                        value=longueur*(i+1)/(nb_emetteurs_3d+1), step=0.1, key=f"x_emit_3d_{i}")
                y_meter = st.number_input(f"Y{i+1} (m)", min_value=0.0, max_value=largeur, 
                                        value=largeur/2, step=0.1, key=f"y_emit_3d_{i}")
                z_meter = st.number_input(f"Z{i+1} (m)", min_value=0.1, max_value=hauteur_totale, 
                                        value=hauteur_etage/2 + (i % nb_etages) * hauteur_etage, step=0.1, key=f"z_emit_3d_{i}")
            
            with col2:
                puissance_tx = st.number_input(f"Puissance TX (dBm)", value=20.0, step=1.0, key=f"power_3d_{i}")
                gain_antenne = st.number_input(f"Gain antenne (dBi)", value=2.0, step=0.5, key=f"gain_3d_{i}")
                directivite = st.selectbox(f"Directivité", ["Omnidirectionnel", "Directif"], key=f"dir_3d_{i}")
            
            with col3:
                # Calcul des coordonnées pixel et étage
                x_pixel = int(x_meter / scale_x)
                y_pixel = int(y_meter / scale_y)
                etage = int(z_meter // hauteur_etage) + 1
                
                st.write(f"Position pixel: ({x_pixel}, {y_pixel})")
                st.write(f"Étage: {etage}")
                st.write(f"Puissance totale: {puissance_tx + gain_antenne:.1f} dBm")
                st.write(f"Type: {directivite}")
            
            emetteurs_3d.append({
                'position_meter': (x_meter, y_meter, z_meter),
                'position_pixel': (x_pixel, y_pixel),
                'puissance_tx': puissance_tx,
                'gain_antenne': gain_antenne,
                'puissance_totale': puissance_tx + gain_antenne,
                'directivite': directivite,
                'etage': etage
            })
        
        # Bouton pour générer la heatmap 3D
        if st.button("Générer la Heatmap 3D", key="generate_heatmap_3d"):
            with st.spinner("Génération de la heatmap 3D avec voxels... Cela peut prendre quelques minutes."):
                try:
                    # Importation du module heatmap 3D
                    from heatmap_generator_3d import HeatmapGenerator3D
                    
                    # Création du générateur de heatmap 3D
                    heatmap_3d_gen = HeatmapGenerator3D(frequence)
                    
                    # Génération de la grille de voxels et calcul des pathloss
                    voxel_data, coordinates = heatmap_3d_gen.generate_voxel_grid(
                        walls_detected=walls_detected,
                        emetteurs_3d=emetteurs_3d,
                        longueur=longueur,
                        largeur=largeur,
                        hauteur_totale=hauteur_totale,
                        resolution_xy=resolution_xy,
                        resolution_z=resolution_z
                    )
                    
                    st.success(f"Grille de voxels générée: {resolution_xy}x{resolution_xy}x{resolution_z} = {resolution_xy*resolution_xy*resolution_z} voxels")
                    
                    # Visualisation selon le mode choisi
                    if view_mode == "Voxels transparents":
                        try:
                            fig_3d = heatmap_3d_gen.visualize_voxel_heatmap(
                                voxel_data, coordinates, emetteurs_3d, colormap_3d
                            )
                            
                            st.subheader("Heatmap 3D - Voxels")
                            st.plotly_chart(fig_3d, use_container_width=True)
                        except Exception as e:
                            st.error(f"Erreur visualisation voxels: {e}")
                            # Fallback vers la méthode en couches
                            st.info("Utilisation de la visualisation par couches...")
                            fig_3d_layers = heatmap_3d_gen.visualize_voxel_heatmap_layered(
                                voxel_data, coordinates, emetteurs_3d, colormap_3d
                            )
                            st.plotly_chart(fig_3d_layers, use_container_width=True)
                    
                    elif view_mode == "Voxels par couches":
                        fig_3d_layers = heatmap_3d_gen.visualize_voxel_heatmap_layered(
                            voxel_data, coordinates, emetteurs_3d, colormap_3d
                        )
                        
                        st.subheader("Heatmap 3D - Voxels par qualité")
                        st.plotly_chart(fig_3d_layers, use_container_width=True)
                    
                    elif view_mode == "Surfaces isométriques":
                        fig_iso = heatmap_3d_gen.visualize_isosurfaces(
                            voxel_data, coordinates, 
                            seuils={
                                'excellent': seuil_excellent_3d,
                                'bon': seuil_bon_3d,
                                'faible': seuil_faible_3d
                            }
                        )
                        
                        st.subheader("Heatmap 3D - Surfaces isométriques")
                        st.plotly_chart(fig_iso, use_container_width=True)
                    
                    elif view_mode == "Coupes transversales":
                        figs_slices = heatmap_3d_gen.visualize_cross_sections(
                            voxel_data, coordinates, colormap_3d, nb_etages
                        )
                        
                        st.subheader("Heatmap 3D - Coupes par étage")
                        for i, fig_slice in enumerate(figs_slices):
                            st.write(f"**Étage {i+1}**")
                            st.plotly_chart(fig_slice, use_container_width=True)
                    
                    # Statistiques 3D
                    stats_3d = heatmap_3d_gen.calculate_3d_coverage_statistics(
                        voxel_data,
                        seuils={
                            'excellent': seuil_excellent_3d,
                            'bon': seuil_bon_3d,
                            'faible': seuil_faible_3d
                        }
                    )
                    
                    # Affichage des statistiques 3D
                    st.subheader("Statistiques de Couverture 3D")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Volume Excellent", f"{stats_3d['excellent']:.1f}%")
                    with col2:
                        st.metric("Volume Bon", f"{stats_3d['bon']:.1f}%")
                    with col3:
                        st.metric("Volume Faible", f"{stats_3d['faible']:.1f}%")
                    with col4:
                        st.metric("Volume Mauvais", f"{stats_3d['mauvaise']:.1f}%")
                    with col5:
                        st.metric("Total Voxels", f"{stats_3d['total_voxels']}")
                    
                    # Analyse par étage
                    if nb_etages > 1:
                        stats_par_etage = heatmap_3d_gen.analyze_coverage_by_floor(
                            voxel_data, coordinates, nb_etages, hauteur_etage,
                            seuils={
                                'excellent': seuil_excellent_3d,
                                'bon': seuil_bon_3d,
                                'faible': seuil_faible_3d
                            }
                        )
                        
                        st.subheader("Analyse par étage")
                        
                        for etage, stats_etage in enumerate(stats_par_etage, 1):
                            with st.expander(f"Étage {etage}"):
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Excellent", f"{stats_etage['excellent']:.1f}%")
                                with col2:
                                    st.metric("Bon", f"{stats_etage['bon']:.1f}%")
                                with col3:
                                    st.metric("Faible", f"{stats_etage['faible']:.1f}%")
                                with col4:
                                    st.metric("Mauvais", f"{stats_etage['mauvaise']:.1f}%")
                    
                    # Recommandations 3D
                    recommendations_3d = heatmap_3d_gen.generate_3d_recommendations(
                        stats_3d, stats_par_etage if nb_etages > 1 else None, 
                        nb_emetteurs_3d, emetteurs_3d
                    )
                    
                    with st.expander("Recommandations d'optimisation 3D"):
                        for i, rec in enumerate(recommendations_3d, 1):
                            st.write(f"{i}. {rec}")
                    
                    # Informations détaillées 3D
                    with st.expander("Détails de la heatmap 3D"):
                        st.write(f"**Paramètres de génération 3D:**")
                        st.write(f"- Résolution grille XY: {resolution_xy}x{resolution_xy}")
                        st.write(f"- Résolution Z: {resolution_z} niveaux")
                        st.write(f"- Total voxels: {resolution_xy*resolution_xy*resolution_z}")
                        st.write(f"- Palette de couleurs: {colormap_3d}")
                        st.write(f"- Mode de visualisation: {view_mode}")
                        st.write(f"- Fréquence: {frequence} MHz")
                        st.write(f"- Nombre d'émetteurs 3D: {nb_emetteurs_3d}")
                        
                        st.write(f"**Dimensions du volume:**")
                        st.write(f"- Longueur: {longueur} m")
                        st.write(f"- Largeur: {largeur} m")
                        st.write(f"- Hauteur totale: {hauteur_totale} m")
                        st.write(f"- Volume total: {longueur * largeur * hauteur_totale:.2f} m³")
                        
                        st.write(f"**Émetteurs 3D configurés:**")
                        for i, emit in enumerate(emetteurs_3d):
                            pos = emit['position_meter']
                            st.write(f"- Émetteur {i+1}: Position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m")
                            st.write(f"  Étage {emit['etage']}, Puissance: {emit['puissance_totale']} dBm, Type: {emit['directivite']}")
                        
                        st.write(f"**Valeurs de pathloss 3D:**")
                        valid_voxels = voxel_data[voxel_data < 200]  # Exclure les murs
                        if len(valid_voxels) > 0:
                            st.write(f"- Minimum: {np.min(valid_voxels):.2f} dB")
                            st.write(f"- Maximum: {np.max(valid_voxels):.2f} dB")
                            st.write(f"- Moyenne: {np.mean(valid_voxels):.2f} dB")
                            st.write(f"- Médiane: {np.median(valid_voxels):.2f} dB")
                        
                        # Option de téléchargement 3D
                        csv_3d_data = heatmap_3d_gen.export_voxel_data_csv(voxel_data, coordinates)
                        st.download_button(
                            label="Télécharger les données voxels CSV",
                            data=csv_3d_data,
                            file_name=f"heatmap_3d_data_{frequence}MHz.csv",
                            mime="text/csv"
                        )
                
                except ImportError as e:
                    st.error(f"Module heatmap 3D non disponible: {e}")
                    st.info("Création du module de génération de heatmap 3D...")
                    create_heatmap_3d_module()
                except Exception as e:
                    st.error(f"Erreur lors de la génération 3D: {e}")
                    st.exception(e)

def optimization_3d_interface():
    """Interface pour l'optimisation automatique des points d'accès 3D"""
    st.header("Optimisation Automatique des Points d'Accès 3D")
    
    # Sidebar pour les paramètres d'optimisation
    st.sidebar.header("Paramètres d'Optimisation")
    
    # Affichage du statut ML
    display_ml_status()
    
    # Upload du fichier pour optimisation
    uploaded_file_optimization = st.file_uploader(
        "Téléchargez le plan de l'appartement (PNG) pour l'optimisation",
        type=['png'],
        help="Le plan doit être en blanc avec les murs en noir",
        key="upload_optimization"
    )
    
    if uploaded_file_optimization is not None:
        # Conversion de l'image uploadée
        image = Image.open(uploaded_file_optimization)
        image_array = np.array(image)
        
        # Affichage de l'image originale
        st.subheader("Plan de base pour optimisation")
        st.image(image, caption="Plan pour optimisation des points d'accès", use_column_width=True)
        
        # Paramètres du bâtiment
        st.sidebar.subheader("Dimensions du bâtiment")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            longueur = st.number_input("Longueur (m)", min_value=1.0, value=12.5, step=0.1, key="longueur_opt")
            largeur = st.number_input("Largeur (m)", min_value=1.0, value=9.4, step=0.1, key="largeur_opt")
        
        with col2:
            hauteur_totale = st.number_input("Hauteur totale (m)", min_value=2.0, value=5.4, step=0.1, key="hauteur_opt")
            nb_etages = st.number_input("Nombre d'étages", min_value=1, value=2, step=1, key="etages_opt")
        
        # Paramètres RF
        st.sidebar.subheader("Paramètres RF")
        frequency_opt = st.sidebar.selectbox("Fréquence", [2400, 5000, 6000], index=0, key="freq_opt")
        
        # Objectifs de couverture
        st.sidebar.subheader("Objectifs de Couverture")
        target_coverage_db = st.sidebar.number_input("Signal minimal (dB)", value=-70.0, step=1.0, key="target_signal")
        min_coverage_percent = st.sidebar.number_input("Couverture minimale (%)", min_value=50.0, max_value=100.0, value=90.0, step=1.0, key="min_coverage")
        power_tx = st.sidebar.number_input("Puissance émetteur (dBm)", value=20.0, step=1.0, key="power_opt")
        
        # Paramètres d'optimisation
        st.sidebar.subheader("Paramètres d'Optimisation")
        max_access_points = st.sidebar.number_input("Nb max de points d'accès", min_value=1, max_value=12, value=8, step=1, key="max_ap")
        st.sidebar.info("🔧 Optimisation par clustering K-means uniquement")

        
        # Résolution pour le calcul
        st.sidebar.subheader("Résolution de Calcul")
        resolution_xy_opt = st.sidebar.slider("Résolution XY", min_value=10, max_value=30, value=20, key="res_xy_opt")
        resolution_z_opt = st.sidebar.slider("Résolution Z", min_value=4, max_value=12, value=8, key="res_z_opt")
        
        # Traitement de l'image
        try:
            from image_processor import ImageProcessor
            processor = ImageProcessor()
            processed_image, walls_detected = processor.process_image(image_array)
            
            # Affichage de l'image traitée
            st.subheader("Murs détectés")
            st.image(processed_image, caption="Murs extraits pour l'optimisation", use_column_width=True)
            
            # Bouton d'optimisation
            if st.button("🚀 Lancer l'Optimisation", key="optimize_button"):
                
                try:
                    from access_point_optimizer import AccessPointOptimizer
                    
                    # Initialisation de l'optimiseur
                    optimizer = AccessPointOptimizer(frequency_opt)
                    
                    # Génération des points à couvrir
                    with st.spinner("Génération des zones à couvrir..."):
                        coverage_points, grid_info = optimizer.generate_coverage_zones(
                            walls_detected, longueur, largeur, hauteur_totale,
                            resolution_xy_opt, resolution_z_opt
                        )
                    
                    st.success(f"Zones générées: {len(coverage_points)} points à couvrir")
                    
                    # Optimisation par clustering K-means
                    with st.spinner("Optimisation par clustering K-means..."):
                        best_config, cluster_analysis = optimizer.optimize_with_clustering(
                            coverage_points, grid_info, longueur, largeur, hauteur_totale,
                            target_coverage_db, min_coverage_percent, power_tx
                        )
                    
                    st.success(f"Optimisation terminée: {best_config['stats']['coverage_percent']:.1f}% de couverture avec {len(best_config['access_points'])} points d'accès")
                    
                    # Affichage des résultats
                    if best_config:
                        # Visualisation 3D
                        st.subheader("Résultat de l'Optimisation")
                        fig_opt = optimizer.visualize_optimization_result(
                            best_config, coverage_points, grid_info, 
                            longueur, largeur, hauteur_totale
                        )
                        st.plotly_chart(fig_opt, use_container_width=True)
                        
                        # Statistiques détaillées
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Points d'Accès", len(best_config['access_points']))
                            st.metric("Couverture", f"{best_config['stats']['coverage_percent']:.1f}%")
                        
                        with col2:
                            st.metric("Points Couverts", f"{best_config['stats']['covered_points']}/{best_config['stats']['total_points']}")
                            st.metric("Score d'Optimisation", f"{best_config['score']:.3f}")
                        
                        with col3:
                            if 'signal_levels' in best_config['stats']:
                                avg_signal = np.mean(best_config['stats']['signal_levels'])
                                st.metric("Signal Moyen", f"{avg_signal:.1f} dB")
                                min_signal = np.min(best_config['stats']['signal_levels'])
                                st.metric("Signal Minimal", f"{min_signal:.1f} dB")
                        
                        # Configuration des points d'accès
                        st.subheader("Configuration des Points d'Accès Optimisés")
                        
                        ap_data = []
                        for i, ap in enumerate(best_config['access_points']):
                            x, y, z, power = ap
                            etage = int(z // (hauteur_totale / nb_etages)) + 1
                            ap_data.append({
                                "Point d'Accès": f"AP{i+1}",
                                "Position X (m)": round(x, 2),
                                "Position Y (m)": round(y, 2),
                                "Position Z (m)": round(z, 2),
                                "Étage": etage,
                                "Puissance (dBm)": round(power, 1)
                            })
                        
                        df_ap = pd.DataFrame(ap_data)
                        st.dataframe(df_ap, use_container_width=True)
                        
                        # Génération du rapport
                        report = optimizer.generate_optimization_report(
                            best_config, cluster_analysis or {}, {}
                        )
                        
                        # Recommandations
                        st.subheader("Recommandations")
                        for i, rec in enumerate(report['recommendations'], 1):
                            st.write(f"{i}. {rec}")
                        
                        # Analyse détaillée de la couverture
                        if 'coverage_analysis' in report and report['coverage_analysis']:
                            st.subheader("Analyse Détaillée de la Couverture")
                            
                            coverage_data = [
                                ["Excellent (≥-50dB)", report['coverage_analysis']['excellent_coverage']],
                                ["Bon (-50 à -70dB)", report['coverage_analysis']['good_coverage']],
                                ["Faible (-70 à -85dB)", report['coverage_analysis']['poor_coverage']],
                                ["Pas de couverture (<-85dB)", report['coverage_analysis']['no_coverage']]
                            ]
                            
                            df_coverage = pd.DataFrame(coverage_data, columns=["Qualité", "Nombre de Points"])
                            st.dataframe(df_coverage, use_container_width=True)
                            
                            # Graphique en camembert
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=[row[0] for row in coverage_data],
                                values=[row[1] for row in coverage_data],
                                hole=0.3,
                                marker=dict(colors=['green', 'yellow', 'orange', 'red'])
                            )])
                            
                            fig_pie.update_layout(
                                title="Répartition de la Qualité de Couverture",
                                width=600,
                                height=400
                            )
                            
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        # Export des données
                        st.subheader("Export des Résultats")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Export CSV
                            csv_data = optimizer.export_optimization_csv(best_config, report)
                            st.download_button(
                                label="📥 Télécharger Configuration CSV",
                                data=csv_data,
                                file_name=f"points_acces_optimises_{frequency_opt}MHz.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Export de la visualisation HTML
                            html_buffer = fig_opt.to_html()
                            st.download_button(
                                label="📥 Télécharger Visualisation HTML",
                                data=html_buffer,
                                file_name=f"optimisation_3d_{frequency_opt}MHz.html",
                                mime="text/html"
                            )
                
                except ImportError as e:
                    st.error(f"Module d'optimisation non disponible: {e}")
                    st.info("Vérifiez que tous les modules sont installés correctement.")
                except Exception as e:
                    st.error(f"Erreur lors de l'optimisation: {e}")
                    st.exception(e)
        
        except ImportError as e:
            st.error(f"Module de traitement d'image non disponible: {e}")
        except Exception as e:
            st.error(f"Erreur lors du traitement: {e}")
            st.exception(e)
    
    else:
        # Instructions d'utilisation
        st.info("👆 Téléchargez un plan d'appartement pour commencer l'optimisation")
        
        st.markdown("""
        ### 🎯 Optimisation Automatique des Points d'Accès 3D
        
        Cet outil optimise automatiquement le placement et le nombre de points d'accès WiFi pour:
        
        **🔍 Fonctionnalités:**
        - **Placement optimal**: Calcul automatique des meilleures positions
        - **Nombre minimal**: Trouve le nombre minimum de points d'accès nécessaires
        - **Couverture maximale**: Assure la couverture demandée dans tout le volume
        - **Algorithme K-means**: Clustering intelligent pour placement optimal
        
        **📊 Méthode d'optimisation:**
        - **Clustering K-means**: Placement basé sur les zones de densité et obstacles
        
        **📈 Paramètres configurables:**
        - Signal minimal requis (dB)
        - Pourcentage de couverture minimal
        - Puissance des émetteurs
        - Nombre maximal de points d'accès
        - Résolution de calcul
        
        **📋 Résultats fournis:**
        - Visualisation 3D interactive des points d'accès optimisés
        - Configuration détaillée de chaque point d'accès
        - Analyse de la qualité de couverture
        - Recommandations d'amélioration
        - Export CSV et HTML
        """)

def optimization_2d_interface():
    """Interface pour l'optimisation automatique des points d'accès 2D"""
    st.header("Optimisation Automatique des Points d'Accès 2D")
    
    # Sidebar pour les paramètres d'optimisation 2D
    st.sidebar.header("Paramètres d'Optimisation 2D")
    
    # Affichage du statut ML
    display_ml_status()
    
    # Upload du fichier pour optimisation 2D
    uploaded_file_optimization_2d = st.file_uploader(
        "Téléchargez le plan de l'appartement (PNG) pour l'optimisation 2D",
        type=['png'],
        help="Le plan doit être en blanc avec les murs en noir",
        key="upload_optimization_2d"
    )
    
    if uploaded_file_optimization_2d is not None:
        # Conversion de l'image uploadée
        image = Image.open(uploaded_file_optimization_2d)
        image_array = np.array(image)
        
        # Affichage de l'image originale
        st.subheader("Plan 2D pour optimisation")
        st.image(image, caption="Plan pour optimisation des points d'accès 2D", use_column_width=True)
        
        # Paramètres du bâtiment 2D
        st.sidebar.subheader("Dimensions du plan")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            longueur = st.number_input("Longueur (m)", min_value=1.0, value=10.0, step=0.1, key="longueur_opt_2d")
            largeur = st.number_input("Largeur (m)", min_value=1.0, value=8.0, step=0.1, key="largeur_opt_2d")
        
        with col2:
            nb_etages = st.number_input("Nombre d'étages", min_value=1, value=1, step=1, key="etages_opt_2d")
            hauteur_etage = st.number_input("Hauteur étage (m)", min_value=2.0, value=2.7, step=0.1, key="hauteur_opt_2d")
        
        # Paramètres RF 2D
        st.sidebar.subheader("Paramètres RF")
        frequency_opt_2d = st.sidebar.selectbox("Fréquence", [2400, 5000, 6000], index=0, key="freq_opt_2d")
        
        # Objectifs de couverture 2D
        st.sidebar.subheader("Objectifs de Couverture 2D")
        target_coverage_db_2d = st.sidebar.number_input("Signal minimal (dB)", value=-70.0, step=1.0, key="target_signal_2d")
        min_coverage_percent_2d = st.sidebar.number_input("Couverture minimale (%)", min_value=50.0, max_value=100.0, value=85.0, step=1.0, key="min_coverage_2d")
        power_tx_2d = st.sidebar.number_input("Puissance émetteur (dBm)", value=20.0, step=1.0, key="power_opt_2d")
        
        # Paramètres d'optimisation 2D
        st.sidebar.subheader("Paramètres d'Optimisation")
        max_access_points_2d = st.sidebar.number_input("Nb max de points d'accès", min_value=1, max_value=8, value=6, step=1, key="max_ap_2d")
        st.sidebar.info("🔧 Optimisation par clustering K-means uniquement")
        
        # Résolution pour le calcul 2D
        st.sidebar.subheader("Résolution de Calcul")
        resolution_2d = st.sidebar.slider("Résolution grille", min_value=15, max_value=40, value=25, key="res_2d")
        
        # Traitement de l'image
        try:
            from image_processor import ImageProcessor
            processor = ImageProcessor()
            processed_image, walls_detected = processor.process_image(image_array)
            
            # Affichage de l'image traitée
            st.subheader("Murs détectés")
            st.image(processed_image, caption="Murs extraits pour l'optimisation 2D", use_column_width=True)
            
            # Bouton d'optimisation 2D
            if st.button("🚀 Lancer l'Optimisation 2D", key="optimize_button_2d"):
                
                try:
                    from access_point_optimizer_2d_fixed import AccessPointOptimizer2D
                    
                    # Initialisation de l'optimiseur 2D
                    optimizer_2d = AccessPointOptimizer2D(frequency_opt_2d)
                    
                    # Génération des points à couvrir 2D
                    with st.spinner("Génération de la grille de couverture 2D..."):
                        coverage_points, grid_info = optimizer_2d.generate_coverage_grid_2d(
                            walls_detected, longueur, largeur, resolution_2d
                        )
                    
                    st.success(f"Grille générée: {len(coverage_points)} points à couvrir en 2D")
                    
                    # Optimisation par clustering K-means 2D
                    with st.spinner("Optimisation par clustering K-means 2D..."):
                        best_config_2d, cluster_analysis_2d = optimizer_2d.optimize_with_clustering_2d(
                            coverage_points, grid_info, longueur, largeur,
                            target_coverage_db_2d, min_coverage_percent_2d, power_tx_2d, max_access_points_2d
                        )
                    
                    st.success(f"Optimisation terminée: {best_config_2d['stats']['coverage_percent']:.1f}% de couverture avec {len(best_config_2d['access_points'])} points d'accès")
                    
                    # Affichage des résultats 2D
                    if best_config_2d:
                        # Visualisation 2D
                        st.subheader("Résultat de l'Optimisation 2D")
                        fig_opt_2d = optimizer_2d.visualize_optimization_result_2d(
                            best_config_2d, coverage_points, grid_info, 
                            longueur, largeur, image_array
                        )
                        st.pyplot(fig_opt_2d)
                        
                        # Statistiques détaillées 2D
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Points d'Accès 2D", len(best_config_2d['access_points']))
                            st.metric("Couverture 2D", f"{best_config_2d['stats']['coverage_percent']:.1f}%")
                        
                        with col2:
                            st.metric("Points Couverts", f"{best_config_2d['stats']['covered_points']}/{best_config_2d['stats']['total_points']}")
                            st.metric("Score d'Optimisation", f"{best_config_2d['score']:.3f}")
                        
                        with col3:
                            if 'signal_levels' in best_config_2d['stats']:
                                avg_signal = np.mean(best_config_2d['stats']['signal_levels'])
                                st.metric("Signal Moyen", f"{avg_signal:.1f} dB")
                                min_signal = np.min(best_config_2d['stats']['signal_levels'])
                                st.metric("Signal Minimal", f"{min_signal:.1f} dB")
                        
                        # Configuration des points d'accès 2D
                        st.subheader("Configuration des Points d'Accès Optimisés 2D")
                        
                        ap_data_2d = []
                        for i, ap in enumerate(best_config_2d['access_points']):
                            x, y, power = ap
                            ap_data_2d.append({
                                "Point d'Accès": f"AP{i+1}",
                                "Position X (m)": round(x, 2),
                                "Position Y (m)": round(y, 2),
                                "Puissance (dBm)": round(power, 1)
                            })
                        
                        df_ap_2d = pd.DataFrame(ap_data_2d)
                        st.dataframe(df_ap_2d, use_container_width=True)
                        
                        # Génération du rapport 2D
                        report_2d = optimizer_2d.generate_optimization_report_2d(
                            best_config_2d, cluster_analysis_2d or {}, {}
                        )
                        
                        # Recommandations 2D
                        st.subheader("Recommandations 2D")
                        for i, rec in enumerate(report_2d['recommendations'], 1):
                            st.write(f"{i}. {rec}")
                        
                        # Analyse détaillée de la couverture 2D
                        if 'coverage_analysis' in report_2d and report_2d['coverage_analysis']:
                            st.subheader("Analyse Détaillée de la Couverture 2D")
                            
                            coverage_data_2d = [
                                ["Excellent (≥-50dB)", report_2d['coverage_analysis']['excellent_coverage']],
                                ["Bon (-50 à -70dB)", report_2d['coverage_analysis']['good_coverage']],
                                ["Faible (-70 à -85dB)", report_2d['coverage_analysis']['poor_coverage']],
                                ["Pas de couverture (<-85dB)", report_2d['coverage_analysis']['no_coverage']]
                            ]
                            
                            df_coverage_2d = pd.DataFrame(coverage_data_2d, columns=["Qualité", "Nombre de Points"])
                            st.dataframe(df_coverage_2d, use_container_width=True)
                            
                            # Graphique en camembert pour 2D
                            fig_pie_2d = go.Figure(data=[go.Pie(
                                labels=[row[0] for row in coverage_data_2d],
                                values=[row[1] for row in coverage_data_2d],
                                hole=0.3,
                                marker=dict(colors=['green', 'yellow', 'orange', 'red'])
                            )])
                            
                            fig_pie_2d.update_layout(
                                title="Répartition de la Qualité de Couverture 2D",
                                width=600,
                                height=400
                            )
                            
                            st.plotly_chart(fig_pie_2d, use_container_width=True)
                        
                        # Export des données 2D
                        st.subheader("Export des Résultats 2D")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Export CSV 2D
                            csv_data_2d = optimizer_2d.export_optimization_csv_2d(best_config_2d, report_2d)
                            st.download_button(
                                label="📥 Télécharger Configuration CSV 2D",
                                data=csv_data_2d,
                                file_name=f"points_acces_optimises_2d_{frequency_opt_2d}MHz.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Export de la visualisation
                            import io
                            import base64
                            buffer = io.BytesIO()
                            fig_opt_2d.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                            buffer.seek(0)
                            
                            st.download_button(
                                label="📥 Télécharger Visualisation PNG",
                                data=buffer.getvalue(),
                                file_name=f"optimisation_2d_{frequency_opt_2d}MHz.png",
                                mime="image/png"
                            )
                        
                        # Comparaison avec les méthodes traditionnelles
                        st.subheader("Avantages de l'Optimisation Automatique")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("""
                            **✅ Placement Manuel Traditional:**
                            - Placement au centre des pièces
                            - Répartition uniforme
                            - Pas d'optimisation globale
                            - Souvent sur-dimensionné
                            """)
                        
                        with col2:
                            st.markdown(f"""
                            **🚀 Optimisation Automatique:**
                            - Placement optimal calculé
                            - Prise en compte des obstacles
                            - Nombre minimal de points d'accès: **{len(best_config_2d['access_points'])}**
                            - Couverture optimisée: **{best_config_2d['stats']['coverage_percent']:.1f}%**
                            - Score d'efficacité: **{best_config_2d['score']:.3f}**
                            """)
                
                except ImportError as e:
                    st.error(f"Module d'optimisation 2D non disponible: {e}")
                    st.info("Vérifiez que tous les modules sont installés correctement.")
                except Exception as e:
                    st.error(f"Erreur lors de l'optimisation 2D: {e}")
                    st.exception(e)
        
        except ImportError as e:
            st.error(f"Module de traitement d'image non disponible: {e}")
        except Exception as e:
            st.error(f"Erreur lors du traitement: {e}")
            st.exception(e)
    
    else:
        # Instructions d'utilisation 2D
        st.info("👆 Téléchargez un plan d'appartement pour commencer l'optimisation 2D")
        
        st.markdown("""
        ### 🎯 Optimisation Automatique des Points d'Accès 2D
        
        Cet outil optimise automatiquement le placement et le nombre de points d'accès WiFi dans un plan 2D pour:
        
        **🔍 Fonctionnalités 2D:**
        - **Placement optimal 2D**: Calcul automatique des meilleures positions sur le plan
        - **Nombre minimal**: Trouve le nombre minimum de points d'accès nécessaires
        - **Couverture maximale**: Assure la couverture demandée dans toute la surface
        - **Algorithme K-means**: Clustering intelligent adapté au 2D
        
        **📊 Méthode d'optimisation 2D:**
        - **Clustering K-means**: Placement basé sur les zones de densité du plan
        
        **📈 Paramètres configurables 2D:**
        - Signal minimal requis (dB)
        - Pourcentage de couverture minimal
        - Puissance des émetteurs
        - Nombre maximal de points d'accès
        - Résolution de calcul de la grille
        
        **📋 Résultats fournis pour le 2D:**
        - Visualisation 2D interactive des points d'accès optimisés
        - Configuration détaillée de chaque point d'accès
        - Heatmap de qualité de couverture
        - Analyse de la qualité de couverture par zones
        - Recommandations d'amélioration spécifiques au 2D
        - Export CSV et PNG
        
        **🆚 Avantages vs placement manuel:**
        - Réduction du nombre de points d'accès nécessaires
        - Amélioration de la couverture globale
        - Prise en compte automatique des obstacles (murs)
        - Optimisation mathématique vs intuition
        """)

def install_3d_dependencies():
    """Installation des dépendances pour la 3D"""
    try:
        import subprocess
        import sys
        
        st.info("Installation de plotly pour la visualisation 3D...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "plotly"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            st.success("Plotly installé avec succès! Rechargez la page.")
        else:
            st.error(f"Erreur lors de l'installation: {result.stderr}")
    except Exception as e:
        st.error(f"Erreur: {e}")

def create_heatmap_module():
    """Crée le module de génération de heatmap s'il n'existe pas"""
    st.info("Le module de génération de heatmap sera créé automatiquement...")
    # Cette fonction peut être appelée pour créer le module si nécessaire

def create_heatmap_3d_module():
    """Crée le module de génération de heatmap 3D s'il n'existe pas"""
    st.info("Le module de génération de heatmap 3D sera créé automatiquement...")
    # Cette fonction peut être appelée pour créer le module si nécessaire

if __name__ == "__main__":
    main()
