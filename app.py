import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import io

# Configuration de la page (layout standard, pas wide)
st.set_page_config(
    page_title="Analyseur de Pathloss",
    page_icon="📡",
    layout="centered",  # Layout standard, pas wide
    initial_sidebar_state="expanded"
)

# Chargement du CSS personnalisé
def load_css():
    try:
        with open('styles.css', 'r', encoding='utf-8') as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Fichier CSS non trouvé. Styles par défaut utilisés.")

# Application du CSS
load_css()

def add_custom_html():
    """Ajoute des éléments HTML personnalisés pour améliorer l'interface"""
    st.markdown("""
    <div class="header-container">
        <div class="animated-background"></div>
    </div>
    """, unsafe_allow_html=True)

def create_custom_button(text, key, button_type="primary"):
    """Crée un bouton personnalisé avec style"""
    if button_type == "primary":
        class_name = "custom-button-primary"
    else:
        class_name = "custom-button-secondary"
    
    return st.button(text, key=key)

from image_processor import ImageProcessor
from pathloss_calculator import PathlossCalculator
from visualization import Visualizer
from ml_pathloss_predictor_2d import ml_predictor_2d
from ml_pathloss_predictor_3d import ml_predictor_3d
from auto_optimization_interface import auto_optimization_2d_interface
from auto_optimization_interface_3d import auto_optimization_3d_interface

def check_if_file_uploaded():
    """
    Vérifie si un fichier est uploadé dans n'importe quel onglet.
    """
    # Vérifier dans le session state de Streamlit si des fichiers sont uploadés
    uploaded_keys = [
        "upload_2d", "upload_3d", "upload_heatmap", "upload_heatmap_3d", 
        "upload_optimization_2d", "upload_optimization", 
        "upload_auto_opt", "upload_auto_opt_3d_unique"
    ]
    
    for key in uploaded_keys:
        if key in st.session_state and st.session_state[key] is not None:
            return True
    return False

def display_about_section():
    """
    Affiche la section 'À propos des fonctionnalités' dans la sidebar.
    """
    st.sidebar.subheader("À propos des fonctionnalités")

    with st.sidebar.expander("📡 Calculateurs de Pathloss"):
        st.write("""
        **2D** : Analyse point-à-point sur un plan
                 
        **3D** : Extension multi-étages avec propagation verticale
        """)
    
    with st.sidebar.expander("🌈 Générateurs de Heatmap"):
        st.write("""
        **2D** : Cartes thermiques de couverture radio
                 
        **3D** : Visualisation volumétrique par voxels
        """)
    
    with st.sidebar.expander("🗺️ Optimiseurs de Points d'Accès"):
        st.write("""
        **Classique** : Algorithmes K-means, GMM, Greedy
                 
        **Automatique** : Centré sur vos récepteurs spécifiques
        """)
    
    with st.sidebar.expander("🧠 Intelligence Artificielle"):
        st.write("""
        **Modèles ML** : XGBoost et régression linéaire
                 
        **Fallback** : Modèles théoriques de propagation
        """)
    
    st.sidebar.markdown("---")

def display_ml_status():
    """
    Affiche le statut des modèles ML dans la sidebar.
    """
    st.sidebar.subheader("Modèles ML")
    
    # Statut du modèle 2D
    model_2d_info = ml_predictor_2d.get_model_info()
    if model_2d_info['status'] == 'chargé':
        st.sidebar.success("Modèle 2D: Chargé")
    else:
        st.sidebar.warning("Modèle 2D: Fallback théorique")
    
    # Statut du modèle 3D
    model_3d_info = ml_predictor_3d.get_model_info()
    if model_3d_info['status'] == 'chargé':
        st.sidebar.success("Modèle 3D: Chargé")
        st.sidebar.caption(f"Type: {model_3d_info['model_type']}")
        if 'metrics' in model_3d_info:
            st.sidebar.caption(f"R²: {model_3d_info['metrics'].get('r2_score', 'N/A'):.3f}")
    else:
        st.sidebar.warning("Modèle 3D: Fallback théorique")

def main():
    # Afficher les sections générales seulement si aucun fichier n'est uploadé
    if not check_if_file_uploaded():
        # Affichage de la section à propos
        display_about_section()
        
        # Affichage du statut ML
        display_ml_status()

    st.title("Analyseur de Pathloss")
    st.markdown("---")
    # Création des onglets (incluant l'optimisation automatique 3D)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Calcul Pathloss 2D", 
        "Calcul Pathloss 3D", 
        "Génération Heatmap 2D", 
        "Génération Heatmap 3D", 
        "Optimisation Wifi 2D", 
        "Optimisation Wifi 3D",
        "Optimisation Automatique 2D",
        "Optimisation Automatique 3D"
    ])

    with tab1:
        pathloss_2d_interface()
    with tab2:
        pathloss_3d_interface()
    with tab3:
        heatmap_2d_interface()
    with tab4:
        heatmap_3d_interface()
    with tab5:
        optimization_2d_interface()
    with tab6:
        optimization_3d_interface()
    with tab7:
        auto_optimization_2d_interface()
    with tab8:
        auto_optimization_3d_interface()

def pathloss_2d_interface():
    """Interface pour l'analyse 2D du pathloss"""
    st.header("Analyse 2D du Pathloss")

    # Explication de la section
    st.info("""
    **Calculateur de Perte de Signal 2D** : Analysez la propagation radio entre deux points sur un plan. 
    Chargez votre plan d'étage, placez émetteur et récepteur, et obtenez instantanément le calcul précis 
    des pertes avec visualisation du trajet et détection automatique des obstacles.
    """)

    # Upload du fichier
    uploaded_file = st.file_uploader(
        "Téléchargez le plan de l'appartement (PNG)",
        type=['png'],
        help="Le plan doit être en blanc avec les murs en noir",
        key="upload_2d"
    )
    
    if uploaded_file is not None:
        # Affichage des paramètres spécifiques dans la sidebar
        st.sidebar.header("Paramètres du bâtiment (2D)")
        
        # Conversion de l'image uploadée
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Affichage de l'image originale
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Plan original")
            st.image(image, caption="Plan téléchargé", use_container_width=True)
        
        # Traitement de l'image pour extraire les murs
        processor = ImageProcessor()
        processed_image, walls_detected = processor.process_image(image_array)
        
        with col2:
            st.subheader("Murs détectés")
            st.image(processed_image, caption="Murs extraits", use_container_width=True)
        
        
        # Paramètres du bâtiment
        col1, col2 = st.sidebar.columns(2)
        with col1:
            longueur = st.number_input("Longueur (m)", min_value=1.0, value=10.0, step=0.1)
            
        with col2:
            largeur = st.number_input("Largeur (m)", min_value=1.0, value=8.0, step=0.1)
        
        frequence = st.sidebar.selectbox("Fréquence (MHz)", [2400, 5000], index=0)


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
        
        if st.button("Calculer le Pathloss", key="calc_2d_main"):
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

            st.subheader("Visualisation du trajet")
            st.image(result_image, caption="Trajet et points", use_container_width=True)
            
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
            

def pathloss_3d_interface():
    """Interface pour l'analyse 3D du pathloss"""
    st.header("Analyse 3D du Pathloss")

    # Explication de la section
    st.info("""
    **Calculateur de Perte de Signal 3D** : Extension du calcul 2D pour bâtiments multi-étages. 
    Votre plan devient un modèle 3D avec reproduction automatique sur chaque niveau. 
    Positionnez vos équipements dans l'espace et visualisez la propagation radio en volume.
    """)
    
    # Upload du fichier pour 3D
    uploaded_file_3d = st.file_uploader(
        "Téléchargez le plan de l'appartement (PNG) pour la 3D",
        type=['png'],
        help="Le plan doit être en blanc avec les murs en noir",
        key="upload_3d"
    )
    
    if uploaded_file_3d is not None:
        # Affichage des paramètres spécifiques dans la sidebar
        st.sidebar.header("Paramètres du bâtiment (3D)")
        
        # Conversion de l'image uploadée
        image = Image.open(uploaded_file_3d)
        image_array = np.array(image)

        # Affichage de l'image originale
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Plan 2D original")
            st.image(image, caption="Plan téléchargé pour conversion 3D", use_container_width=True)
        
        # Traitement de l'image pour extraire les murs
        processor = ImageProcessor()
        processed_image, walls_detected = processor.process_image(image_array)
        
        with col2:
            st.subheader("Murs détectés")
            st.image(processed_image, caption="Murs extraits pour duplication 3D", use_container_width=True)
        
        # Paramètres du bâtiment 3D
        st.sidebar.subheader("Dimensions du bâtiment")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            longueur = st.number_input("Longueur (m)", min_value=1.0, value=10.0, step=0.1, key="longueur_3d")
            largeur = st.number_input("Largeur (m)", min_value=1.0, value=8.0, step=0.1, key="largeur_3d")
        
        with col2:
            nb_etages = st.number_input("Nombre d'étages", min_value=1, value=2, step=1, key="etages_3d")
            hauteur_etage = st.number_input("Hauteur étage (m)", min_value=2.0, value=2.7, step=0.1, key="hauteur_3d")
        
        frequence = st.sidebar.selectbox("Fréquence (MHz)", options=[2400, 5000], index=0, key="freq_3d")
        
        
        # Visualisation interactive du plan
        if st.button("Visualisation Interactive 3D", key="interactive_3d"):
            with st.spinner("Génération de la visualisation interactive..."):
                try:
                    from visualization_3d import Visualizer3D
                    import plotly.express as px
                    
                    visualizer_3d = Visualizer3D()
                    
                    # Création d'une visualisation interactive permettant de naviguer
                    fig_interactive = visualizer_3d.create_3d_building(
                        walls_detected, longueur, largeur, nb_etages, hauteur_etage
                    )
                    
                    # Configuration pour l'interactivité
                    fig_interactive.update_layout(
                        title="Visualisation 3D Interactive - Naviguez avec la souris",
                        scene=dict(
                            xaxis_title="Longueur (m)",
                            yaxis_title="Largeur (m)",
                            zaxis_title="Hauteur (m)",
                            aspectmode="data",
                            camera=dict(
                                eye=dict(x=1.5, y=1.5, z=1.2)
                            ),
                            dragmode="orbit"  # Permet la rotation
                        ),
                        updatemenus=[
                            dict(
                                type="buttons",
                                direction="left",
                                buttons=list([
                                    dict(
                                        args=[{"scene.camera.eye": {"x": 1.5, "y": 1.5, "z": 1.2}}],
                                        label="Vue Perspective",
                                        method="relayout"
                                    ),
                                    dict(
                                        args=[{"scene.camera.eye": {"x": 0, "y": 0, "z": 2.5}}],
                                        label="Vue du Dessus",
                                        method="relayout"
                                    ),
                                    dict(
                                        args=[{"scene.camera.eye": {"x": 2.5, "y": 0, "z": 0}}],
                                        label="Vue de Face",
                                        method="relayout"
                                    ),
                                    dict(
                                        args=[{"scene.camera.eye": {"x": 0, "y": 2.5, "z": 0}}],
                                        label="Vue de Côté",
                                        method="relayout"
                                    )
                                ]),
                                pad={"r": 10, "t": 10},
                                showactive=True,
                                x=0.01,
                                xanchor="left",
                                y=1.02,
                                yanchor="top"
                            ),
                        ],
                        width=1000,
                        height=800
                    )
                    st.plotly_chart(fig_interactive, use_container_width=True)
                    
                    
                except Exception as e:
                    st.error(f"❌ Erreur dans la visualisation interactive: {str(e)}")
        
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
                    
                    # Calcul de la puissance reçue (supposant 20dBm d'émission)
                    power_tx = 20.0
                    received_power = power_tx - pathloss_3d
                    
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
                    st.subheader("Visualisation 3D du trajet")
                    st.plotly_chart(fig_3d_path, use_container_width=True)
                    
                    

                    # Affichage des résultats avec métriques améliorées
                    st.subheader("Résultats du calcul 3D")
                    
                    # Métriques principales
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Distance 3D", f"{distance_3d:.2f} m")
                    with col2:
                        st.metric("Murs traversés", wall_count_2d)
                    with col3:
                        st.metric("Diff. étages", floor_difference)
                    with col4:
                        st.metric("Pathloss 3D", f"{pathloss_3d:.1f} dB")
                    with col5:
                        st.metric("Signal reçu", f"{received_power:.1f} dBm")
                    
                    st.markdown("---")

                    # Analyse de la qualité du signal
                    st.subheader("Analyse de la qualité du signal")
                    
                    if received_power >= -50:
                        signal_quality = "Excellent"
                        signal_color = "🟢"
                    elif received_power >= -70:
                        signal_quality = "Bon"
                        signal_color = "🟡"
                    elif received_power >= -85:
                        signal_quality = "Moyen"
                        signal_color = "🟠"
                    else:
                        signal_quality = "Faible"
                        signal_color = "🔴"
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Qualité signal", f"{signal_color} {signal_quality}")
                    with col2:
                        st.metric("Fréquence", f"{frequence} MHz")
                    with col3:
                        st.metric("Puissance TX", f"{power_tx} dBm")
                    
                except ImportError:
                    st.error("❌ Modules de calcul 3D non disponibles")
                except Exception as e:
                    st.error(f"❌ Erreur lors du calcul 3D: {str(e)}")
                    st.info("💡 Vérifiez que les paramètres sont corrects")
                    
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
                    
                except ImportError as e:
                    st.error(f"Erreur d'importation: {e}")
                    st.info("Installation des dépendances 3D requise...")

def heatmap_2d_interface():
    """Interface pour la génération de heatmap 2D"""
    st.header("Génération Heatmap 2D")

    # Explication de la section
    st.info("""
    **Carte de Couverture Radio 2D** : Générez une carte thermique complète de votre réseau WiFi. 
    Placez vos points d'accès sur le plan et visualisez instantanément les zones de couverture 
    avec statistiques détaillées et classification automatique de la qualité du signal.
    """)
    
    # Upload du fichier pour heatmap
    uploaded_file_heatmap = st.file_uploader(
        "Téléchargez le plan de l'appartement (PNG) pour la heatmap",
        type=['png'],
        help="Le plan doit être en blanc avec les murs en noir",
        key="upload_heatmap"
    )
    
    if uploaded_file_heatmap is not None:
        # Affichage des paramètres spécifiques dans la sidebar
        st.sidebar.header("Paramètres Heatmap 2D")
        
        # Conversion de l'image uploadée
        image = Image.open(uploaded_file_heatmap)
        image_array = np.array(image)

        # Affichage de l'image originale
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Plan original")
            st.image(image, caption="Plan pour génération de heatmap", use_container_width=True)
        
        # Traitement de l'image pour extraire les murs
        processor = ImageProcessor()
        processed_image, walls_detected = processor.process_image(image_array)
        
        with col2:
            st.subheader("Murs détectés")
            st.image(processed_image, caption="Murs extraits pour calcul heatmap", use_container_width=True)  
        
        # Paramètres du bâtiment pour heatmap
        st.sidebar.subheader("Dimensions du bâtiment")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            longueur = st.number_input("Longueur (m)", min_value=1.0, value=50.0, step=0.1, key="longueur_heatmap")
            
        with col2:
            largeur = st.number_input("Largeur (m)", min_value=1.0, value=50.0, step=0.1, key="largeur_heatmap")
        
        frequence = st.sidebar.selectbox("Fréquence (MHz)", options=[2400, 5000], index=0, key="freq_heatmap")
        
        # Paramètres de la heatmap
        st.sidebar.subheader("Paramètres de la heatmap")
        resolution = st.sidebar.slider("Résolution de la grille", min_value=20, max_value=100, value=100, key="resolution_heatmap")
        colormap = st.sidebar.selectbox("Palette de couleurs", 
                                       ["RdYlGn_r","plasma", "viridis", "hot", "coolwarm"], 
                                       index=0, key="colormap_heatmap")
        
        # Seuils de qualité du signal
        st.sidebar.subheader("Seuils de signal")
        seuil_excellent = st.sidebar.number_input("Excellent (dB max)", value=-50.0, step=1.0, key="seuil_excellent")
        seuil_bon = st.sidebar.number_input("Bon (dB max)", value=-70.0, step=1.0, key="seuil_bon")
        seuil_faible = st.sidebar.number_input("Faible (dB max)", value=-90.0, step=1.0, key="seuil_faible")

        
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
                st.write(f"Puissance totale: {puissance_tx + gain_antenne:.1f} dBm")
            
            emetteurs.append({
                'position_meter': (x_meter, y_meter),
                'position_pixel': (x_pixel, y_pixel),
                'puissance_tx': puissance_tx,
                'gain_antenne': gain_antenne,
                'puissance_totale': puissance_tx + gain_antenne
            })
        
        # Bouton pour générer la heatmap
        if st.button("Générer la Heatmap 2D", key="generate_heatmap_2d_main"):
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
                
                except ImportError as e:
                    st.error(f"Module heatmap non disponible: {e}")
                    st.info("Création du module de génération de heatmap...")
                except Exception as e:
                    st.error(f"Erreur lors de la génération: {e}")
                    st.exception(e)

def heatmap_3d_interface():
    """Interface pour la génération de heatmap 3D avec voxels"""
    st.header("Génération Heatmap 3D")

    # Explication de la section
    st.info("""
    **Visualisation 3D de Couverture Radio** : Explorez votre couverture WiFi en trois dimensions avec des voxels colorés. 
    Analysez la propagation radio étage par étage, identifiez les zones mortes et optimisez 
    la couverture volumétrique de vos bâtiments multi-niveaux.
    """)
    
    # Upload du fichier pour heatmap 3D
    uploaded_file_heatmap_3d = st.file_uploader(
        "Téléchargez le plan de l'appartement (PNG) pour la heatmap 3D",
        type=['png'],
        help="Le plan doit être en blanc avec les murs en noir",
        key="upload_heatmap_3d"
    )
    
    if uploaded_file_heatmap_3d is not None:
        # Affichage des paramètres spécifiques dans la sidebar
        st.sidebar.header("Paramètres Heatmap 3D")
        
        # Conversion de l'image uploadée
        image = Image.open(uploaded_file_heatmap_3d)
        image_array = np.array(image)

        # Affichage de l'image originale
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Plan 2D de base")
            st.image(image, caption="Plan pour génération de heatmap 3D", use_container_width=True)
        
        # Traitement de l'image pour extraire les murs
        processor = ImageProcessor()
        processed_image, walls_detected = processor.process_image(image_array)
        
        with col2:
            st.subheader("Murs détectés")
            st.image(processed_image, caption="Murs extraits pour duplication en 3D", use_container_width=True)

        # Paramètres du bâtiment pour heatmap 3D
        st.sidebar.subheader("Dimensions du bâtiment")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            longueur = st.number_input("Longueur (m)", min_value=1.0, value=50.0, step=0.1, key="longueur_heatmap_3d")
            largeur = st.number_input("Largeur (m)", min_value=1.0, value=50.0, step=0.1, key="largeur_heatmap_3d")
        
        with col2:
            nb_etages = st.number_input("Nombre d'étages", min_value=1, value=4, step=1, key="etages_heatmap_3d")
            hauteur_etage = st.number_input("Hauteur étage (m)", min_value=2.0, value=2.7, step=0.1, key="hauteur_heatmap_3d")
        
        frequence = st.sidebar.selectbox("Fréquence (MHz)", options=[2400, 5000], index=0, key="freq_heatmap_3d")
        
        # Paramètres de la heatmap 3D
        st.sidebar.subheader("Paramètres des voxels")
        resolution_xy = st.sidebar.slider("Résolution XY", min_value=15, max_value=50, value=25, key="resolution_xy_3d")
        resolution_z = st.sidebar.slider("Résolution Z", min_value=5, max_value=20, value=10, key="resolution_z_3d")
        colormap_3d = st.sidebar.selectbox("Palette de couleurs 3D", 
                                          ["jet", "plasma", "viridis", "hot", "coolwarm"], 
                                          index=0, key="colormap_3d")
        
        # Mode de visualisation
        st.sidebar.subheader("Visualisation")
        view_mode = st.sidebar.selectbox("Mode d'affichage", 
                                        ["Voxels transparents", "Voxels par couches", "Coupes transversales"], 
                                        key="view_mode_3d")
        
        # Seuils de qualité du signal 3D
        st.sidebar.subheader("Seuils de signal 3D")
        seuil_excellent_3d = st.sidebar.number_input("Excellent (dB max)", value=-40.0, step=1.0, key="seuil_excellent_3d")
        seuil_bon_3d = st.sidebar.number_input("Bon (dB max)", value=-65.0, step=1.0, key="seuil_bon_3d")
        seuil_faible_3d = st.sidebar.number_input("Faible (dB max)", value=-85.0, step=1.0, key="seuil_faible_3d")

        
        # Interface pour les points d'accès 3D (émetteurs)
        st.subheader("Configuration des points d'accès 3D")
        
        # Conversion des coordonnées pour l'affichage
        height, width = image_array.shape[:2]
        scale_x = longueur / width
        scale_y = largeur / height
        hauteur_totale = nb_etages * hauteur_etage
        
        # Option pour multiple émetteurs 3D
        nb_emetteurs_3d = st.number_input("Nombre de points d'accès", min_value=1, max_value=4, value=1, key="nb_emetteurs_3d")
        
        emetteurs_3d = []
        for i in range(nb_emetteurs_3d):
            st.write(f"**Point d'accès {i+1}**")
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
                #directivite = st.selectbox(f"Directivité", ["Omnidirectionnel", "Directif"], key=f"dir_3d_{i}")
            
            with col3:
                # Calcul des coordonnées pixel et étage
                x_pixel = int(x_meter / scale_x)
                y_pixel = int(y_meter / scale_y)
                etage = int(z_meter // hauteur_etage) + 1
                
                st.write(f"Étage: {etage}")
                st.write(f"Puissance totale: {puissance_tx + gain_antenne:.1f} dBm")
                #st.write(f"Type: {directivite}")
            
            emetteurs_3d.append({
                'position_meter': (x_meter, y_meter, z_meter),
                'position_pixel': (x_pixel, y_pixel),
                'puissance_tx': puissance_tx,
                'gain_antenne': gain_antenne,
                'puissance_totale': puissance_tx + gain_antenne,
                #'directivite': directivite,
                'etage': etage
            })
        
        # Bouton pour générer la heatmap 3D
        if st.button("Générer la Heatmap 3D", key="generate_heatmap_3d_main"):
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
                
                except ImportError as e:
                    st.error(f"Module heatmap 3D non disponible: {e}")
                    st.info("Création du module de génération de heatmap 3D...")
                except Exception as e:
                    st.error(f"Erreur lors de la génération 3D: {e}")
                    st.exception(e)

def optimization_2d_interface():
    """Interface pour l'optimisation automatique des points d'accès 2D"""
    st.header("Optimisation des Points d'Accès 2D")

    # Explication de la section
    st.info("""
    **Optimiseur Intelligent de Réseau WiFi 2D** : Laissez l'IA placer automatiquement vos points d'accès. 
    Définissez vos objectifs de couverture, choisissez votre algorithme d'optimisation et obtenez 
    la configuration réseau optimale avec visualisation complète et métriques de performance.
    """)
    
    # Upload du fichier pour optimisation 2D
    uploaded_file_optimization_2d = st.file_uploader(
        "Téléchargez le plan de l'appartement (PNG) pour l'optimisation 2D",
        type=['png'],
        help="Le plan doit être en blanc avec les murs en noir",
        key="upload_optimization_2d"
    )
    
    if uploaded_file_optimization_2d is not None:
        # Affichage des paramètres spécifiques dans la sidebar
        st.sidebar.header("Paramètres d'Optimisation 2D")
        
        # Conversion de l'image uploadée
        image = Image.open(uploaded_file_optimization_2d)
        image_array = np.array(image)

        # Affichage de l'image originale
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Plan original")
            st.image(image, caption="Plan pour optimisation des points d'accès 2D", use_container_width=True)
        
        # Traitement de l'image pour extraire les murs
        processor = ImageProcessor()
        processed_image, walls_detected = processor.process_image(image_array)
        
        with col2:
            st.subheader("Murs détectés")
            st.image(processed_image, caption="Murs extraits pour l'optimisation 2D", use_container_width=True)

        
        # Paramètres du bâtiment 2D
        st.sidebar.subheader("Dimensions du plan")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            longueur = st.number_input("Longueur (m)", min_value=1.0, value=50.0, step=0.1, key="longueur_opt_2d")
        
        with col2:
            largeur = st.number_input("Largeur (m)", min_value=1.0, value=50.0, step=0.1, key="largeur_opt_2d")
        
        # Paramètres RF 2D
        frequency_opt_2d = st.sidebar.selectbox("Fréquence", [2400, 5000], index=0, key="freq_opt_2d")
        
        # Objectifs de couverture 2D
        st.sidebar.subheader("Objectifs de Couverture 2D")
        target_coverage_db_2d = st.sidebar.number_input("Signal minimal (dB)", value=-45.0, step=1.0, key="target_signal_2d")
        min_coverage_percent_2d = st.sidebar.number_input("Couverture minimale (%)", min_value=50.0, max_value=100.0, value=60.0, step=1.0, key="min_coverage_2d")
        power_tx_2d = st.sidebar.number_input("Puissance émetteur (dBm)", value=20.0, step=1.0, key="power_opt_2d")
        
        # Paramètres d'optimisation 2D
        st.sidebar.subheader("Paramètres d'Optimisation")
        max_access_points_2d = st.sidebar.number_input("Nb max de points d'accès", min_value=1, max_value=8, value=5, step=1, key="max_ap_2d")

        # Choix de l'algorithme
        algorithm_choice = st.sidebar.selectbox(
            "Algorithme d'optimisation",
            ["K-means", "GMM + EM", "Greedy"],
            index=0,
            help="K-means: Rapide, clusters sphériques\nGMM: Plus précis, clusters ellipsoïdaux\nGreedy: Placement séquentiel optimisé",
            key="algorithm_choice_2d"
        )
        
        if algorithm_choice == "K-means":
            st.sidebar.info("🔧 Optimisation par clustering K-means")
        elif algorithm_choice == "GMM + EM":
            st.sidebar.info("🧠 Optimisation par Gaussian Mixture Model + EM")
        elif algorithm_choice == "Greedy":
            st.sidebar.info("🎯 Optimisation par placement séquentiel Greedy")
        
        # Résolution pour le calcul 2D
        st.sidebar.subheader("Résolution de Calcul")
        resolution_2d = st.sidebar.slider("Résolution grille", min_value=15, max_value=40, value=25, key="res_2d")
        
        # Traitement de l'image
        try:
            # Bouton d'optimisation 2D
            if st.button("Lancer l'Optimisation 2D", key="optimize_button_2d_main"):
                
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
                    
                   
                    if algorithm_choice == "GMM + EM":
                        algorithm_key = 'gmm'
                        algorithm_name = "GMM + EM"
                    elif algorithm_choice == "Greedy":
                        algorithm_key = 'greedy'
                        algorithm_name = "Greedy"
                    else:
                        algorithm_key = 'kmeans'
                        algorithm_name = "K-means"
                    
                    with st.spinner(f"Optimisation par {algorithm_name}..."):
                        best_config_2d, cluster_analysis_2d = optimizer_2d.optimize_with_algorithm_choice_2d(
                            coverage_points, grid_info, longueur, largeur,
                            target_coverage_db_2d, min_coverage_percent_2d, power_tx_2d, max_access_points_2d,
                            algorithm=algorithm_key
                        )
                    
                    st.success(f"Optimisation {algorithm_name} terminée: {best_config_2d['stats']['coverage_percent']:.1f}% de couverture avec {len(best_config_2d['access_points'])} points d'accès")
                    
                    # Affichage des résultats 2D
                    if best_config_2d :
                        # Visualisation 2D
                        st.subheader("Résultat de l'Optimisation 2D")
                        fig_opt_2d = optimizer_2d.visualize_optimization_result_2d(
                            best_config_2d, coverage_points, grid_info, 
                            longueur, largeur, image_array
                        )
                        st.pyplot(fig_opt_2d)
                        
                        # Export de la visualisation
                        import io
                        buffer = io.BytesIO()
                        fig_opt_2d.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                        buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Télécharger Visualisation PNG",
                            data=buffer.getvalue(),
                            file_name=f"optimisation_2d_{frequency_opt_2d}MHz.png",
                            mime="image/png"
                        )

                        # Statistiques détaillées 2D
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Nbr de Points d'Accès", len(best_config_2d['access_points']))
                            st.metric("Couverture", f"{best_config_2d['stats']['coverage_percent']:.1f}%")
                        
                        with col2:
                            st.metric("Points Couverts", f"{best_config_2d['stats']['covered_points']}/{best_config_2d['stats']['total_points']}")
                            st.metric("Score d'Optimisation", f"{best_config_2d['score']:.3f}")
                        
                        with col3:
                            if 'signal_levels' in best_config_2d['stats']:
                                avg_signal = np.mean(best_config_2d['stats']['signal_levels'])
                                st.metric("Signal Moyen", f"{avg_signal:.1f} dB")
                                min_signal = np.min(best_config_2d['stats']['signal_levels'])
                                st.metric("Signal Minimal", f"{min_signal:.1f} dB")

                        # Informations sur l'algorithme utilisé
                        if 'algorithm_used' in best_config_2d:
                            algorithm_used = best_config_2d['algorithm_used']
                            st.info(f"🧠 **Algorithme utilisé:** {algorithm_used}")
                               
                        else:
                            algorithm_display = "GMM + EM" if algorithm_choice == "GMM + EM" else "K-means"
                            st.info(f"🧠 **Algorithme utilisé:** {algorithm_display}")
                        
                        # Configuration des points d'accès 2D
                        st.subheader("Configuration des Points d'Accès Optimisés")
                        
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


def optimization_3d_interface():
    """Interface pour l'optimisation automatique des points d'accès 3D"""
    st.header("Optimisation des Points d'Accès 3D")

    # Explication de la section
    st.info("""
    **Optimiseur Avancé de Réseau WiFi 3D** : Optimisation intelligente pour bâtiments complexes. 
    Algorithmes génétiques et machine learning pour placer automatiquement vos équipements 
    dans l'espace 3D et maximiser la couverture volumétrique avec un minimum de points d'accès.
    """)
    
    # Upload du fichier pour optimisation
    uploaded_file_optimization = st.file_uploader(
        "Téléchargez le plan de l'appartement (PNG) pour l'optimisation",
        type=['png'],
        help="Le plan doit être en blanc avec les murs en noir",
        key="upload_optimization"
    )
    
    if uploaded_file_optimization is not None:
        # Affichage des paramètres spécifiques dans la sidebar
        st.sidebar.header("Paramètres d'Optimisation 3D")
        
        # Conversion de l'image uploadée
        image = Image.open(uploaded_file_optimization)
        image_array = np.array(image)

        # Affichage de l'image originale
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Plan 2D original")
            st.image(image, caption="Plan pour optimisation des points d'accès", use_container_width=True)
        
        # Traitement de l'image pour extraire les murs
        processor = ImageProcessor()
        processed_image, walls_detected = processor.process_image(image_array)
        
        with col2:
            st.subheader("Murs détectés")
            st.image(processed_image, caption="Murs extraits pour l'optimisation 3D", use_container_width=True)

        
        # Paramètres du bâtiment
        st.sidebar.subheader("Dimensions du bâtiment")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            longueur = st.number_input("Longueur (m)", min_value=1.0, value=50.0, step=0.1, key="longueur_opt")
            largeur = st.number_input("Largeur (m)", min_value=1.0, value=50.0, step=0.1, key="largeur_opt")
        
        with col2:
            hauteur_totale = st.number_input("Hauteur totale (m)", min_value=2.0, value=5.4, step=0.1, key="hauteur_opt")
            nb_etages = st.number_input("Nombre d'étages", min_value=1, value=2, step=1, key="etages_opt")
        
        frequency_opt = st.sidebar.selectbox("Fréquence", [2400, 5000], index=0, key="freq_opt")
        
        # Objectifs de couverture
        st.sidebar.subheader("Objectifs de Couverture")
        target_coverage_db = st.sidebar.number_input("Signal minimal (dB)", value=-65.0, step=1.0, key="target_signal")
        min_coverage_percent = st.sidebar.number_input("Couverture minimale (%)", min_value=50.0, max_value=100.0, value=60.0, step=1.0, key="min_coverage")
        power_tx = st.sidebar.number_input("Puissance émetteur (dBm)", value=20.0, step=1.0, key="power_opt")
        
        # Paramètres d'optimisation
        st.sidebar.subheader("Paramètres d'Optimisation")
        max_access_points = st.sidebar.number_input("Nb max de points d'accès", min_value=1, max_value=12, value=5, step=1, key="max_ap")
        # Choix de l'algorithme d'optimisation
        algorithm_choice = st.sidebar.selectbox(
            "Algorithme d'optimisation",
            ["kmeans", "gmm", "greedy"],
            index=1,  # K-means par défaut
            help="Choisissez l'algorithme d'optimisation pour le placement des points d'accès",
            key="algorithm_choice"
        )
        
        # Informations sur les algorithmes
        algorithm_info = {
            "kmeans": "📊 K-means clustering - Regroupement par proximité",
            "gmm": "🧠 Gaussian Mixture Model - Modélisation probabiliste avancée",
            "greedy": "🎯 Greedy (Glouton) - Placement séquentiel optimal"
        }
        st.sidebar.info(algorithm_info[algorithm_choice])
        

        # Résolution pour le calcul
        st.sidebar.subheader("Résolution de Calcul")
        resolution_xy_opt = st.sidebar.slider("Résolution XY", min_value=10, max_value=30, value=20, key="res_xy_opt")
        resolution_z_opt = st.sidebar.slider("Résolution Z", min_value=4, max_value=12, value=8, key="res_z_opt")
        
        # Traitement de l'image
        try:
            # Bouton d'optimisation
            if st.button("Lancer l'Optimisation 3D", key="optimize_button"):
                
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
                    # Mode algorithme unique
                    algorithm_name = algorithm_choice.upper()
                    with st.spinner(f"Optimisation avec {algorithm_name}..."):
                        best_config, cluster_analysis = optimizer.optimize_with_algorithm_choice_3d(
                            algorithm_choice, coverage_points, grid_info, longueur, largeur, hauteur_totale,
                            target_coverage_db, min_coverage_percent, max_access_points, power_tx
                        )
                    
                    if best_config:
                        st.success(f"Optimisation {algorithm_name} terminée: {best_config['stats']['coverage_percent']:.1f}% de couverture avec {len(best_config['access_points'])} points d'accès")
                    else:
                        st.error(f"❌ L'optimisation {algorithm_name} a échoué")
                    
                    # Affichage des résultats
                    if best_config:
                        # Visualisation 3D
                        st.subheader("Résultat de l'Optimisation")
                        fig_opt = optimizer.visualize_optimization_result(
                            best_config, coverage_points, grid_info, 
                            longueur, largeur, hauteur_totale
                        )
                        st.plotly_chart(fig_opt, use_container_width=True)

                        html_buffer = fig_opt.to_html()
                        st.download_button(
                            label="📥 Télécharger Visualisation HTML",
                            data=html_buffer,
                            file_name=f"optimisation_3d_{frequency_opt}MHz.html",
                            mime="text/html"
                        )
                        
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

if __name__ == "__main__":
    main()
