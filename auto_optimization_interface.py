"""
Interface utilisateur pour l'optimisation automatique des points d'accès 2D.

Ce module contient l'interface Streamlit pour l'onglet d'optimisation automatique.
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from auto_optimizer_2d import AutoOptimizer2D
from image_processor import ImageProcessor


def auto_optimization_2d_interface():
    """Interface pour l'optimisation automatique 2D des points d'accès"""
    st.header("Optimisation Auto des Points d'Accès 2D")
    
    # Explication de la section
    st.info("""
    **Optimiseur Intelligent Centré sur les Récepteurs 2D** : Définissez vos appareils/récepteurs 
    et laissez l'IA placer automatiquement les points d'accès optimaux. Algorithme de descente de gradient 
    avec placement intelligent pour minimiser le pathloss vers vos équipements spécifiques.
    """)
    

    
    # Upload du fichier pour optimisation automatique
    uploaded_file_auto = st.file_uploader(
        "Téléchargez le plan de l'appartement (PNG)",
        type=['png'],
        help="Le plan doit être en blanc avec les murs en noir",
        key="upload_auto_opt"
    )
    
    if uploaded_file_auto is not None:
        # Affichage des paramètres spécifiques dans la sidebar
        st.sidebar.header("Paramètres d'Optimisation 2D")
        
        # Conversion de l'image uploadée
        image = Image.open(uploaded_file_auto)
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
        st.sidebar.subheader("Dimensions du bâtiment")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            longueur = st.number_input("Longueur (m)", min_value=1.0, value=10.0, step=0.5, key="longueur_auto")
        with col2:
            largeur = st.number_input("Largeur (m)", min_value=1.0, value=10.0, step=0.5, key="largeur_auto")
        
        # Paramètres RF
        frequence = st.sidebar.selectbox("Fréquence (MHz)", options=[2400, 5000], index=0, key="freq_auto")
        power_tx = st.sidebar.number_input("Puissance TX (dBm)", min_value=10.0, max_value=30.0, 
                                          value=20.0, step=1.0, key="power_auto")
        
        # Paramètres d'optimisation
        st.sidebar.subheader("Paramètres d'optimisation")
        max_access_points = st.sidebar.slider("Max points d'accès", min_value=1, max_value=8, 
                                             value=5, key="max_aps_auto")
        
        st.sidebar.info("🔧 Algorithme: Descente de Gradient avec placement intelligent")
        
        # Interface pour définir les récepteurs
        st.subheader("Configuration des Récepteurs")
        st.markdown("Définissez les positions des récepteurs (appareils) à couvrir.")
        
        # Conversion des coordonnées pour l'affichage
        height, width = image_array.shape[:2]
        scale_x = longueur / width
        scale_y = largeur / height
        
        # Gestion dynamique des récepteurs
        if 'receivers_auto' not in st.session_state:
            # Initialisation avec 4 récepteurs par défaut
            st.session_state.receivers_auto = [
                (longueur/4, largeur/4),
                (3*longueur/4, 3*largeur/4),
                (1.0, 9.0),   # Récepteur 3 demandé
                (9.0, 1.0)    # Récepteur 4 demandé
            ]
        
        # Options de gestion des récepteurs
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("➕ Ajouter Récepteur", key="add_receiver"):
                st.session_state.receivers_auto.append((longueur/2, largeur/2))
                st.rerun()
        
        with col2:
            if st.button("➖ Supprimer Dernier", key="remove_receiver"):
                if len(st.session_state.receivers_auto) > 1:
                    st.session_state.receivers_auto.pop()
                    st.rerun()
        
        with col3:
            if st.button("🔄 Réinitialiser", key="reset_receivers"):
                st.session_state.receivers_auto = [(longueur/4, largeur/4), (3*longueur/4, 3*largeur/4)]
                st.rerun()
        
        # Interface pour chaque récepteur
        receivers = []
        for i in range(len(st.session_state.receivers_auto)):
            st.write(f"**Récepteur {i+1}**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_meter = st.number_input(f"X{i+1} (m)", min_value=0.5, max_value=longueur-0.5, 
                                         value=float(st.session_state.receivers_auto[i][0]), 
                                         step=0.1, key=f"rx_x_{i}")
            
            with col2:
                y_meter = st.number_input(f"Y{i+1} (m)", min_value=0.5, max_value=largeur-0.5, 
                                         value=float(st.session_state.receivers_auto[i][1]), 
                                         step=0.1, key=f"rx_y_{i}")
            
            with col3:
                # Calcul des coordonnées pixel correspondantes
                x_pixel = int(x_meter / scale_x)
                y_pixel = int(y_meter / scale_y)
                st.write(f"Position pixel:")
                st.write(f"({x_pixel}, {y_pixel})")
            
            # Mise à jour de la position dans le state
            st.session_state.receivers_auto[i] = (x_meter, y_meter)
            receivers.append((x_meter, y_meter))
        
        # Affichage des récepteurs sur le plan
        if receivers:
            st.subheader("Aperçu des récepteurs sur le plan")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(walls_detected, cmap='gray_r', extent=[0, longueur, largeur, 0])
            
            # Affichage des récepteurs
            for i, (rx_x, rx_y) in enumerate(receivers):
                ax.scatter(rx_x, rx_y, c='blue', s=150, marker='o', 
                          edgecolors='white', linewidth=2, zorder=5)
                ax.annotate(f'R{i+1}', (rx_x, rx_y), xytext=(5, 5), 
                           textcoords='offset points', fontweight='bold', 
                           color='blue', fontsize=12)
            
            ax.set_xlabel('Longueur (m)')
            ax.set_ylabel('Largeur (m)')
            ax.set_title(f'Plan avec {len(receivers)} récepteur(s) défini(s)')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
        
        
        # Bouton d'optimisation
        if st.button("Lancer l'Optimisation", type="primary", key="launch_optimization"):
            if len(receivers) == 0:
                st.error("❌ Veuillez définir au moins un récepteur!")
                return
            
            with st.spinner("🔄 Optimisation en cours avec l'algorithme de descente de gradient..."):
                try:
                    # Création de l'optimiseur
                    optimizer = AutoOptimizer2D(frequence)
                    
                    # Lancement de l'optimisation
                    result = optimizer.optimize_access_points(
                        walls_detected=walls_detected,
                        receivers=receivers,
                        longueur=longueur,
                        largeur=largeur,
                        max_access_points=max_access_points,
                        power_tx=power_tx
                    )
                    
                    # Vérification des résultats
                    if result['best_config'] is None:
                        st.error("❌ L'optimisation a échoué. Veuillez ajuster les paramètres.")
                        return
                    
                    best_config = result['best_config']
                    
                    # Affichage des résultats principaux
                    st.success("✅ Optimisation terminée avec succès!")
                    
                    # Métriques de performance
                    st.subheader("Résultats de l'optimisation")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Points d'accès", best_config['num_access_points'])
                    with col2:
                        st.metric("Pathloss moyen", f"{best_config['avg_pathloss']:.1f} dB")
                    with col3:
                        st.metric("Pathloss max", f"{best_config['max_pathloss']:.1f} dB")
                    with col4:
                        st.metric("Pathloss min", f"{best_config['min_pathloss']:.1f} dB")
                    with col5:
                        st.metric("Score total", f"{best_config['total_score']:.1f}")
                    
                    # Visualisation du résultat
                    st.subheader("Visualisation de la solution optimale")
                    fig_result = optimizer.visualize_optimization_result(result, walls_detected, longueur, largeur)
                    st.pyplot(fig_result)
                    plt.close()
                    
                    # Détails de la configuration optimale
                    st.subheader("Configuration des points d'accès optimisés")
                    
                    access_points = best_config['access_points']
                    for i, (ap_x, ap_y, ap_power) in enumerate(access_points):
                        with st.expander(f"📡 Point d'accès {i+1}"):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Position X", f"{ap_x:.2f} m")
                            with col2:
                                st.metric("Position Y", f"{ap_y:.2f} m")
                            with col3:
                                st.metric("Puissance", f"{ap_power:.1f} dBm")
                            with col4:
                                # Coordonnées pixel pour référence
                                ap_x_pixel = int(ap_x / scale_x)
                                ap_y_pixel = int(ap_y / scale_y)
                                st.metric("Position pixel", f"({ap_x_pixel}, {ap_y_pixel})")
                    
                    # Analyse détaillée des récepteurs
                    st.subheader("Analyse des récepteurs")
                    
                    receiver_stats = best_config['stats']['receiver_stats']
                    for i, stats in enumerate(receiver_stats):
                        with st.expander(f"📱 Récepteur {i+1} - Pathloss: {stats['pathloss']:.1f} dB"):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Position", f"({stats['position'][0]:.1f}, {stats['position'][1]:.1f}) m")
                            with col2:
                                st.metric("Pathloss", f"{stats['pathloss']:.1f} dB")
                            with col3:
                                st.metric("Signal reçu", f"{stats['received_power']:.1f} dBm")
                            with col4:
                                st.metric("Distance AP", f"{stats['distance_to_best_ap']:.1f} m")
                            
                            # Qualité du signal
                            received_power = stats['received_power']
                            if received_power >= -50:
                                signal_quality = "🟢 Excellent"
                            elif received_power >= -70:
                                signal_quality = "🟡 Bon"
                            elif received_power >= -85:
                                signal_quality = "🟠 Moyen"
                            else:
                                signal_quality = "🔴 Faible"
                            
                            st.info(f"Qualité du signal: {signal_quality}")
                    
                    # Comparaison des algorithmes si plusieurs résultats
                    if len(result['all_results']) > 1:
                        st.subheader("Comparaison des configurations")
                        
                        comparison_data = []
                        for num_aps, config in result['all_results'].items():
                            comparison_data.append({
                                'Points d\'accès': num_aps,
                                'Pathloss moyen (dB)': f"{config['avg_pathloss']:.1f}",
                                'Pathloss max (dB)': f"{config['max_pathloss']:.1f}",
                                'Score total': f"{config['total_score']:.1f}",
                                'Optimal': '✅' if config == best_config else '❌'
                            })
                        
                        st.table(comparison_data)
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'optimisation: {str(e)}")
                    st.info("💡 Vérifiez que tous les paramètres sont corrects et réessayez.")
                    
                    # Debug info pour développeur
                    if st.checkbox("🔧 Afficher les détails de l'erreur (debug)"):
                        st.exception(e)
