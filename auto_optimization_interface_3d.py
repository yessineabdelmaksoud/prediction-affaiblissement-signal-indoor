#!/usr/bin/env python3
"""
Interface utilisateur pour l'optimisation automatique des points d'accès 3D.
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from auto_optimizer_3d import AutoOptimizer3D
from image_processor import ImageProcessor

def auto_optimization_3d_interface():
    st.header("🎯 Optimisation Automatique des Points d'Accès 3D")
    st.markdown("""
    Cette fonctionnalité optimise automatiquement le nombre et la position des points d'accès (x, y, z)
    pour minimiser le pathloss vers vos récepteurs définis en 3D.
    """)

    uploaded_file_auto = st.file_uploader(
        "Téléchargez le plan de l'appartement (PNG)",
        type=['png'],
        help="Le plan doit être en blanc avec les murs en noir",
        key="upload_auto_opt_3d_unique"
    )

    if uploaded_file_auto is not None:
        st.sidebar.header("🔧 Paramètres d'Optimisation 3D")
        image = Image.open(uploaded_file_auto)
        image_array = np.array(image)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 Plan original")
            st.image(image, caption="Plan téléchargé", use_column_width=True)
        processor = ImageProcessor()
        processed_image, walls_detected = processor.process_image(image_array)
        with col2:
            st.subheader("🏗️ Murs détectés")
            st.image(processed_image, caption="Murs extraits", use_column_width=True)
        st.sidebar.subheader("📐 Dimensions du bâtiment")
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            longueur = st.number_input("Longueur (m)", min_value=1.0, value=10.0, step=0.5, key="longueur_auto_3d")
        with col2:
            largeur = st.number_input("Largeur (m)", min_value=1.0, value=10.0, step=0.5, key="largeur_auto_3d")
        with col3:
            nb_etages = st.number_input("Nb étages", min_value=1, value=2, step=1, key="nb_etages_auto_3d")
        hauteur_etage = st.sidebar.number_input("Hauteur d'un étage (m)", min_value=2.0, value=3.0, step=0.1, key="hauteur_etage_auto_3d")
        hauteur_totale = nb_etages * hauteur_etage
        st.sidebar.subheader("📡 Paramètres RF")
        frequence = st.sidebar.selectbox("Fréquence (MHz)", options=[2400, 5000], index=0, key="freq_auto_3d")
        power_tx = st.sidebar.number_input("Puissance TX (dBm)", min_value=10.0, max_value=30.0, value=20.0, step=1.0, key="power_auto_3d")
        st.sidebar.subheader("⚙️ Paramètres d'optimisation")
        max_access_points = st.sidebar.slider("Max points d'accès", min_value=1, max_value=8, value=5, key="max_aps_auto_3d")
        st.sidebar.info("🔧 Algorithme: Descente de Gradient avec placement intelligent (3D)")
        st.subheader("📍 Configuration des Récepteurs 3D")
        st.markdown("Définissez les positions des récepteurs (x, y, z) à couvrir.")
        height, width = image_array.shape[:2]
        scale_x = longueur / width
        scale_y = largeur / height
        if 'receivers_auto_3d' not in st.session_state:
            st.session_state.receivers_auto_3d = [
                (longueur/4, largeur/4, hauteur_totale/2),
                (3*longueur/4, 3*largeur/4, hauteur_totale/2),
                (1.0, 9.0, hauteur_totale-0.5),
                (9.0, 1.0, 0.5)
            ]
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("➕ Ajouter Récepteur 3D", key="add_receiver_3d"):
                st.session_state.receivers_auto_3d.append((longueur/2, largeur/2, hauteur_totale/2))
                st.rerun()
        with col2:
            if st.button("➖ Supprimer Dernier 3D", key="remove_receiver_3d"):
                if len(st.session_state.receivers_auto_3d) > 1:
                    st.session_state.receivers_auto_3d.pop()
                    st.rerun()
        with col3:
            if st.button("🔄 Réinitialiser 3D", key="reset_receivers_3d"):
                st.session_state.receivers_auto_3d = [
                    (longueur/4, largeur/4, hauteur_totale/2),
                    (3*longueur/4, 3*largeur/4, hauteur_totale/2)
                ]
                st.rerun()
        receivers = []
        for i in range(len(st.session_state.receivers_auto_3d)):
            st.write(f"**Récepteur {i+1}**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                x_meter = st.number_input(f"X{i+1} (m)", min_value=0.5, max_value=longueur-0.5, value=float(st.session_state.receivers_auto_3d[i][0]), step=0.1, key=f"rx3d_x_{i}")
            with col2:
                y_meter = st.number_input(f"Y{i+1} (m)", min_value=0.5, max_value=largeur-0.5, value=float(st.session_state.receivers_auto_3d[i][1]), step=0.1, key=f"rx3d_y_{i}")
            with col3:
                z_meter = st.number_input(f"Z{i+1} (m)", min_value=0.1, max_value=hauteur_totale-0.1, value=float(st.session_state.receivers_auto_3d[i][2]), step=0.1, key=f"rx3d_z_{i}")
            with col4:
                x_pixel = int(x_meter / scale_x)
                y_pixel = int(y_meter / scale_y)
                st.write(f"Pixel: ({x_pixel}, {y_pixel})")
            st.session_state.receivers_auto_3d[i] = (x_meter, y_meter, z_meter)
            receivers.append((x_meter, y_meter, z_meter))
        if receivers:
            st.subheader("🗺️ Aperçu des récepteurs sur le plan (projection XY)")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(walls_detected, cmap='gray_r', extent=[0, longueur, largeur, 0])
            for i, (rx_x, rx_y, rx_z) in enumerate(receivers):
                ax.scatter(rx_x, rx_y, c='blue', s=150, marker='o', edgecolors='white', linewidth=2, zorder=5)
                ax.annotate(f'R{i+1}\nZ={rx_z:.1f}m', (rx_x, rx_y), xytext=(5, 5), textcoords='offset points', fontweight='bold', color='blue', fontsize=12)
            ax.set_xlabel('Longueur (m)')
            ax.set_ylabel('Largeur (m)')
            ax.set_title(f'Plan XY avec {len(receivers)} récepteur(s) défini(s)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

            # --- Visualisation 3D du plan avec points d'accès et récepteurs ---
            st.subheader("🏢 Visualisation 3D du plan avec AP et récepteurs")
            try:
                from visualization_3d import Visualizer3D
                vis3d = Visualizer3D()
                # Points d'accès (avant optimisation, on affiche les positions actuelles si dispo)
                ap_points = []
                if 'last_ap_points' in st.session_state:
                    ap_points = st.session_state['last_ap_points']
                # Si pas encore optimisé, on n'affiche que les récepteurs
                fig3d = vis3d.create_3d_building(walls_detected, longueur, largeur, nb_etages, hauteur_etage)
                # Ajout des récepteurs
                for i, (rx_x, rx_y, rx_z) in enumerate(receivers):
                    fig3d.add_trace({
                        'type': 'scatter3d',
                        'x': [rx_x],
                        'y': [rx_y],
                        'z': [rx_z],
                        'mode': 'markers+text',
                        'marker': {'size': 10, 'color': 'blue', 'symbol': 'diamond'},
                        'name': f"Récepteur {i+1}",
                        'text': [f"R{i+1}"],
                        'textposition': 'top center'
                    })
                # Ajout des points d'accès si dispo
                for i, (ap_x, ap_y, ap_z, ap_power) in enumerate(ap_points):
                    fig3d.add_trace({
                        'type': 'scatter3d',
                        'x': [ap_x],
                        'y': [ap_y],
                        'z': [ap_z],
                        'mode': 'markers+text',
                        'marker': {'size': 12, 'color': 'red', 'symbol': 'circle'},
                        'name': f"AP {i+1}",
                        'text': [f"AP{i+1}"],
                        'textposition': 'top center'
                    })
                vis3d.add_floor_color_legend(fig3d, nb_etages)
                st.plotly_chart(fig3d, use_container_width=True)
            except Exception as e:
                st.info(f"Visualisation 3D non disponible: {e}")
        if st.button("🚀 Lancer l'Optimisation 3D", type="primary", key="launch_optimization_3d"):
            if len(receivers) == 0:
                st.error("❌ Veuillez définir au moins un récepteur!")
                return
            with st.spinner("🔄 Optimisation 3D en cours..."):
                try:
                    optimizer = AutoOptimizer3D(frequence)
                    result = optimizer.optimize_access_points(
                        walls_detected=walls_detected,
                        receivers=receivers,
                        longueur=longueur,
                        largeur=largeur,
                        hauteur=hauteur_totale,
                        max_access_points=max_access_points,
                        power_tx=power_tx
                    )
                    if result['best_config'] is None:
                        st.error("❌ L'optimisation a échoué. Veuillez ajuster les paramètres.")
                        return
                    best_config = result['best_config']
                    # Sauvegarder les points d'accès pour la visualisation 3D
                    st.session_state['last_ap_points'] = best_config['access_points']
                    st.success("✅ Optimisation 3D terminée avec succès!")
                    st.subheader("📊 Résultats de l'optimisation 3D")
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
                    st.subheader("📋 Configuration des points d'accès optimisés (3D)")
                    access_points = best_config['access_points']
                    for i, (ap_x, ap_y, ap_z, ap_power) in enumerate(access_points):
                        with st.expander(f"📡 Point d'accès {i+1}"):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("X", f"{ap_x:.2f} m")
                            with col2:
                                st.metric("Y", f"{ap_y:.2f} m")
                            with col3:
                                st.metric("Z", f"{ap_z:.2f} m")
                            with col4:
                                st.metric("Puissance", f"{ap_power:.1f} dBm")
                    st.subheader("📱 Analyse des récepteurs (3D)")
                    receiver_stats = best_config['stats']['receiver_stats']
                    for i, stats in enumerate(receiver_stats):
                        with st.expander(f"📱 Récepteur {i+1} - Pathloss: {stats['pathloss']:.1f} dB"):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Position", f"({stats['position'][0]:.1f}, {stats['position'][1]:.1f}, {stats['position'][2]:.1f}) m")
                            with col2:
                                st.metric("Pathloss", f"{stats['pathloss']:.1f} dB")
                            with col3:
                                st.metric("Signal reçu", f"{stats['received_power']:.1f} dBm")
                            with col4:
                                st.metric("Distance AP", f"{stats['distance_to_best_ap']:.1f} m")
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

                    # --- Visualisation 3D du plan avec AP optimisés et récepteurs ---
                    st.subheader("🏢 Visualisation 3D du plan optimisé avec AP et récepteurs")
                    try:
                        from visualization_3d import Visualizer3D
                        vis3d = Visualizer3D()
                        fig3d = vis3d.create_3d_building(walls_detected, longueur, largeur, nb_etages, hauteur_etage)
                        # Ajout des récepteurs
                        for i, (rx_x, rx_y, rx_z) in enumerate(receivers):
                            fig3d.add_trace({
                                'type': 'scatter3d',
                                'x': [rx_x],
                                'y': [rx_y],
                                'z': [rx_z],
                                'mode': 'markers+text',
                                'marker': {'size': 10, 'color': 'blue', 'symbol': 'diamond'},
                                'name': f"Récepteur {i+1}",
                                'text': [f"R{i+1}"],
                                'textposition': 'top center'
                            })
                        # Ajout des points d'accès optimisés
                        for i, (ap_x, ap_y, ap_z, ap_power) in enumerate(access_points):
                            fig3d.add_trace({
                                'type': 'scatter3d',
                                'x': [ap_x],
                                'y': [ap_y],
                                'z': [ap_z],
                                'mode': 'markers+text',
                                'marker': {'size': 12, 'color': 'red', 'symbol': 'circle'},
                                'name': f"AP {i+1}",
                                'text': [f"AP{i+1}"],
                                'textposition': 'top center'
                            })
                        vis3d.add_floor_color_legend(fig3d, nb_etages)
                        st.plotly_chart(fig3d, use_container_width=True)
                    except Exception as e:
                        st.info(f"Visualisation 3D non disponible: {e}")
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'optimisation: {str(e)}")
                    st.info("💡 Vérifiez que tous les paramètres sont corrects et réessayez.")
                    if st.checkbox("🔧 Afficher les détails de l'erreur (debug)"):
                        st.exception(e)
