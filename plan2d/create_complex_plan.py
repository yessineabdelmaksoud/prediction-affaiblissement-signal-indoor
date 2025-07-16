import numpy as np
from PIL import Image, ImageDraw

def create_complex_floorplan():
    """
    Crée un plan d'appartement plus complexe pour tester la heatmap.
    """
    # Dimensions de l'image
    width, height = 1000, 750
    
    # Créer une image blanche
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Épaisseur des murs
    wall_thickness = 10
    
    # Murs extérieurs
    draw.rectangle([0, 0, width, wall_thickness], fill='black')  # Haut
    draw.rectangle([0, height-wall_thickness, width, height], fill='black')  # Bas
    draw.rectangle([0, 0, wall_thickness, height], fill='black')  # Gauche
    draw.rectangle([width-wall_thickness, 0, width, height], fill='black')  # Droite
    
    # Murs intérieurs - Salon
    # Mur vertical principal (séparation salon/chambres)
    draw.rectangle([width//2-wall_thickness//2, wall_thickness, 
                   width//2+wall_thickness//2, height*2//3], fill='black')
    
    # Mur horizontal (séparation salon/cuisine)
    draw.rectangle([wall_thickness, height//3-wall_thickness//2, 
                   width//2, height//3+wall_thickness//2], fill='black')
    
    # Murs des chambres
    # Chambre 1
    draw.rectangle([width//2, height//3-wall_thickness//2, 
                   width*3//4, height//3+wall_thickness//2], fill='black')
    
    # Chambre 2  
    draw.rectangle([width*3//4-wall_thickness//2, height//3, 
                   width*3//4+wall_thickness//2, height*2//3], fill='black')
    
    # Salle de bain
    draw.rectangle([width*3//4, height*2//3-wall_thickness//2, 
                   width-wall_thickness, height*2//3+wall_thickness//2], fill='black')
    
    # Cuisine - murs en L
    draw.rectangle([width//4-wall_thickness//2, wall_thickness, 
                   width//4+wall_thickness//2, height//3], fill='black')
    
    # Couloir central
    draw.rectangle([width//2, height*2//3-wall_thickness//2, 
                   width*7//8, height*2//3+wall_thickness//2], fill='black')
    
    # Portes (ouvertures dans les murs)
    door_width = 35
    
    # Porte d'entrée
    entrance_x = width//8
    draw.rectangle([entrance_x, 0, entrance_x+door_width, wall_thickness], fill='white')
    
    # Porte salon-cuisine
    door_x = width//6
    draw.rectangle([door_x, height//3-wall_thickness//2, 
                   door_x+door_width, height//3+wall_thickness//2], fill='white')
    
    # Porte salon-couloir
    door_y = height//2
    draw.rectangle([width//2-wall_thickness//2, door_y, 
                   width//2+wall_thickness//2, door_y+door_width], fill='white')
    
    # Porte chambre 1
    door_x_ch1 = width*5//8
    draw.rectangle([door_x_ch1, height//3-wall_thickness//2, 
                   door_x_ch1+door_width, height//3+wall_thickness//2], fill='white')
    
    # Porte chambre 2
    door_y_ch2 = height//2
    draw.rectangle([width*3//4-wall_thickness//2, door_y_ch2, 
                   width*3//4+wall_thickness//2, door_y_ch2+door_width], fill='white')
    
    # Porte salle de bain
    door_x_sdb = width*13//16
    draw.rectangle([door_x_sdb, height*2//3-wall_thickness//2, 
                   door_x_sdb+door_width, height*2//3+wall_thickness//2], fill='white')
    
    # Ajout d'éléments fixes (pour réalisme)
    # Îlot de cuisine
    draw.rectangle([width//8, height//6, width//4, height//4], fill='gray')
    
    # Plans de travail
    draw.rectangle([wall_thickness+5, wall_thickness+5, 
                   width//4-5, wall_thickness+25], fill='gray')
    
    return image

def create_positions_suggestions():
    """
    Retourne des suggestions de positions d'émetteurs optimales.
    """
    # Positions en coordonnées relatives (0 à 1)
    suggestions = {
        'optimal_1_emetteur': [
            {'x_rel': 0.3, 'y_rel': 0.4, 'description': 'Centre salon, couverture globale'}
        ],
        'optimal_2_emetteurs': [
            {'x_rel': 0.25, 'y_rel': 0.25, 'description': 'Salon principal'},
            {'x_rel': 0.75, 'y_rel': 0.55, 'description': 'Zone chambres'}
        ],
        'optimal_3_emetteurs': [
            {'x_rel': 0.2, 'y_rel': 0.2, 'description': 'Salon/Cuisine'},
            {'x_rel': 0.6, 'y_rel': 0.2, 'description': 'Chambre 1'},
            {'x_rel': 0.8, 'y_rel': 0.8, 'description': 'Fond appartement'}
        ]
    }
    return suggestions

if __name__ == "__main__":
    # Créer le plan complexe
    complex_plan = create_complex_floorplan()
    
    # Sauvegarder
    complex_plan.save("plan_complexe_exemple.png")
    print("Plan complexe créé : plan_complexe_exemple.png")
    
    # Afficher les informations
    print("\nCaractéristiques du plan complexe :")
    print("- Dimensions : 1000x750 pixels")
    print("- Représente un appartement d'environ 12.5m x 9.4m")
    print("- Plusieurs pièces : salon, cuisine, 2 chambres, salle de bain, couloir")
    print("- Murs en noir (épaisseur 10px), espaces libres en blanc")
    print("- Portes représentées par des ouvertures")
    print("- Éléments fixes en gris (cuisine, plans de travail)")
    
    # Suggestions de positions
    suggestions = create_positions_suggestions()
    print("\nSuggestions de positions d'émetteurs :")
    
    for config, positions in suggestions.items():
        print(f"\n{config.replace('_', ' ').title()} :")
        for i, pos in enumerate(positions, 1):
            x_pixel = int(pos['x_rel'] * 1000)
            y_pixel = int(pos['y_rel'] * 750)
            x_meter = pos['x_rel'] * 12.5
            y_meter = pos['y_rel'] * 9.4
            print(f"  {i}. {pos['description']}")
            print(f"     Position: ({x_meter:.1f}m, {y_meter:.1f}m) ou ({x_pixel}px, {y_pixel}px)")
    
    print(f"\nPour utiliser ce plan dans l'application :")
    print(f"1. Lancez 'streamlit run app.py'")
    print(f"2. Allez dans l'onglet 'Génération Heatmap 2D'")
    print(f"3. Téléchargez le fichier 'plan_complexe_exemple.png'")
    print(f"4. Configurez les dimensions : 12.5m x 9.4m")
    print(f"5. Utilisez les positions suggérées ci-dessus")
