import numpy as np
from PIL import Image, ImageDraw

def create_sample_floorplan():
    """
    Crée un plan d'appartement exemple pour tester l'application.
    """
    # Dimensions de l'image
    width, height = 800, 600
    
    # Créer une image blanche
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Épaisseur des murs
    wall_thickness = 8
    
    # Murs extérieurs
    # Mur du haut
    draw.rectangle([0, 0, width, wall_thickness], fill='black')
    # Mur du bas
    draw.rectangle([0, height-wall_thickness, width, height], fill='black')
    # Mur de gauche
    draw.rectangle([0, 0, wall_thickness, height], fill='black')
    # Mur de droite
    draw.rectangle([width-wall_thickness, 0, width, height], fill='black')
    
    # Murs intérieurs - Salon/Cuisine
    # Mur vertical séparant salon et chambre
    draw.rectangle([width//2-wall_thickness//2, wall_thickness, width//2+wall_thickness//2, height//2], fill='black')
    
    # Mur horizontal séparant salon et cuisine
    draw.rectangle([wall_thickness, height//3-wall_thickness//2, width//2, height//3+wall_thickness//2], fill='black')
    
    # Mur de la salle de bain
    draw.rectangle([width*3//4-wall_thickness//2, height//2, width*3//4+wall_thickness//2, height*3//4], fill='black')
    draw.rectangle([width//2, height*3//4-wall_thickness//2, width*3//4, height*3//4+wall_thickness//2], fill='black')
    
    # Portes (ouvertures dans les murs)
    door_width = 30
    
    # Porte entre salon et chambre
    door_y = height//4
    draw.rectangle([width//2-wall_thickness//2, door_y, width//2+wall_thickness//2, door_y+door_width], fill='white')
    
    # Porte entre salon et cuisine
    door_x = width//4
    draw.rectangle([door_x, height//3-wall_thickness//2, door_x+door_width, height//3+wall_thickness//2], fill='white')
    
    # Porte d'entrée
    entrance_x = 50
    draw.rectangle([entrance_x, 0, entrance_x+door_width, wall_thickness], fill='white')
    
    return image

if __name__ == "__main__":
    # Créer le plan exemple
    floorplan = create_sample_floorplan()
    
    # Sauvegarder
    floorplan.save("plan_exemple.png")
    print("Plan d'exemple créé : plan_exemple.png")
    
    # Afficher les informations
    print("\nCaractéristiques du plan :")
    print("- Dimensions : 800x600 pixels")
    print("- Représente un appartement d'environ 10m x 7.5m")
    print("- Murs en noir, espaces libres en blanc")
    print("- Plusieurs pièces : salon, cuisine, chambre, salle de bain")
    print("- Portes représentées par des ouvertures dans les murs")
