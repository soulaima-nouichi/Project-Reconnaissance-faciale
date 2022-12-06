from PIL import Image, ImageDraw
import face_recognition
import numpy as np
# Chargez un exemple d'image et apprenez a la reconnaitre
image_soulaima = face_recognition.load_image_file("/Users/lenovo/Desktop/Project-reconnaissance faciale/soul.jpg")
encodage_visage_soulaima = face_recognition.face_encodings(image_soulaima)[0]
image_chiraz = face_recognition.load_image_file("/Users/lenovo/Desktop/Project-reconnaissance faciale/chiraz.jpg")
encodage_visage_chiraz = face_recognition.face_encodings(image_chiraz)[0]
# Creer une liste d'encodages de visage connus et leurs noms
encodage_visage_connu = [
    encodage_visage_soulaima,
    encodage_visage_chiraz
]
nom_visage_connu = [
    "Soulaima Nouichi",
    "Chiraz Dridi"
]


# Charger une image avec un visage inconnu
image_inconnu = face_recognition.load_image_file("/Users/lenovo/Desktop/Project-reconnaissance faciale/inconnu3.jpg")

# Trouver tous les visages et encodages de visage dans l'image inconnue
emp_visage_inconnu = face_recognition.face_locations(image_inconnu)
encodage_visage_inconnu = face_recognition.face_encodings(image_inconnu, emp_visage_inconnu)

image_pil = Image.fromarray(image_inconnu)
draw = ImageDraw.Draw(image_pil)


# Traverser chaque visage trouve dans l'image inconnue
for (haut, droite, bas, gauche), encodage_visage in zip(emp_visage_inconnu, encodage_visage_inconnu):
    # Voir si le visage correspond au visage connu
    corresp = face_recognition.compare_faces(encodage_visage_connu, encodage_visage)
    # [True, False]
    
    nom = "Inconnu"
    # Ou a la place, utilisez le visage connu avec la plus petite distance par rapport au nouveau visage
    distances_visages = face_recognition.face_distance(encodage_visage_connu, encodage_visage)
    meilleur_indice = np.argmin(distances_visages)
    if corresp[meilleur_indice]:
        nom = nom_visage_connu[meilleur_indice]
    # Dessinez une boite autour du visage a l'aide du module Pillow
    draw.rectangle(((gauche, haut), (droite, bas)), outline=(0, 0, 255))
    # Dessinez une etiquette avec un nom sous le visage
    largeur_texte, hauteur_texte = draw.textsize(nom)
    draw.text((gauche + 6, bas - hauteur_texte - 5), nom, fill=(255, 255, 255, 255))
# Afficher l'image resultante
image_pil.show()
#image_pil.save("/Users/lenovo/Desktop/Project-reconnaissance faciale/im2.jpg") #- Enregistrer l'image



print("hello success", encodage_visage_soulaima)