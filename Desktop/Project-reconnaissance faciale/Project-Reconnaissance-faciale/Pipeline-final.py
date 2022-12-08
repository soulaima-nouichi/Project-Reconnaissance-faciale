import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#===> Etape 1 
#on va Spécifier le chemin de notre dataset des images
path = '/Users/lenovo/Desktop/Project-reconnaissance faciale/Project-Reconnaissance-faciale/img/'
images = []
classNames = []
# on va lister les sous-dossiers dans une liste
myList = os.listdir(path)
#print("Liste des images",myList) #cap1
#on va parcourir notre liste et on charge les images 
for cl in myList:
    #ce variable contient les images sous forme des matrices 
    curImg = cv2.imread(f'{path}/{cl}')
    #print("****",curImg) #cap2
    #on va affecter ces derniers à une liste
    images.append(curImg)
    #on va mettre dans la liste juste les noms des images avec la méthode splitext et on prend la partie de l'indice 0 pour que aprés on les met des data.csv
    classNames.append(os.path.splitext(cl)[0])
#print("***Les Noms des images :",classNames) #cap3


#====> Etape2 : cette fonction permet de calculer tous les codeges pour tous les images qui se trouve ds la liste images 
def findEncodings(images):
    encodeList = []
    for img in images:
        #aprés le read des images on va les convertir les img RGB (bleu,green,red) car on a utilisé openCV et si on utilisé pillow donc RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #Trouver l'encodage
        encode = face_recognition.face_encodings(img)[0]
        #cette liste contient tous les encadages des images
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
#print("Encoding Complete",len(encodeListKnown)) : 4


def markAttendance(name):
    #ouvrir csv pour read et write
    with open('/Users/lenovo/Desktop/Project-reconnaissance faciale/Project-Reconnaissance-faciale/data.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        #print("***liste-name", nameList)
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



#====> Etape 3 : trouver les correspondances entre les codages, donc on va comparer l'image qui proviendra de notre webcam avec ce qu'on a 
#initialisation with web cam 
#Traitement sur l'image webcam
cap = cv2.VideoCapture(0) #ID
#Boucle while pour obtenir chaque image one by one 
while True:
    success, img = cap.read()
    #img = captureScreen()
    #on va reduire la taille de notre image cela nous aide à accélérer le process
    imgS = cv2.resize(img,(0,0),None,0.25,0.25) #taille de pixel, echelle de sorte
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    #face_locations c pour localiser et detecter le visage dans l'image
    facesCurFrame = face_recognition.face_locations(imgS)
    #trouver leur encodage
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    #parcourir tous les visages que nous avons trouvé
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        #on va comparer les images => retour true ou false
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        #on va calculer la distance entre les images (car on plusieurs img) pour savoir a quel point ces images sont similaires
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis) 
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)




    cv2.imshow('Webcam',img)
    cv2.waitKey(1)



"""image_soulaima = face_recognition.load_image_file("/Users/lenovo/Desktop/Project-reconnaissance faciale/img/soul.jpg")
image_soulaima=cv2.cvtColor(image_soulaima,cv2.COLOR_BGR2RGB)

image_test = face_recognition.load_image_file("/Users/lenovo/Desktop/Project-reconnaissance faciale/img/chiraz.jpg")
image_test=cv2.cvtColor(image_test,cv2.COLOR_BGR2RGB)"""