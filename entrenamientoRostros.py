import cv2
import os
import numpy as np

dataPath = "/home/cristhiamdaniel/Desarrollo/ComputerVision/imagenesRostros"
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = [] #Lista para almacenar las etiquetas
facesData = []  #Lista para almacenar los rostros
label = 0 #Etiqueta para cada persona

for nameDir in peopleList: #Recorremos cada persona
    personPath = dataPath + '/' + nameDir
    print('Leyendo las imágenes')

    for fileName in os.listdir(personPath): #Recorremos cada imagen
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label) #Agregamos la etiqueta
        facesData.append(cv2.imread(personPath+'/'+fileName,0)) #Agregamos el rostro

    label = label + 1 #Aumentamos la etiqueta

#print('labels= ',labels)
#print("Numero de etiquetas 0: ",np.count_nonzero(np.array(labels)==0))
#print("Numero de etiquetas 1: ",np.count_nonzero(np.array(labels)==1))

# Metodo para entrenar el reconocedor
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido
#face_recognizer.write('modeloEigenFace.xml')
#face_recognizer.write('modeloFisherFace.xml')
face_recognizer.write('modeloLBPHFace_clase.xml')
print("Modelo almacenado...")
