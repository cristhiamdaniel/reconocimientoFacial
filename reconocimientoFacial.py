import cv2
import os

dataPath = "/home/cristhiamdaniel/Desarrollo/ComputerVision/imagenesRostros" #Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath) #Lista con las rutas de las imagenes
print('imagePaths=',imagePaths) #Imprimimos la lista

#face_recognizer = cv2.face.EigenFaceRecognizer_create() #Creamos el reconocedor
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create() #Creamos el reconocedor

# Leyendo el modelo
#face_recognizer.read('modeloEigenFace.xml') #Cargamos el modelo
#face_recognizer.read('modeloFisherFace.xml') #Cargamos el modelo
#face_recognizer.read('modeloLBPHFace.xml') #Cargamos el modelo
face_recognizer.read('modeloLBPHFace_clase.xml') #Cargamos el modelo

cap = cv2.VideoCapture(0,cv2.CAP_V4L) # Abrimos la camara

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') #Cargamos el clasificador

while True:
    ret, frame = cap.read() #Leemos el frame
    if ret == False: break #Si el frame no llega salimos
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convertimos a escala de grises
    auxFrame = gray.copy() #Hacemos una copia para no modificar el original

    faces = faceClassif.detectMultiScale(gray, 1.1, 5) #Buscamos las coordenadas de los rostros

    for (x,y,w,h) in faces:
        rostro = auxFrame[y:y+h,x:x+w] #Obtenemos el rostro
        rostro = cv2.resize(rostro, (150,150), interpolation = cv2.INTER_CUBIC) #Lo redimensionamos
        result = face_recognizer.predict(rostro) #Usamos el modelo para predecir

        cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA) #Mostramos el resultado en la imagen

        # EigenFaces
        if result[1] < 5700: # Si el resultado es menor a 5700 es porque es una persona conocida
            cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA) #Mostramos el nombre de la persona
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2) #Dibujamos el rectangulo
        else:
            cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA) # Mostramos el nombre de la persona
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2) #Dibujamos el rectangulo

    cv2.imshow('frame', frame) #Mostramos la imagen

    k =  cv2.waitKey(1) #Esperamos a que se presione una tecla
    if k == 27: #Si es ESC salimos
        break #Salimos del while

cap.release() #Liberamos la camara
cv2.destroyAllWindows() #Cerramos todas las ventanas