# Reconocimiento Facial

```detectarRostroVideo.py``` es un programa que detecta el rostro de una persona usando la camara web del computador.

```almacenarRostros.py``` es un programa que detecta el rostro de una persona usando la camara web y usando la tecla ```s``` almacena la captura del rostro en un directorio que el sistema lo crea ```imagenes/```

Para automatizar el proceso de crear un banco de imagenes, se crea el siguiente programa: ```bancoImagenes.py``` para generar 300 capturas de manera automatica durante los 27 segundos que dura el streaming. Previamente debes asignar la etiqueta (nombre) de la persona a la cual se estan extrayendo las imagenes.

Una vez que se cuenta con un banco de imagenes de los rostros de cada persona, realizamos el entrenamiento del modelo con ```entrenamientoRostros.py``` usandoo algoritmos como: ```EigenFaces```, ```FisherFaces``` y ```LBPH```. El programa ```entrenamiento.py``` realiza el entrenamiento de los algoritmos y genera un archivo ```modeloEigenFaces.xml``` que contiene el modelo entrenado.

Finalmente usamos el programa ```reconocimientoFacial.py``` para reconocer el rostro de una persona usando la camara web del computador.
