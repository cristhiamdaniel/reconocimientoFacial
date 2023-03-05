import cv2


class DetectorDeCaras:

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                                  + 'haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0)

    def detectar_caras(self):
        while True:
            ret, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()



detector = DetectorDeCaras()
detector.detectar_caras()
