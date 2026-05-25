"""
reconocimiento.py
Responsabilidad: Detectar personas en frames usando HOG + SVM de OpenCV.
Dibuja rectángulos y etiquetas. NO captura, NO preprocesa para TFLite.
"""

import cv2
import numpy as np
from camara import Camara


class ReconocimientoPersonas:
    def __init__(self):
        # Detector HOG pre-entrenado para personas
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detectar(self, frame):
        """
        Detecta personas en un frame.
        :param frame: Imagen BGR de OpenCV
        :return: (frame_dibujado, lista_boxes, cantidad_personas)
        """
        # Detectar personas: boxes = [(x, y, w, h), ...]
        boxes, pesos = self.hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.05
        )

        cantidad = len(boxes)

        for i, (x, y, w, h) in enumerate(boxes):
            # Dibujar rectángulo verde
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Etiqueta con número de persona
            etiqueta = f"Persona {i + 1}"
            cv2.putText(frame, etiqueta, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Mensaje general si hay personas
        if cantidad > 0:
            cv2.putText(frame, f"Personas Detectadas: {cantidad}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return frame, boxes, cantidad


# --- Ejecución directa ---
if __name__ == "__main__":
    cam = Camara().conectar()
    detector = ReconocimientoPersonas()

    print("[Reconocimiento] Presiona 'q' para salir.")

    while True:
        ret, frame = cam.leer_frame()
        if not ret:
            break

        frame_procesado, boxes, total = detector.detectar(frame.copy())

        cv2.imshow("Reconocimiento de Personas", frame_procesado)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.liberar()