"""
detector.py
Responsabilidad: Detectar MULTIPLES personas usando YOLOv8 + MoveNet TF Lite.
- YOLOv8-nano: Detecta cuántas personas hay en la escena
- MoveNet: Extrae 17 keypoints de CADA persona detectada
- Colorea bounding box según clasificación (verde/amarillo/rojo)
"""

import os
import cv2
import numpy as np
import tensorflow as tf


class DetectorPersonas:
    def __init__(self, ruta_movenet="modelos/movenet_lightning.tflite", umbral_confianza=0.3):
        self.umbral_confianza = umbral_confianza
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.yolo = None
        self._cargar_movenet(ruta_movenet)
        self._cargar_yolo()

    def _cargar_movenet(self, ruta):
        if not os.path.exists(ruta):
            raise FileNotFoundError(
                "No se encontro MoveNet: " + ruta + ""
                "Descargalo desde: https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning/3"
            )
        self.interpreter = tf.lite.Interpreter(model_path=ruta)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("[Detector] MoveNet cargado.")

    def _cargar_yolo(self):
        try:
            from ultralytics import YOLO
            self.yolo = YOLO("yolov8n.pt")
            print("[Detector] YOLOv8-nano cargado (multi-persona).")
        except ImportError:
            print("[ADVERTENCIA] ultralytics no instalado. Instala con: pip install ultralytics")
            self.yolo = None

    def detectar(self, frame, clasificaciones=None, renderizar=True):
        """
        Detecta multiples personas.
        :param frame: Imagen BGR de OpenCV
        :param clasificaciones: Lista de dicts del clasificador (opcional, para colorear boxes)
        :param renderizar: Si True, dibuja keypoints, boxes y esqueleto. Si False, retorna frame limpio.
        :return: (frame_dibujado_o_limpio, boxes, total_personas, lista_keypoints)
        """
        h, w, _ = frame.shape
        boxes = []
        keypoints_list = []

        # --- PASO 1: YOLOv8 detecta TODAS las personas ---
        if self.yolo is not None:
            resultados_yolo = self.yolo(frame, verbose=False, classes=[0])
            for r in resultados_yolo:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    if conf > 0.5:
                        boxes.append((x1, y1, x2 - x1, y2 - y1))

        if len(boxes) == 0:
            boxes = [(0, 0, w, h)]

        # --- PASO 2: MoveNet extrae keypoints de CADA persona ---
        for (x, y, bw, bh) in boxes:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                keypoints_list.append(None)
                continue

            roi_resized = cv2.resize(roi, (192, 192))
            input_img = np.expand_dims(roi_resized, axis=0).astype(np.float32)

            self.interpreter.set_tensor(self.input_details[0]["index"], input_img)
            self.interpreter.invoke()
            kps_raw = self.interpreter.get_tensor(self.output_details[0]["index"])
            kps = np.squeeze(kps_raw)

            rh, rw = roi.shape[:2]
            scale_x = rw / 192.0
            scale_y = rh / 192.0

            kps_ajustados = np.zeros_like(kps)
            for i in range(17):
                kps_ajustados[i][0] = (kps[i][0] * 192.0 * scale_y + y1) / h
                kps_ajustados[i][1] = (kps[i][1] * 192.0 * scale_x + x1) / w
                kps_ajustados[i][2] = kps[i][2]

            keypoints_list.append(kps_ajustados)

        total = len(boxes)

        if not renderizar:
            return frame, boxes, total, keypoints_list

        # --- PASO 3: Dibujar ---
        frame_dibujado = self.renderizar(frame, boxes, keypoints_list, clasificaciones)
        return frame_dibujado, boxes, total, keypoints_list

    def renderizar(self, frame, boxes, keypoints_list, clasificaciones=None):
        """
        Dibuja boxes, keypoints y esqueleto sobre una copia del frame.
        Permite dibujar por separado sin re-ejecutar la detección.
        """
        frame_dibujado = frame.copy()
        h, w = frame.shape[:2]
        total = len(boxes)

        colores_clase = {
            0: (0, 255, 0),     # Quieta -> VERDE
            1: (0, 255, 255),   # Movimiento -> AMARILLO
            2: (0, 0, 255)      # Sospechoso -> ROJO
        }

        for i, (x, y, bw, bh) in enumerate(boxes):
            color = (0, 255, 0)  # Default verde
            etiqueta_clase = ""

            if clasificaciones and i < len(clasificaciones):
                c = clasificaciones[i]
                color = colores_clase.get(c["clase_idx"], (0, 255, 0))
                etiqueta_clase = " [" + c["nombre"] + "]"

                # Si es sospechoso, dibujar alerta grande
                if c["clase_idx"] == 2 and c["confianza"] > 0.6:
                    cv2.putText(frame_dibujado, "!!! ALERTA SOSPECHOSO !!!", 
                                (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.8, (0, 0, 255), 3)

            # Bounding box con color de clasificación
            cv2.rectangle(frame_dibujado, (x, y), (x + bw, y + bh), color, 3)

            # Etiqueta
            texto = "Persona " + str(i + 1) + etiqueta_clase
            cv2.putText(frame_dibujado, texto, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Keypoints y esqueleto
            if keypoints_list[i] is not None:
                self._dibujar_keypoints(frame_dibujado, keypoints_list[i], h, w)
                self._dibujar_esqueleto(frame_dibujado, keypoints_list[i], h, w)

        # Mensaje general
        if total > 0:
            cv2.putText(frame_dibujado, "Personas: " + str(total), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return frame_dibujado

    def _dibujar_keypoints(self, frame, keypoints, h, w):
        for kp in keypoints:
            if kp[2] > self.umbral_confianza:
                px = int(kp[1] * w)
                py = int(kp[0] * h)
                cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)
                cv2.circle(frame, (px, py), 5, (255, 255, 255), 1)

    def _dibujar_esqueleto(self, frame, keypoints, h, w):
        conexiones = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (3, 5), (4, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 6), (5, 11),
            (6, 12), (11, 13), (13, 15), (12, 14),
            (14, 16), (11, 12),
        ]
        for a, b in conexiones:
            if keypoints[a][2] > self.umbral_confianza and keypoints[b][2] > self.umbral_confianza:
                pa = (int(keypoints[a][1] * w), int(keypoints[a][0] * h))
                pb = (int(keypoints[b][1] * w), int(keypoints[b][0] * h))
                cv2.line(frame, pa, pb, (255, 0, 0), 2)


if __name__ == "__main__":
    from camara import Camara
    cam = Camara().conectar()
    detector = DetectorPersonas()
    print("[Detector] Presiona 'q' para salir.")
    while True:
        ret, frame = cam.leer_frame()
        if not ret:
            break
        frame_procesado, boxes, total, kps = detector.detectar(frame.copy())
        cv2.imshow("Multi-Persona: YOLO + MoveNet", frame_procesado)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.liberar()