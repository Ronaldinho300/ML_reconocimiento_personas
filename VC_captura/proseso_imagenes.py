"""
proceso_imagenes.py
Responsabilidad: Recibir frames o imágenes y transformarlas al formato
exacto que espera un modelo TensorFlow/Keras (resize, normalizar, batch).
"""

import cv2
import numpy as np


class ProcesadorImagenes:
    def __init__(self, tamano_modelo=(224, 224), normalizar=True):
        """
        :param tamano_modelo: (alto, ancho) que espera el modelo TF/Keras
        :param normalizar: Si True, escala píxeles a [0, 1]
        """
        self.tamano = tamano_modelo
        self.normalizar = normalizar

    def recortar_persona(self, frame, box):
        """
        Recorta la región de interés (ROI) de una persona detectada.
        :param frame: Imagen completa (BGR)
        :param box: (x, y, w, h)
        :return: Imagen recortada (BGR)
        """
        x, y, w, h = box
        alto, ancho = frame.shape[:2]

        # Asegurar límites
        x = max(0, x)
        y = max(0, y)
        w = min(w, ancho - x)
        h = min(h, alto - y)

        roi = frame[y:y + h, x:x + w]
        return roi

    def preprocesar(self, imagen):
        """
        Pipeline completo para UNA imagen: resize → float32 → normalizar.
        :param imagen: Imagen BGR de OpenCV (cualquier tamaño)
        :return: np.array con shape (224, 224, 3) listo para apilar en batch
        """
        # 1. Resize al tamaño del modelo
        imagen_redimensionada = cv2.resize(imagen, self.tamano)

        # 2. Convertir a float32 (Keras/TF espera float32)
        imagen_float = imagen_redimensionada.astype(np.float32)

        # 3. Normalizar [0, 255] → [0, 1]
        if self.normalizar:
            imagen_float = imagen_float / 255.0

        return imagen_float

    def preprocesar_roi(self, frame, box):
        """
        Pipeline completo desde frame + box hasta tensor individual.
        :return: np.array shape (224, 224, 3)
        """
        roi = self.recortar_persona(frame, box)
        return self.preprocesar(roi)

    def preprocesar_batch(self, frame, boxes):
        """
        Procesa MÚLTIPLES ROIs en un solo batch.
        MÁS EFICIENTE para TensorFlow que inferir una por una.
        :param frame: Imagen completa
        :param boxes: Lista de boxes [(x,y,w,h), ...]
        :return: np.array shape (N, 224, 224, 3) donde N = cantidad de personas
        """
        imagenes_procesadas = []

        for box in boxes:
            tensor_individual = self.preprocesar_roi(frame, box)
            imagenes_procesadas.append(tensor_individual)

        if len(imagenes_procesadas) == 0:
            return np.array([])  # Batch vacío

        # Apilar en un solo array: (N, 224, 224, 3)
        batch = np.stack(imagenes_procesadas, axis=0)
        return batch

    def info_tensor(self, tensor, nombre="Tensor"):
        """Muestra información del tensor generado."""
        print(f"[Proceso] {nombre} | Shape: {tensor.shape} | Dtype: {tensor.dtype} | "
              f"Rango: {tensor.min():.4f} - {tensor.max():.4f}")


# ============================================================
# EJEMPLO DE USO CON TENSORFLOW (no TFLite)
# ============================================================

if __name__ == "__main__":
    from camara import Camara
    from reconocimiento import ReconocimientoPersonas
    import tensorflow as tf

    # --- 1. Conectar cámara y detector ---
    cam = Camara().conectar()
    detector = ReconocimientoPersonas()
    procesador = ProcesadorImagenes(tamano_modelo=(224, 224))

    # --- 2. Cargar modelo TensorFlow/Keras ---
    # Reemplaza 'modelo_comportamiento.h5' por tu modelo real
    # modelo = tf.keras.models.load_model('modelo_comportamiento.h5')

    # Para prueba, usamos un modelo dummy con las 3 clases
    print("[TF] Creando modelo dummy para demostración...")
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(3, activation='softmax')  # [quieta, movimiento, sospechoso]
    ])
    modelo.compile(optimizer='adam', loss='categorical_crossentropy')

    clases = ["Persona Quieta", "Persona en Movimiento", "Movimiento Sospechoso"]

    print("[TF] Presiona 'q' para salir.")

    while True:
        ret, frame = cam.leer_frame()
        if not ret:
            break

        # --- 3. Detectar personas ---
        frame_dibujado, boxes, total = detector.detectar(frame.copy())

        if total > 0:
            # --- 4. Preprocesar TODAS las personas en un solo batch ---
            batch = procesador.preprocesar_batch(frame, boxes)
            procesador.info_tensor(batch, "Batch TF")

            # --- 5. Inferencia con TensorFlow ---
            predicciones = modelo.predict(batch, verbose=0)

            # --- 6. Interpretar resultados ---
            for i, (box, pred) in enumerate(zip(boxes, predicciones)):
                clase_idx = np.argmax(pred)
                confianza = np.max(pred)
                etiqueta = f"{clases[clase_idx]} ({confianza:.2%})"

                x, y, w, h = box
                # Dibujar etiqueta de comportamiento encima del rectángulo
                cv2.putText(frame_dibujado, etiqueta, (x, y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Mostrar ROI individual
                roi = procesador.recortar_persona(frame, box)
                roi_preview = cv2.resize(roi, (224, 224))
                cv2.imshow(f"ROI Persona {i + 1}", roi_preview)

        # --- 7. Mostrar frame principal ---
        cv2.imshow("Vigilancia TF - Detecciones + Clasificacion", frame_dibujado)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.liberar()
    cv2.destroyAllWindows()