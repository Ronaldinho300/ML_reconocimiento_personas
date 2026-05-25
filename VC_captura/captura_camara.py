"""
captura_camara.py
Responsabilidad: Capturar frames de la cámara y guardarlos como archivos .jpg.
Útil para crear datasets de entrenamiento (persona quieta, movimiento, etc.).
"""

import cv2
import os
import time
from datetime import datetime
from camara import Camara


class Capturador:
    def __init__(self, carpeta_salida="dataset_crudo"):
        self.carpeta_salida = carpeta_salida
        self._crear_carpeta()

    def _crear_carpeta(self):
        """Crea la carpeta de salida si no existe."""
        if not os.path.exists(self.carpeta_salida):
            os.makedirs(self.carpeta_salida)
            print(f"[Captura] Carpeta creada: {self.carpeta_salida}")

    def capturar_unica(self, frame, etiqueta="frame"):
        """
        Guarda un único frame como imagen.
        :param frame: Imagen BGR
        :param etiqueta: Prefijo del nombre (ej: 'quieta', 'movimiento', 'sospechoso')
        :return: Ruta del archivo guardado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        nombre_archivo = f"{etiqueta}_{timestamp}.jpg"
        ruta = os.path.join(self.carpeta_salida, nombre_archivo)

        cv2.imwrite(ruta, frame)
        print(f"[Captura] Guardado: {ruta}")
        return ruta

    def captura_continua(self, camara, intervalo_seg=2, etiqueta="frame"):
        """
        Captura automáticamente cada X segundos.
        :param camara: Instancia de Camara ya conectada
        :param intervalo_seg: Tiempo entre capturas
        :param etiqueta: Categoría de la imagen
        """
        print(f"[Captura] Modo continuo cada {intervalo_seg}s. Presiona 'q' para salir.")
        ultima_captura = 0

        while True:
            ret, frame = camara.leer_frame()
            if not ret:
                break

            # Mostrar preview
            preview = frame.copy()
            tiempo_actual = time.time()

            # Indicador de próxima captura
            if tiempo_actual - ultima_captura >= intervalo_seg:
                self.capturar_unica(frame, etiqueta)
                ultima_captura = tiempo_actual
                cv2.putText(preview, "CAPTURADO!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Mostrar countdown
            restante = intervalo_seg - (tiempo_actual - ultima_captura)
            cv2.putText(preview, f"Proxima captura: {restante:.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Captura Continua", preview)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# --- Ejecución directa ---
if __name__ == "__main__":
    cam = Camara().conectar()
    capturador = Capturador(carpeta_salida="dataset_entrenamiento")

    # Modo manual: presiona ESPACIO para capturar, 'q' para salir
    print("[Captura Manual] ESPACIO = capturar | Q = salir")

    while True:
        ret, frame = cam.leer_frame()
        if not ret:
            break

        preview = frame.copy()
        cv2.putText(preview, "ESPACIO: Capturar | Q: Salir", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Captura Manual", preview)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord(' '):  # ESPACIO
            capturador.capturar_unica(frame, etiqueta="manual")
        elif tecla == ord('q'):
            break

    cam.liberar()