"""
camara.py
Responsabilidad: ÚNICAMENTE gestionar la conexión con la cámara web.
No detecta, no procesa, solo entrega frames crudos.
"""

import cv2


class Camara:
    def __init__(self, indice=0, ancho=640, alto=480):
        """
        Inicializa la cámara web.
        :param indice: 0 = cámara integrada, 1 = cámara externa, etc.
        :param ancho: Resolución horizontal
        :param alto: Resolución vertical
        """
        self.indice = indice
        self.ancho = ancho
        self.alto = alto
        self.cap = None

    def conectar(self):
        """Abre la conexión con la cámara."""
        self.cap = cv2.VideoCapture(self.indice)

        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara en el índice {self.indice}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.ancho)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.alto)

        print(f"[Camara] Conectada en índice {self.indice} ({self.ancho}x{self.alto})")
        return self

    def leer_frame(self):
        """
        Lee un frame de la cámara.
        :return: (bool, frame) - (éxito, imagen)
        """
        if self.cap is None:
            raise RuntimeError("La cámara no está conectada. Llama a .conectar() primero.")
        return self.cap.read()

    def liberar(self):
        """Libera la cámara y cierra ventanas."""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
            print("[Camara] Cámara liberada.")


# --- Ejecución directa (prueba) ---
if __name__ == "__main__":
    cam = Camara().conectar()

    while True:
        ret, frame = cam.leer_frame()
        if not ret:
            break

        cv2.imshow("Camara - Solo conexion", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.liberar()