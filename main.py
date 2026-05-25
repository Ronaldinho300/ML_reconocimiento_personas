"""
main.py
Orquestador principal - VERSION MULTI-PERSONA.
- YOLOv8: Detecta cuantas personas hay
- MoveNet: Pose estimation por persona
- Clasificador: Alertas individuales por persona sospechosa
"""

import sys
import os

raiz = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(raiz, "VC_captura"))
sys.path.insert(0, raiz)

import cv2
import numpy as np
import time
import platform
from datetime import datetime

try:
    from camara import Camara
    from detector import DetectorPersonas
    from captura_camara import Capturador
    from clasificador import ClasificadorComportamiento
    MODULOS_OK = True
except ImportError as e:
    print("[ERROR] No se pudo importar: " + str(e))
    MODULOS_OK = False
    exit(1)


def sonido_alerta():
    sistema = platform.system()
    try:
        if sistema == "Windows":
            import winsound
            winsound.Beep(1000, 500)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


class SistemaVigilancia:
    def __init__(self):
        print("=" * 60)
        print("  SISTEMA DE VIGILANCIA INTELIGENTE - MULTI-PERSONA")
        print("  (YOLOv8 + MoveNet + Clasificacion geometrica)")
        print("=" * 60)

        print()
        print("[1/4] Iniciando camara...")
        self.camara = Camara(indice=0, ancho=640, alto=480)
        self.camara.conectar()
        print("      ✓ Camara lista.")

        print("[2/4] Cargando detector (YOLOv8 + MoveNet)...")
        self.detector = DetectorPersonas()
        print("      ✓ Detector listo.")

        print("[3/4] Preparando capturador...")
        self.capturador = Capturador(carpeta_salida="dataset_capturado")
        print("      ✓ Capturador listo.")

        print("[4/4] Iniciando clasificador...")
        self.clasificador = ClasificadorComportamiento(
            umbral_quieta=150.0,
            umbral_sospechoso=800.0,
            historial_frames=15
        )
        print("      ✓ Clasificador listo.")

        self.modo_captura = False
        self.modo_clasificacion = False
        self.tracker_presencia = {}
        self.contador_frames = 0
        self.ultimo_log = 0
        self.ultimo_sonido = 0

        print()
        print("=" * 60)
        print("  TODOS LOS MODULOS CARGADOS")
        print("=" * 60)

    def mostrar_menu(self, frame, total_personas, num_sospechosos):
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (420, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        alerta_texto = ""
        if num_sospechosos > 0:
            alerta_texto = " | ⚠ SOSPECHOSOS: " + str(num_sospechosos)

        lineas = [
            "Personas detectadas: " + str(total_personas) + alerta_texto,
            "Modo captura: " + ("ON" if self.modo_captura else "OFF") + " (ESPACIO)",
            "Modo clasif: " + ("ON" if self.modo_clasificacion else "OFF") + " (C)",
            "Detector: YOLOv8 + MoveNet",
            "[Q] Salir | [G] Guardar frame | [L] Log",
        ]

        for i, texto in enumerate(lineas):
            color = (0, 255, 255) if i == 0 else (200, 200, 200)
            if num_sospechosos > 0 and i == 0:
                color = (0, 0, 255)
            cv2.putText(frame, texto, (15, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)

        return frame

    def verificar_alertas(self, frame, boxes, total_personas, clasificaciones):
        alertas = []
        color_borde = (0, 255, 0)
        ahora = time.time()
        num_sospechosos = 0

        # Alerta 1: Multiples personas
        if total_personas > 3:
            alertas.append("ALERTA: Multiples personas (" + str(total_personas) + ")!")
            color_borde = (0, 0, 255)

        # Alerta 2: Personas sospechosas individuales
        if clasificaciones:
            for c in clasificaciones:
                if c["nombre"] == "Sospechoso" and c["confianza"] > 0.6:
                    num_sospechosos += 1
                    alertas.append("ALERTA: Persona " + str(c["id"]) + " SOSPECHOSA!")
                    color_borde = (0, 0, 255)
                    x, y, w, h = c["box"]
                    cv2.putText(frame, "!!! SOSPECHOSO !!!", (x, y - 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                    if ahora - self.ultimo_sonido > 3:
                        sonido_alerta()
                        self.ultimo_sonido = ahora
                    self.log_evento("Persona " + str(c["id"]) + " - Movimiento sospechoso")

        # Alerta 3: Presencia prolongada
        ids_activos = set()
        for i, (x, y, w, h) in enumerate(boxes):
            pid = "p_" + str(x) + "_" + str(y)
            ids_activos.add(pid)
            if pid not in self.tracker_presencia:
                self.tracker_presencia[pid] = ahora
            else:
                duracion = ahora - self.tracker_presencia[pid]
                if duracion > 10:
                    alertas.append("ALERTA: Presencia prolongada (" + str(int(duracion)) + "s)")
                    color_borde = (0, 165, 255)
                    cv2.putText(frame, str(int(duracion)) + "s", (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        self.tracker_presencia = {k: v for k, v in self.tracker_presencia.items() if k in ids_activos}

        if alertas:
            for i, alerta in enumerate(alertas[-3:]):
                cv2.putText(frame, alerta, (15, 460 - i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(frame, (0, 0), (639, 479), color_borde, 4)

        return frame, num_sospechosos

    def log_evento(self, mensaje):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("log_vigilancia.txt", "a", encoding="utf-8") as f:
            f.write("[" + timestamp + "] " + mensaje + "")

    def ejecutar(self):
        print()
        print("[INICIO] Controles: ESPACIO=captura | C=clasificar | G=guardar | Q=salir")
        print()

        while True:
            ret, frame = self.camara.leer_frame()
            if not ret:
                print("[ERROR] Frame no leido. Reintentando...")
                continue

            self.contador_frames += 1
            h, w, _ = frame.shape

            # DETECCION
            frame_dibujado, boxes, total, keypoints_list = self.detector.detectar(frame.copy())
            clasificaciones = []

            # CLASIFICACION
            if self.modo_clasificacion and total > 0:
                clasificaciones = self.clasificador.clasificar(boxes, keypoints_list, h, w)

                for c in clasificaciones:
                    x, y, bw, bh = c["box"]
                    etiqueta = c["nombre"] + " (" + str(int(c["confianza"] * 100)) + "%)"
                    colores = [(0, 255, 0), (0, 255, 255), (0, 0, 255)]
                    color = colores[c["clase_idx"]]

                    cv2.putText(frame_dibujado, etiqueta, (x, y - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # ALERTAS
            num_sospechosos = 0
            frame_dibujado, num_sospechosos = self.verificar_alertas(
                frame_dibujado, boxes, total, clasificaciones
            )

            # MENU
            frame_final = self.mostrar_menu(frame_dibujado, total, num_sospechosos)

            cv2.imshow("Sistema de Vigilancia Inteligente", frame_final)

            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q"):
                print()
                print("[FIN] Cerrando sistema...")
                break
            elif tecla == ord(" "):
                self.capturador.capturar_unica(frame, etiqueta="manual")
                self.log_evento("Captura manual - " + str(total) + " personas")
            elif tecla == ord("g"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre = "frame_guardado_" + ts + ".jpg"
                cv2.imwrite(nombre, frame)
                print("[INFO] Frame guardado: " + nombre)
            elif tecla == ord("c"):
                self.modo_clasificacion = not self.modo_clasificacion
                estado = "ACTIVADA" if self.modo_clasificacion else "DESACTIVADA"
                print("[INFO] Clasificacion " + estado)
                if not self.modo_clasificacion:
                    self.clasificador.reset()

            if time.time() - self.ultimo_log > 5:
                if total > 0:
                    self.log_evento("Estado: " + str(total) + " persona(s), " + str(num_sospechosos) + " sospechosas")
                self.ultimo_log = time.time()

        self.camara.liberar()
        print("[FIN] Sistema cerrado.")
        print("[INFO] Frames procesados: " + str(self.contador_frames))
        print("[INFO] Log: log_vigilancia.txt")


if __name__ == "__main__":
    if not MODULOS_OK:
        exit(1)
    sistema = SistemaVigilancia()
    sistema.ejecutar()