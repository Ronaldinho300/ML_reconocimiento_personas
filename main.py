"""
main.py
Orquestador principal - VERSION COMPLETAMENTE AUTOMATICA.
- No requiere pulsar teclas para clasificar ni capturar
- Clasificacion activa siempre
- Guardado automatico de sospechosos con cooldown por persona
- Presiona 'q' para salir (unica tecla necesaria)
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
    """Emite un beep de alerta."""
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000, 600)
            winsound.Beep(1200, 400)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


class SistemaVigilancia:
    def __init__(self, guardar_sospechosos=True, cooldown_captura_seg=5):
        """
        :param guardar_sospechosos: Si True, guarda capturas automaticamente
        :param cooldown_captura_seg: Segundos entre capturas del mismo sospechoso
        """
        self.guardar_sospechosos = guardar_sospechosos
        self.cooldown_captura = cooldown_captura_seg

        print("=" * 60)
        print("  SISTEMA DE VIGILANCIA INTELIGENTE - AUTOMATICO")
        print("  (YOLOv8 + MoveNet + Clasificacion en tiempo real)")
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
        self.capturador = Capturador(carpeta_salida="alertas_sospechosos")
        print("      ✓ Capturador listo.")

        print("[4/4] Iniciando clasificador...")
        self.clasificador = ClasificadorComportamiento(
            umbral_quieta=100.0,
            umbral_sospechoso=600.0,
            umbral_acercamiento=1.25,
            umbral_desplazamiento=300.0,
            historial_frames=10
        )
        print("      ✓ Clasificador listo.")

        # Estado del sistema
        self.tracker_presencia = {}
        self.contador_frames = 0
        self.ultimo_log = 0
        self.ultimo_sonido = 0
        self.sospechoso_activo = False

        # Cooldown de captura por persona (id -> timestamp ultima captura)
        self.ultima_captura_sospechoso = {}

        print()
        print("=" * 60)
        print("  SISTEMA LISTO - Presiona 'q' para salir")
        print("=" * 60)

    def mostrar_hud(self, frame, total_personas, num_sospechosos, clasificaciones):
        """Dibuja el panel de informacion en pantalla."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (480, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Color segun estado
        color_estado = (0, 255, 0)  # verde = todo normal
        estado_texto = "NORMAL"
        if num_sospechosos > 0:
            color_estado = (0, 0, 255)  # rojo = alerta
            estado_texto = "ALERTA!"
        elif total_personas > 0:
            color_estado = (0, 255, 255)  # amarillo = personas detectadas
            estado_texto = "VIGILANDO"

        lineas = [
            "ESTADO: " + estado_texto,
            "Personas: " + str(total_personas) + " | Sospechosos: " + str(num_sospechosos),
            "Modo: AUTOMATICO (sin intervencion)",
            "[q] Salir del sistema",
        ]

        for i, texto in enumerate(lineas):
            color = color_estado if i == 0 else (200, 200, 200)
            if i == 1 and num_sospechosos > 0:
                color = (0, 0, 255)
            cv2.putText(frame, texto, (15, 35 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)

        # Mostrar info de cada persona
        y_offset = 140
        for c in clasificaciones:
            info = "P" + str(c["id"]) + ": " + c["nombre"]
            if c["nombre"] == "Sospechoso":
                color_info = (0, 0, 255)
                info += " (!)"
            elif c["nombre"] == "Movimiento":
                color_info = (0, 255, 255)
            else:
                color_info = (0, 255, 0)

            cv2.putText(frame, info, (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color_info, 1)
            y_offset += 22

        return frame

    def verificar_alertas(self, frame, boxes, total_personas, clasificaciones):
        """Verifica alertas y guarda capturas automaticas."""
        alertas = []
        color_borde = (0, 255, 0)
        ahora = time.time()
        num_sospechosos = 0
        hay_sospechoso_ahora = False

        # Alerta 1: Multiples personas
        if total_personas > 3:
            alertas.append("ALERTA: Multiples personas (" + str(total_personas) + ")!")
            color_borde = (0, 0, 255)

        # Alerta 2: Personas sospechosas
        if clasificaciones:
            for c in clasificaciones:
                if c["nombre"] == "Sospechoso" and c["confianza"] > 0.5:
                    num_sospechosos += 1
                    hay_sospechoso_ahora = True
                    pid = c["id"]

                    # Dibujar alerta visual
                    x, y, w, h = c["box"]
                    cv2.line(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
                    cv2.line(frame, (x + w, y), (x, y + h), (0, 0, 255), 4)

                    # Guardar captura automatica con cooldown
                    if self.guardar_sospechosos:
                        ultima = self.ultima_captura_sospechoso.get(pid, 0)
                        if ahora - ultima > self.cooldown_captura:
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            nombre = "SOSPECHOSO_P" + str(pid) + "_" + ts + ".jpg"
                            cv2.imwrite(nombre, frame)
                            self.ultima_captura_sospechoso[pid] = ahora
                            self.log_evento("CAPTURA AUTO: Persona " + str(pid) + " -> " + nombre)
                            print("[ALERTA] Captura guardada: " + nombre)

                    # Log
                    self.log_evento("SOSPECHOSO P" + str(pid) + " | munecas=" + 
                                    str(int(c["velocidad_munecas"])) + " despl=" + 
                                    str(int(c["velocidad_desplazamiento"])))

        # Sonido de alerta (solo cuando aparece nuevo sospechoso)
        if hay_sospechoso_ahora and not self.sospechoso_activo:
            if ahora - self.ultimo_sonido > 2:
                sonido_alerta()
                self.ultimo_sonido = ahora
            self.sospechoso_activo = True
        elif not hay_sospechoso_ahora:
            self.sospechoso_activo = False

        # Alerta 3: Presencia prolongada
        ids_activos = set()
        for i, (x, y, w, h) in enumerate(boxes):
            pid_str = "p_" + str(x) + "_" + str(y)
            ids_activos.add(pid_str)
            if pid_str not in self.tracker_presencia:
                self.tracker_presencia[pid_str] = ahora
            else:
                duracion = ahora - self.tracker_presencia[pid_str]
                if duracion > 10:
                    alertas.append("ALERTA: Presencia prolongada (" + str(int(duracion)) + "s)")
                    color_borde = (0, 165, 255)
                    cv2.putText(frame, str(int(duracion)) + "s", (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        self.tracker_presencia = {k: v for k, v in self.tracker_presencia.items() if k in ids_activos}

        # Dibujar alertas en pantalla
        if alertas:
            for i, alerta in enumerate(alertas[-3:]):
                cv2.putText(frame, alerta, (15, 460 - i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(frame, (0, 0), (639, 479), color_borde, 4)

        return frame, num_sospechosos

    def log_evento(self, mensaje):
        """Registra eventos en archivo de log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("log_vigilancia.txt", "a", encoding="utf-8") as f:
            f.write("[" + timestamp + "] " + mensaje + "")

    def ejecutar(self):
        """Bucle principal automatico."""
        print()
        print("[INICIO] Sistema corriendo en modo AUTOMATICO")
        print("         Clasificacion activa | Capturas auto de sospechosos")
        print("         Presiona 'q' para salir")
        print()

        while True:
            ret, frame = self.camara.leer_frame()
            if not ret:
                print("[ERROR] Frame no leido. Reintentando...")
                time.sleep(0.1)
                continue

            self.contador_frames += 1
            h, w, _ = frame.shape

            # --- DETECCION + CLASIFICACION (siempre activa) ---
            frame_dibujado, boxes, total, keypoints_list = self.detector.detectar(frame.copy())
            clasificaciones = []

            if total > 0:
                clasificaciones = self.clasificador.clasificar(boxes, keypoints_list, h, w)
                # Redibujar con colores de clasificacion
                frame_dibujado, _, _, _ = self.detector.detectar(frame.copy(), clasificaciones)

            # --- ALERTAS ---
            frame_dibujado, num_sospechosos = self.verificar_alertas(
                frame_dibujado, boxes, total, clasificaciones
            )

            # --- HUD ---
            frame_final = self.mostrar_hud(frame_dibujado, total, num_sospechosos, clasificaciones)

            cv2.imshow("Sistema de Vigilancia - AUTOMATICO", frame_final)

            # --- UNICA TECLA: 'q' para salir ---
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print()
                print("[FIN] Cerrando sistema...")
                break

            # --- LOG AUTOMATICO cada 5 segundos ---
            if time.time() - self.ultimo_log > 5:
                if total > 0:
                    self.log_evento("Estado: " + str(total) + " personas, " + str(num_sospechosos) + " sospechosas")
                self.ultimo_log = time.time()

        self.camara.liberar()
        print("[FIN] Sistema cerrado correctamente.")
        print("[INFO] Frames procesados: " + str(self.contador_frames))
        print("[INFO] Log guardado en: log_vigilancia.txt")
        if self.guardar_sospechosos:
            print("[INFO] Capturas de sospechosos guardadas en: alertas_sospechosos/")


if __name__ == "__main__":
    if not MODULOS_OK:
        exit(1)

    # Configuracion
    GUARDAR_SOSPECHOSOS = True      # False = no guarda capturas
    COOLDOWN_CAPTURA_SEG = 5        # Segundos entre capturas del mismo sospechoso

    sistema = SistemaVigilancia(
        guardar_sospechosos=GUARDAR_SOSPECHOSOS,
        cooldown_captura_seg=COOLDOWN_CAPTURA_SEG
    )
    sistema.ejecutar()