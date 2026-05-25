"""
main.py
Orquestador principal del Sistema de Vigilancia Inteligente.
Integra: camara.py | reconocimiento.py | captura_camara.py | proceso_imagenes.py
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime

# ============================================================
# IMPORTAR MODULOS PROPIOS
# ============================================================
try:
    from camara import Camara
    from reconocimiento import ReconocimientoPersonas
    from captura_camara import Capturador
    from proceso_imagenes import ProcesadorImagenes
    MODULOS_OK = True
except ImportError as e:
    print(f"[ERROR] No se pudo importar un modulo: {e}")
    print("Asegurate de que los 4 archivos esten en la misma carpeta.")
    MODULOS_OK = False
    exit(1)


class SistemaVigilancia:
    """
    Sistema completo que une captura, deteccion, clasificacion y alertas.
    """

    def __init__(self):
        print("=" * 60)
        print("  SISTEMA DE VIGILANCIA INTELIGENTE - ORQUESTADOR")
        print("=" * 60)

        # --- 1. Camara ---
        print("\n[1/4] Iniciando camara...")
        self.camara = Camara(indice=0, ancho=640, alto=480)
        self.camara.conectar()
        print("      ✓ Camara lista.")

        # --- 2. Detector de personas ---
        print("[2/4] Cargando detector de personas (HOG)...")
        self.detector = ReconocimientoPersonas()
        print("      ✓ Detector listo.")

        # --- 3. Capturador de imagenes ---
        print("[3/4] Preparando capturador de dataset...")
        self.capturador = Capturador(carpeta_salida="dataset_capturado")
        print("      ✓ Capturador listo.")

        # --- 4. Procesador de imagenes ---
        print("[4/4] Iniciando procesador para TensorFlow...")
        self.procesador = ProcesadorImagenes(tamano_modelo=(224, 224), normalizar=True)
        print("      ✓ Procesador listo.")

        # --- Estado del sistema ---
        self.modo_captura = False          # Si True, guarda frames al presionar ESPACIO
        self.modo_clasificacion = False    # Si True, usa modelo TF (dummy por ahora)
        self.tracker = {}                  # Seguimiento de presencia prolongada
        self.contador_frames = 0
        self.ultimo_log = 0

        print("\n" + "=" * 60)
        print("  TODOS LOS MODULOS CARGADOS CORRECTAMENTE")
        print("=" * 60)

    def mostrar_menu(self, frame, total_personas):
        """Dibuja el panel de informacion en pantalla."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (340, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        lineas = [
            f"Personas detectadas: {total_personas}",
            f"Modo captura: {'ON' if self.modo_captura else 'OFF'} (barra espaciadora)",
            f"Modo clasif: {'ON' if self.modo_clasificacion else 'OFF'} (tecla C)",
            "[Q] Salir  |  [G] Guardar frame  |  [L] Ver log",
        ]

        for i, texto in enumerate(lineas):
            color = (0, 255, 255) if i == 0 else (200, 200, 200)
            cv2.putText(frame, texto, (15, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        return frame

    def verificar_alertas(self, frame, boxes, total_personas):
        """Genera alertas visuales segun reglas de negocio."""
        alertas = []
        color_borde = (0, 255, 0)

        # Alerta 1: Multiples personas
        if total_personas > 3:
            alertas.append("ALERTA: Multiples personas detectadas!")
            color_borde = (0, 0, 255)

        # Alerta 2: Presencia prolongada
        ahora = time.time()
        ids_activos = set()

        for i, (x, y, w, h) in enumerate(boxes):
            pid = f"p_{x}_{y}"
            ids_activos.add(pid)

            if pid not in self.tracker:
                self.tracker[pid] = ahora
            else:
                duracion = ahora - self.tracker[pid]
                if duracion > 10:
                    alertas.append(f"ALERTA: Presencia prolongada ({duracion:.0f}s)")
                    color_borde = (0, 165, 255)
                    cv2.putText(frame, f"{duracion:.0f}s", (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        self.tracker = {k: v for k, v in self.tracker.items() if k in ids_activos}

        if alertas:
            for i, alerta in enumerate(alertas[-3:]):
                cv2.putText(frame, alerta, (15, 460 - i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(frame, (0, 0), (639, 479), color_borde, 4)

        return frame

    def log_evento(self, mensaje):
        """Registra eventos en archivo de log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("log_vigilancia.txt", "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {mensaje}\n")

    def ejecutar(self):
        """Bucle principal del sistema."""
        print("\n[INICIO] Sistema corriendo. Controles:")
        print("         ESPACIO = Capturar imagen")
        print("         C       = Activar/Desactivar clasificacion TF")
        print("         G       = Guardar frame actual")
        print("         Q       = Salir\n")

        clases = ["Quieta", "Movimiento", "Sospechoso"]

        while True:
            ret, frame = self.camara.leer_frame()
            if not ret:
                print("[ERROR] Frame no leido. Reintentando...")
                continue

            self.contador_frames += 1
            frame_salida = frame.copy()

            # --- DETECCION DE PERSONAS ---
            frame_dibujado, boxes, total = self.detector.detectar(frame_salida)

            # --- CLASIFICACION (si esta activa) ---
            if self.modo_clasificacion and total > 0:
                batch = self.procesador.preprocesar_batch(frame, boxes)

                if batch.size > 0:
                    # DEMO: Simulacion de clasificacion aleatoria
                    predicciones = np.random.rand(len(boxes), 3)
                    predicciones = predicciones / predicciones.sum(axis=1, keepdims=True)

                    for i, (box, pred) in enumerate(zip(boxes, predicciones)):
                        clase_idx = np.argmax(pred)
                        confianza = np.max(pred)
                        etiqueta = f"{clases[clase_idx]} ({confianza:.0%})"

                        x, y, w, h = box
                        colores = [(0, 255, 0), (0, 255, 255), (0, 0, 255)]
                        color = colores[clase_idx]

                        cv2.putText(frame_dibujado, etiqueta, (x, y - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                        if clase_idx == 2 and confianza > 0.6:
                            cv2.putText(frame_dibujado, "!!! SOSPECHOSO !!!", (x, y - 45),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                            self.log_evento("Movimiento sospechoso detectado")

            # --- ALERTAS ---
            frame_dibujado = self.verificar_alertas(frame_dibujado, boxes, total)

            # --- MENU / HUD ---
            frame_final = self.mostrar_menu(frame_dibujado, total)

            # --- MOSTRAR ---
            cv2.imshow("Sistema de Vigilancia Inteligente", frame_final)

            # --- CONTROLES ---
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord('q'):
                print("\n[FIN] Cerrando sistema...")
                break

            elif tecla == ord(' '):
                self.capturador.capturar_unica(frame, etiqueta="manual")
                self.log_evento(f"Captura manual - {total} personas")

            elif tecla == ord('g'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre = f"frame_guardado_{ts}.jpg"
                cv2.imwrite(nombre, frame)
                print(f"[INFO] Frame guardado: {nombre}")

            elif tecla == ord('c'):
                self.modo_clasificacion = not self.modo_clasificacion
                estado = "ACTIVADA" if self.modo_clasificacion else "DESACTIVADA"
                print(f"[INFO] Clasificacion {estado}")

            # --- LOG AUTOMATICO cada 5 segundos ---
            if time.time() - self.ultimo_log > 5:
                if total > 0:
                    self.log_evento(f"Estado: {total} persona(s) detectada(s)")
                self.ultimo_log = time.time()

        # --- CIERRE ---
        self.camara.liberar()
        print("[FIN] Sistema cerrado correctamente.")
        print(f"[INFO] Total de frames procesados: {self.contador_frames}")
        print(f"[INFO] Log guardado en: log_vigilancia.txt")


# ============================================================
# PUNTO DE ENTRADA
# ============================================================
if __name__ == "__main__":
    if not MODULOS_OK:
        exit(1)

    sistema = SistemaVigilancia()
    sistema.ejecutar()