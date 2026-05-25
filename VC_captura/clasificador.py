"""
clasificador.py
Responsabilidad: Clasificar comportamiento de MULTIPLES personas.
Usa tracking por bounding box + velocidad de muñecas.
Muestra alertas individuales por persona sospechosa.
"""

import time
import numpy as np


class ClasificadorComportamiento:
    def __init__(self, umbral_quieta=150.0, umbral_sospechoso=800.0, historial_frames=15):
        self.umbral_quieta = umbral_quieta
        self.umbral_sospechoso = umbral_sospechoso
        self.historial_frames = historial_frames
        self.historial = {}   # id_persona -> dict
        self.contador_id = 0

    def _asignar_ids(self, boxes):
        """Asigna IDs persistentes basándose en proximidad del centroide del box."""
        centroides_actuales = []
        for (x, y, w, h) in boxes:
            centroides_actuales.append((x + w / 2.0, y + h / 2.0))

        if not self.historial:
            for i in range(len(centroides_actuales)):
                self.contador_id += 1
                self.historial[self.contador_id] = {
                    "centroide": centroides_actuales[i],
                    "lw": None,
                    "rw": None,
                    "t": time.time(),
                    "velocidades": [],
                    "clase_previa": "Quieta",
                    "sospechoso_count": 0
                }
            return list(range(1, len(centroides_actuales) + 1))

        ids_usados = set()
        asignaciones = []
        ids_disponibles = list(self.historial.keys())

        for c_actual in centroides_actuales:
            mejor_id = None
            mejor_dist = float("inf")
            for pid in ids_disponibles:
                if pid in ids_usados:
                    continue
                c_previo = self.historial[pid]["centroide"]
                dist = np.sqrt((c_actual[0] - c_previo[0])**2 + (c_actual[1] - c_previo[1])**2)
                if dist < mejor_dist and dist < 150:
                    mejor_dist = dist
                    mejor_id = pid

            if mejor_id is not None:
                asignaciones.append(mejor_id)
                ids_usados.add(mejor_id)
                self.historial[mejor_id]["centroide"] = c_actual
            else:
                self.contador_id += 1
                self.historial[self.contador_id] = {
                    "centroide": c_actual,
                    "lw": None,
                    "rw": None,
                    "t": time.time(),
                    "velocidades": [],
                    "clase_previa": "Quieta",
                    "sospechoso_count": 0
                }
                asignaciones.append(self.contador_id)
                ids_usados.add(self.contador_id)

        for pid in list(self.historial.keys()):
            if pid not in ids_usados:
                del self.historial[pid]

        return asignaciones

    def clasificar(self, boxes, keypoints_list, h, w):
        """
        :param boxes: Lista de (x, y, w, h)
        :param keypoints_list: Lista de arrays (17,3) o None
        :param h, w: Dimensiones del frame
        :return: Lista de dicts con info completa por persona
        """
        if not boxes:
            return []

        ids = self._asignar_ids(boxes)
        resultados = []
        t_actual = time.time()

        for i, pid in enumerate(ids):
            (x, y, bw, bh) = boxes[i]
            kps = keypoints_list[i] if i < len(keypoints_list) else None
            hist = self.historial[pid]

            vel_max = 0.0
            confianza = 0.5
            nombre = "Desconocido"
            clase_idx = 1

            if kps is not None:
                lw = kps[9]
                rw = kps[10]

                # Calcular velocidad muñeca izquierda
                if lw[2] > 0.3 and hist["lw"] is not None:
                    dt = t_actual - hist["t"]
                    if dt > 0:
                        px_actual = (lw[1] * w, lw[0] * h)
                        px_previo = hist["lw"]
                        dist = np.sqrt((px_actual[0] - px_previo[0])**2 + (px_actual[1] - px_previo[1])**2)
                        vel = dist / dt
                        vel_max = max(vel_max, vel)

                # Calcular velocidad muñeca derecha
                if rw[2] > 0.3 and hist["rw"] is not None:
                    dt = t_actual - hist["t"]
                    if dt > 0:
                        px_actual = (rw[1] * w, rw[0] * h)
                        px_previo = hist["rw"]
                        dist = np.sqrt((px_actual[0] - px_previo[0])**2 + (px_actual[1] - px_previo[1])**2)
                        vel = dist / dt
                        vel_max = max(vel_max, vel)

                # Guardar velocidad
                if vel_max > 0:
                    hist["velocidades"].append(vel_max)
                    if len(hist["velocidades"]) > self.historial_frames:
                        hist["velocidades"].pop(0)

                # Variabilidad
                var = 0.0
                if len(hist["velocidades"]) >= 5:
                    var = np.std(hist["velocidades"])

                # Clasificación
                if vel_max < self.umbral_quieta:
                    clase_idx = 0
                    nombre = "Quieta"
                    confianza = min(0.99, 1.0 - (vel_max / self.umbral_quieta) * 0.3)
                elif vel_max > self.umbral_sospechoso or var > 400.0:
                    clase_idx = 2
                    nombre = "Sospechoso"
                    confianza = min(0.99, 0.7 + (vel_max / 2000.0) * 0.3)
                    hist["sospechoso_count"] += 1
                else:
                    clase_idx = 1
                    nombre = "Movimiento"
                    confianza = min(0.95, 0.6 + (vel_max / self.umbral_sospechoso) * 0.3)

                # Actualizar historial
                if lw[2] > 0.3:
                    hist["lw"] = (lw[1] * w, lw[0] * h)
                if rw[2] > 0.3:
                    hist["rw"] = (rw[1] * w, rw[0] * h)

            hist["t"] = t_actual
            hist["clase_previa"] = nombre

            resultados.append({
                "id": pid,
                "clase_idx": clase_idx,
                "nombre": nombre,
                "confianza": confianza,
                "velocidad": vel_max,
                "box": (x, y, bw, bh),
                "sospechoso_count": hist["sospechoso_count"]
            })

        return resultados

    def reset(self):
        self.historial.clear()
        self.contador_id = 0


if __name__ == "__main__":
    print("[Clasificador] Multi-persona listo.")