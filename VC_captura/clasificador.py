"""
clasificador.py
Clasifica comportamiento de múltiples personas:
- Quieta
- Movimiento normal  
- Sospechoso (muñecas rápidas, acercamiento brusco o desplazamiento rápido)
"""

import time
import numpy as np

class ClasificadorComportamiento:
    def __init__(self,
                 umbral_quieta=100.0,          # REDUCIDO: mas sensible a movimiento
                 umbral_sospechoso=600.0,      # REDUCIDO: detecta sospechosos mas facil
                 umbral_acercamiento=1.25,     # REDUCIDO: 25% de crecimiento en 1s
                 umbral_desplazamiento=300.0,  # REDUCIDO: detecta desplazamiento rapido
                 historial_frames=10):         # REDUCIDO: reacciona mas rapido

        self.umbral_quieta = umbral_quieta
        self.umbral_sospechoso = umbral_sospechoso
        self.umbral_acercamiento = umbral_acercamiento
        self.umbral_desplazamiento = umbral_desplazamiento
        self.historial_frames = historial_frames

        self.historial = {}
        self.contador_id = 0

    def _asignar_ids(self, boxes):
        """Asigna IDs persistentes usando proximidad de centroides."""
        centroides = [(x + w/2, y + h/2) for (x, y, w, h) in boxes]
        areas = [w * h for (_, _, w, h) in boxes]

        if not self.historial:
            for i in range(len(centroides)):
                self.contador_id += 1
                self.historial[self.contador_id] = {
                    "centroide": centroides[i],
                    "area": areas[i],
                    "lw": None, "rw": None,
                    "t": time.time(),
                    "velocidades": [],
                    "areas": [areas[i]],
                    "desplazamientos": [],
                    "clase_previa": "Quieta",
                    "sospechoso_count": 0
                }
            return list(range(1, len(centroides) + 1))

        ids_usados = set()
        asignaciones = []
        ids_disponibles = list(self.historial.keys())

        for c_act, a_act in zip(centroides, areas):
            mejor_id = None
            mejor_dist = float("inf")
            for pid in ids_disponibles:
                if pid in ids_usados:
                    continue
                c_prev = self.historial[pid]["centroide"]
                dist = np.hypot(c_act[0] - c_prev[0], c_act[1] - c_prev[1])
                if dist < mejor_dist and dist < 150:
                    mejor_dist = dist
                    mejor_id = pid

            if mejor_id is not None:
                asignaciones.append(mejor_id)
                ids_usados.add(mejor_id)
                hist = self.historial[mejor_id]
                hist["centroide"] = c_act
                hist["area"] = a_act
                hist["areas"].append(a_act)
                if len(hist["areas"]) > self.historial_frames:
                    hist["areas"].pop(0)
            else:
                self.contador_id += 1
                self.historial[self.contador_id] = {
                    "centroide": c_act,
                    "area": a_act,
                    "lw": None, "rw": None,
                    "t": time.time(),
                    "velocidades": [],
                    "areas": [a_act],
                    "desplazamientos": [],
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
        if not boxes:
            return []

        ids = self._asignar_ids(boxes)
        resultados = []
        t_actual = time.time()

        for i, pid in enumerate(ids):
            (x, y, bw, bh) = boxes[i]
            kps = keypoints_list[i] if i < len(keypoints_list) else None
            hist = self.historial[pid]

            vel_muneca_max = 0.0
            vel_centroide = 0.0
            factor_acercamiento = 1.0
            nombre = "Desconocido"
            clase_idx = 1
            confianza = 0.5

            # --- Velocidad de muñecas ---
            if kps is not None:
                lw = kps[9]
                rw = kps[10]
                dt = t_actual - hist["t"]
                if dt > 0:
                    if lw[2] > 0.3 and hist["lw"] is not None:
                        px_act = (lw[1] * w, lw[0] * h)
                        px_prev = hist["lw"]
                        dist = np.hypot(px_act[0] - px_prev[0], px_act[1] - px_prev[1])
                        vel_muneca_max = max(vel_muneca_max, dist / dt)
                    if rw[2] > 0.3 and hist["rw"] is not None:
                        px_act = (rw[1] * w, rw[0] * h)
                        px_prev = hist["rw"]
                        dist = np.hypot(px_act[0] - px_prev[0], px_act[1] - px_prev[1])
                        vel_muneca_max = max(vel_muneca_max, dist / dt)

                    if lw[2] > 0.3:
                        hist["lw"] = (lw[1] * w, lw[0] * h)
                    if rw[2] > 0.3:
                        hist["rw"] = (rw[1] * w, rw[0] * h)

                if vel_muneca_max > 0:
                    hist["velocidades"].append(vel_muneca_max)
                    if len(hist["velocidades"]) > self.historial_frames:
                        hist["velocidades"].pop(0)

            # --- Velocidad de desplazamiento ---
            if len(hist["centroide"]) > 0 and hist["t"] != t_actual:
                dt = t_actual - hist["t"]
                if dt > 0:
                    cx_prev, cy_prev = hist["centroide"]
                    cx_act, cy_act = (x + bw/2, y + bh/2)
                    dist_centroide = np.hypot(cx_act - cx_prev, cy_act - cy_prev)
                    vel_centroide = dist_centroide / dt
                    hist["desplazamientos"].append(vel_centroide)
                    if len(hist["desplazamientos"]) > self.historial_frames:
                        hist["desplazamientos"].pop(0)

            # --- Factor de acercamiento ---
            if len(hist["areas"]) >= 2:
                area_prev = hist["areas"][-2]
                area_act = hist["areas"][-1]
                if area_prev > 0:
                    factor_acercamiento = area_act / area_prev

            dt_aprox = max(0.05, t_actual - hist["t"])
            tasa_crecimiento = (factor_acercamiento - 1) / dt_aprox

            # --- CLASIFICACION ---
            es_acercamiento = (tasa_crecimiento > (self.umbral_acercamiento - 1))
            es_muneca_rapida = (vel_muneca_max > self.umbral_sospechoso)
            es_desplazamiento = (vel_centroide > self.umbral_desplazamiento)

            # DEBUG: Mostrar valores en consola para ajuste
            if vel_muneca_max > 50 or vel_centroide > 50:
                print("[DEBUG] P" + str(pid) + " | muneca=" + str(int(vel_muneca_max)) + 
                      " | despl=" + str(int(vel_centroide)) + " | acerc=" + str(round(factor_acercamiento, 2)))

            if es_acercamiento or es_muneca_rapida or es_desplazamiento:
                clase_idx = 2
                nombre = "Sospechoso"
                confianza = min(0.99, 0.7 + (vel_muneca_max / 1500.0) * 0.3)
                hist["sospechoso_count"] += 1
            elif vel_muneca_max < self.umbral_quieta and vel_centroide < self.umbral_quieta:
                clase_idx = 0
                nombre = "Quieta"
                confianza = min(0.99, 1.0 - (vel_muneca_max / self.umbral_quieta) * 0.3)
            else:
                clase_idx = 1
                nombre = "Movimiento"
                confianza = min(0.95, 0.6 + (vel_muneca_max / self.umbral_sospechoso) * 0.3)

            hist["t"] = t_actual
            hist["clase_previa"] = nombre

            resultados.append({
                "id": pid,
                "clase_idx": clase_idx,
                "nombre": nombre,
                "confianza": confianza,
                "velocidad_munecas": vel_muneca_max,
                "velocidad_desplazamiento": vel_centroide,
                "factor_acercamiento": factor_acercamiento,
                "box": (x, y, bw, bh),
                "sospechoso_count": hist["sospechoso_count"]
            })

        return resultados

    def reset(self):
        self.historial.clear()
        self.contador_id = 0