"""
================================================================================
  SISTEMA DE VIGILANCIA INTELIGENTE - DOCUMENTACION TECNICA COMPLETA
================================================================================
  Proyecto: ML_reconocimiento_personas
  Arquitectura: YOLOv8 (deteccion multi-persona) + MoveNet Lightning (pose)
                + Clasificador geometrico OpenCV (comportamiento)
  Procesamiento: 100% local (sin nube)
  Requisitos: Python 3.10+, Webcam, 4GB RAM minimo
================================================================================


INDICE
======
1. Estructura de Carpetas
2. Dependencias e Instalacion
3. Arquitectura del Sistema
4. Flujo de Datos (Pipeline)
5. Modulo: camara.py
6. Modulo: detector.py
7. Modulo: clasificador.py
8. Modulo: captura_camara.py
9. Modulo: main.py (Orquestador)
10. Clasificacion de Comportamientos
11. Sistema de Alertas
12. Registro de Eventos (Log)
13. Controles de Usuario
14. Rendimiento y Optimizacion
15. Solucion de Problemas
16. Extensiones Futuras


================================================================================
1. ESTRUCTURA DE CARPETAS
================================================================================

ML_reconocimiento_personas/
|
|-- main.py                          <- Punto de entrada principal
|-- requirements.txt                 <- Dependencias del proyecto
|-- README.txt                       <- Este archivo (documentacion)
|-- log_vigilancia.txt               <- Archivo de registro (generado auto)
|
|-- modelos/
|   |-- movenet_lightning.tflite     <- Modelo TF Lite de pose estimation
|   |                                 [Descargar de Kaggle/TF Hub]
|
|-- VC_captura/                      <- Modulos del sistema de vision
|   |-- camara.py                    <- Gestion de webcam (OpenCV)
|   |-- detector.py                  <- Deteccion multi-persona (YOLOv8 + MoveNet)
|   |-- clasificador.py              <- Clasificacion de comportamiento (OpenCV)
|   |-- captura_camara.py            <- Captura de imagenes para dataset
|   |
|   |-- __pycache__/                 <- Cache de Python (auto-generado)
|
|-- dataset_capturado/               <- Imagenes capturadas (auto-generado)
|-- yolov8n.pt                       <- Modelo YOLOv8 nano (auto-descarga)


================================================================================
2. DEPENDENCIAS E INSTALACION
================================================================================

2.1 Requisitos del sistema
--------------------------
- Sistema operativo: Windows 10/11, Linux, macOS
- Python: 3.10 o superior (recomendado 3.11)
- RAM: 4GB minimo, 8GB recomendado
- Camara: Webcam integrada o USB
- GPU: Opcional (CPU suficiente para YOLOv8-nano + MoveNet)

2.2 Instalacion paso a paso
---------------------------
Paso 1: Crear entorno virtual (recomendado)
    python -m venv venv
    venv\Scripts\activate        (Windows)
    source venv/bin/activate       (Linux/Mac)

Paso 2: Instalar dependencias
    pip install -r requirements.txt

    Contenido de requirements.txt:
    ------------------------------
    opencv-python      -> Vision por computadora (captura, dibujo, GUI)
    numpy              -> Operaciones matematicas con arrays
    tensorflow         -> Interprete TF Lite para MoveNet
    ultralytics        -> Framework YOLOv8 para deteccion de objetos

Paso 3: Descargar modelo MoveNet
    Opcion A - Descarga directa (PowerShell):
        Invoke-WebRequest -Uri "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/movenet/singlepose/lightning/tflite/float16/3.tflite" -OutFile "modelos/movenet_lightning.tflite"

    Opcion B - Manual desde Kaggle:
        URL: https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning/3
        Descargar: 3.tflite
        Renombrar a: movenet_lightning.tflite
        Mover a: modelos/movenet_lightning.tflite

Paso 4: Verificar instalacion
    python -c "import cv2, numpy, tensorflow, ultralytics; print('OK')"

Paso 5: Ejecutar sistema
    python main.py


================================================================================
3. ARQUITECTURA DEL SISTEMA
================================================================================

El sistema utiliza una arquitectura HIBRIDA de tres capas:

    +-------------------------------------------------------------+
    |  CAPA 1: DETECCION DE PERSONAS (YOLOv8-nano)               |
    |  - Detecta CUANTAS personas hay en la escena               |
    |  - Genera bounding boxes (x, y, w, h) para cada persona    |
    |  - Procesa hasta 30+ FPS en CPU                            |
    +-------------------------------------------------------------+
                              |
                              v
    +-------------------------------------------------------------+
    |  CAPA 2: POSE ESTIMATION (MoveNet Lightning TF Lite)       |
    |  - Procesa CADA persona detectada por YOLO                 |
    |  - Extrae 17 puntos clave del cuerpo (keypoints)           |
    |  - Keypoints: nariz, ojos, orejas, hombros, codos,         |
    |    muñecas, caderas, rodillas, tobillos                    |
    |  - Devuelve coordenadas normalizadas [y, x, confianza]     |
    +-------------------------------------------------------------+
                              |
                              v
    +-------------------------------------------------------------+
    |  CAPA 3: CLASIFICACION DE COMPORTAMIENTO (OpenCV)          |
    |  - Tracking de personas frame a frame (asignacion de IDs)  |
    |  - Calculo de velocidad de muñecas (px/segundo)            |
    |  - Clasificacion: Quieta / Movimiento / Sospechoso         |
    |  - Deteccion de movimiento erratico (variabilidad)         |
    +-------------------------------------------------------------+
                              |
                              v
    +-------------------------------------------------------------+
    |  CAPA 4: ALERTAS Y REGISTRO                                |
    |  - Alertas visuales en pantalla (texto, colores, bordes)   |
    |  - Alertas sonoras (beep en Windows)                       |
    |  - Registro en archivo log_vigilancia.txt                  |
    +-------------------------------------------------------------+


================================================================================
4. FLUJO DE DATOS (PIPELINE)
================================================================================

Cada frame de video sigue este recorrido:

    Frame BGR (640x480)
         |
         v
    [1] Camara.leer_frame()  ->  ret, frame
         |
         v
    [2] Detector.detectar(frame)
         |
         |---> YOLOv8: Detecta personas -> lista de boxes [(x,y,w,h), ...]
         |
         |---> Por cada persona:
         |       Recortar ROI del frame
         |       Resize a 192x192
         |       MoveNet: Extrae 17 keypoints
         |       Ajustar coordenadas al frame original
         |
         v
    Retorna: frame_dibujado, boxes, total_personas, lista_keypoints
         |
         v
    [3] Clasificador.clasificar(boxes, keypoints_list, h, w)
         |
         |---> Tracking: Asignar IDs persistentes por proximidad
         |---> Por cada persona:
         |       Calcular velocidad de muñecas (px/seg)
         |       Calcular variabilidad del movimiento (std)
         |       Clasificar: Quieta / Movimiento / Sospechoso
         |
         v
    Retorna: lista de diccionarios con info por persona
         |
         v
    [4] Main.verificar_alertas()
         |
         |---> Multiples personas (>3)?
         |---> Algun sospechoso?
         |---> Presencia prolongada (>10 seg)?
         |---> Dibujar alertas en frame
         |
         v
    [5] Main.mostrar_menu()
         |
         |---> HUD con: personas detectadas, sospechosos, controles
         |
         v
    [6] cv2.imshow()  ->  Muestra en pantalla
         |
         v
    [7] cv2.waitKey(1)  ->  Lee teclas de control


================================================================================
5. MODULO: camara.py
================================================================================

Responsabilidad: Gestionar la conexion con la webcam. NADA MAS.
No detecta, no procesa, solo entrega frames crudos.

Clase: Camara
-------------
Atributos:
    indice      -> Indice de la camara (0=integrada, 1=externa)
    ancho       -> Resolucion horizontal (default: 640)
    alto        -> Resolucion vertical (default: 480)
    cap         -> Objeto VideoCapture de OpenCV

Metodos:
    conectar()       -> Abre la camara, configura resolucion
    leer_frame()     -> Lee un frame: ret (bool), frame (numpy array)
    liberar()        -> Cierra camara y destruye ventanas OpenCV

Flujo interno:
    cv2.VideoCapture(indice)
        |
        |---> cap.set(CAP_PROP_FRAME_WIDTH, ancho)
        |---> cap.set(CAP_PROP_FRAME_HEIGHT, alto)
        |
        v
    cap.read()  ->  Frame en formato BGR (Blue, Green, Red)

Nota: OpenCV captura en BGR por defecto. MoveNet espera RGB,
      por lo que detector.py hace la conversion.


================================================================================
6. MODULO: detector.py
================================================================================

Responsabilidad: Detectar MULTIPLES personas y extraer pose de cada una.

Arquitectura hibrida:
---------------------
    YOLOv8-nano (deteccion de objetos)
    + MoveNet Lightning TF Lite (pose estimation)

Por que dos modelos?
--------------------
- YOLOv8 detecta multiples personas pero NO da pose (articulaciones)
- MoveNet da pose excelente pero SOLO para 1 persona
- Solucion: YOLO detecta TODAS, MoveNet analiza CADA UNA

Clase: DetectorPersonas
-----------------------
Atributos:
    umbral_confianza    -> Minima confianza para aceptar keypoint (0.3)
    interpreter         -> Interprete TF Lite de MoveNet
    input_details       -> Metadata de entrada del modelo
    output_details      -> Metadata de salida del modelo
    yolo                -> Modelo YOLOv8-nano de Ultralytics

Metodos:
    _cargar_movenet(ruta)   -> Carga modelo TF Lite
    _cargar_yolo()          -> Carga YOLOv8-nano (auto-descarga yolov8n.pt)
    detectar(frame)         -> Pipeline principal de deteccion
    _dibujar_keypoints()    -> Dibuja puntos del cuerpo
    _dibujar_esqueleto()    -> Dibuja lineas entre articulaciones

Pipeline de deteccion (metodo detectar):
----------------------------------------
    Entrada: frame BGR (640x480)
        |
        v
    [PASO 1] YOLOv8 detecta personas
        |
        |   yolo(frame, classes=[0])   # clase 0 = persona
        |   -> boxes: [(x1,y1,x2,y2), ...]
        |   -> Filtra por confianza > 0.5
        |
        v
    [PASO 2] Por cada persona detectada:
        |
        |   Recortar ROI del frame original
        |       roi = frame[y1:y2, x1:x2]
        |
        |   Preprocesar para MoveNet:
        |       resize a 192x192
        |       expand_dims -> [1, 192, 192, 3]
        |       astype(float32)
        |
        |   Inferencia MoveNet:
        |       interpreter.set_tensor(input, roi_procesado)
        |       interpreter.invoke()
        |       output: [1, 1, 17, 3]
        |
        |   Ajustar coordenadas al frame original:
        |       keypoint_y = (kp_y * 192 * scale_y + y1) / h_frame
        |       keypoint_x = (kp_x * 192 * scale_x + x1) / w_frame
        |
        v
    [PASO 3] Dibujar resultados
        |
        |   Por cada persona:
        |       - Bounding box verde
        |       - Etiqueta "Persona N"
        |       - Keypoints como circulos rojos
        |       - Esqueleto como lineas azules
        |
        v
    Salida: frame_dibujado, boxes, total, keypoints_list

17 Keypoints de MoveNet (orden):
--------------------------------
Indice | Nombre          | Uso en clasificador
-------|-----------------|----------------------
  0    | Nariz           | Orientacion de cabeza
  1    | Ojo izquierdo   | -
  2    | Ojo derecho     | -
  3    | Oreja izquierda | -
  4    | Oreja derecha   | -
  5    | Hombro izq      | Torso (tracking)
  6    | Hombro der      | Torso (tracking)
  7    | Codo izq        | Brazo
  8    | Codo der        | Brazo
  9    | Muñeca izq      | VELOCIDAD (clasificacion)
 10    | Muñeca der      | VELOCIDAD (clasificacion)
 11    | Cadera izq      | Torso (tracking)
 12    | Cadera der      | Torso (tracking)
 13    | Rodilla izq     | Pierna
 14    | Rodilla der     | Pierna
 15    | Tobillo izq     | Pierna
 16    | Tobillo der     | Pierna

Esqueleto dibujado (conexiones):
--------------------------------
    Nariz -> Ojos -> Orejas -> Hombros
    Hombros -> Codos -> Muñecas
    Hombros -> Caderas -> Rodillas -> Tobillos
    Hombros conectados entre si
    Caderas conectadas entre si


================================================================================
7. MODULO: clasificador.py
================================================================================

Responsabilidad: Clasificar el comportamiento de CADA persona detectada.
NO usa TensorFlow ni dataset. Usa pura geometria y fisica.

Concepto matematico:
--------------------
    Velocidad = Distancia / Tiempo

    v = d / t

    Donde:
    - d = distancia euclidiana entre posicion actual y anterior de la muñeca
    - t = tiempo transcurrido entre frames (segundos)
    - Unidades: pixeles / segundo

Clase: ClasificadorComportamiento
---------------------------------
Atributos:
    umbral_quieta       -> Velocidad maxima para "Quieta" (150 px/seg)
    umbral_sospechoso   -> Velocidad minima para "Sospechoso" (800 px/seg)
    historial_frames    -> Cuántos frames recordar para variabilidad (15)
    historial           -> Dict: id_persona -> datos de tracking
    contador_id         -> Contador auto-incremental de IDs

Tracking de personas (metodo _asignar_ids):
-------------------------------------------
Problema: Como saber si la "Persona 1" del frame actual es la misma
          que la "Persona 1" del frame anterior?

Solucion: Tracking por proximidad de centroide
    1. Calcular centroide de cada bounding box
       centroide = (x + w/2, y + h/2)

    2. Comparar con centroides del frame anterior
       distancia = sqrt((x1-x2)^2 + (y1-y2)^2)

    3. Si distancia < 150 pixeles -> Misma persona, mismo ID
       Si distancia >= 150 pixeles -> Nueva persona, nuevo ID

    4. Eliminar IDs que no aparecen en el frame actual

Metodo clasificar (pipeline):
-----------------------------
    Entrada: boxes, keypoints_list, h, w
        |
        v
    [1] Asignar IDs persistentes (tracking)
        |
        v
    [2] Por cada persona:
        |
        |   Obtener muñecas del frame actual:
        |       muñeca_izq  = keypoints[9]   [y, x, conf]
        |       muñeca_der  = keypoints[10]  [y, x, conf]
        |
        |   Obtener muñecas del frame anterior (del historial):
        |       prev_izq = historial[id]["lw"]
        |       prev_der = historial[id]["rw"]
        |
        |   Calcular velocidad muñeca izquierda:
        |       dt = tiempo_actual - tiempo_anterior
        |       dist = sqrt((x_actual - x_previo)^2 + (y_actual - y_previo)^2)
        |       vel_izq = dist / dt
        |
        |   Calcular velocidad muñeca derecha (igual)
        |       vel_der = dist / dt
        |
        |   vel_max = max(vel_izq, vel_der)
        |
        v
    [3] Guardar velocidad en historial
        |
        |   historial[id]["velocidades"].append(vel_max)
        |   Mantener solo ultimos 15 valores
        |
        v
    [4] Calcular variabilidad (desviacion estandar)
        |
        |   Si hay >= 5 velocidades guardadas:
        |       var = std(velocidades)
        |   Alta variabilidad = movimiento ERRATICO/REPETITIVO
        |
        v
    [5] Clasificar
        |
        |   SI vel_max < 150:
        |       -> "Quieta" (verde)
        |       confianza = 1.0 - (vel_max / 150) * 0.3
        |
        |   SI vel_max > 800 O var > 400:
        |       -> "Sospechoso" (rojo)
        |       confianza = 0.7 + (vel_max / 2000) * 0.3
        |       # Movimiento rapido O erratico
        |
        |   EN OTRO CASO:
        |       -> "Movimiento" (amarillo)
        |       confianza = 0.6 + (vel_max / 800) * 0.3
        |
        v
    [6] Actualizar historial para siguiente frame
        |
        |   Guardar posicion actual de muñecas
        |   Guardar tiempo actual
        |   Guardar clase detectada
        |
        v
    Salida: Lista de diccionarios
        [
            {
                "id": 1,
                "clase_idx": 2,
                "nombre": "Sospechoso",
                "confianza": 0.92,
                "velocidad": 1250.5,
                "box": (x, y, w, h),
                "sospechoso_count": 3
            },
            ...
        ]


================================================================================
8. MODULO: captura_camara.py
================================================================================

Responsabilidad: Guardar frames como imagenes .jpg para crear datasets.

Clase: Capturador
-----------------
Metodos:
    capturar_unica(frame, etiqueta)
        -> Guarda: dataset_capturado/etiqueta_YYYYMMDD_HHMMSS_mmm.jpg

    captura_continua(camara, intervalo_seg, etiqueta)
        -> Captura automatica cada X segundos

Uso independiente:
    python VC_captura/captura_camara.py
    ESPACIO = capturar manual
    Q = salir


================================================================================
9. MODULO: main.py (ORQUESTADOR)
================================================================================

Responsabilidad: Coordinar todos los modulos en un bucle infinito.

Inicializacion (SistemaVigilancia.__init__):
--------------------------------------------
    [1] Camara -> conectar()
    [2] Detector -> cargar YOLOv8 + MoveNet
    [3] Capturador -> crear carpeta dataset_capturado
    [4] Clasificador -> configurar umbrales

Bucle principal (ejecutar()):
-----------------------------
    MIENTRAS True:
        |
        |---> Leer frame de camara
        |---> Detectar personas (YOLO + MoveNet)
        |---> Clasificar comportamiento (si modo activo)
        |---> Verificar alertas
        |---> Dibujar HUD/menu
        |---> Mostrar en pantalla
        |---> Procesar teclas de control
        |---> Log automatico cada 5 segundos

Controles de teclado:
---------------------
    ESPACIO  -> Capturar imagen manual
    C        -> Activar/Desactivar clasificacion
    G        -> Guardar frame actual como .jpg
    Q        -> Salir del sistema

Alertas implementadas:
----------------------
    1. Multiples personas (>3):
       - Texto: "ALERTA: Multiples personas (N)!"
       - Color borde: ROJO

    2. Movimiento sospechoso:
       - Texto: "ALERTA: Persona X SOSPECHOSA!"
       - Color borde: ROJO
       - Sonido: Beep (Windows) o bell (Linux/Mac)
       - Cooldown: Max 1 beep cada 3 segundos

    3. Presencia prolongada (>10 seg):
       - Texto: "ALERTA: Presencia prolongada (Xs)"
       - Color borde: NARANJA
       - Contador de segundos sobre la persona

HUD (Heads-Up Display):
-----------------------
    Panel negro translucido en esquina superior izquierda:

    +------------------------------------------+
    | Personas detectadas: 3 | SOSPECHOSOS: 1  |
    | Modo captura: OFF (ESPACIO)              |
    | Modo clasif: ON (C)                      |
    | Detector: YOLOv8 + MoveNet               |
    | [Q] Salir | [G] Guardar | [L] Log        |
    +------------------------------------------+

    Si hay sospechosos, la primera linea se pone ROJA.


================================================================================
10. CLASIFICACION DE COMPORTAMIENTOS
================================================================================

Tres clases definidas:

    +----------------+--------------------------------+------------------+
    | Clase          | Criterio matematico            | Color en pantalla|
    +----------------+--------------------------------+------------------+
    | Quieta         | vel_muñecas < 150 px/seg      | VERDE            |
    | Movimiento     | 150 <= vel < 800 px/seg       | AMARILLO         |
    | Sospechoso     | vel >= 800 px/seg O           | ROJO             |
    |                | variabilidad > 400            |                  |
    +----------------+--------------------------------+------------------+

Ejemplos practicos:
-------------------
    Persona parada quieto:
        vel = 20 px/seg  ->  "Quieta" (verde)

    Persona caminando normal:
        vel = 300 px/seg  ->  "Movimiento" (amarillo)

    Persona corriendo:
        vel = 1200 px/seg  ->  "Sospechoso" (rojo)

    Persona moviendo manos rapidamente (erratico):
        vel = 200, 800, 150, 900, 300...  ->  var = 350
        Como var > 400 (ejemplo), -> "Sospechoso" (rojo)

Ajuste de umbrales:
-------------------
    En main.py, al crear el clasificador:

    self.clasificador = ClasificadorComportamiento(
        umbral_quieta=150.0,        # Ajustar segun sensibilidad deseada
        umbral_sospechoso=800.0,    # Ajustar segun escenario
        historial_frames=15         # Memoria de frames para variabilidad
    )

    Valores mas bajos = Mas sensible (mas alertas)
    Valores mas altos = Menos sensible (menos alertas)


================================================================================
11. SISTEMA DE ALERTAS
================================================================================

Tipos de alerta:
----------------

1. ALERTA VISUAL (en pantalla)
   - Texto rojo en parte inferior del frame
   - Borde rojo/naranja alrededor de toda la pantalla
   - Etiqueta "!!! SOSPECHOSO !!!" sobre la persona
   - Contador de segundos de presencia prolongada

2. ALERTA SONORA
   - Windows: Beep de 1000Hz, 500ms (winsound.Beep)
   - Linux/Mac: Caracter bell (\a)
   - Cooldown: Maximo 1 sonido cada 3 segundos para no saturar

3. ALERTA EN LOG
   - Archivo: log_vigilancia.txt
   - Formato: [YYYY-MM-DD HH:MM:SS] Mensaje
   - Ejemplo:
     [2025-05-25 14:30:15] Persona 2 - Movimiento sospechoso
     [2025-05-25 14:30:18] Estado: 3 persona(s), 1 sospechosas

Reglas de alerta:
-----------------
    Regla 1: Multiples personas
        Condicion: total_personas > 3
        Accion: Texto rojo + borde rojo
        Log: automatico cada 5 segundos

    Regla 2: Movimiento sospechoso
        Condicion: clase == "Sospechoso" AND confianza > 0.6
        Accion: Texto rojo + borde rojo + SONIDO + log inmediato
        Nota: Se evalua por CADA persona individualmente

    Regla 3: Presencia prolongada
        Condicion: persona detectada > 10 segundos seguidos
        Accion: Texto naranja + borde naranja + contador de segundos
        Tracking: Por posicion del bounding box (x, y)


================================================================================
12. REGISTRO DE EVENTOS (LOG)
================================================================================

Archivo: log_vigilancia.txt
Ubicacion: Raiz del proyecto (mismo nivel que main.py)

Eventos registrados:
--------------------
- Capturas manuales (ESPACIO)
- Frames guardados (G)
- Movimientos sospechosos (inmediato)
- Estado general cada 5 segundos (si hay personas)

Formato:
--------
    [2025-05-25 14:30:10] Captura manual - 2 personas
    [2025-05-25 14:30:12] Persona 2 - Movimiento sospechoso
    [2025-05-25 14:30:15] Estado: 3 persona(s), 1 sospechosas
    [2025-05-25 14:30:20] Estado: 3 persona(s), 1 sospechosas

El archivo crece indefinidamente. Para reiniciar, borrar manualmente.


================================================================================
13. CONTROLES DE USUARIO
================================================================================

Durante la ejecucion (con la ventana de video activa):

    Tecla        Accion
    ---------    -------------------------------------------
    ESPACIO      Capturar imagen y guardar en dataset_capturado/
    C            Activar/Desactivar clasificacion de comportamiento
    G            Guardar frame actual como frame_guardado_YYYYMMDD_HHMMSS.jpg
    Q            Salir del sistema (libera camara y cierra ventanas)
    ---------    -------------------------------------------

    Nota: La ventana de video debe estar enfocada para capturar teclas.


================================================================================
14. RENDIMIENTO Y OPTIMIZACION
================================================================================

Modelos utilizados:
-------------------
    YOLOv8-nano:    ~6MB,  ~30 FPS en CPU,  deteccion de 80 clases
    MoveNet Light:  ~9MB,  ~30+ FPS en CPU, 17 keypoints

Rendimiento esperado:
---------------------
    Hardware              | FPS aprox | Latencia
    ----------------------|-----------|----------
    Laptop moderna (CPU)  | 15-25     | ~50ms
    PC gamer (GPU)        | 30+       | ~30ms
    Raspberry Pi 4        | 5-10      | ~150ms

Optimizaciones aplicadas:
-------------------------
    1. YOLOv8-nano (el mas ligero de la familia YOLOv8)
    2. MoveNet Lightning (optimizado para edge devices)
    3. TF Lite (inferencia optimizada, no TensorFlow completo)
    4. Resize a 192x192 para MoveNet (baja resolucion = rapido)
    5. Tracking por proximidad (evita re-inferencia si persona no se mueve mucho)
    6. Cooldown en alertas sonoras (evita bloqueo por beeps)

Posibles mejoras de rendimiento:
--------------------------------
    - Reducir resolucion de camara a 320x240
    - Procesar 1 de cada 2 frames (skip frames)
    - Usar GPU si esta disponible (CUDA)
    - Compilar modelo con TF Lite delegates


================================================================================
15. SOLUCION DE PROBLEMAS
================================================================================

Problema: "No module named 'ultralytics'"
Solucion: pip install ultralytics

Problema: "No se encontro el modelo: modelos/movenet_lightning.tflite"
Solucion: Descargar modelo desde Kaggle y guardar en carpeta modelos/

Problema: "Cannot set tensor: Got value of type X but expected type Y"
Solucion: Verificar version del modelo. v3 usa float32, otras versiones pueden usar uint8.
          Revisar mensaje de carga que indica el dtype esperado.

Problema: Camara no se abre (indice 0)
Solucion: Probar con indice=1 en Camara(indice=1) o verificar permisos de camara

Problema: FPS muy bajo (<10)
Solucion: Cerrar otras aplicaciones, reducir resolucion de camara, o usar PC mas potente

Problema: Deteccion de persona inestable (parpadea)
Solucion: Ajustar umbral_confianza en DetectorPersonas (subir a 0.4 o 0.5)

Problema: Falsos positivos (detecta persona donde no hay)
Solucion: YOLOv8 es muy preciso, pero si ocurre, ajustar confianza de YOLO
          en detector.py: yolo(frame, verbose=False, classes=[0], conf=0.6)

Problema: No suena la alerta
Solucion: En Windows, winsound requiere focus de ventana. En Linux/Mac,
          el bell puede estar silenciado en el sistema.


================================================================================
16. EXTENSIONES FUTURAS
================================================================================

Posibles mejoras:
-----------------
1. Reconocimiento facial: Identificar personas especificas
2. Deteccion de objetos: Armas, bolsas sospechosas, etc.
3. Seguimiento de trayectoria: Mapear donde se mueve cada persona
4. Grabacion automatica: Guardar video cuando hay alerta
5. Notificaciones push: Enviar alerta a telefono/email
6. Interfaz web: Ver camara desde navegador
7. Base de datos: Almacenar historial de eventos en SQLite
8. Entrenamiento personalizado: Fine-tuning de YOLO con escenario especifico
9. Multi-camara: Soportar varias camaras simultaneamente
10. Analisis post-evento: Revisar grabaciones con heatmaps de movimiento


================================================================================
                    FIN DE LA DOCUMENTACION
================================================================================
"""

# Este archivo es solo documentacion. No se ejecuta.
# Para ejecutar el sistema: python main.py
