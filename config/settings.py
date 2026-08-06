#Cámara
CAMERA_INDEX   = 0        # Índice de la webcam (0 = cámara por defecto)
FRAME_WIDTH    = 640      # Ancho de salida del frame (píxeles)
FRAME_HEIGHT   = 480      # Alto de salida del frame (píxeles)
CAMERA_FPS     = 30       # FPS solicitados a la webcam (el driver puede ignorarlo)

#Preprocesamiento (M1)
APPLY_GAUSSIAN  = False           # Activarlo solo si el ruido mejora al detectarlo
GAUSSIAN_KERNEL = (5, 5)          # Tamaño del kernel (debe ser impar)

#MediaPipe Pose (M2)
MIN_DETECTION_CONFIDENCE = 0.7    # Confianza mínima de detección
MIN_TRACKING_CONFIDENCE  = 0.5    # Evita redetecciones costosas demasiado frecuentes
POSE_MODEL_COMPLEXITY    = 0      # Modelo ligero para CPU de laptop
HAND_MODEL_COMPLEXITY    = 0      # Modelo ligero para CPU de laptop
HANDS_PROCESS_INTERVAL   = 2      # Ejecutar Hands cada N frames y reutilizar el último resultado
LANDMARKS_OF_INTEREST    = [11, 13, 15, 17, 19, 21]  # Brazo derecho

#Cálculo de ángulos (M3)
SMOOTHING_WINDOW = 5              # Ventana de media móvil (frames)
VISIBILITY_THRESHOLD = 0.5        # Visibilidad mínima de landmark válido
GRIPPER_D_MIN = 20                # Distancia mínima en px (pinza cerrada)
GRIPPER_D_MAX = 120               # Distancia máxima en px (pinza abierta)

#CoppeliaSim (M4)
COPPELIASIM_ENABLED = True
COPPELIASIM_HOST = "localhost"
COPPELIASIM_PORT = 23000
COPPELIASIM_CONNECTION_TIMEOUT_MS = 3000
COPPELIASIM_UPDATE_HZ = 20.0
COPPELIASIM_JOINT_COUNT = 6
COPPELIASIM_SHOULDER_INDEX = 1
COPPELIASIM_ELBOW_INDEX = 2
COPPELIASIM_SHOULDER_MIN_DEG = -60.0  # Rango validado manualmente
COPPELIASIM_SHOULDER_MAX_DEG = 0.0
COPPELIASIM_ELBOW_MIN_DEG = -60.0     # Rango validado manualmente
COPPELIASIM_ELBOW_MAX_DEG = 0.0
COPPELIASIM_GRIPPER_THRESHOLD = 0.5
COPPELIASIM_MAX_VELOCITY_DEG = 45.0
COPPELIASIM_MAX_ACCELERATION_DEG = 90.0
COPPELIASIM_MAX_JERK_DEG = 360.0

#Visualización y métricas (M5)
FPS_SMOOTHING_WINDOW = 30          # Ventana para estabilizar el FPS mostrado
