# Robot Mimic

Sistema de imitación de movimientos que captura el brazo derecho y la mano con
una cámara, estima su pose mediante MediaPipe y envía los movimientos a un robot
UR5 con gripper RG2 en CoppeliaSim.

## Características

- Captura de video en tiempo real con OpenCV.
- Detección del brazo derecho y la mano con MediaPipe Pose y Hands.
- Cálculo y suavizado de los ángulos de hombro y codo.
- Estimación de la apertura del gripper con la distancia entre índice y pulgar.
- Control no bloqueante de un UR5 y un RG2 mediante la ZeroMQ Remote API.
- Límites de posición, velocidad, aceleración y jerk configurables.
- Visualización de FPS, ángulos, detección de mano y estado de la imitación.
- Funcionamiento de la cámara aunque CoppeliaSim no esté disponible.

## Flujo del sistema

```text
Cámara (M1)
    ↓
MediaPipe Pose + Hands (M2)
    ↓
Ángulos y apertura del gripper (M3)
    ↓
UR5 + RG2 en CoppeliaSim (M4)
```

## Requisitos

- Python 3.10 o 3.11.
- Una cámara compatible con OpenCV.
- CoppeliaSim con soporte para ZeroMQ Remote API.
- La escena incluida en `simulation/robot_mimic_ur5.ttt`.

Las dependencias de Python se encuentran en `requirements.txt`:

- OpenCV
- MediaPipe
- NumPy
- Cliente de la ZeroMQ Remote API de CoppeliaSim

## Instalación

En Windows PowerShell:

```powershell
git clone <URL_DEL_REPOSITORIO>
cd robot-mimic

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Preparar CoppeliaSim

1. Abre CoppeliaSim.
2. Carga la escena `simulation/robot_mimic_ur5.ttt`.
3. Inicia la simulación con el botón **Play**.
4. Verifica que el servidor ZeroMQ utilice el puerto `23000`.
5. Ejecuta la aplicación mientras la simulación continúa activa.

La escena debe contener un objeto `/UR5` con sus seis articulaciones y el
gripper RG2. El programa busca estos objetos al conectarse y conserva la postura
inicial para restaurarla al finalizar.

Si CoppeliaSim no está abierto, la escena no está ejecutándose o la conexión
falla, el procesamiento de cámara continúa sin controlar el robot virtual.

## Ejecución

Con el entorno virtual activo:

```powershell
python main.py
```

Controles:

| Tecla | Acción |
| --- | --- |
| `Espacio` | Inicia o pausa la imitación |
| `Q` | Cierra la aplicación y restaura la postura inicial del robot |

Para una detección estable, coloca el brazo derecho completo frente a la cámara
y procura mantener visibles el hombro, el codo, la muñeca, el pulgar y el índice.

## Configuración

Los parámetros principales están en `config/settings.py`:

| Grupo | Parámetros relevantes |
| --- | --- |
| Cámara | índice, resolución y FPS solicitados |
| Preprocesamiento | filtro gaussiano opcional |
| MediaPipe | confianza, complejidad de modelos e intervalo de Hands |
| Ángulos | ventana de suavizado, visibilidad y rango del gripper |
| CoppeliaSim | host, puerto, frecuencia y límites seguros del UR5 |
| Visualización | ventana usada para suavizar el cálculo de FPS |

Para ejecutar únicamente la captura y detección, cambia
`COPPELIASIM_ENABLED = False`.

Los ángulos humanos de hombro y codo se limitan al intervalo de 0° a 180° y se
escalan al rango seguro configurado para el UR5. Ajusta esos límites solo después
de validar la escena y evitar colisiones.

## Pruebas

Las pruebas unitarias del mapeo y los comandos del robot no necesitan una
instancia activa de CoppeliaSim:

```powershell
python -m unittest discover -s tests -v
```

Los scripts `test_ur5.py` y `test_rg2.py` sirven para validación manual con la
escena activa. Ejecútalos únicamente después de revisar que el espacio de trabajo
del robot esté libre de obstáculos.

## Estructura

```text
robot-mimic/
├── config/
│   └── settings.py
├── modules/
│   ├── m1_capture.py
│   ├── m2_mediapipe.py
│   ├── m3_angles.py
│   └── m4_coppeliasim.py
├── simulation/
│   └── robot_mimic_ur5.ttt
├── tests/
│   └── test_m4_coppeliasim.py
├── main.py
└── requirements.txt
```

## Solución de problemas

- **La cámara no abre:** cambia `CAMERA_INDEX` y cierra otras aplicaciones que
  puedan estar utilizando la webcam.
- **No se conecta a CoppeliaSim:** inicia la simulación antes de ejecutar
  `main.py` y confirma el host y el puerto configurados.
- **No mueve el robot:** comprueba que el brazo detectado sea válido y que el
  UR5 de la escena tenga la ruta `/UR5`.
- **El gripper no responde:** mantén visibles el pulgar y el índice y verifica
  que la escena use la señal `signal.RG2_open`.
- **Movimiento inestable:** aumenta `SMOOTHING_WINDOW` o reduce
  `COPPELIASIM_UPDATE_HZ`.

## Seguridad

Este proyecto controla un robot simulado. Antes de adaptar el código a hardware
real, añade límites físicos, parada de emergencia, detección de colisiones y una
validación independiente de cada comando.
