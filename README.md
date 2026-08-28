# Robot Mimic

Sistema de imitación de movimientos que captura el brazo derecho y la mano con
una cámara, estima su pose mediante MediaPipe y envía los movimientos a un robot
UR5 con gripper RG2 en CoppeliaSim.

## Características

- Captura de video en tiempo real con OpenCV.
- Detección del brazo derecho y la mano con MediaPipe Pose y Hands.
- Cálculo y suavizado de los ángulos de hombro y codo.
- Estimación de la apertura del gripper con la distancia entre índice y pulgar.
- Control asíncrono de un UR5 y un RG2 mediante la ZeroMQ Remote API.
- Validación independiente del brazo y el gripper antes de cada orden.
- Límites efectivos obtenidos de la intersección entre la configuración y la escena.
- Parada de emergencia enclavada y watchdog de pérdida de pose.
- Detección reactiva de colisiones del robot con el entorno.
- Dashboard Tkinter con dos vistas, telemetría y controles visibles.
- Vista fija del UR5 integrada mediante un Vision Sensor de CoppeliaSim.
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
- Pillow
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

La escena debe contener un objeto `/UR5` con seis articulaciones revolutas y un
gripper RG2 dentro de su jerarquía. Al conectarse, el programa comprueba los
handles, los intervalos físicos, el modo de control y el estado de la simulación.
La escena incluida también contiene `/RobotMimicVisionSensor`, configurado a
640x480 para alimentar a 10 Hz el panel derecho del dashboard. La ausencia del
sensor no deshabilita el control seguro: la interfaz muestra un placeholder.
Al conectarse, el controlador recoloca ese sensor en una vista oblicua hacia la
pinza RG2, alinea su eje vertical con el eje Z del robot y usa un ángulo de
visión de 50° para mostrar el robot completo sin llenar el cuadro de piso.
La escena versionada ya deja los seis joints en control dinámico de posición con
perfil de movimiento activo (`dynPosMode = 1`). Si se sustituye la escena y ese
perfil no está activo, el controlador permanece fuera de `READY` y no acepta
movimientos, porque velocidad, aceleración y jerk sólo están garantizados en
[modos compatibles](https://manual.coppeliarobotics.com/en/sim/simSetJointTargetPosition.htm).

Las colecciones de colisión se crean durante la ejecución: los elementos móviles
del UR5/RG2 se comparan con los objetos externos al árbol `/UR5`. Las
autocolisiones y la predicción de trayectorias no forman parte de este proyecto.
Los aliases se precargan durante el preflight para que ninguna RPC opcional se
interponga entre [`sim.checkCollision`](https://manual.coppeliarobotics.com/en/sim/simCheckCollision.htm)
y la parada de la simulación.

Si CoppeliaSim no está abierto, la escena no está ejecutándose o la conexión
falla, el procesamiento de cámara continúa sin controlar el robot virtual.

## Ejecución

Con el entorno virtual activo:

```powershell
python main.py
```

Controles por teclado:

| Tecla | Acción |
| --- | --- |
| `Espacio` | Inicia la imitación o pausa haciendo `hold` |
| `E` | Enclava la parada y detiene la simulación |
| `R` | Reconecta después de pulsar **Play** nuevamente |
| `Q` | Restaura `home` solo si no hubo emergencia o fallo |

La leyenda de atajos permanece visible en la franja inferior. La ventana aparece
antes de que termine el preflight. La captura, MediaPipe y la conexión se
ejecutan fuera del hilo de Tkinter, por lo que el teclado sigue respondiendo
durante la inicialización y el rearme. Tres fallos consecutivos de cámara
enclavan la emergencia y exigen reiniciar la aplicación.

Después de una parada de emergencia, pulsar **Play** no reactiva por sí solo el
control. Primero pulsa `R` para repetir el preflight y luego `Espacio`.

### Estados de seguridad

| Estado | Significado |
| --- | --- |
| `DISCONNECTED` | No existe un runtime validado; no se aceptan órdenes |
| `READY` | Preflight superado y sin colisión; espera `Espacio` |
| `RUNNING` | Acepta comandos frescos y vigila pose y colisiones |
| `PAUSED` | Cola vacía y postura actual mantenida con `hold` |
| `ESTOP` | Parada enclavada; la simulación recibió `stopSimulation(False)` |
| `FAULT` | Fallo fatal de RPC, escena, cámara o cierre del worker |

`R` solo está permitido desde `ESTOP`, `FAULT` o `DISCONNECTED`. El rearme no
pulsa **Play** automáticamente ni inicia la imitación: vuelve a `READY` después
de repetir todas las comprobaciones. Si la RPC de parada no pudo confirmarse,
el `FAULT` permanece enclavado durante toda esa instancia y `R` se rechaza para
no declarar segura una simulación que podría seguir moviéndose; detén la escena
manualmente y reinicia la aplicación.

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
| CoppeliaSim | endpoint, frecuencias, timeouts y límites operativos del UR5 |
| Orientación | giro global de la base del UR5 alrededor del eje vertical |
| Seguridad | edad de comandos, watchdog, colisiones y retorno a `home` |
| Visualización | posición, objetivo, zoom, resolución y frecuencia del Vision Sensor |

Defaults de seguridad:

| Parámetro | Valor |
| --- | ---: |
| Presupuesto de conexión y preflight | 10 s |
| Timeout RPC operativo | 500 ms |
| Edad máxima de un comando | 250 ms |
| Watchdog sin brazo válido | ESTOP a los 750 ms |
| Detección de colisiones | 20 Hz y después de cada lote |
| Fallo fatal de cámara | 3 lecturas consecutivas |
| Readback de `home` | tolerancia 1°; plazo 5 s |

Para ejecutar únicamente la captura y detección, cambia
`COPPELIASIM_ENABLED = False`.

El hombro humano se expresa como elevación firmada respecto a la vertical hacia
abajo y pertenece a `[-180°, 180°]`; el signo coincide con la vista espejada del
dashboard. El codo conserva el ángulo interno `[0°, 180°]`, donde `180°` es un
brazo recto. Cuando MediaPipe entrega landmarks mundiales, el codo se calcula en
3D para reducir el error al acercar el brazo a la cámara. Un valor ausente,
booleano, no finito o fuera de rango se rechaza, no se recorta.

La transformación al UR5 usa neutros y direcciones calibrables. Con los valores
iniciales, cuando el codo humano está a la misma altura que el hombro, la
elevación es `+90°` y el joint de hombro del UR5 queda en `0°`, manteniendo recto
el primer tramo desde la base. Una zona neutral de ±15° absorbe las pequeñas
diferencias de postura y detección: cualquier elevación entre `75°` y `105°`
mantiene ese joint exactamente en `0°`. A la vez, `codo humano 90°` produce
`joint UR5 +90°`, flexionando el antebrazo virtual hacia el mismo lado que el
humano en la vista espejada. El rango operativo final de cada joint sigue siendo la
intersección entre los límites de `settings.py` y `sim.getJointInterval` de la
escena. El dashboard muestra ambos valores como `humano → UR5` para que la
conversión sea visible durante la imitación.

`COPPELIASIM_BASE_YAW_DEG` orienta el root `/UR5` respecto al mundo durante el
preflight. El controlador fuerza roll y pitch a cero para conservar la base y el
eje Z verticales, y aplica este valor únicamente como yaw. El giro afecta a toda
la jerarquía del robot y no cambia los ejes locales ni los límites de sus joints.

`COPPELIASIM_VIEW_CAMERA_POSITION` y `COPPELIASIM_VIEW_TARGET_POSITION` se
expresan respecto al root `/UR5`. Los valores iniciales colocan la cámara en una
vista oblicua frontal-dominante, a unos 32° de una vista completamente frontal:
el gripper se ve casi de frente, mientras la componente lateral permite leer la
flexión del antebrazo. El robot completo queda centrado, con la base abajo y Z
vertical en la imagen. Para acercar o abrir el encuadre, ajusta
`COPPELIASIM_VIEW_ANGLE_DEG`: un valor menor produce más zoom y uno mayor muestra
más escena.

## Pruebas

Las pruebas unitarias de visión, configuración, validación, estados y transporte
usan dobles de prueba y no necesitan una instancia activa de CoppeliaSim:

```powershell
python -m unittest discover -s tests -v
```

Los scripts `test_ur5.py` y `test_rg2.py` usan el mismo controlador seguro de la
aplicación para validación manual con la escena activa. Ambos aceptan `e` para la
parada de emergencia y nunca restauran `home` después de una emergencia o fallo.

## Estructura

```text
robot-mimic/
├── config/
│   └── settings.py
├── modules/
│   ├── m1_capture.py
│   ├── m2_mediapipe.py
│   ├── m3_angles.py
│   ├── commands.py
│   ├── m4_coppeliasim.py
│   └── m5_gui.py
├── simulation/
│   └── robot_mimic_ur5.ttt
├── tests/
│   ├── test_m2_mediapipe.py
│   ├── test_m3_angles.py
│   ├── test_m4_coppeliasim.py
│   ├── test_m5_gui.py
│   ├── test_main.py
│   └── test_settings.py
├── main.py
└── requirements.txt
```

## Solución de problemas

- **La cámara no abre:** cambia `CAMERA_INDEX` y cierra otras aplicaciones que
  puedan estar utilizando la webcam.
- **No se conecta a CoppeliaSim:** inicia la simulación antes de ejecutar
  `main.py` y confirma el host y el puerto configurados.
- **No mueve el robot:** comprueba que el brazo detectado sea válido y que el
  UR5 de la escena tenga la ruta `/UR5`; revisa también el mensaje del preflight.
- **El gripper no responde:** mantén visibles el pulgar y el índice y verifica
  que la escena use la señal `signal.RG2_open`.
- **Movimiento inestable:** aumenta `SMOOTHING_WINDOW` o reduce
  `COPPELIASIM_UPDATE_HZ`.
- **La simulación se detuvo:** revisa el HUD para distinguir una tecla `E`, una
  colisión, pérdida prolongada de pose o un fallo de comunicación.
- **No permite rearmar:** retira la colisión, pulsa **Play** en CoppeliaSim y
  luego `R`; el controlador no inicia la simulación automáticamente.

## Seguridad

Las protecciones descritas aquí pertenecen exclusivamente a CoppeliaSim. La
parada depende de la conexión con el simulador, la detección es reactiva después
del contacto y no existe planificación de trayectorias ni comprobación de
autocolisiones.
Este código no es un sistema de seguridad certificado y no debe conectarse a un
robot físico sin una arquitectura de seguridad independiente y específica para
el hardware.
