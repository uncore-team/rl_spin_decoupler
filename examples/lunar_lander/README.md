# LunarLander + Stable-Baselines3 with rl_spin_decoupler

Este ejemplo completo y ejecutable demuestra el patron de desacoplo RL/agente
con Gymnasium `LunarLander-v3` y SB3 (PPO).

## Que demuestra

Arquitectura del ejemplo:

- `rl_side_lunarlander.py`: proceso RL. Abre el socket servidor con `RLSide`,
  entrena PPO y calcula la senal de aprendizaje.
- `agent_side_lunarlander.py`: proceso agente/entorno. Ejecuta el spin loop,
  aplica acciones y publica observaciones.
- `reward.py`: logica de recompensa, `terminated` y `truncated` en el lado RL.

Decision central de diseno:

- El agente solo transporta observaciones y tiempo de agente (mas LAT).
- La recompensa y la terminacion se calculan en el lado RL desde la
  observacion recibida.

Esto es coherente con la filosofia del decoupler: el agente queda agnostico a
la tarea de aprendizaje, y la logica especifica de RL vive en el proceso RL.

## Requisitos e instalacion

Estas dependencias son opcionales y aplican solo a este ejemplo. El core de
`rl_spin_decoupler` sigue siendo stdlib puro.

Instalacion:

```bash
pip install -r examples/lunar_lander/requirements.txt
```

Nota Box2D:

`gymnasium[box2d]` puede requerir herramientas de compilacion del sistema (por
ejemplo `swig`, `build-essential`, `python3-dev` o equivalentes segun distro).

## Ejecucion (orden correcto)

Abre dos terminales en la raiz del repositorio.

1) Terminal 1 (primero, lado RL)

```bash
python examples/lunar_lander/rl_side_lunarlander.py
```

2) Terminal 2 (despues, lado agente)

```bash
python examples/lunar_lander/agent_side_lunarlander.py
```

Si quieres ver la parte grafica (ventana de LunarLander), activa render en el
proceso agente:

```bash
python examples/lunar_lander/agent_side_lunarlander.py --render
```

Nota: en entornos sin display (por ejemplo algunos servidores o WSL sin
servidor X/Wayland), el render puede no estar disponible.

Por que este orden:

`RLSide` abre el socket servidor y bloquea esperando conexion; `AgentSide`
conecta como cliente.

## Que deberias ver

- Logs de entrenamiento de SB3/PPO en el proceso RL.
- Metadatos temporales por paso (incluyendo `lat`) dentro de `info`.
- Si usas `--rollout-steps`, resumen con LAT medio al final del rollout.

## Alcance del ejemplo

Este ejemplo prioriza ilustrar el patron de desacoplo, no alcanzar SOTA.

- La recompensa se reconstruye en el lado RL y es una aproximacion didactica.
- Para resolver LunarLander de forma robusta normalmente se necesitan mas pasos
  y ajuste de hiperparametros.

## Referencias

- README principal del proyecto: [../../README.md](../../README.md)
- API de la libreria: [../../docs/api.rst](../../docs/api.rst)
