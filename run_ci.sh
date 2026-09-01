#!/usr/bin/env bash
#
# run_ci.sh — Runs all checks from .github/workflows/ci.yml locally for
# rl_spin_decoupler.
#
# Reproduces the three CI jobs:
#   1. lint          -> ruff check  +  ruff format --check
#   2. test-core     -> pytest WITHOUT example tests (coverage gate >=95%)
#   3. test-examples -> pytest for example tests (WITHOUT coverage gate)
#
# Limitation: CI uses an OS matrix (ubuntu/macos/windows) and Python versions
# (3.8-3.13). This script runs with ONE interpreter (the current one or the one
# selected by --python / $PYTHON). It does not replace the matrix, but validates
# that everything passes in your environment before pushing.

set -uo pipefail

# --------------------------- default configuration ---------------------------
PYTHON="${PYTHON:-python3}"
VENV_DIR=".ci-venv"
USE_VENV=1
DO_INSTALL=1
RUN_LINT=1
RUN_CORE=1
RUN_EXAMPLES=1

usage() {
  cat <<'EOF'
Uso: ./run_ci.sh [opciones]

Reproduce localmente los jobs de .github/workflows/ci.yml:
  lint           ruff check + ruff format --check
  test-core      pytest sin los tests de ejemplos (gate de cobertura >=95%)
  test-examples  pytest de los tests de ejemplos (sin gate de cobertura)

Opciones:
  --no-venv       Usa el intérprete actual en vez de crear/usar .ci-venv
  --no-install    No (re)instala dependencias (asume el entorno ya preparado)
  --python BIN    Intérprete base a usar (por defecto: $PYTHON o python3)
  --only STAGE    Ejecuta solo una etapa: lint | core | examples
  -h, --help      Muestra esta ayuda
EOF
}

# --------------------------------- arguments ------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --no-venv)    USE_VENV=0 ;;
    --no-install) DO_INSTALL=0 ;;
    --python)     PYTHON="$2"; shift ;;
    --only)
      RUN_LINT=0; RUN_CORE=0; RUN_EXAMPLES=0
      case "$2" in
        lint)     RUN_LINT=1 ;;
        core)     RUN_CORE=1 ;;
        examples) RUN_EXAMPLES=1 ;;
        *) echo "Etapa desconocida: $2 (usa lint|core|examples)" >&2; exit 2 ;;
      esac
      shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Opción desconocida: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

# ------------------------------------ colors -------------------------------------
if [ -t 1 ]; then
  BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'
  YELLOW=$'\033[33m'; BLUE=$'\033[34m'; RESET=$'\033[0m'
else
  BOLD=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; RESET=""
fi

section() { echo ""; echo "${BOLD}${BLUE}==> $1${RESET}"; }

RESULTS=()
overall=0
record() {  # $1 = name, $2 = exit code
  if [ "$2" -eq 0 ]; then
    RESULTS+=("${GREEN}PASS${RESET}  $1")
  else
    RESULTS+=("${RED}FAIL${RESET}  $1  (exit $2)")
    overall=1
  fi
}

# ------------------------- repository root check -------------------------
if [ ! -f pyproject.toml ]; then
  echo "${RED}Error:${RESET} no encuentro pyproject.toml." >&2
  echo "Ejecuta el script desde la raíz del repositorio." >&2
  exit 1
fi

# -------------------------------- interpreter ------------------------------------
if [ "$USE_VENV" -eq 1 ]; then
  section "Preparando entorno virtual ($VENV_DIR)"
  if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON" -m venv "$VENV_DIR" || { echo "${RED}No se pudo crear el venv${RESET}" >&2; exit 1; }
  fi
  if   [ -x "$VENV_DIR/bin/python" ];        then PY="$VENV_DIR/bin/python"
  elif [ -x "$VENV_DIR/Scripts/python.exe" ]; then PY="$VENV_DIR/Scripts/python.exe"
  else echo "${RED}No encuentro el python del venv${RESET}" >&2; exit 1
  fi
else
  PY="$PYTHON"
fi

echo "Intérprete: $("$PY" --version 2>&1)  ->  $PY"
PYVER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"

# -------------------------------- installation -----------------------------------
if [ "$DO_INSTALL" -eq 1 ]; then
  section "Instalando paquete y dependencias"
  "$PY" -m pip install --upgrade pip
  # .[dev] includes ruff + pytest + pytest-cov: it covers lint and test-core.
  "$PY" -m pip install -e ".[dev]"
  if [ "$RUN_EXAMPLES" -eq 1 ]; then
    case "$PYVER" in
      3.8|3.9)
        echo "${YELLOW}Aviso:${RESET} Python $PYVER < 3.10. gymnasium/box2d de los"
        echo "        ejemplos pueden no tener wheels y fallar al instalar."
        echo "        El CI solo prueba los ejemplos en 3.10-3.13." ;;
    esac
    "$PY" -m pip install \
      -r examples/first_order_plant_control/requirements.txt \
      -r examples/lunar_lander/requirements.txt
  fi
fi

# ------------------------------------- lint --------------------------------------
if [ "$RUN_LINT" -eq 1 ]; then
  section "Lint — ruff check ."
  "$PY" -m ruff check .
  record "lint: ruff check" $?

  section "Lint — ruff format --check ."
  "$PY" -m ruff format --check .
  record "lint: ruff format --check" $?
fi

# ---------------------------------- test-core ------------------------------------
if [ "$RUN_CORE" -eq 1 ]; then
  section "test-core — pytest (gate de cobertura >=95%)"
  "$PY" -m pytest \
    --ignore=tests/test_fopcontrol_smoke.py \
    --ignore=tests/test_fopcontrol_reward_unit.py \
    --ignore=tests/test_lunarlander_smoke.py \
    --ignore=tests/test_lunarlander_reward_unit.py
  record "test-core" $?
fi

# -------------------------------- test-examples ----------------------------------
if [ "$RUN_EXAMPLES" -eq 1 ]; then
  section "test-examples — pytest de ejemplos (sin gate de cobertura)"
  "$PY" -m pytest \
    tests/test_fopcontrol_smoke.py \
    tests/test_fopcontrol_reward_unit.py \
    tests/test_lunarlander_smoke.py \
    tests/test_lunarlander_reward_unit.py \
    --override-ini addopts=""
  record "test-examples" $?
fi

# ------------------------------------ resumen ------------------------------------
section "Resumen"
if [ "${#RESULTS[@]}" -gt 0 ]; then
  for line in "${RESULTS[@]}"; do echo "  $line"; done
fi

echo ""
if [ "$overall" -eq 0 ]; then
  echo "${GREEN}${BOLD}Todo OK.${RESET}"
else
  echo "${RED}${BOLD}Hay fallos.${RESET} Revisa la salida anterior."
fi
exit $overall
