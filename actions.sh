# 1. Instalar todas las dependencias
python -m pip install -e ".[dev,docs]" -r examples/first_order_plant_control/requirements.txt -r examples/lunar_lander/requirements.txt && \
# 2. Linter y formato (Job: lint)
python -m ruff check . && \
python -m ruff format --check . && \
# 3. Tests Core (Job: test-core)
python -m pytest --ignore=tests/test_fopcontrol_smoke.py --ignore=tests/test_fopcontrol_reward_unit.py --ignore=tests/test_lunarlander_smoke.py --ignore=tests/test_lunarlander_reward_unit.py && \
# 4. Tests de Ejemplos (Job: test-examples)
python -m pytest tests/test_fopcontrol_smoke.py tests/test_fopcontrol_reward_unit.py tests/test_lunarlander_smoke.py tests/test_lunarlander_reward_unit.py --override-ini addopts="" && \
# 5. Generar Documentación (Job: build de Docs)
sphinx-build -b html docs docs/_build/html
