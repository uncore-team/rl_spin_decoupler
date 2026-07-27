Installation
============

Requirements
------------

- Python 3.8 or newer.
- No mandatory third-party dependencies for the communication core.

Install from a local checkout
-----------------------------

Clone the repository and install the package in editable mode:

.. code-block:: bash

   git clone https://github.com/uncore-team/rl_spin_decoupler.git
   cd rl_spin_decoupler
   python -m pip install -e .

Install optional RL dependencies
--------------------------------

The core package does not require Stable-Baselines3 or Gymnasium. Install them
only if your training scripts depend on them:

.. code-block:: bash

   python -m pip install -e ".[rl]"

Install documentation or test dependencies
------------------------------------------

For contributor workflows:

.. code-block:: bash

   python -m pip install -e ".[dev]"
   python -m pip install -e ".[docs]"

Import paths
------------

The installed package exposes both of these import styles:

.. code-block:: python

   import spindecoupler
   from spindecoupler import RLSide, AgentSide

   import rl_spin_decoupler
   from rl_spin_decoupler import RLSide, AgentSide