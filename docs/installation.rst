Installation
============

Requirements
------------

Python 3.8 or newer. The communication core uses only the standard library.

Install from a checkout
-----------------------

.. code-block:: bash

   git clone https://github.com/uncore-team/rl_spin_decoupler.git
   cd rl_spin_decoupler
   python -m pip install -e .

Optional dependencies
---------------------

Stable-Baselines3 and Gymnasium are not needed by the library. Install them only if
your training scripts use them:

.. code-block:: bash

   python -m pip install -e ".[rl]"

To run the tests or build these pages, install the development or documentation extras:

.. code-block:: bash

   python -m pip install -e ".[dev]"
   python -m pip install -e ".[docs]"

Import paths
------------

The package can be imported under two names:

.. code-block:: python

   from spindecoupler import RLSide, AgentSide
   from rl_spin_decoupler import RLSide, AgentSide
