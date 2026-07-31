# SPDX-License-Identifier: GPL-3.0-only

"""
SYSTEM FOR DECOUPLING RL SPIN LOOP FROM AGENT SPIN LOOP.

(c) Juan-Antonio Fernández-Madrigal
Uncore Team, 2025
"""

from enum import Enum
from importlib.metadata import PackageNotFoundError, version

try:
    from rl_spin_decoupler.socketcomms.comms import (
        BaseCommPoint,
        ClientCommPoint,
        ServerCommPoint,
    )
except ImportError:  # pragma: no cover
    from socketcomms.comms import BaseCommPoint, ClientCommPoint, ServerCommPoint


try:
    # Read version from installed package metadata to avoid hardcoding.
    __version__ = version("rl-spin-decoupler")
except PackageNotFoundError:  # pragma: no cover
    # Fallback for source-only execution when package metadata is unavailable.
    __version__ = "0+unknown"
__all__ = [
    "AgentSide",
    "BaseCommPoint",
    "ClientCommPoint",
    "RLSide",
    "ServerCommPoint",
]


# -------------------------------------------------------------------------------
#
# 	Base Class: RLSide
#
# -------------------------------------------------------------------------------


class RLSide:
    """
    Communication endpoint used by the reinforcement-learning process.

    An :class:`RLSide` instance owns the server side of the TCP connection and
    coordinates one high-level RL loop with one external agent loop. The typical
    call sequence in an environment wrapper is:

    1. Construct :class:`RLSide` before training starts.
    2. Call :meth:`resetGetObs` from ``reset()``.
    3. Call :meth:`stepSendActGetObs` from each ``step()``.
    4. Call :meth:`stepExpFinished` once the experiment is over.

    The class does not define rewards or episode termination on its own. It only
    transports actions, observations, and timing metadata between processes.
    """

    def __init__(self, port: int, verbose: bool = False):
        """
        Create the RL-side communication endpoint and wait for the agent.

        Args:
                port: TCP port used by the RL process to listen for the agent
                        connection. Valid ports are restricted by the underlying
                        transport layer to the range ``20000`` to ``49151``. The
                        socket always binds to every interface (``0.0.0.0``), so this
                        works unmodified whether the process runs on a bare host or
                        inside a container publishing the port to another host.
                verbose: If ``True``, print lifecycle messages while waiting for the
                        agent and while closing the connection.

        Raises:
                RuntimeError: If no agent connects before the transport timeout or if
                        the underlying server endpoint cannot be started.
        """

        self._verbose = verbose
        self._rlcomm = ServerCommPoint(port)  # not connected yet
        # if socket in use, repeatedly wait
        # until free
        if self._verbose:
            print("RL decoupler enabled. Waiting for agent connection...")
        res = self._rlcomm.begin(timeoutaccept=60.0)  # blocks for agent
        if len(res) > 0:
            raise RuntimeError("No agent connection: " + res)
        if self._verbose:
            print("Agent connected to this RL")

    def __del__(self):
        """Best-effort shutdown during garbage collection."""

        rlcomm = getattr(self, "_rlcomm", None)
        if rlcomm is None:
            return
        res = rlcomm.end()
        if len(res) > 0:
            print("Error closing communications with the agent: " + res)
        if self._verbose:
            print("Communications closed in the RL side.")

    def resetGetObs(self, timeout: float = 10.0):
        """
        Request the first observation after an episode reset.

        This method is usually called from the RL environment ``reset()`` method.
        It blocks until the agent acknowledges the reset and sends back the first
        observation.

        Args:
                timeout: Timeout in seconds for each communication operation involved
                        in the request. Values less than or equal to ``0.0`` disable the
                        timeout at the transport layer.

        Returns:
                Tuple[dict, float]: A pair ``(obs, ato)`` where ``obs`` is the first
                observation dictionary after the reset and ``ato`` is the agent time of
                observation, that is, the timestamp measured by the agent-side clock
                when that observation was sampled.

        Raises:
                RuntimeError: If the reset command cannot be sent or if the reply is
                        not received successfully.
        """

        res = self._rlcomm.sendData(dict({"stepkind": "reset"}))
        if len(res) > 0:
            raise RuntimeError("Error sending what to do to the agent. " + res)

        res, obsato = self._rlcomm.readData(timeout)
        if len(res) > 0:
            raise RuntimeError(
                "Error reading after-reset observation from the agent. " + res
            )

        return obsato["obs"], obsato["ato"]  # return tuple

    def stepSendActGetObs(self, action, timeout: float = 10.0):
        """
        Send a new action and wait for the corresponding observation.

        This method is usually called from the RL environment ``step()`` method.
        It first transfers the new action to the agent and then blocks until the
        agent reports two pieces of information:

        1. ``LAT``: the actual execution duration of the previous action.
        2. The observation collected after applying the new action, optionally
                accompanied by a reward computed on the agent side.

        Args:
                action: Action object to send to the agent. It must be serializable
                        by Python ``pickle`` because the transport exchanges pickled
                        payloads.
                timeout: Timeout in seconds for each communication operation involved
                        in the request. Values less than or equal to ``0.0`` disable the
                        timeout at the transport layer.

        Returns:
                Tuple[float, dict, float, float]: A tuple ``(lat, obs, rew, ato)``.
                ``lat`` is the duration of the previous action as measured by the
                agent clock, ``obs`` is the observation dictionary after executing the
                new action, ``rew`` is the reward reported by the agent for the current
                action, and ``ato`` is the agent timestamp when the observation was
                captured.

                If your RL code also records a local wall-clock timestamp such as
                ``t_wall = time.time()``, keep in mind that ``t_wall`` belongs to the
                RL process clock, while ``lat`` and ``ato`` come from the agent-side
                clock domain.

        Raises:
                RuntimeError: If sending the action fails or if either response
                        message cannot be received successfully.
        """

        # send a STEP indicator to the agent interface, that should use
        # readWhatToDo() to get the indicator
        res = self._rlcomm.sendData(dict({"stepkind": "step", "action": action}))
        if len(res) > 0:
            raise RuntimeError("Error sending step action: " + res)

        res, lat = self._rlcomm.readData(timeout)  # blocks
        if len(res) > 0:
            raise RuntimeError("Error receiving last action duration: " + res)

        res, obsrewato = self._rlcomm.readData(timeout)  # blocks
        if len(res) > 0:
            raise RuntimeError("Error receiving step observation: " + res)

        return lat["lat"], obsrewato["obs"], obsrewato["rew"], obsrewato["ato"]

    def stepExpFinished(self, timeout: float = 10.0):
        """
        Notify the agent that the experiment has finished.

        Call this method once, after the final RL step, when no more actions will
        be sent. The method sends a finish indicator to the agent loop so it can
        perform its own shutdown procedure.

        Args:
                timeout: Reserved for API symmetry with the other high-level methods.
                        The current implementation sends the finish indicator
                        immediately and does not consume the timeout value.

        Raises:
                RuntimeError: Not raised directly by this wrapper, but downstream
                        socket errors may still surface during shutdown in user-managed
                        teardown code.
        """

        self._rlcomm.sendData(dict({"stepkind": "finish"}))


# -------------------------------------------------------------------------------
#
# 	Base Class: AgentSide
#
# -------------------------------------------------------------------------------


class AgentSide:
    """
    Communication endpoint used by the external agent process.

    An :class:`AgentSide` instance owns the client side of the TCP connection and
    lets an agent loop poll for new RL commands without blocking when no command
    is pending. The agent remains responsible for applying actions, resetting its
    workspace, sampling observations, and optionally computing rewards.
    """

    class WhatToDo(Enum):
        """
        Command kinds that the RL process can send to the agent loop.

        Attributes:
                REC_ACTION_SEND_OBS: Receive a new action, report the duration of the
                        previous action, and later send the observation that results
                        from the new action.
                RESET_SEND_OBS: Reset the agent-side episode state and send the first
                        observation after that reset.
                FINISH: Stop the experiment and terminate the control loop gracefully.
        """

        REC_ACTION_SEND_OBS = 0  # receive action from RL, executes it and sends
        # back resulting observation and other stuff
        RESET_SEND_OBS = 1  # reset episode and send observation back to RL
        FINISH = 2  # finish experiment (and comms)

    def __init__(
        self, ipbaselinespart: str, portbaselinespart: int, verbose: bool = False
    ):
        """
        Create the agent-side communication endpoint and connect to the RL side.

        Args:
                ipbaselinespart: IPv4 address of the RL process that owns the
                        listening server socket. A common pattern is to pass
                        ``BaseCommPoint.get_ip()`` when both processes run on
                        the same host.
                portbaselinespart: TCP port exposed by the RL process.
                verbose: If ``True``, print lifecycle messages while connecting and
                        while closing the connection.

        Raises:
                RuntimeError: If the client socket cannot connect to the RL process.
        """

        self._verbose = verbose
        self._rlcomm = ClientCommPoint(ipbaselinespart, portbaselinespart)

        if self._verbose:
            print("Agent decoupler enabled.")

        res = self._rlcomm.begin()
        if len(res) > 0:
            raise RuntimeError("Error starting connection with RL. " + res)

        if self._verbose:
            print("Agent decoupler connected to RL decoupler")

    def __del__(self):
        """Best-effort shutdown during garbage collection."""

        rlcomm = getattr(self, "_rlcomm", None)
        if rlcomm is None:
            return
        res = rlcomm.end()
        if len(res) > 0:
            raise RuntimeError("Error stopping connection with RL: " + res)
        if self._verbose:
            print("Connection with RL finished.")

    def readWhatToDo(self, timeout: float = 10.0):
        """
        Poll the RL side for the next command.

        This method is intended to be called from every agent-loop iteration.
        The initial poll is non-blocking: if no data are pending in the socket,
        the method returns ``None`` immediately. Once pending data are detected,
        reading the command itself may block up to ``timeout`` seconds.

        Args:
                timeout: Maximum number of seconds allowed for reading a pending
                        command once the socket indicates that data are available.
                        Values less than or equal to ``0.0`` disable the timeout at
                        the transport layer.

        Returns:
                Optional[Tuple[AgentSide.WhatToDo, Any]]: ``None`` if no command is
                pending. Otherwise, a tuple whose first element is a
                :class:`WhatToDo` value and whose second element is the payload:

                - ``(WhatToDo.REC_ACTION_SEND_OBS, action)`` for a normal step.
                - ``(WhatToDo.RESET_SEND_OBS, None)`` for a reset request.
                - ``(WhatToDo.FINISH, None)`` for shutdown.

        Raises:
                RuntimeError: If reading the pending command fails.
                ValueError: If the RL side sends an unknown command indicator.
        """

        if not self._rlcomm.checkDataToRead():
            return None

        # read last (pending) step()/reset() msg and then proceed accordingly
        res, ind = self._rlcomm.readData(timeout)
        # read a dict: { 'stepkind' : 'reset', 'step' or 'finish' ,
        # 			      'action' : <action> if any}
        if len(res) > 0:
            raise RuntimeError("Error receiving what-to-do from RL: " + res)

        if ind["stepkind"] == "step":
            return (AgentSide.WhatToDo.REC_ACTION_SEND_OBS, ind["action"])
        elif ind["stepkind"] == "reset":
            return (AgentSide.WhatToDo.RESET_SEND_OBS, None)
        elif ind["stepkind"] == "finish":
            return (AgentSide.WhatToDo.FINISH, None)
        else:
            raise (ValueError("Unknown what-to-do indicator [" + ind["stepkind"] + "]"))

    def stepSendLastActDur(self, lat: float):
        """
        Send ``LAT``, the duration of the previous action, back to the RL side.

        This method should be called immediately after receiving
        ``WhatToDo.REC_ACTION_SEND_OBS`` and before the new action starts running
        for its full control interval.

        Args:
                lat: Actual duration, in seconds, of the action that was being executed
                        before the newly received action replaced it. This value belongs
                        to the agent clock domain.

        Raises:
                RuntimeError: If the timing payload cannot be sent.
        """

        res = self._rlcomm.sendData(dict({"lat": lat}))
        if len(res) > 0:
            raise RuntimeError("Error sending lat to RL. " + res)

    def stepSendObs(self, obs, agenttime: float = 0.0, rew: float = 0.0):
        """
        Send the observation obtained after executing a step action.

        This method completes the response cycle that starts with
        ``WhatToDo.REC_ACTION_SEND_OBS``. It should be called after the agent has
        applied the action long enough to produce the next observation.

        Args:
                obs: Observation dictionary produced by the agent or simulator.
                agenttime: Agent-side timestamp for when ``obs`` was sampled. This is
                        the value that the RL side receives as ``ATO``.
                rew: Reward associated with the current action, when reward
                        computation is delegated to the agent side. If the RL process
                        owns reward computation, leave this value at its default.

        Raises:
                RuntimeError: If the observation payload cannot be sent.
        """

        res = self._rlcomm.sendData(dict({"obs": obs, "rew": rew, "ato": agenttime}))
        if len(res) > 0:
            raise RuntimeError("Error sending observation/reward to RL. " + res)

    def resetSendObs(self, obs, agenttime=0.0):
        """
        Send the first observation collected after a reset.

        This method should be called after receiving
        ``WhatToDo.RESET_SEND_OBS`` and completing the agent-side reset logic.

        Args:
                obs: Observation dictionary collected immediately after the reset.
                agenttime: Agent-side timestamp for when the reset observation was
                        sampled. This is the value that the RL side receives as ``ATO``.

        Raises:
                RuntimeError: If the reset observation cannot be sent.
        """

        res = self._rlcomm.sendData({"obs": obs, "ato": agenttime})
        if len(res) > 0:
            raise RuntimeError("Error sending observation to RL. " + res)
