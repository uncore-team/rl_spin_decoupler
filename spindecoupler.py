# SPDX-License-Identifier: GPL-3.0-only

"""
Decouple the spin loop of a reinforcement-learning algorithm from the spin loop
of the agent (a robot or a simulator) that runs in another process.

The RL process uses :class:`RLSide` and the agent process uses :class:`AgentSide`.
They exchange commands, actions and observations over one TCP connection, and the
agent also reports two timing values with each step: ``LAT``, how long the
previous action actually lasted, and ``ATO``, when the observation was sampled.
Both are measured with the agent's clock.

(c) Juan-Antonio Fernández-Madrigal
Uncore Team, 2025
"""

from enum import Enum
from importlib.metadata import PackageNotFoundError, version

try:
    from rl_spin_decoupler.socketcomms.comms import (
        DEFAULT_MAX_FRAME_SIZE,
        BaseCommPoint,
        ClientCommPoint,
        ServerCommPoint,
        enable_console_logging,
        logger,
    )
except ImportError:  # pragma: no cover
    from socketcomms.comms import (
        DEFAULT_MAX_FRAME_SIZE,
        BaseCommPoint,
        ClientCommPoint,
        ServerCommPoint,
        enable_console_logging,
        logger,
    )


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
    "enable_console_logging",
]


class RLSide:
    """
    Endpoint used by the reinforcement-learning process.

    It owns the server side of the connection and drives the agent from the
    environment wrapper:

    1. Create the object before training starts.
    2. Call :meth:`resetGetObs` from ``reset()``.
    3. Call :meth:`stepSendActGetObs` from each ``step()``.
    4. Call :meth:`stepExpFinished` when the experiment is over.

    It defines no reward and no episode termination; it only moves actions,
    observations and timing values between the two processes.
    """

    def __init__(
        self,
        port: int,
        verbose: bool = False,
        *,
        accept_timeout: float = 60.0,
        max_frame_size: int = DEFAULT_MAX_FRAME_SIZE,
        allow_pickle: bool = False,
        send_timeout: float = 10.0,
    ):
        """
        Open the server socket and wait for the agent to connect.

        Args:
            port: TCP port to listen on, between 20000 and 49151. The socket binds
                to every interface, so this works on a bare host and inside a
                container that publishes the port.
            verbose: Log a message when waiting starts and when the agent
                connects, and when the connection is closed. The messages go to
                the ``spindecoupler`` logger, and ``verbose`` also shows them on the
                standard output.
            accept_timeout: Seconds to wait for the agent to connect. Raise it when
                the agent starts on another host after a delay.
            max_frame_size: Largest message, in bytes, that is sent or accepted.
            allow_pickle: Accept any pickle from the agent. By default only plain
                data and NumPy arrays are accepted. Enable it only if you trust
                the agent: a pickle can run arbitrary code.
            send_timeout: Seconds a send may block if the agent stops reading.

        Raises:
            ValueError: If ``accept_timeout`` is not positive.
            RuntimeError: If the agent does not connect in time.
            OSError: If the port stays in use (see :class:`ServerCommPoint`).
        """

        if accept_timeout <= 0.0:
            raise ValueError("accept_timeout must be > 0.0")
        self._verbose = verbose
        if self._verbose:
            enable_console_logging()
        self._rlcomm = ServerCommPoint(
            port,
            max_frame_size=max_frame_size,
            allow_pickle=allow_pickle,
            send_timeout=send_timeout,
        )  # not connected yet
        if self._verbose:
            logger.info("RL decoupler enabled. Waiting for agent connection...")
        res = self._rlcomm.begin(timeoutaccept=accept_timeout)  # blocks for agent
        if len(res) > 0:
            raise RuntimeError("No agent connection: " + res)
        if self._verbose:
            logger.info("Agent connected to this RL")

    def __del__(self):
        """Close the connection when the object is collected."""

        rlcomm = getattr(self, "_rlcomm", None)
        if rlcomm is None:
            return
        res = rlcomm.end()
        if len(res) > 0:
            logger.error("Error closing communications with the agent: " + res)
        if self._verbose:
            logger.info("Communications closed in the RL side.")

    def resetGetObs(self, timeout: float = 10.0):
        """
        Ask the agent to reset the episode and wait for the first observation.

        Call it from the environment's ``reset()``.

        Args:
            timeout: Seconds to wait for the reply. A value of 0 or less waits
                without limit.

        Returns:
            A pair ``(obs, ato)``: the first observation after the reset and the
            agent time at which it was sampled.

        Raises:
            RuntimeError: If the request cannot be sent or no valid reply arrives.
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
        Send an action and wait for the resulting observation.

        Call it from the environment's ``step()``. The agent answers with two
        messages: first ``LAT``, the duration of the previous action, then the
        observation that follows the new action.

        Args:
            action: Any picklable object.
            timeout: Seconds to wait for each of the two replies. A value of 0 or
                less waits without limit.

        Returns:
            A tuple ``(lat, obs, rew, ato)``: the duration of the previous action,
            the new observation, the reward the agent attached to it (0.0 unless
            the agent computes one) and the agent time at which the observation
            was sampled.

            ``lat`` and ``ato`` come from the agent's clock. A ``time.time()``
            taken in the RL process belongs to another clock, so the two are not
            directly comparable.

        Raises:
            RuntimeError: If the action cannot be sent or a reply does not arrive.
        """

        # The agent picks this up with readWhatToDo().
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
        Tell the agent that the experiment is over.

        Call it once, after the last step. It returns as soon as the message is
        written. A failure to send it is not reported, because by then the agent
        may already be gone.

        Args:
            timeout: Not used; kept so the three calls have the same signature.
        """

        self._rlcomm.sendData(dict({"stepkind": "finish"}))


class AgentSide:
    """
    Endpoint used by the agent process.

    It owns the client side of the connection. The agent calls
    :meth:`readWhatToDo` on every iteration of its own loop; the call returns at
    once when the RL side has sent nothing, so the loop never stalls waiting for
    the learner. The agent is responsible for applying actions, resetting its
    workspace and sampling observations.
    """

    class WhatToDo(Enum):
        """Commands the RL side can send to the agent."""

        REC_ACTION_SEND_OBS = 0  # new action: reply with the previous action's
        # duration, then with the observation that follows it
        RESET_SEND_OBS = 1  # reset the episode and reply with the first observation
        FINISH = 2  # the experiment is over

    def __init__(
        self,
        ipbaselinespart: str,
        portbaselinespart: int,
        verbose: bool = False,
        *,
        max_frame_size: int = DEFAULT_MAX_FRAME_SIZE,
        allow_pickle: bool = False,
        send_timeout: float = 10.0,
    ):
        """
        Connect to the RL side.

        Start the RL process first: it is the one that listens.

        Args:
            ipbaselinespart: IPv4 address of the RL process. When both processes
                run on the same host, ``BaseCommPoint.get_ip()`` gives a valid one.
            portbaselinespart: TCP port the RL process listens on.
            verbose: Log a message when the connection is opened and closed (see
                :class:`RLSide` for where the messages go).
            max_frame_size: Largest message, in bytes, that is sent or accepted.
            allow_pickle: Accept any pickle from the RL side. By default only plain
                data and NumPy arrays are accepted. Enable it only if you trust
                the RL side.
            send_timeout: Seconds a send may block if the RL side stops reading.

        Raises:
            RuntimeError: If the connection cannot be established.
        """

        self._verbose = verbose
        if self._verbose:
            enable_console_logging()
        self._rlcomm = ClientCommPoint(
            ipbaselinespart,
            portbaselinespart,
            max_frame_size=max_frame_size,
            allow_pickle=allow_pickle,
            send_timeout=send_timeout,
        )

        if self._verbose:
            logger.info("Agent decoupler enabled.")

        res = self._rlcomm.begin()
        if len(res) > 0:
            raise RuntimeError("Error starting connection with RL. " + res)

        if self._verbose:
            logger.info("Agent decoupler connected to RL decoupler")

    def __del__(self):
        """Close the connection when the object is collected."""

        rlcomm = getattr(self, "_rlcomm", None)
        if rlcomm is None:
            return
        res = rlcomm.end()
        if len(res) > 0:
            raise RuntimeError("Error stopping connection with RL: " + res)
        if self._verbose:
            logger.info("Connection with RL finished.")

    def readWhatToDo(self, timeout: float = 10.0):
        """
        Check whether the RL side has sent a command and read it.

        It returns ``None`` immediately if nothing is pending. Once something is
        pending, reading it can block for up to ``timeout`` seconds.

        Args:
            timeout: Seconds to wait for a command that has started to arrive. A
                value of 0 or less waits without limit.

        Returns:
            ``None`` if no command is pending, otherwise a pair
            ``(what, payload)`` where ``what`` is a :class:`WhatToDo` value:

            - ``(WhatToDo.REC_ACTION_SEND_OBS, action)`` for a step,
            - ``(WhatToDo.RESET_SEND_OBS, None)`` for a reset,
            - ``(WhatToDo.FINISH, None)`` for the end of the experiment.

        Raises:
            RuntimeError: If the command cannot be read.
            ValueError: If the command is not one of the three above.
        """

        if not self._rlcomm.checkDataToRead():
            return None

        # The message is a dict: {'stepkind': 'reset' | 'step' | 'finish',
        # 'action': <action>} where 'action' is present only for a step.
        res, ind = self._rlcomm.readData(timeout)
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
        Send ``LAT``, the duration of the action that the new one replaces.

        Call it right after receiving ``REC_ACTION_SEND_OBS``, before sending the
        observation.

        Args:
            lat: Seconds the previous action actually lasted, by the agent's clock.

        Raises:
            RuntimeError: If it cannot be sent.
        """

        res = self._rlcomm.sendData(dict({"lat": lat}))
        if len(res) > 0:
            raise RuntimeError("Error sending lat to RL. " + res)

    def stepSendObs(self, obs, agenttime: float = 0.0, rew: float = 0.0):
        """
        Send the observation that follows a step action.

        It completes the reply to ``REC_ACTION_SEND_OBS``; call it once the action
        has been applied for its whole interval.

        Args:
            obs: The observation. Any picklable object.
            agenttime: Time at which ``obs`` was sampled, by the agent's clock. The
                RL side receives it as ``ato``.
            rew: Reward for the action, if the agent computes it. Leave the
                default when the RL side does.

        Raises:
            RuntimeError: If it cannot be sent.
        """

        res = self._rlcomm.sendData(dict({"obs": obs, "rew": rew, "ato": agenttime}))
        if len(res) > 0:
            raise RuntimeError("Error sending observation/reward to RL. " + res)

    def resetSendObs(self, obs, agenttime=0.0):
        """
        Send the first observation after a reset.

        Call it once the agent has finished handling ``RESET_SEND_OBS``.

        Args:
            obs: The observation. Any picklable object.
            agenttime: Time at which ``obs`` was sampled, by the agent's clock.

        Raises:
            RuntimeError: If it cannot be sent.
        """

        res = self._rlcomm.sendData({"obs": obs, "ato": agenttime})
        if len(res) > 0:
            raise RuntimeError("Error sending observation to RL. " + res)
