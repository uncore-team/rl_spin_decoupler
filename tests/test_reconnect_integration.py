# SPDX-License-Identifier: GPL-3.0-only

"""A lost connection, recovered with ``reconnect()`` over real sockets."""

import threading
import time

import pytest

from spindecoupler import AgentSide, BaseCommPoint, ConnectionLost, RLSide


def connect_agent_with_retry(ip, port, attempts=50, delay=0.05):
    last_error = None
    for _ in range(attempts):
        try:
            return AgentSide(ip, port)
        except RuntimeError as exc:
            last_error = exc
            time.sleep(delay)
    raise last_error


def serve_one_step(agent, deadline_s=5.0):
    """Serve a reset and a step, then return when the step has been answered."""
    deadline = time.time() + deadline_s
    steps = 0
    while time.time() < deadline:
        command = agent.readWhatToDo(timeout=1.0)
        if command is None:
            time.sleep(0.005)
            continue
        what, _payload = command
        if what == AgentSide.WhatToDo.RESET_SEND_OBS:
            agent.resetSendObs({"phase": "reset"}, agenttime=1.0)
        elif what == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
            agent.stepSendLastActDur(0.1)
            agent.stepSendObs({"phase": "step"}, agenttime=2.0)
            steps += 1
            return steps
    raise AssertionError("the agent was not asked for a step in time")


def run_scenario(port, make_second_agent):
    """The agent drops in the middle of a step; both sides recover."""
    results = {}
    errors = []

    def rl_worker():
        try:
            rl = RLSide(port, accept_timeout=5.0)
            results["reset1"] = rl.resetGetObs(timeout=2.0)
            try:
                rl.stepSendActGetObs("a", timeout=2.0)
            except ConnectionLost as exc:
                results["lost"] = str(exc)
            rl.reconnect(timeout=5.0)
            results["reset2"] = rl.resetGetObs(timeout=2.0)
            results["step2"] = rl.stepSendActGetObs("b", timeout=2.0)
            rl.stepExpFinished()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=rl_worker, daemon=True)
    thread.start()

    agent = connect_agent_with_retry(BaseCommPoint.get_ip(), port)
    # Serve the reset, then take the step command and vanish without answering.
    deadline = time.time() + 5.0
    got_step = False
    while time.time() < deadline and not got_step:
        command = agent.readWhatToDo(timeout=1.0)
        if command is None:
            time.sleep(0.005)
            continue
        what, _payload = command
        if what == AgentSide.WhatToDo.RESET_SEND_OBS:
            agent.resetSendObs({"phase": "reset"}, agenttime=1.0)
        else:
            got_step = True
    assert got_step
    agent._rlcomm.end()

    with pytest.raises(ConnectionLost):
        agent.readWhatToDo()

    second = make_second_agent(agent, port)
    assert serve_one_step(second) == 1
    second._rlcomm.end()

    thread.join(timeout=10.0)
    assert not thread.is_alive(), "RL thread did not finish"
    assert errors == []
    assert "Error" in results["lost"] or "Connection" in results["lost"]
    assert results["reset1"] == ({"phase": "reset"}, 1.0)
    assert results["reset2"] == ({"phase": "reset"}, 1.0)
    assert results["step2"][1] == {"phase": "step"}


def test_same_agent_reconnects_after_a_drop(free_tcp_port, force_localhost):
    def same_agent(agent, _port):
        agent.reconnect(timeout=5.0, retry_interval=0.05)
        return agent

    run_scenario(free_tcp_port, same_agent)


def test_a_new_agent_is_accepted_after_a_drop(free_tcp_port, force_localhost):
    def new_agent(_old, port):
        return connect_agent_with_retry(BaseCommPoint.get_ip(), port)

    run_scenario(free_tcp_port, new_agent)
