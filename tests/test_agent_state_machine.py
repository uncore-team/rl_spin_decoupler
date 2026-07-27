from enum import Enum

import pytest

from spindecoupler import AgentSide


class RecordingComm:
	def __init__(self, commands):
		self.commands = list(commands)
		self.calls = []

	def readWhatToDo(self):
		self.calls.append(("readWhatToDo", None))
		if self.commands:
			return self.commands.pop(0)
		return None

	def stepSendLastActDur(self, lat):
		self.calls.append(("stepSendLastActDur", lat))

	def stepSendObs(self, obs, agenttime):
		self.calls.append(("stepSendObs", obs, agenttime))

	def resetSendObs(self, obs, agenttime):
		self.calls.append(("resetSendObs", obs, agenttime))


class AgentHarness:
	class StepState(Enum):
		READYFORRLCOMMAND = 0
		EXECUTINGLASTACTION = 1
		AFTERRESET = 2

	def __init__(self, comm, rltimestep=0.5):
		self._commstoRL = comm
		self._rltimestep = rltimestep
		self._stepstate = AgentHarness.StepState.READYFORRLCOMMAND
		self._lastaction = "idle"
		self._lastactiont0 = 0.0
		self._starttimecurepisode = 0.0

	def _null_action(self):
		return "idle"

	def _build_observation(self):
		return {"action": self._lastaction, "episode_start": self._starttimecurepisode}

	def step(self, curtime):
		act = self._lastaction

		if self._stepstate == AgentHarness.StepState.EXECUTINGLASTACTION:
			if curtime - self._lastactiont0 >= self._rltimestep:
				self._commstoRL.stepSendObs(self._build_observation(), curtime)
				self._stepstate = AgentHarness.StepState.READYFORRLCOMMAND

		elif self._stepstate == AgentHarness.StepState.READYFORRLCOMMAND:
			whattodo = self._commstoRL.readWhatToDo()
			if whattodo is not None:
				what, payload = whattodo
				if what == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
					lat = curtime - self._lastactiont0
					self._lastactiont0 = curtime
					self._commstoRL.stepSendLastActDur(lat)
					self._stepstate = AgentHarness.StepState.EXECUTINGLASTACTION
					act = payload
				elif what == AgentSide.WhatToDo.RESET_SEND_OBS:
					act = self._null_action()
					self._starttimecurepisode = curtime
					self._stepstate = AgentHarness.StepState.AFTERRESET
				elif what == AgentSide.WhatToDo.FINISH:
					raise RuntimeError("Experiment finished")
				else:
					raise ValueError("Unknown indicator data")

		elif self._stepstate == AgentHarness.StepState.AFTERRESET:
			self._commstoRL.resetSendObs(self._build_observation(), curtime)
			self._stepstate = AgentHarness.StepState.READYFORRLCOMMAND

		self._lastaction = act
		return act


def test_state_machine_ignores_empty_polling_cycle():
	comm = RecordingComm(commands=[])
	agent = AgentHarness(comm)

	assert agent.step(curtime=0.1) == "idle"
	assert agent._stepstate == AgentHarness.StepState.READYFORRLCOMMAND
	assert comm.calls == [("readWhatToDo", None)]


def test_state_machine_handles_reset_then_sends_observation():
	comm = RecordingComm(commands=[(AgentSide.WhatToDo.RESET_SEND_OBS, None)])
	agent = AgentHarness(comm)

	assert agent.step(curtime=1.0) == "idle"
	assert agent._stepstate == AgentHarness.StepState.AFTERRESET
	assert agent.step(curtime=1.1) == "idle"
	assert agent._stepstate == AgentHarness.StepState.READYFORRLCOMMAND
	assert comm.calls[-1] == (
		"resetSendObs",
		{"action": "idle", "episode_start": 1.0},
		1.1,
	)


def test_state_machine_computes_lat_and_waits_before_sending_observation():
	comm = RecordingComm(commands=[(AgentSide.WhatToDo.REC_ACTION_SEND_OBS, "turn-left")])
	agent = AgentHarness(comm, rltimestep=0.5)
	agent._lastactiont0 = 1.0

	assert agent.step(curtime=2.0) == "turn-left"
	assert agent._stepstate == AgentHarness.StepState.EXECUTINGLASTACTION
	assert ("stepSendLastActDur", 1.0) in comm.calls

	assert agent.step(curtime=2.2) == "turn-left"
	assert agent._stepstate == AgentHarness.StepState.EXECUTINGLASTACTION

	assert agent.step(curtime=2.6) == "turn-left"
	assert agent._stepstate == AgentHarness.StepState.READYFORRLCOMMAND
	assert comm.calls[-1] == (
		"stepSendObs",
		{"action": "turn-left", "episode_start": 0.0},
		2.6,
	)


def test_state_machine_raises_on_finish_command():
	comm = RecordingComm(commands=[(AgentSide.WhatToDo.FINISH, None)])
	agent = AgentHarness(comm)

	with pytest.raises(RuntimeError, match="Experiment finished"):
		agent.step(curtime=0.0)