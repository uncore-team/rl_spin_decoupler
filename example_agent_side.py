""" 
AGENT PART OF THE RL-AGENT DECOUPLING EXAMPLE
Start this after the other.
"""

# imports
from spindecoupler import AgentSide



class Agent:
	"""The agent"""

	def __init__(self, debug = False) -> None:
	
		self._control_timestep = ... # timestep of the step() cycle in the agent
		self._rltimestep = ... # timestep of a RL step (not of an Agent step; the timestep of the Agent step is self._control_timestep)
		if self._rltimestep <= self._control_timestep:
			raise(ValueError("RL timestep must be > control timestep"))

		self._waitingforrlcommands = True
		self._lastaction = None
		self._lastactiont0 = 0.0
			
		self._commstoRL = AgentSide(BaseCommPoint.get_ip(),49054,verbose = debug) # wait til connecting to RL
	
		self._debug = debug
		
		print("Agent created.")



	def step(self, timestep) -> np.ndarray:
		"""A step for the agent: timestep includes the observation).
		It also manages episodic resets."""

		if self._debug:
			print("Step")

		if not self._waitingforrlcommands: # not waiting new commands from RL, just executing last action
		
			if (... agent physical time ... - self._lastactiont0 >= self._rltimestep): # last action finished
			
				observation = ...
				self._commstoRL.stepSendObs(observation, ... reward...) # RL was waiting for this; no reward is actually needed here
				self._waitingforrlcommands = True
				act = self._lastaction
				
			else: # still doing the action
				act = self._lastaction		

		else:  # waiting for new RL step() or reset()
			
			# read the last (pending) step()/reset() indicator and then proceed accordingly
			whattodo = self._commstoRL.readWhatToDo()
			if whattodo[0] == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:

				act = whattodo[1]
				self._lastaction = act
				lat = ... agent physical time ... - self._lastactiont0
				self._lastactiont0 = ... agent physical time ...
				self._waitingforrlcommands = False # from now on, we are waiting to execute the action
				self._commstoRL.stepSendLastActDur(lat)

			elif whattodo[0] == AgentSide.WhatToDo.RESET_SEND_OBS:

				# reset the agent and the scenario

				print("\tInitialized OK")

				observation = ...
				self._commstoRL.resetSendObs(observation)
				act = ... null act ...

			elif whattodo[0] == AgentSide.WhatToDo.FINISH:
			
				raise RuntimeError("Experiment finished")
				
			else:
				raise(ValueError("Unknown indicator data"))
				
		if self._debug:
			print("Step panda -- end")
		return act	 



if __name__ == '__main__':

	# initialize agent

	"""
	main loop
	"""

	agent = Agent(env,False)

	agent.spinloop()...


