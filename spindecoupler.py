"""
SYSTEM FOR DECOUPLING RL SPIN LOOP FROM AGENT SPIN LOOP.

v1.0.0

(c) Juan-Antonio Fernández-Madrigal
Uncore Team, 2025
"""

from enum import Enum
from socketcomms.comms import ClientCommPoint,ServerCommPoint


#-------------------------------------------------------------------------------
#
#	Base Class: RLSide
#
#-------------------------------------------------------------------------------

class RLSide:
	"""
	Your RL class must create an instance of this in order to communicate
	with the agent that actually produces observations and executes actions.
	"""
	
	def __init__(self, port: int, verbose: bool = False):
		"""
		In PORT, the number of the port to use for comms., e.g., 49054.
		"""
		
		self._verbose = verbose
		self._rlcomm = ServerCommPoint(port) # socket not connected yet
											 # if socket in use, repeatedly wait
											 # until free
		if self._verbose:
			print("RL decoupler enabled. Waiting for agent connection...")
		res = self._rlcomm.begin(timeoutaccept = 60.0)
		if len(res) > 0:
			raise RuntimeError("No agent connection: " + res)
		if self._verbose:
			print("Agent connected to this RL")
			
			
	def __del__(self):
	
		res = self._rlcomm.end()
		if len(res) > 0:
			print("Error closing communications with the agent: " + res)
		if self._verbose:
			print("Communications closed in the RL side.")

		
	def resetGetObs(self, timeout: float = 10.0):
		"""
		Call this method at the start of your RL reset() to get from the agent
		the first observation after a reset, as a dictionary.
		TIMEOUT is the timeout in seconds used for communication operations that
		admit a timeout.
		It raises RuntimeError() if any error in communications.
		"""
		
		res = self._rlcomm.sendData(dict({"stepkind": "reset"}))
		if len(res) > 0:
			raise RuntimeError("Error sending what to do to the agent. " + res)	
			
		res,obs = self._rlcomm.readData(timeout)
		if len(res) > 0:
			raise RuntimeError("Error reading after-reset observation from the agent. " + res)
						
		return obs			

	
	def stepSendActGetObs(self, action,timeout:float = 10.0):
		"""
		Call this method at the start of your RL step() to send the action
		to the agent and then get the resulting observation, both as 
		dictionaries.
		Return the final duration of the action previous to this one, the
		observation obtained after executing this one for some time, and the 
		reward calculated by the agent for that.
		TIMEOUT is the timeout in seconds used for communication operations that
		admit a timeout.
		It raises RuntimeError() if any error in communications.
		""" 
		
		# send a STEP indicator to the agent interface, that should be blocked
		# in a readWhatToDo()
		res = self._rlcomm.sendData(dict({"stepkind": "step",
										  "action": action}))
		if len(res) > 0:
			raise RuntimeError("Error sending step action: " + res)

		res,lat = self._rlcomm.readData(timeout) 
		if len(res) > 0:
			raise RuntimeError("Error receiving last action duration: " + res)
			
		res,obsrew = self._rlcomm.readData(timeout) 
		if len(res) > 0:
			raise RuntimeError("Error receiving step observation: " + res)

		return lat["lat"],obsrew["obs"],obsrew["rew"]

				
	def stepExpFinished(self, timeout:float = 10.0):
		""" 
		Call this method at the end of your RL step() ONLY IF the learning
		has finished completely after that step.
		TIMEOUT is the timeout in seconds used for communication operations that
		admit a timeout.
		"""
		
		self._rlcomm.sendData(dict({"stepkind": "finish"}))
		 	
		 	
#-------------------------------------------------------------------------------
#
#	Base Class: AgentSide
#
#-------------------------------------------------------------------------------

class AgentSide:
	"""
	Your agent interface (e.g., with a robot or a simulation), in charge of
	getting observations and executing actions, must contain an instance of
	this class to communicate with the RL spin loop, which provides the actions.
	"""
	
	class WhatToDo(Enum):
		"""
		Things that RL is intending for the agent interface to do.
		"""
		
		REC_ACTION_SEND_OBS = 0	# receive action from RL, executes it and sends back resulting observation
		RESET_SEND_OBS = 1		# reset episode and then send observation back to RL
		FINISH = 2				# finish experiment (and comms)
	
	
	def __init__(self, ipbaselinespart:str, portbaselinespart:int, verbose:bool = False):
		"""
		IPBASELINESPART is the IPv4 of the baselines part of the system, e.g.,
		"BaseCommPoint.get_ip()".
		PORTBASELINESPART is the port, e.g., 49054.
		"""
		
		self._verbose = verbose
		self._rlcomm = ClientCommPoint(ipbaselinespart,portbaselinespart)
		
		if self._verbose:
			print("Agent decoupler enabled.")
		
		res = self._rlcomm.begin()
		if len(res) > 0:
			raise RuntimeError("Error starting connection with RL. " + res)
		
		if self._verbose:
			print("Agent decoupler connected to RL decoupler")
		
					
	def __del__(self):
	
		res = self._rlcomm.end()
		if len(res) > 0:
			raise RuntimeError("Error stopping connection with RL: " + res)
		if self._verbose:
			print("Connection with RL finished.")

 	
	def readWhatToDo(self, timeout: float = 10.0): 	
		""" 
		Call this method at each iteration of the agent spin loop if you need
		to receive new commands from the RL side. 
		It returns a tuple with an indicator plus possibly some data received
		(or None if none).
		Depending on the indicator, you must:
			REC_ACTION_SEND_OBS : take the action from the second element of the
								  tuple, start its execution, return the actual
								  duration of the action executed before that
								  (see stepSendLastActDur()), execute the new
								  action during some time, and then send back an 
								  observation read after that time (see 
								  stepSendObs()).
			RESET_SEND_OBS:	reset the episode for the agent, read an observation
							turn it into a dictionary and call the 
							resetSendObs() method with that.
			FINISH:	nothing besides the needed final arrangements of the agent
					to finish the experiment (the comms are closed automatically
					in this case).
		TIMEOUT is the timeout in seconds for the operation of starting comms.
		This method can raise RuntimeError if any error occurs in comms.
		"""
		
		# read the last (pending) step()/reset() msg and then proceed accordingly
		res,ind = self._rlcomm.readData() # read a dict: { 'stepkind' : 'reset', 'step' or 'finish' ,
										  #			       'action' : <action> if any}
		if len(res) > 0:
			raise RuntimeError("Error receiving what-to-do from RL: " + res)
				
		if ind["stepkind"] == "step":
			return (AgentSide.WhatToDo.REC_ACTION_SEND_OBS, ind["action"])
		elif ind["stepkind"] == "reset":
			return (AgentSide.WhatToDo.RESET_SEND_OBS, None)
		elif ind["stepkind"] == "finish":
			return (AgentSide.WhatToDo.FINISH, None)
		else:
			raise(ValueError("Unknown what-to-do indicator [" + ind["stepkind"] + "]"))

	def stepSendLastActDur(self, lat:float, timeout:float = 10.0):
		"""
		Call this method after receiving a REC_ACTION_SEND_OBS and starting the
		action, being LAT the actual time during which the action previous to 
		that one was executed before being substituted by the one in 
		REC_ACTION_SEND_OBS.
		TIMEOUT is the timeout in seconds for the operations with comms.
		This method can raise RuntimeError if any error occurs in comms.
		"""

		res = self._rlcomm.sendData(dict({"lat": lat}))
		if len(res) > 0:
			raise RuntimeError("Error sending lat to RL. " + res)	


	def stepSendObs(self, obs, rew:float = 0.0, timeout:float = 10.0):		
		"""
		Call this method if readWhatToDo() returned REC_ACTION_SEND_OBS, after
		executing the action, with the observation (a dictionary) to be sent 
		back to the RL and reward calculated for that action, if any (usually,
		reward is calculated at the RL side, but in some situations it could be
		interesting to calculate it at the agent side).
		TIMEOUT is the timeout in seconds for the operations with comms.
		This method can raise RuntimeError if any error occurs in comms.
		"""
		
		res = self._rlcomm.sendData(dict({"obs":obs,"rew":rew}))
		if len(res) > 0:
			raise RuntimeError("Error sending observation/reward to RL. " + res)	

					
	def resetSendObs(self, obs):
		"""
		Call this method if readWhatToDo() returned RESET_SEND_OBS to send back
		the first observation (OBS, a dictionary) got after an episode reset.
		This method can raise RuntimeError if any error occurs in comms.
		"""

		res = self._rlcomm.sendData(obs)
		if len(res) > 0:
			raise RuntimeError("Error sending observation to RL. " + res)	
 	
 			 	
