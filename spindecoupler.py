"""
RL-BASELINES-SOCKET SYSTEM FOR DECOUPLING RL FROM AGENT.

v1.0.0

(c) Juan-Antonio Fernández-Madrigal
Uncore Team, 2025
"""

from enum import Enum
from socketcomms.comms import ClientCommPoint,ServerCommPoint


#-------------------------------------------------------------------------------
#
#	Base Class: BaselinesSide
#
#-------------------------------------------------------------------------------

class BaselinesSide:
	"""
	Your gym.Env class must create an instance of this class to communicate
	with the agent interface that actually produces observations and executes 
	actions.
	"""
	
	def __init__(self, port: int):
		"""
		In PORT, the number of the port to use for comms., e.g., 49054.
		"""
		self._rlcomm = ServerCommPoint(port)
		
	def resetGetObs(self, timeout: float = 10.0):
		"""
		Call this method at the start of your gym.Env reset() to get the 
		observation as a dictionary.
		TIMEOUT is the timeout in seconds used for communication operations that
		admit a timeout.
		It raises RuntimeError() if any error in communications.
		"""
		res = self._rlcomm.begin(timeout)
		if len(res) > 0:
			raise RuntimeError("Begin connection error. " + res)
			
		res = self._rlcomm.sendData(dict({"stepkind": "reset"}))
		if len(res) > 0:
			raise RuntimeError("Sending error. " + res)	
			
		res,obs = self._rlcomm.readData(timeout)
		if len(res) > 0:
			raise RuntimeError("Reading error. " + res)
						
		res = self._rlcomm.end()
		if len(res) > 0:
			raise RuntimeError("End connection error in reset. " + res)

		return obs			
	
	def stepSendActGetObs(self, action,timeout:float = 10.0):
		"""
		Call this method at the start of your gym.Env step() to send the action
		to the agent and then get the resulting observation, both as 
		dictionaries.
		TIMEOUT is the timeout in seconds used for communication operations that
		admit a timeout.
		It raises RuntimeError() if any error in communications.
		""" 
		res = self._rlcomm.begin(timeout)
		if len(res) > 0:
			raise RuntimeError("Begin connection error. " + res)
			
		# send a STEP indicator to the agent interface, that should be blocked
		# in a readWhatToDo()
		res = self._rlcomm.sendData(dict({"stepkind": "step"}))
		if len(res) > 0:
			raise RuntimeError("Sending what-to-do error. " + res)
			
		res = self._rlcomm.sendData(action)
		if len(res) > 0:
			raise RuntimeError("Sending action error. " + res)

		res,obs = self._rlcomm.readData(timeout) 
		if len(res) > 0:
			raise RuntimeError("Reading observation error. " + res)

		res = self._rlcomm.end()
		if len(res) > 0:
			raise RuntimeError("Ending connection error in step. " + res)
			
		return obs
				
	def stepExpFinished(self, timeout:float = 10.0):
		""" 
		Call this method at the end of your gym.Env step() ONLY IF the learning
		has finished completely after that step.
		TIMEOUT is the timeout in seconds used for communication operations that
		admit a timeout.
		"""
		self._rlcomm.begin(timeout)
		self._rlcomm.sendData(dict({"stepkind": "finish"}))
		self._rlcomm.end()
		 	
		 	
#-------------------------------------------------------------------------------
#
#	Base Class: AgentSide
#
#-------------------------------------------------------------------------------

class AgentSide:
	"""
	Your agent interface (e.g., with a robot or a simulation), in charge of
	getting observations and executing actions, must contain an instance of
	this class to communicate with Baselines, which provides the actions.
	"""
	
	class WhatToDo(Enum):
		"""
		Things that baselines is intending for the agent interface to do.
		"""
		REC_ACTION_SEND_OBS = 0	# receive action from baselines, executes it and sends back resulting observation
		RESET_SEND_OBS = 1		# reset episode and then send observation to baselines
		FINISH = 2				# finish experiment (and comms)
	
	
	def __init__(self, ipbaselinespart:str, portbaselinespart:int):
		"""
		IPBASELINESPART is the IPv4 of the baselines part of the system, e.g.,
		"BaseCommPoint.get_ip()".
		PORTBASELINESPART is the port, e.g., 49054.
		"""
		self._rlcomm = ClientCommPoint(ipbaselinespart,portbaselinespart)
		self._commstarted = False
			
	def startComms(self):
		"""
		Call this when you need to communicate with the baselines part. 
		"""
		if self._commstarted:
			raise RuntimeError("Cannot start twice the comms to baselines")	
			
		res = self._rlcomm.begin()
		if len(res) > 0:
			raise RuntimeError("Error starting connection with baselines. " + res)
		self._commstarted = True
		
	def stopComms(self):
		"""
		Call this after you need to signal end of comms to the baselines part.
		"""
		if not self._commstarted:
			raise RuntimeError("Cannot stop comms to baselines if not started before")	
			
		res = self._rlcomm.end()
		if len(res) > 0:
			raise RuntimeError("Error stopping connection with baselines. " + res)
		self._commstarted = False
 	
	def readWhatToDo(self, timeout: float = 10.0): 	
		""" 
		Call this method repeatedly while comms are started to receive commands
		from the baselines and execute them. Commands can be received 
		asynchronously, so this method blocks the caller (until timeout anyway).
		It must be called at the start of each of those iterations.
		It returns an indicator of what to do. Depending on that, you must:
			REC_ACTION_SEND_OBS : read an action (see stepRecAct()), execute it
								  and then send back an observation (see 
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
		if not self._commstarted:
			raise RuntimeError("Cannot read what to do with comms shut down")
		
		# read the last (pending) step()/reset() msg and then proceed accordingly
		res,ind = self._rlcomm.readData() # read a dict: { 'stepkind' : 'reset', 'step' or 'finish' }
		if len(res) > 0:
			raise RuntimeError("Error receiving what-to-do indicator from baselines. " + res)
				
		if ind["stepkind"] == "step":
			return AgentSide.WhatToDo.REC_ACTION_SEND_OBS
		elif ind["stepkind"] == "reset":
			return AgentSide.WhatToDo.RESET_SEND_OBS
		elif ind["stepkind"] == "finish":
			res = self._rlcomm.end()
			if len(res) > 0:
				raise RuntimeError("Error closing comms with baselines. " + res)
			return AgentSide.WhatToDo.FINISH
		else:
			raise(ValueError("Unknown what-to-do indicator [" + ind["stepkind"] + "]"))

	def stepRecAct(self, timeout:float = 10.0):
		"""
		Call this method if readWhatToDo() returned REC_ACTION_SEND_OBS. It
		returns the action (a dictionary) received from Baselines.
		TIMEOUT is the timeout in seconds for the operations with comms.
		This method can raise RuntimeError if any error occurs in comms.
		"""
		if not self._commstarted:
			raise RuntimeError("Cannot send/receive data with comms shut down")

		res,actrec = self._rlcomm.readData(timeout) 
		if len(res) > 0:
			raise RuntimeError("Error reading action from baselines. " + res)	
		return actrec

	def stepSendObs(self, obs, timeout:float = 10.0):		
		"""
		Call this method if readWhatToDo() returned REC_ACTION_SEND_OBS, after
		executing the action (after calling recAct()), and with the observation
		(a dictionary) to be sent back to baselines.
		TIMEOUT is the timeout in seconds for the operations with comms.
		This method can raise RuntimeError if any error occurs in comms.
		"""
		if not self._commstarted:
			raise RuntimeError("Cannot send/receive data with comms shut down")
		
		res = self._rlcomm.sendData(obs)
		if len(res) > 0:
			raise RuntimeError("Error sending observation to baselines. " + res)	
					
	def resetSendObs(self, obs):
		"""
		Call this method if readWhatToDo() returned RESET_SEND_OBS to send back
		the first observation (OBS, a dictionary) got after an episode reset.
		This method can raise RuntimeError if any error occurs in comms.
		"""
		if not self._commstarted:
			raise RuntimeError("Cannot send/receive data with comms shut down")

		res = self._rlcomm.sendData(obs)
		if len(res) > 0:
			raise RuntimeError("Error sending observation to baselines. " + res)	
 	
 			 	
