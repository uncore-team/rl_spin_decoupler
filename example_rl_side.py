"""
EXAMPLE (STRUCTURE) OF THE RL SIDE THAT IS DECOUPLED
To be launched first, then the agent part.
"""


# imports here

from spindecoupler import RLSide


class RLEnv:
	"""
	RL environment.
	"""

	def __init__(self,debug = False):
		"""
		Constructor. Define observations and actions.
		"""
		
		self._commstoagent = RLSide(49054,verbose = debug) # blocks until receiving connection from panda

		self._debug = debug		
		if self._debug:
			print("RL side started ok.")
			

	def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
	
		obs = self._commstoagent.resetGetObs()

		if self._debug:
			print("\tobs read: {}".format(obs))
			
		return obs

        
	def step(self, action):
	
		lat,obs,rew = self._commstoagent.stepSendActGetObs(action) 

		if self._debug:
			print("\taction executed: {}; last action dur: {}; agent reward: {}".format(action,lat,rew))

		terminated = ...
		reward = ...
		
		if ..finished... :
			print("\Experiment finished")
			self._commstoagent.stepExpFinished()
		
		return obs,reward,terminated,truncated,info ...
	
	
	
if __name__ == '__main__':

	print("Learning...")

	env = RLEnv(True)
	model = ...
	
	# to use tensorboard afterwards to see logs:
	checkpoint_callback=CheckpointCallback(save_freq=10000,save_path="logs",name_prefix="example")
	
	numstepsexp = 1_000 
	model.learn(total_timesteps=numstepsexp, 
				callback=checkpoint_callback)
	model.save(...)

	print("Press Enter to end...")
	input()

