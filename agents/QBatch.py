# -*- coding: utf-8 -*-
"""
Batch learner for temporal difference Q learning

Should converge to standard temporal difference Q learning for batchsize=1
"""

import numpy as np

class QBatch():
    '''
    Nice doc string.
    '''
    
    def __init__(self,
                 obs_action_space,
                 alpha,
                 beta,
                 gamma,
                 batchsize=1,
                 Xinit=None,
                 Qoa=None):

        self.alpha = alpha       # learning stepsize / rate
        self.beta = beta         # intensity of choice
        self.gamma = gamma       # discout factor
        
        # value table: gets updateded while acting with the same policy 
        self.valQoa = self._init_ObsActionValues(obs_action_space)

        # actor table: used for acting, gets updated in learning step
        self.actQoa = self._init_ObsActionValues(obs_action_space)
        
        if Xinit is not None and Xinit.shape == self.valQoa.shape:
            assert np.allclose(Xinit.sum(-1), 1), 'Xinit must be probabiliy'
            self.valQoa = (np.log(Xinit)/self.beta)\
                - np.mean(np.log(Xinit)/self.beta, axis=-1, keepdims=True)
            self.actQoa = (np.log(Xinit)/self.beta)\
                - np.mean(np.log(Xinit)/self.beta, axis=-1, keepdims=True)
        
        elif Qoa is not None and Qoa.shape == self.valQoa.shape:
            self.valQoa = Qoa.copy()
            self.actQoa = Qoa.copy()
        
        # transformation of actQoa
        self.X = self.current_policy()

        # ++++++++++++++++++++++++++    
        self.batchsize = batchsize
        self.last_Amax = None
        # GERNEAL AGENT
        self.current_act = None
        self.current_obs = None
        self.next_obs = None
        self.As =  np.arange(obs_action_space.shape[1])

        self.batch_step = 0
        self.total_step = 0
        self.episode = 0
        self.ret = 0
        
        # batch
        Z, M = self._init_ObsActionValues(obs_action_space).shape
        self.count_oa = 0 * self._init_ObsActionValues(obs_action_space)
        self.count_oao = np.zeros((Z, M, Z))
        self.reward_oa = 0 * self._init_ObsActionValues(obs_action_space)
        # self.nextQoa = 0 * self._init_ObsActionValues(obs_action_space)


    @staticmethod
    def _init_ObsActionValues(obs_action_space):
        """Initialize state/observation-action values to zero."""
        _OA = 0 * obs_action_space.copy()
        # _OA.set_fill_value(0)
        return _OA
    
    def current_policy(self):
        expQoa = np.exp(self.beta * self.actQoa) 
        assert not np.any(np.isinf(expQoa)), "behavior policy contains infs"
        return expQoa / expQoa.sum(axis=-1, keepdims=True)
    
    
    def interact(self, observation, reward):
        """
        ADD: if reward=None: do only the acting
        """
        
        if reward is not None:
            self.batchstore(reward, observation)
        
            if self.batch_step == self.batchsize:
                print("\r>> batchlearn at {}".format(self.total_step))
                self.batchlearn()        
        
        # returns with the next action
        action = self.act(observation)
        return action


    def act(self, observation):
        """
        Choose action for given observation with boltzmanm probabilites.

        Parameters
        ----------
        observation : int
            the agent's observation of the state of the environment
        """
        action = np.random.choice(self.As, p=self.X[observation])
        
        self.current_obs = observation
        self.current_act = action

        return action
    
    
    def batchstore(self, reward, next_obs):
        self.count_oa[self.current_obs, self.current_act] += 1
        self.count_oao[self.current_obs, self.current_act, next_obs] += 1
        self.reward_oa[self.current_obs, self.current_act] += reward     
        
        # updating the value table, estiamting the current state-action values
        self.valQoa[self.current_obs, self.current_act]\
            += self.alpha * ((1-self.gamma) * reward\
            + self.gamma * np.dot(self.X[next_obs], self.valQoa[next_obs])\
            - self.valQoa[self.current_obs, self.current_act])
            
        self.next_obs = next_obs  # just for consistency checking
        
        self.ret = (1-self.gamma)*reward + self.gamma * self.ret
        self.batch_step += 1
        self.total_step += 1
        
        
    def batchlearn(self, final_state=False):
        # what to do anyway with finals states in batch learning ???
        
        TDe = self.estimate_TDisa()     
        self.actQoa += self.alpha * TDe
        self.X = self.current_policy()
               
        #re-init
        self.count_oa.fill(0)
        self.count_oao.fill(0)
        self.reward_oa.fill(0)
        # self.nextQoa.fill(0)
        self.valQoa = self.actQoa.copy()
        self.batch_step = 0


    def estimate_X(self):
        divcount = self.count_oa.sum(-1, keepdims=True).copy()
        divcount[np.where(divcount == 0)] = 1
        return self.count_oa / divcount
       
    def estimate_T(self):
        divcount = self.count_oao.sum(-1, keepdims=True).copy()
        divcount[np.where(divcount == 0)] = 1
        return self.count_oao / divcount
    
    def estimate_Risa(self):
        divcount = self.count_oa.copy()
        divcount[np.where(self.count_oa == 0)] = 1
        return self.reward_oa / divcount
    
    def estimate_Qisa(self):
        return self.valQoa
    
    def estimate_MaxQisa(self):
        divcount = self.count_oa.copy()
        divcount[np.where(self.count_oa == 0)] = 1
        
        #return self.nextQoa / divcount   
        return np.dot(self.estimate_T(), self.valQoa.max(-1))
    
    def estimate_TDisa(self):
        divcount = self.count_oa.copy()
        divcount[np.where(self.count_oa == 0)] = 1
               
        TDe = ((1-self.gamma) * self.estimate_Risa()
               + self.gamma * self.estimate_MaxQisa()
               - self.actQoa)
        
        return (self.count_oa / divcount) * TDe