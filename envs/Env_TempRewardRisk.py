"""
Single Agent Tipping element : temporal reward-risk dilemma
"""

import numpy as np


class TemporalRewardRiskDilemma(object):

    def __init__(self,
                 rh=1,
                 rl=0.5,
                 pc=0.2,
                 pr=0.1,
                 ):
        self.N = 1
        self.M = 2
        self.Z = 2

        self.rh = rh
        self.rl = rl
        self.pc = pc
        self.pr = pr
        
        # -- 
        self.T = self.TransitionTensor()
        self.R = self.RewardTensor()
        self.state = 1 # inital state


    def actionticks(self):
        return [0, 1], ["low", "high"]

    def stateticks(self):
        return [0, 1], ["deg.", "prosp."]
    
    def state_action_space(self):
        return np.zeros((self.Z, self.M))

    def TransitionTensor(self):
        """Get the Transition Tensor."""
        dim = np.concatenate(([self.Z],
                              [self.M for _ in range(self.N)],
                              [self.Z]))
        Tsas = np.ones(dim, dtype=object) * (-1)

        Tsas[1, 0, 1] = 1
        Tsas[1, 0, 0] = 0
        Tsas[1, 1, 1] = 1-self.pc
        Tsas[1, 1, 0] = self.pc
        Tsas[0, 0, 1] = self.pr
        Tsas[0, 0, 0] = 1-self.pr
        Tsas[0, 1, 0] = 1
        Tsas[0, 1, 1] = 0

        return Tsas

    def RewardTensor(self):
        """Get the Reward Tensor R[i,s,a1,...,aN,s']."""
        dim = np.concatenate(([self.N],
                              [self.Z],
                              [self.M for _ in range(self.N)],
                              [self.Z]))
        Risas = np.zeros(dim, dtype=object)

        for index, _ in np.ndenumerate(Risas):
            Risas[index] = self._reward(index[0], index[1], index[2:-1],
                                        index[-1])
        return Risas

    def _reward(self, i, s, jA, sprim):

        if s == 1 and sprim == 1:
            # while remaining at the prosperous state
            # standard public good
            reward = self.rh if jA[i] else self.rl
        else:
            reward = 0

        return reward
    
    def step(self, jA):
        """
        iterate env for one step
        
        jA : joint action as an iterable
        """
        tps = self.T[tuple([self.state]+list(jA))].astype(float)
        # final state: if tps = 0 everywhere we arrived at a final state
        next_state = np.random.choice(range(len(tps)), p=tps)
        obs = np.array([next_state]).repeat(self.N)
        
        rewards = self.R[tuple([slice(self.N),self.state]+list(jA)
                                +[next_state])]

        self.state = next_state
        
        return obs, rewards.astype(float)
    
    
    def observation(self):
        obs = np.array([self.state]).repeat(self.N)
        return obs
                

        