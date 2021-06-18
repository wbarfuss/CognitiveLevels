"""
The 2-state Matching Pennies according to HennesEtAl2010
"""

import numpy as np

class TwoStateMatchingPennies(object):

    def __init__(self):
        self.N = 2
        self.M = 2
        self.Z = 2
        
        # -- 
        self.T = self.TransitionTensor()
        self.R = self.RewardTensor()
        self.state = 1 # inital state


    def actionticks(self): 
        return [0, 1], ["A", "B"]
    
    def stateticks(self):
        return [0, 1], ["one", "two"]
    
    def state_action_space(self):
        return np.zeros((self.Z, self.M))

    def TransitionTensor(self):
        """Get the Transition Tensor."""
        Tsas = np.ones((2, 2, 2, 2)) * (-1)

        T1 = np.array([[1.0, 1.0],
                       [0.0, 0.0]])
        T2 = np.array([[0.0, 0.0],
                       [1.0, 1.0]])

        Tsas[0, :, :, 1] = T1
        Tsas[0, :, :, 0] = 1-T1
        Tsas[1, :, :, 0] = T2
        Tsas[1, :, :, 1] = 1-T2
        
        return Tsas

    def RewardTensor(self):
        """Get the Reward Tensor R[i,s,a1,...,aN,s']."""

        R = np.zeros((2, 2, 2, 2, 2))

        R[0, 0, :, :, 0] = [[1 , 0 ],
                            [0 , 1 ]]
        R[1, 0, :, :, 0] = [[0 , 1 ],
                            [1 , 0 ]]

        R[:, 0, :, :, 1] = R[:, 0, :, :, 0]

        R[0, 1, :, :, 1] = [[0 , 1 ],
                            [1 , 0 ]]
        R[1, 1, :, :, 1] = [[1 , 0 ],
                            [0 , 1 ]]

        R[:, 1, :, :, 0] = R[:, 1, :, :, 1]

        return R


    def step(self, jA):
        """
        iterate env for one step
        
        jA : joint action as an iterable
        """
        tps = self.T[tuple([self.state]+list(jA))].astype(float)
        # final state: if tps = 0 everywhere we arrived at a final state
        next_state = np.random.choice(range(len(tps)), p=tps)
        
        rewards = self.R[tuple([slice(self.N),self.state]+list(jA)
                               +[next_state])]
            
        self.state = next_state
        obs = self.observation()        
        
        return obs, rewards.astype(float)

    
    def observation(self):
        obs = np.array([self.state]).repeat(self.N)
        return obs
                