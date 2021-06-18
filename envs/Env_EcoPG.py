"""
2 Agent Eco Public Good for inequality studies
"""

import numpy as np

class EcologicalPublicGood(object):
    """classic PD"""

    def __init__(self,
                 NrOfAgents=2,
                 Rfactor=1.2,
                 Cost=5,
                 Impact=[-1, -1],
                 CRprob=[0.01,0.01],
                 DCprob=[0.2,0.2]
                 ):
        self.N = NrOfAgents
        self.M = 2
        self.Z = 2

        self.r = Rfactor
        self.c = Cost
        self.s = Impact

        self.qr = CRprob
        self.qc = DCprob

        # -- 
        self.T = self.TransitionTensor()
        self.R = self.RewardTensor()
        self.state = 1 # inital state

    def actionticks(self): 
        return [0, 1], ["coop.", "defect."]

    def stateticks(self):
        return [0, 1], ["deg.", "prosp."]

    def state_action_space(self):
        return np.zeros((self.Z, self.M))

    def TransitionTensor(self):
        """Get the Transition Tensor."""
        dim = np.concatenate(([self.Z],
                              [self.M for _ in range(self.N)],
                              [self.Z]))
        Tsas = np.ones(dim) * (-1)

        for index, _ in np.ndenumerate(Tsas):
            Tsas[index] = self._transition_probability(index[0],
                                                       index[1:-1],
                                                       index[-1])
        return Tsas

    def _transition_probability(self, s, jA, sprim):

        if s == 1:
            #q = ( jA[0]*self.Pdc0 + jA[1]*self.Pdc1 ) / self.N
            q = np.dot(self.qc, jA) / self.N
            p = min(q, 1)

            if sprim == 0:
            # collapse
                pass
            if sprim == 1:
            # remain at the prosp. state
                p = 1-p
            
        if s == 0:
            #q = ( (1-jA[0])*self.Pcr0 + (1-jA[1])*self.Pcr1 ) / self.N
            q = np.dot(self.qr, 1-np.array(jA)) / self.N
            p = min(q, 1)

            if sprim == 1:
            # recovery
                pass
            if sprim == 0:
                p = 1-p

        return p


    def RewardTensor(self):
        """Get the Reward Tensor R[i,s,a1,...,aN,s']."""
        dim = np.concatenate(([self.N],
                              [self.Z],
                              [self.M for _ in range(self.N)],
                              [self.Z]))
        Risas = np.zeros(dim)

        for index, _ in np.ndenumerate(Risas):
            Risas[index] = self._reward(index[0], index[1], index[2:-1],
                                        index[-1])
        return Risas


    def _reward(self, i, s, jA, sprim):


        if s == 1 and sprim == 1:
            # while remaining at the prosperous state
            # standard public good
            Nc = self.N - sum(jA)
            # Nd = sum(jA)

            Rd = self.r * Nc * self.c / self.N
            Rc = Rd - self.c

            reward = Rd if jA[i] else Rc
        else:
            reward = self.s[i]

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