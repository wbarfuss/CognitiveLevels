"""
This is detQ.py, the determinsitic limit of Q learning.
"""
import itertools as it
import numpy as np
from .detMAE import detMAE

class detQ(detMAE):

    def __init__(self,
                 TranstionTensor,
                 RewardTensor,
                 alpha,
                 beta,
                 gamma,
                 roundingprec=9):

        detMAE.__init__(self,
                        TranstionTensor,
                        RewardTensor,
                        alpha,
                        beta,
                        gamma,
                        roundingprec)

    # =========================================================================
    #   Temporal difference error
    # =========================================================================

    def TDerror(self, X, norm=True):
        Risa = self.obtain_Risa(X)
        MaxQisa = self.obtain_MaxQisa(X)
        
        n = np.newaxis
        TDe = (1-self.gamma[:,n,n])*Risa + self.gamma[:,n,n]*MaxQisa\
            - 1/self.beta * np.ma.log(X)
            
        # # TEST : adaptive beta
        # TDe = (1-self.gamma[:,n,n])*Risa + self.gamma[:,n,n]*MaxQisa
        # TDe = TDe - TDe.mean(axis=2, keepdims=True)
        # TDe -= self.R.mean()*np.abs(TDe).mean() * np.ma.log(X)          
       
        if norm:
            TDe = TDe - TDe.mean(axis=2, keepdims=True)
        TDe = TDe.filled(0)
        return TDe

    # =========================================================================
    #   Behavior profile averages
    # =========================================================================

    def obtain_MaxQisa(self, X):
        """
        For q learning
        """
        Qisa = self.obtain_Qisa(X)

        i = 0  # agent i
        a = 1  # its action a
        s = 2  # the current state
        sprim = 3  # the next state
        j2k = list(range(4, 4+self.N-1))                      # other agents
        b2d = list(range(4+self.N-1, 4+self.N-1 + self.N))    # all actions
        e2f = list(range(3+2*self.N, 3+2*self.N + self.N-1))  # all other acts

        # get arguments ready for function call
        # # 1# other policy X
        sumsis = [[j2k[o], s, e2f[o]] for o in range(self.N-1)]  # sum ids
        otherX = list(it.chain(*zip((self.N-1)*[X], sumsis)))

        args = [self.Omega, [i]+j2k+[a]+b2d+e2f,
                Qisa.max(axis=-1), [i, sprim],
                self.T, [s]+b2d+[sprim]] + otherX + [[i, s, a]]

        return np.einsum(*args)



