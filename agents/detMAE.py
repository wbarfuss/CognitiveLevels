"""
This is detMAE.py, a class for obtaining various bits for the deterministic
limit of temporal difference reinforcement learning.
"""

import itertools as it
import numpy as np
import scipy.linalg as la

class detMAE(object):

    def __init__(self,
                 TranstionTensor,
                 RewardTensor,
                 alpha,
                 beta,
                 gamma,
                 roundingprec=9):
        """doc."""
        assert len(np.unique(TranstionTensor.shape[1:-1])) == 1,\
            "Transition tensor has different action sets sizes"
        assert len(np.unique(RewardTensor.shape[2:-1])) == 1,\
            "Reward tensor has different action sets sizes"

        self.R = RewardTensor
        self.T = TranstionTensor

        self.N = self.R.shape[0]  # the number of agents
        self.Z = self.T.shape[0]  # the number of states
        self.M = self.T.shape[1]  # the number of actions for each agent

        self.alpha = alpha  # the agent's learning rate
        self.beta = beta  # the agent's exploitation level


#        assert hasattr(gamma, "__iter__"), "gamma needs __iter__"
#        self.gamma = gamma  
        # the agent's discout factor
        if hasattr(gamma, '__iter__'):
            self.gamma = gamma
        else:
            self.gamma = np.repeat(gamma, self.N)
            


        self.Omega = self._obtain_OtherAgentsActionsSummationTensor()
        self.OmegaD = self._obtain_OtherAgentsDerivativeSummationTensor()

        self.roundingprec = roundingprec


    # =========================================================================
    #   Behavior profiles
    # =========================================================================

    def zeroIntelligence_behavior(self):
        """Behavior profile with equal probabilities."""
        return np.ones((self.N, self.Z, self.M)) / float(self.M)

    def random_behavior(self, method="norm"):
        """Behavior profile with random probabilities."""
        if method=="norm":
            X = np.random.rand(self.N, self.Z, self.M)
            X = X / X.sum(axis=2).repeat(self.M).reshape(self.N, self.Z, self.M)
        elif method == "diff":
            X = np.random.rand(self.N, self.Z, self.M-1)
            X = np.concatenate((np.zeros((self.N, self.Z, 1)),
                                np.sort(X, axis=-1),
                                np.ones((self.N, self.Z, 1))), axis=-1)
            X = X[:, :, 1:] - X[:, :, :-1]
        return X


    # =========================================================================
    #   Behavior profile averages
    # =========================================================================

    def obtain_Tss(self, X):
        """Effective Markov Chain transition tensor."""
        x4einsum = list(it.chain(*zip(X,
                                      [[0, i+1] for i in range(self.N)])))
        x4einsum.append(np.arange(self.N+1).tolist())  # output format
        Xs = np.einsum(*x4einsum)

        return np.einsum(self.T, np.arange(self.N+2).tolist(),  # trans. tensor
                         Xs, np.arange(self.N+1).tolist(),      # policies
                         [0, self.N+1])                         # output format

    def obtain_Tisas(self, X):
        """
        Effective environmental transition model for agent i
        """
        i = 0  # agent i
        a = 1  # its action a
        s = 2  # the current state
        sprim = 3  # the next state
        j2k = list(range(4, 4+self.N-1))  # other agents
        b2d = list(range(4+self.N-1, 4+self.N-1 + self.N))  # all actions
        e2f = list(range(3+2*self.N, 3+2*self.N + self.N-1))  # all other acts

        # get arguments ready for function call
        # # 1# other policy X
        sumsis = [[j2k[o], s, e2f[o]] for o in range(self.N-1)]  # sum inds
        otherX = list(it.chain(*zip((self.N-1)*[X], sumsis)))

        args = [self.Omega, [i]+j2k+[a]+b2d+e2f,
                self.T, [s]+b2d+[sprim]] + otherX + [[i, s, a, sprim]]

        return np.einsum(*args)

    def obtain_Risa(self, X):
        """
        Reward of agent i in state s under action a,
        given other agents behavior according to X.
        """
        i = 0  # agent i
        a = 1  # its action a
        s = 2  # the current state
        sprim = 3  # the next state
        j2k = list(range(4, 4+self.N-1))  # other agents
        b2d = list(range(4+self.N-1, 4+self.N-1 + self.N))  # all actions
        e2f = list(range(3+2*self.N, 3+2*self.N + self.N-1))  # all other acts

        # get arguments ready for function call
        # # 1# other policy X
        sumsis = [[j2k[o], s, e2f[o]] for o in range(self.N-1)]  # sum inds
        otherX = list(it.chain(*zip((self.N-1)*[X], sumsis)))

        args = [self.Omega, [i]+j2k+[a]+b2d+e2f,
                self.R, [i, s]+b2d+[sprim],
                self.T, [s]+b2d+[sprim]] + otherX + [[i, s, a]]

        return np.einsum(*args)

    def obtain_Ris(self, X):
        """
        Reward of agent i in state s, given all agents behavior according to X.
        """
        i = 0  # agent i
        s = 1  # state s
        sprim = 2  # next state s'
        b2d = list(range(3, 3+self.N))  # all actions

        x4einsum = list(it.chain(*zip(X,
                                      [[s, b2d[a]] for a in range(self.N)])))

        # einsum argument
        arg = [self.R, [i, s]+b2d+[sprim],
               self.T, [s]+b2d+[sprim]] +\
            x4einsum +\
            [[i, s]]

        return np.einsum(*arg)

    def obtain_Vis(self, X):
        """State Value of state s for agent i"""
        i = 0
        s = 1
        sp = 2

        Ris = self.obtain_Ris(X)
        Tss = self.obtain_Tss(X)
        # new
        n = np.newaxis
        Miss = np.eye(self.Z)[n,:,:] - self.gamma[:, n,n] * Tss[n,:,:]
        invMiss = self._vect_matrix_inverse(Miss)

        return (1-self.gamma[:, n]) * np.einsum(invMiss, [i, s, sp],
                                                Ris, [i, sp],
                                                [i, s])

        # old
        #invM = np.linalg.inv(np.eye(self.Z) - self.gamma*Tss)

        #return (1-self.gamma) * np.einsum(invM, [s, sp], Ris, [i, sp], [i, s])

    def obtain_Qisa(self, X):
        """Current state action value

        Q = (1-gamma)*Risa + gamma * VXT
        """
        Risa = self.obtain_Risa(X)
        Vis = self.obtain_Vis(X)
        
        # Tss = self.obtain_Tss(X)
        # nextVis = np.einsum(Tss, [1, 3], Vis, [0, 3], [0, 1])
        
        Tisas = self.obtain_Tisas(X)
        nextQisa = np.einsum(Tisas, [0,1,2,3], Vis, [0,3], [0,1,2])
        
        # nextV = nextV.repeat(self.M).reshape(self.N, self.Z, self.M)

        n = np.newaxis
        # return (1-self.gamma[:,n,n]) * Risa + self.gamma[:,n,n]*nextVis[:,:,n]
        return (1-self.gamma[:,n,n]) * Risa + self.gamma[:,n,n]*nextQisa

    # =========================================================================
    #   HELPER
    # =========================================================================
    @staticmethod
    def _vect_matrix_inverse(A):
        """
        Vectorized matrix inverse.

        first dimension of A is running index, the last two dimensions are
        the matrix dimensions
        """
        identity = np.identity(A.shape[2], dtype=A.dtype)
        return np.array([np.linalg.solve(x, identity) for x in A])

    def _obtain_OtherAgentsActionsSummationTensor(self):
        """For use in Einstein Summation Convention.

        To sum over the other agents and their respective actions.
        """
        dim = np.concatenate(([self.N],  # agent i
                              [self.N for _ in range(self.N-1)],  # other agnt
                              [self.M],  # agent a of agent i
                              [self.M for _ in range(self.N)],  # all acts
                              [self.M for _ in range(self.N-1)]))  # other a's
        Omega = np.zeros(dim.astype(int), int)

        for index, _ in np.ndenumerate(Omega):
            I = index[0]
            notI = index[1:self.N]
            A = index[self.N]
            allA = index[self.N+1:2*self.N+1]
            notA = index[2*self.N+1:]

            if len(np.unique(np.concatenate(([I], notI)))) is self.N:
                # all agents indicides are different

                if A == allA[I]:
                    # action of agent i equals some other action
                    cd = allA[:I] + allA[I+1:]  # other actionss
                    areequal = [cd[k] == notA[k] for k in range(self.N-1)]
                    if np.all(areequal):
                        Omega[index] = 1

        return Omega

    def obtain_statdist(self, X, tol=1e-10, adapt_tol=False,
                        verbose=False):
        """Obtain stationary distribution for X"""
        Tss = self.obtain_Tss(X)

        oeival, oeivec = la.eig(Tss, right=False, left=True)
        oeival = oeival.real
        oeivec = oeivec.real
        mask = abs(oeival - 1) < tol

        if adapt_tol and np.sum(mask) != 1:
            # not ONE eigenvector found AND tolerance adaptation true
            sign = 2*(np.sum(mask)<1) - 1 # 1 if sum(mask)<1, -1 if sum(mask)>1

            while np.sum(mask) != 1 and tol < 1.0 and tol > 0.0:
                tol = tol * 10**(int(sign))
                if verbose:
                    print("Adapting tolerance to {}".format(tol))
                mask = abs(oeival - 1) < tol

        meivec = oeivec[:, mask]
        dist = meivec / meivec.sum(axis=0, keepdims=True)
        dist[dist < tol] = 0
        dist = dist / dist.sum(axis=0, keepdims=True)

        if verbose:
            if len(dist[0]) > 1:
                print("more than 1 eigenvector found")
            if len(dist[0]) < 1:
                print("less than 1 eigenvector found")

        return dist

    def TDstep(self, X):
        TDe = self.TDerror(X)

        num = X * np.exp(self.alpha * self.beta * TDe)
        num = np.round(num, self.roundingprec)

        den = np.reshape(np.repeat(np.sum(num, axis=2), self.M),
                         (self.N, self.Z, self.M))

        return num / den


    def _obtain_OtherAgentsDerivativeSummationTensor(self):
        """For use in Einstein Summation Convention.

        To sum over the other agents and their respective actionsself.
        """
        dim = np.concatenate(([self.N],  # agent i
                              [self.N for _ in range(self.N-1)],  # other agnt
                              [self.M],  # agent a of agent i
                              [self.M for _ in range(self.N)],  # all acts
                              [self.M for _ in range(self.N-1)]))  # other a's
        Omega = np.zeros(dim.astype(int), int)

        for index, _ in np.ndenumerate(Omega):
            I = index[0]
            notI = index[1:self.N]
            A = index[self.N]
            allA = index[self.N+1:2*self.N+1]
            notA = index[2*self.N+1:]

            if len(np.unique(np.concatenate(([I], notI)))) is self.N:
                # all agents indicides are different

                #if A == allA[I]:
                #     # action of agent i equals some other action
                cd = allA[:I] + allA[I+1:]  # other actionss
                areequal = [cd[k] == notA[k] for k in range(self.N-1)]
                if np.all(areequal):
                    # Omega[index] = 1
                    Omega[index] = 2*(A==allA[I])-1

        return Omega

    def obtain_dTss(self, X):
        j = 0
        b = 1
        s = 2
        r = 3
        sprim = 4
        k2l = list(range(5, 5+self.N-1))  # other agents
        c2d = list(range(5+self.N-1, 5+self.N-1 + self.N))  # all actions
        e2f = list(range(4+2*self.N, 4+2*self.N + self.N-1))  # all ot. actions

        sumsis = [[k2l[o], s, e2f[o]] for o in range(self.N-1)]  # sum inds
        otherX = list(it.chain(*zip((self.N-1)*[X], sumsis)))

        args = [self.OmegaD, [j]+k2l+[b]+c2d+e2f,
                np.eye(self.Z), [s, r],
                self.T, [s]+c2d+[sprim]] + otherX + [[s, sprim, j, r, b]]

        dTss = np.einsum(*args)
        return dTss
    
    def obtain_dRis(self, X):
        i = 0  # agent i
        j = 1  # agent j
        a = 2  # agent i's action a
        b = 3  # agent j's action b
        s = 4  # state s
        r = 5  # state r
        sprim = 6  # next state

        k2l = list(range(7, 7+self.N-1))  # other agents but j
        c2d = list(range(7+self.N-1, 7+self.N-1 + self.N))  # all actions
        e2f = list(range(6+2*self.N, 6+2*self.N + self.N-1))  # all actions but j's

        # get arguments ready for function call
        # # 1# other policy X
        sumsis = [[k2l[o], r, e2f[o]] for o in range(self.N-1)]  # sum inds
        otherX = list(it.chain(*zip((self.N-1)*[X], sumsis)))

        args = [self.OmegaD, [j]+k2l+[b]+c2d+e2f,
                np.eye(self.Z), [s, r],
                self.R, [i, s]+c2d+[sprim],
                self.T, [s]+c2d+[sprim]] + otherX + [[i, s, j, r, b]]

        return np.einsum(*args)

    def obtain_dVis(self, X):
        Tss = self.obtain_Tss(X)
        Ris = self.obtain_Ris(X)
        dTss = self.obtain_dTss(X)
        dRis = self.obtain_dRis(X)

        M = np.eye(self.Z) - self.gamma*Tss
        dM = - self.gamma * dTss
        invM = np.linalg.inv(M)
        dinvM = np.einsum(-1 * invM, [0, 1],
                          np.einsum(dM, [0, 1, 2, 3, 4],
                                    invM, [1, 5],
                                    [0, 5, 2, 3, 4]), [1, 2, 3, 4, 5],
                         [0, 2, 3, 4, 5]
                         )

        i, s, sp = 0, 1, 2
        j, r, b = 3, 4, 5

        dinvMR = np.einsum(dinvM, [s, sp, j, r, b], Ris, [i, sp],
                           [i, s, j, r, b])
        invMdR = np.einsum(invM, [s, sp], dRis, [i, sp, j, r, b],
                           [i, s, j, r, b])

        return (1-self.gamma) * (dinvMR + invMdR)