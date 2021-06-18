"""Plot Quiver for DetRL"""
import itertools as it
import numpy as np
import matplotlib.pyplot as plt

# from DeterministicCollectiveAdaptation import \
#     DeterministicCollectiveAdaptation as DCA


def prob_grid():
    """Returns the action probability grid for one axis as a list."""
    return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 


def gridded_policies(pAs, N, Z):
    """Transforms action probabilites (pAs) into policy grid.

    Z : number of states
    """
    Xs = []  # one X has signature Xisa
    for ps in it.product(pAs, repeat=N*Z):
        X = np.zeros((N, Z, 2))
        for i in range(N):
            for s in range(Z):
                X[i, s, 0] = ps[i*Z+s]
                X[i, s, 1] = 1 - ps[i*Z+s]
        Xs.append(X)

    return Xs


def TDerror_difference(Xs, agents):
    """Compute TDerros for Xs and ltype ("A", "Q", "S")."""
    # TDes = []
    # # TDe_func = {"A": "current_acl_TDerror",
    # #             "Q": "current_q_TDerror",
    # #             "S": "current_sarsa_TDerror"}
    
    # for X in Xs:
    #     TDes.append(agents.TDerror(X))
    # return TDes
    return [agents.TDerror(X) for X in Xs]


def DeltaX_difference(Xs, agents):
    """Compute X(t-1)-X(t) for Xs and ltype ("A", "Q", "S")."""
    # DeltaXs = []
    # # step_func = {"A": "actorcritic_step",
    # #              "Q": "q_step",
    # #              "S": "sarsa_step"}

    # for X in Xs:
    #     # dca.X = X
    #     # getattr(dca, step_func[ltype])()
    #     Xnew = agents.TDstep(X)
    #     # Xtp1 = dca.X
    #     DeltaXs.append(Xnew - X)

    # return DeltaXs

    return [agents.TDstep(X) - X for X in Xs]


def quiverformat_difference(diffs, Xs, pAs, Z):
    """Translate differences (diffs) into quiverformat for
    Xs: behaviors
    pAs : action probs (grid of quiver plot)
    Z: nr of states
    """
    n = len(pAs)

    Xagentsdata = np.zeros((Z, n, n, n**(2*(Z-1))))
    Yagentsdata = np.zeros((Z, n, n, n**(2*(Z-1))))
    # for each state(Z): n in X, n in Y direction + values in remaining states

    for i, pX in enumerate(pAs):
        for j, pY in enumerate(pAs):
            k = np.zeros(Z, dtype=int)
            for b, Beh in enumerate(Xs):
                for s in range(Z):
                    if Beh[0, s, 0] == pX and Beh[1, s, 0] == pY:
                        Xagentsdata[s, j, i, k[s]] = diffs[b][0, s, 0]
                        Yagentsdata[s, j, i, k[s]] = diffs[b][1, s, 0]
                        k[s] += 1

    return Xagentsdata, Yagentsdata

def quiverformat_difference_states(diffs, Xs, pAs, N):
    """Translate differences (diffs) into quiverformat for
    Xs: behaviors
    pAs : action probs (grid of quiver plot)
    N: nr of agents
    """
    n = len(pAs)

    Xagentsdata = np.zeros((N, n, n, n**(2*(N-1))))
    Yagentsdata = np.zeros((N, n, n, n**(2*(N-1))))
    # for each state(Z): n in X, n in Y direction + values in remaining states

    for ix, pX in enumerate(pAs):
        for iy, pY in enumerate(pAs):
            k = np.zeros(N, dtype=int)
            for b, Beh in enumerate(Xs):
                for i in range(N):
                    if Beh[i, 0, 0] == pX and Beh[i, 1, 0] == pY:
                        Xagentsdata[i, iy, ix, k[i]] = diffs[b][i, 0, 0]
                        Yagentsdata[i, iy, ix, k[i]] = diffs[b][i, 1, 0]
                        k[i] += 1

    return Xagentsdata, Yagentsdata

def plot_quiver(agents, #T, R, alpha, beta, gamma,
                difftype="TDe",
                # learningtype="Q",
                pAs=prob_grid(),
                axes=None,
                cmap='viridis',
                plot_for='Z', **kwargs):
    # assert R.shape[0] == 2, "Number of agents needs to be 2"
    # assert R.shape[2] == 2, "Nummber of actions needs to be 2"
    # assert agents.R.shape[0] == 2, "Number of agents needs to be 2"
    assert agents.R.shape[2] == 2, "Nummber of actions needs to be 2"
    
    N = agents.R.shape[0]  # Number of agents
    Z = agents.R.shape[1]  # Number of states

    Xs = gridded_policies(pAs, N, Z)

    # dca=DCA(T, R, alpha, beta, gamma, oneminusgammacorr=True)
    diffs = TDerror_difference(Xs, agents) if difftype == "TDe"\
        else DeltaX_difference(Xs, agents)

    X, Y = np.meshgrid(pAs, pAs)
    if plot_for == 'Z':
        dX, dY = quiverformat_difference(diffs, Xs, pAs, Z)
        axes = _do_the_plot(dX, dY, X, Y, pAs, Z, axes, cmap=cmap, **kwargs)
        titl = 'State s='
        ylab = u"$X^1_{s0}$"
        xlab = u"$X^0_{s0}$"
    elif plot_for == 'N':
        dX, dY = quiverformat_difference_states(diffs, Xs, pAs, N)
        axes = _do_the_plot(dX, dY, X, Y, pAs, N, axes, cmap=cmap, **kwargs)
        titl = 'Agent i='
        ylab = u"$X^i_{10}$"
        xlab = u"$X^i_{00}$"
    else:
        raise ValueError('`plot_for` must be either `Z` or `N`')
       

    
    # labeling    
    for i, ax in enumerate(axes):
        ax.set_title(titl + str(i))
        ax.set_xlabel(xlab)
    axes[0].set_ylabel(ylab)
    
    
    return axes

def scale(x, a):
    return np.sign(x) * a * np.abs(x)

def scale2(x, y, a):
    l = (x**2 + y**2)**(1/2)
    k = l**a
    return k/l * x, k/l * y

def _do_the_plot(dX, dY, X, Y, pAs, Z, axes, sf=1, cmap="viridis", **kwargs):
    # figure and axes
    if axes is None:
        fig, axes = plt.subplots(1, Z, figsize=(3*Z, 3))
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    
    assert len(axes) == Z, "Number of axes must equal number of plots"

    # quiver keywords
    qkwargs = {"units":"xy", "angles": "xy", "scale":None, "scale_units": "xy",
               "headwidth": 4.5, "pivot": "tail", **kwargs}

    # individuals
    Nr = len(pAs)**(2*(Z-1))
    for i in range(Nr):
        for s in range(Z):
            DX = dX[s, :, :, i]
            DY = dY[s, :, :, i]
            LEN = (DX**2 + DY**2)**0.5
            axes[s].quiver(X, Y, *scale2(DX, DY, sf), LEN,
                           cmap=cmap, alpha=1/Nr, **qkwargs)           
 
    # averages 
    # lengths = (dX.mean(axis=-1)**2 + dY.mean(axis=-1)**2)**0.5
    for s in range(Z):
        DX = dX[s].mean(axis=-1)
        DY = dY[s].mean(axis=-1)
        LEN = (DX**2 + DY**2)**0.5
        axes[s].quiver(X, Y, *scale2(DX, DY, sf), LEN,
                       cmap=cmap, **qkwargs)

    return axes


def plot_trajectory(agents,
                    Xinit, axes,
                    Tmax=10000, fpepsilon=0.000001,        
                    plot_for="Z", **kwargs):
    
    N = agents.R.shape[0]  # Number of agents
    Z = agents.R.shape[1]  # Number of states
    
    (AXps, AYps), rt, fpr, lastX = trajectory(agents, Xinit,
                                              Tmax=Tmax, plot_for=plot_for,
                                              fpepsilon=fpepsilon)

    K = Z if plot_for == 'Z' else N
    for k in range(K):
        axes[k].plot(AXps[:, k], AYps[:, k], **kwargs)
        axes[k].plot([AXps[0, k]], [AYps[0, k]], 'x', **kwargs)
        if fpr: # fixpointreached
            axes[k].plot([AXps[-1, k]], [AYps[-1, k]], 'o', **kwargs)

    return rt, fpr, lastX


def trajectory(agents, Xinit, plot_for='Z',
               Tmax=10000, fpepsilon=0.000001):
    #dca=DCA(T, R, alpha, beta, gamma, oneminusgammacorr=True)
    #dca.X = Xinit
    X = Xinit
    # step_func = {"A": "actorcritic_step",
    #              "Q": "q_step",
    #              "S": "sarsa_step"}
    # TDe_func = {"A": "current_acl_TDerror",
    #             "Q": "current_q_TDerror",
    #             "S": "current_sarsa_TDerror"}

    agentXprobs, agentYprobs = [], []
    fixpreached = False
    t = 0
    rewardtraj = []
    while not fixpreached and t < Tmax:
        if plot_for == "Z":
            agentXprobs.append(X[0, :, 0])
            agentYprobs.append(X[1, :, 0])
        elif plot_for == "N":
            agentXprobs.append(X[:, 0, 0])
            agentYprobs.append(X[:, 1, 0])

        Ris = agents.obtain_Ris(X)
        δ = agents.obtain_statdist(X)   
        rewardtraj.append(np.sum(δ.T * Ris, axis=1))

        #getattr(dca, step_func[learningtype])()
        Xnew = agents.TDstep(X)
        #dca.TDstep(TDe_func[learningtype], norm=True)
        fixpreached = np.linalg.norm(Xnew - X) < fpepsilon
        X = Xnew
        t += 1

    return (np.array(agentXprobs), np.array(agentYprobs)), rewardtraj,\
        fixpreached, X
