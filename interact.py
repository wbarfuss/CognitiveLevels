# -*- coding: utf-8 -*-
"""
Container for helper functions etc.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from copy import deepcopy

from QuiverPlot import plot_quiver  #, plot_trajectory


def create_name(env, sampsize, learner, detAgents=None):
    envsamps = f"{env.__class__.__name__}_{sampsize}Xtrajs_"
    learName = f"{learner.__class__.__name__}_"
    if hasattr(learner, "batchsize"):
        learName += f"bsize{learner.batchsize}_"
    params = f"alpha{learner.alpha}_beta{learner.beta}_gamma{learner.gamma}"
    
    if detAgents is not None:
        detName = f"{detAgents.__class__.__name__}_"
        return envsamps + learName + detName + params
    
    else: 
        return envsamps + learName + params
        

def interface(agents, envrionment, steps):
    """
    Parameters
    ----------
    agents : iterable of agents
    envrionment : envrionment
    steps : int (Number of interaction steps)
    """
    def allagentinteract(obs, rews):
        actions = []
        for ag in range(envrionment.N):
            actions.append(agents[ag].interact(obs[ag], rews[ag]))
        return np.array(actions)

    # data
    Xtraj = [[agent.X for agent in agents]]

    # INTERACTION
    actions = allagentinteract(envrionment.observation(),
                               np.array([None]).repeat(envrionment.N))
    print(actions)
    for i in range(steps):
        obs, rews = envrionment.step(actions)
        actions = allagentinteract(obs, rews)

        # data
        Xtraj.append([agent.X for agent in agents])

    # data
    return Xtraj
     
# TODO: make storage smaller for batch learners
def compute_Xtrajs(learners, env, length, sampsize, savepath=None):
    """
    Compute and save /load X trajectories of a batch learning algorithm.

    Parameters
    ----------
    learners : learning algo
        Batch reinforcement learning algorithm
    env : env
        Environment.
    length : int
        Number of total interaction steps with the envrionment.
    sampsize : int
        Sample size of identical repetitions.
    savepath : str
        Location of data storage path.

    Returns
    -------
    Xtrajs : list
        List of behavior (X) trajectories. Each list item contains the X-
        trajectory of a sample run.
    """
    # check whether learners is single instance or iterable
    if hasattr(learners, '__iter__'):
        assert len(learners) == env.N
    else:
        learners = [deepcopy(learners) for _ in range(env.N)]
    learner = learners[0]


    if savepath is not None:
        fname = savepath + create_name(env, sampsize, learner) + ".npz"
        save = True
    else:
        save = False
    
    try:
        Xtrajs = np.load(fname)['dat']
        print("Loading")
    except:
        print("Computing: Total number of timesteps: ", length)
        Xtrajs = []
    
        start_time = time.time()
        for _ in range(sampsize):
            #listoflearners = [deepcopy(learner) for _ in range(env.N)]
            listoflearners = deepcopy(learners)
            Xtrajs.append(interface(listoflearners, env, length))
            print()
        if save: np.savez_compressed(fname, dat=np.array(Xtrajs))
    
        now = time.time()
        print(f"--- {now - start_time} seconds ---")
        if sampsize > 0:
            print(f" == {(now - start_time)/sampsize} seconds/sample == ")

    return Xtrajs




def compute_detXtraj(agents, Xinit, learningsteps):
    
    X = Xinit.copy()
    detXtraj = [X]
    for t in range(learningsteps):
        X = agents.TDstep(X)
        detXtraj.append(X)
    
    return detXtraj



#%%

def visualize_BehaviorSpace(
        detAgents, Xinits=[], detXtraj=None, Xtrajs=None, PlotFor="Z",
        DiffType="TDe", pAs = np.linspace(0.01, 0.99, 8), cmap='viridis',
        axes=None,
        detkwargs=dict(color='darkred', lw=3.0, ms=5.5, alpha=0.9),
        algkwargs=dict(color="deepskyblue", ms=0.1, alpha=0.4, zorder=1)):
    
    # quiver
    axes = plot_quiver(detAgents, pAs=pAs, plot_for=PlotFor, sf=0.5,
                       difftype=DiffType, cmap=cmap, axes=axes)
    
    # trajectories
    if PlotFor == "Z":
        ixx = lambda j: (slice(-1), 0, j, 0)
        ixy = lambda j: (slice(-1), 1, j, 0)
    elif PlotFor == "N":
        ixx = lambda j: (slice(-1), j, 0, 0)
        ixy = lambda j: (slice(-1), j, 1, 0)
        
    if detXtraj is not None:
        for j, ax in enumerate(axes):
            ax.plot(np.array(detXtraj)[ixx(j)], np.array(detXtraj)[ixy(j)],
                    "--", **detkwargs);
        
    if Xtrajs is not None:
        for Xtraj in Xtrajs:
            for j, ax in enumerate(axes):
                ax.plot(np.array(Xtraj)[ixx(j)], np.array(Xtraj)[ixy(j)],
                        "-", **algkwargs)
                
    # trajectories
    # for X1 in Xinits:
    #     rt, fpr, lastX = plot_trajectory(detAgents, X1, axes=axes, ls="--",
    #                                      Tmax=5000, **detkwargs,
    #                                      plot_for=PlotFor)

    return axes


#
#   CONVERGENCE
#
    
# helper function
def allagentinteract(learners, env, obs, rews):
    actions = []
    for ag in range(env.N):
        actions.append(learners[ag].interact(obs[ag], rews[ag]))
    return np.array(actions)


def get_convergence_terms(
    detAs, algAs, env, Xinit, testtimes, sampsize, savepath=None):
    
    if savepath is not None:
        fn = lambda x : "_" + x.__class__.__name__
        fid = "CONV" + fn(env) + fn(detAs) + fn(algAs[0]) + f"_samps{sampsize}"
        fname = savepath + fid + ".npz"
        save = True
    else:
        save = False

    try:
        dat = np.load(fname)
        convterms = dict(zip((k for k in dat), (dat[k] for k in dat)))
        print("Loading")
    except:
        print("Computing")
        convterms = compute_convergence_terms(detAs, algAs, env, Xinit,
                                              testtimes, sampsize)
        if save:
            np.savez_compressed(fname, **convterms)
        
    return convterms

def compute_convergence_terms(detAs, algAs, env, Xinit, testtimes, sampsize):
    assert len(algAs) == env.N

    def _diff(esti, tru):
        return np.abs( (esti - tru) / np.abs(tru).mean() ).mean()
        # return np.abs( (esti - tru) ).mean()
        # return np.abs( (esti - tru) / tru.mean() ).mean()

    # terms
    DactX_ss = []
    DXisa_ss = []
    DRisa_ss = []
    DTisas_ss = []
    DQisa_ss = []
    DMaxQisa_ss = []
    DTDisa_ss = []
        
    for samp in range(sampsize):
        print("[compute_convergence_terms] Sample: ", samp)
        j = 0 # testtimes iterator
        learners = deepcopy(algAs)
        X = Xinit.copy()
        
        # containers
        DactX_s = []
        DRisa_s = []
        DXisa_s = []
        DTisas_s = []
        DQisa_s = []
        DMaxQisa_s = []
        DTDisa_s = []
            
        # INTERACTION
        actions = allagentinteract(learners, env, env.observation(),
                                   np.array([None]).repeat(env.N))
        for i in range(testtimes.max()):
            obs, rews = env.step(actions)
            actions = allagentinteract(learners, env, obs, rews)
            
            if i % algAs[0].batchsize == 0:  # learnig adaption takes place
                if i>0:
                    X = detAs.TDstep(X); print("det learn at ", i);
                # updateing determinisitc results
                Tss = detAs.obtain_Tss(X)
                Risa = detAs.obtain_Risa(X)
                Qisa = detAs.obtain_Qisa(X)
                MaxQisa = detAs.obtain_MaxQisa(X)
                TDisa = detAs.TDerror(X, norm=False)
                
            if i == testtimes[j]:
                eXisa = np.array([l.estimate_X() for l in learners])
                eActX = np.array([l.X for l in learners])
                eTss = np.mean([np.einsum(l.estimate_X(), [0,1],
                                          l.estimate_T(), [0,1,2], [0, 2])
                                for l in learners], axis=0)
                eRisa = np.array([l.estimate_Risa() for l in learners])
                eQisa = np.array([l.estimate_Qisa() for l in learners])
                eMaxQisa = np.array([l.estimate_MaxQisa() for l in learners])
                eTDisa = np.array([l.estimate_TDisa() for l in learners])
    
                DactX_s.append(_diff(eActX, X))
                DXisa_s.append(_diff(eXisa, X))
                DTisas_s.append(_diff(eTss, Tss))
                DRisa_s.append(_diff(eRisa, Risa))
                DQisa_s.append(_diff(eQisa, Qisa))
                DMaxQisa_s.append(_diff(eMaxQisa, MaxQisa))
                DTDisa_s.append(_diff(eTDisa, TDisa))
                j += 1
    
        DactX_ss.append(DactX_s)
        DXisa_ss.append(DXisa_s) 
        DTisas_ss.append(DTisas_s)
        DRisa_ss.append(DRisa_s)
        DQisa_ss.append(DQisa_s)
        DMaxQisa_ss.append(DMaxQisa_s)
        DTDisa_ss.append(DTDisa_s)
        
    return dict(DXisa=np.array(DXisa_ss),
                DactX=np.array(DactX_ss),
                DRisa=np.array(DRisa_ss),
                DTisas=np.array(DTisas_ss),
                DQisa=np.array(DQisa_ss),
                DMaxQisa=np.array(DMaxQisa_ss),
                DTDisa=np.array(DTDisa_ss),
                testtimes=testtimes)

def plot_convterms(convterms, keys, colors, labels=None, title="", ax=None):
      
    if ax is None:
        fig, ax = plt.subplots()

    if labels is None:
        labels = keys

    xvs = convterms['testtimes'][:-1]
    for i, k in enumerate(keys):
    
        mean = convterms[k].mean(0)
        std = convterms[k].std(0)
        ax.plot(xvs, mean, color=colors[i])
        ax.fill_between(xvs, mean-0.5*std, mean+0.5*std, alpha=0.4,
                         color=colors[i], label=labels[i])
    
    ax.set_title(title)
    plt.legend()