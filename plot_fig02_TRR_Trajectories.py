# %% imports
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib import ticker

from envs.Env_TempRewardRisk import TemporalRewardRiskDilemma as TRR
from agents.detQ import detQ
from agents.QBatch import QBatch

import interact as ia
np.random.seed(42)

#%% envt
trr = TRR()
Ttrr = trr.TransitionTensor().astype(float)
Rtrr = trr.RewardTensor().astype(float)

#%% PARAMETERS
# org parameters
figfolder = "./"
datfolder = "/Users/wolf/0Data/memoTDdata/"

# init agent
alpha = 0.05
beta = 150
gamma = 0.9
sampsize = 10
agent = detQ(Ttrr, Rtrr, alpha, beta, gamma)
Xinit = np.array([[0.2, 0.8], [0.2, 0.8]])  # inital value estimates

#%% PLOT FUNCTIONS

def _plot_learner(batchsize, ax1, ax2, Nr):
       learner = QBatch(trr.state_action_space(), alpha, beta, gamma,
                     batchsize=batchsize, Xinit=Xinit)
       length = learner.batchsize + 100 * learner.batchsize

       # compute Xtrajs
       Xtrajs = ia.compute_Xtrajs(learner, trr, length, sampsize,
                            savepath=datfolder)
       detXtraj = ia.compute_detXtraj(agent, np.array([Xinit]), 100)

       # plot
       ia.visualize_BehaviorSpace(axes=ax1, PlotFor="N", DiffType="TDe", 
                            detAgents=agent, detXtraj=detXtraj, Xtrajs=Xtrajs,
                            pAs=np.linspace(0.01, 0.99, 8))
       
       # ax2 : timed trajectoies
       for Xtraj in Xtrajs:
              ax2.plot(np.array(Xtraj)[:, 0,0,0], label="X(s=deg., a=low)",
                       color='steelblue', alpha=0.2)
              ax2.plot(np.array(Xtraj)[:, 0,1,0], label="X(s=prosp., a=low)",
                       color='skyblue', alpha=0.2)
              ax2.plot(np.arange(0, length, learner.batchsize),
                      np.array(detXtraj)[:,0,0,0], "--", color="darkred")
              ax2.plot(np.arange(0, length, learner.batchsize),
                       np.array(detXtraj)[:,0,1,0], "--", color="red")

       ax1.set_title(''); ax1.set_xlabel('')
       ax1.set_ylabel("X(s=prosp., a=safe)")   
       ax2.set_ylabel("X(s,a)")   
       ax2.set_ylim(-0.02, 1.02)
       ax2.set_xlim(learner.batchsize-1, length)

       for ax in [ax1, ax2]:
              ax.set_yticks([0., 0.2, 0.4, 0.6, 0.8, 1.0])

       # Exponential on xticks
       g = lambda x,pos : "${}$".format(str('%1.e' % x)\
              .replace('e+0', ' \cdot 10^{')+'}')
       ax2.xaxis.set_major_formatter(plt.FuncFormatter(g))
       
       # Annotations
       lname = f"K={learner.batchsize}"
       ax1.annotate(lname, xy=(-0.5, 0.5),
             fontsize="x-large", rotation=90,
             xycoords="axes fraction", va="center", ha="center",
             bbox=dict(boxstyle='square', fc='white'))

       ax1.annotate(Nr, xy=(-0.5, 1.0),
             fontsize="x-large", 
             xycoords="axes fraction", va="top", ha="right")

#%% PLOT
fsf = 0.6
fig = plt.figure(figsize=(fsf*10, fsf*10))
gs = fig.add_gridspec(3, 5, wspace=2, hspace=.22, left=0.16, right=0.96,
                      top=0.95, bottom=0.08)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#      K = 50
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[0, 2:])
_plot_learner(50, ax1, ax2, "A")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#      K = 500
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
ax3 = fig.add_subplot(gs[1, :2])
ax4 = fig.add_subplot(gs[1, 2:])
_plot_learner(500, ax3, ax4, "B")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#      K = 5000
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
ax5 = fig.add_subplot(gs[2, :2])
ax6 = fig.add_subplot(gs[2, 2:])
_plot_learner(5000, ax5, ax6, "C")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Legend
legend_elements = [
       Line2D([0], [0], color='darkred', ls="--", label=r"X(s=deg., a=safe)"),
       Line2D([0], [0], color='steelblue', label=r"X(s=deg., a=safe)"),
       Line2D([0], [0], color='red', ls="--", label=r"X(s=prosp., a=safe)"),
       Line2D([0], [0], color='skyblue', label=r"X(s=prosp., a=safe)")]
legend1 = ax6.legend(handles=legend_elements, loc='best',
                     frameon=True)

ax1.annotate("Equation", xy=(0.55, 0.62), color="darkred",
             fontsize="large", rotation=40,
             xycoords="axes fraction", va="center", ha="center")
ax1.annotate("Algorithm", xy=(0.7, 0.15), color="deepskyblue",
             fontsize="large", rotation=0,
             xycoords="axes fraction", va="center", ha="center")

ax1.set_title("Policy space"); ax2.set_title("Trajectory")
ax5.set_xlabel("X(s=deg., a=safe)"); ax6.set_xlabel("Interaction time steps")

figname = figfolder+"TRR_Trajectories.png"
#plt.savefig(figname, dpi=300)

# %%
