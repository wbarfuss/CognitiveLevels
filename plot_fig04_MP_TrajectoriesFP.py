# %% imports
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib import ticker

from envs.Env_2StateMatchingPennies import TwoStateMatchingPennies as MP
from agents.detQ import detQ
from agents.QBatch import QBatch

import interact as ia
np.random.seed(42)

#%% env
mp = MP()
Tmp = mp.TransitionTensor().astype(float)
Rmp = mp.RewardTensor().astype(float)

#%% PARAMETERS
# org parameters
figfolder = "./"
datfolder = "/Users/wolf/0Data/memoTDdata/"

# init agent
alpha = 0.05
beta = 25
gamma = 0.75
sampsize = 10
agents = detQ(Tmp, Rmp, alpha, beta, gamma)
Xi = np.array([[0.2, 0.8], [0.2, 0.8]])  # inital value estimates
Xinit = np.array([Xi, Xi])


#%% PLOT FUNCTIONS

def _plot_learner(batchsize, ax1, ax2, ax3, ax4, Nr):
        
    if batchsize >= 1:
        learner = QBatch(mp.state_action_space(), alpha, beta, gamma,
                        batchsize=batchsize, Xinit=Xi)
        length = learner.batchsize + 100 * learner.batchsize

        # compute Xtrajs
        Xtrajs = ia.compute_Xtrajs(learner, mp, length, sampsize,
                                savepath=datfolder)
    else:
        Xtrajs = []

    detXtraj = ia.compute_detXtraj(agents, Xinit, 100)

    # plot
    ia.visualize_BehaviorSpace(axes=[ax1, ax2], PlotFor="Z", DiffType="TDe",
        detAgents=agents, detXtraj=detXtraj, Xtrajs=Xtrajs,
        pAs=np.linspace(0.01, 0.99, 8))

    # ax3+ax4 : timed trajectoies
    for ax, i in [(ax3, 0), (ax4, 1)]:
        for Xtraj in Xtrajs:
                ax.plot(np.array(Xtraj)[:, i,0,0], label="X(s=deg., a=low)",
                        color='steelblue', alpha=0.2)
                ax.plot(np.array(Xtraj)[:, i,1,0], label="X(s=prosp., a=low)",
                        color='skyblue', alpha=0.2)
                ax.plot(np.arange(0, length, learner.batchsize),
                        np.array(detXtraj)[:,i,0,0], "--", color="darkred")
                ax.plot(np.arange(0, length, learner.batchsize),
                        np.array(detXtraj)[:,i,1,0], "--", color="red")

    for ax in [ax1, ax2]:
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels([0, "", 1])
    for ax in [ax3, ax4]:
        ax.set_yticks([0.2, 0.8]); ax.set_ylim(0.1, 0.9)
    ax1.set_title(''); ax2.set_title('')
    ax1.set_ylabel(f"$X^2(s, a=left)$", labelpad=-5); ax2.set_ylabel("")   
    ax1.set_xlabel(""); ax2.set_xlabel(""); ax2.set_yticklabels([])
    if batchsize >= 1:
        ax3.set_xlim(learner.batchsize-1, length)
        ax4.set_xlim(learner.batchsize-1, length)
    ax3.set_xticklabels([])
    ax3.set_ylabel(f"$X^1$", labelpad=-2); ax4.set_ylabel(f"$X^2$", labelpad=-2)

    # Exponential on xticks
    g = lambda x,pos : "${}$".format(str('%1.e' % x)\
          .replace('e+0', ' \cdot 10^{')+'}')
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(g))

    # Annotations            
    if batchsize >= 1:
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
fig = plt.figure(figsize=(fsf*10, fsf*9))

to=0.95; bo=0.08; le=0.16; mi=0.65; ms=0.06; ri=0.99; hs=0.22
gs1 = fig.add_gridspec(3, 2, wspace=0.05, hspace=hs, left=le, right=mi-ms,
                      top=to, bottom=bo)
gs2 = fig.add_gridspec(3, 1, wspace=0.1, hspace=hs, left=mi+ms, right=ri,
                      top=to, bottom=bo)

ax1 = fig.add_subplot(gs1[0, 0])
ax2 = fig.add_subplot(gs1[0, 1])
gs21 = gs2[0].subgridspec(2, 1, hspace=0.05)
ax3 = fig.add_subplot(gs21[0])
ax4 = fig.add_subplot(gs21[1])
_plot_learner(50, ax1, ax2, ax3, ax4, "A")

ax5 = fig.add_subplot(gs1[1, 0])
ax6 = fig.add_subplot(gs1[1, 1])
gs22 = gs2[1].subgridspec(2, 1, hspace=0.05)
ax7 = fig.add_subplot(gs22[0])
ax8 = fig.add_subplot(gs22[1])
_plot_learner(500, ax5, ax6, ax7, ax8, "B")

ax9 = fig.add_subplot(gs1[2, 0])
ax10 = fig.add_subplot(gs1[2, 1])
gs23 = gs2[2].subgridspec(2, 1, hspace=0.05)
ax11 = fig.add_subplot(gs23[0])
ax12 = fig.add_subplot(gs23[1])
_plot_learner(5000, ax9, ax10, ax11, ax12, "C")

ax9.set_xlabel(f"$X^1(s=1, a=left)$", labelpad=4)
ax10.set_xlabel(f"$X^1(s=2, a=left)$", labelpad=4)
ax12.set_xlabel("Interaction time steps", labelpad=4)

ax1.annotate("Policy space", xy=(1.01, 1.1), fontsize="large",
             xycoords="axes fraction", va="center", ha="center")
ax3.annotate("Trajectory", xy=(0.5, 1.2), fontsize="large",
             xycoords="axes fraction", va="center", ha="center")

figname = figfolder+"MP_TrajectoriesFP.png"
# plt.savefig(figname, dpi=300)

#%% LEGEND extra

fig = plt.figure(figsize=(fsf*11.5, fsf*0.8))
ax = fig.add_subplot(frameon=False, xticks=[], yticks=[])

legend_elements = [
       Line2D([0], [0], color='darkred', ls="--", label=r"$X^i(s=1, a=left)$"),
       Line2D([0], [0], color='steelblue', label=r"$X^i(s=1, a=left)$"),
       Line2D([0], [0], color='darkred', ls="--", label=r"$X^i(s=2, a=left)$"),
       Line2D([0], [0], color='steelblue', label=r"$X^i(s=2, a=left)$")]
legend1 = ax.legend(handles=legend_elements,  loc='center right', ncol=4,
                    bbox_to_anchor=(1.3, 0.5), frameon=True)

plt.tight_layout()
# plt.savefig(figfolder+"MP_Trajectories_Legend.png", dpi=300)

#%% LEGEND extra 2
fsf = 0.8
fig = plt.figure(figsize=(fsf*6.2, fsf*0.8))
ax = fig.add_subplot(frameon=False, xticks=[], yticks=[])

legend_elements = [
       Line2D([0], [0], color='darkred', ls="--", label=r"$X^i(s=1, a=left)$"),
       Line2D([0], [0], color='steelblue', label=r"$X^i(s=1, a=left)$"),
       Line2D([0], [0], color='red', ls="--", label=r"$X^i(s=2, a=left)$"),
       Line2D([0], [0], color='skyblue', label=r"$X^i(s=2, a=left)$")]
legend1 = ax.legend(handles=legend_elements,  loc='center right', ncol=2,
                    bbox_to_anchor=(1.3, 0.5), frameon=True)

plt.tight_layout()
# plt.savefig(figfolder+"MP_Trajectories_Legend.png", dpi=300)

# %%
