# -*- coding: utf-8 -*-
"""Test convergence for individual terms of detRL with alog"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import interact as ia
from copy import deepcopy

from agents.detQ import detQ
from agents.QBatch import QBatch

np.random.seed(42)
figfolder = "./"
datfolder = "/Users/wolf/0Data/memoTDdata/"

bsize = 7500
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#%%   ENVS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
from envs.Env_2StateMatchingPennies import TwoStateMatchingPennies as MP
mp = MP()
Tmp = mp.TransitionTensor().astype(float)
Rmp = mp.RewardTensor().astype(float)
Xi = np.array([[0.2, 0.8], [0.2, 0.8]])
Xinit_mp = np.array([Xi, Xi])
alpha_mpFP = 0.05
beta_mpFP = 25
gamma_mpFP = 0.75
detA_mpFP = detQ(Tmp, Rmp, alpha_mpFP, beta_mpFP, gamma_mpFP)
algA_mpFP = QBatch(mp.state_action_space(), alpha_mpFP, beta_mpFP, gamma_mpFP,
                   batchsize=bsize, Xinit=Xinit_mp[0])
algAs_mpFP = [deepcopy(algA_mpFP), deepcopy(algA_mpFP)]

alpha_mpPO = 0.02
beta_mpPO = 45
gamma_mpPO = 0.55
detA_mpPO = detQ(Tmp, Rmp, alpha_mpPO, beta_mpPO, gamma_mpPO)
algA_mpPO = QBatch(mp.state_action_space(), alpha_mpPO, beta_mpPO, gamma_mpPO,
                   batchsize=bsize, Xinit=Xinit_mp[0])
algAs_mpPO = [deepcopy(algA_mpPO), deepcopy(algA_mpPO)]


from envs.Env_TempRewardRisk import TemporalRewardRiskDilemma as TRR
trr = TRR()
Ttrr = trr.TransitionTensor().astype(float)
Rtrr = trr.RewardTensor().astype(float)
Xinit_trr = np.array([[[0.2, 0.8], [0.2, 0.8]]]) # initial behavour
alpha_trr = 0.05
beta_trr = 150
gamma_trr = 0.9
detA_trr = detQ(Ttrr, Rtrr, alpha_trr, beta_trr, gamma_trr)
algA_trr = QBatch(mp.state_action_space(), alpha_trr, beta_trr, gamma_trr,
                   batchsize=bsize, Xinit=Xinit_trr[0])
algAs_trr = [deepcopy(algA_trr)]


from envs.Env_EcoPG import EcologicalPublicGood as EPG
epg = EPG(NrOfAgents=2, Rfactor=1.2, Cost=5, Impact=[-5, -5], 
          CRprob=[0.1,0.1], DCprob=[0.2,0.2])
Tepg = epg.TransitionTensor().astype(float)
Repg = epg.RewardTensor().astype(float)
Xi1 = np.array([[0.1, 0.9], [0.8, 0.2]])  # inital value estimates
Xi2 = np.array([[0.4, 0.6], [0.1, 0.9]])  # inital value estimates
Xinit_epg = np.array([Xi1, Xi2])
alpha_epg = 0.04
beta_epg = 25
gamma_epg = 0.9
detA_epg = detQ(Tepg, Repg, alpha_epg, beta_epg, gamma_epg)
l1 = QBatch(epg.state_action_space(), alpha_epg, beta_epg, gamma_epg,
            batchsize=bsize, Xinit=Xi1)
l2 = QBatch(epg.state_action_space(), alpha_epg, beta_epg, gamma_epg,
            batchsize=bsize, Xinit=Xi2)
algAs_epg = [l1, l2]

#%% compute
testtimes = np.arange(0, bsize, 10)

convterms_trr = ia.get_convergence_terms(
    detA_trr, algAs_trr, trr, Xinit_trr, testtimes, 100, datfolder)

convterms_epg = ia.get_convergence_terms(
    detA_epg, algAs_epg, epg, Xinit_epg, testtimes, 100, datfolder)

convterms_mpFP = ia.get_convergence_terms(
    detA_mpFP, algAs_mpFP, mp, Xinit_mp, testtimes, 100, datfolder)

convterms_mpPO = ia.get_convergence_terms(
    detA_mpPO, algAs_mpPO, mp, Xinit_mp, testtimes, 101, datfolder)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Plot
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -         
fsf = 0.7
fig, axes = plt.subplots(4, 1, figsize=(fsf*8, fsf*8))
plt.subplots_adjust(hspace=0.05, right=0.72, top=0.98, bottom=0.08, left=0.12)

for ax in axes:
    ax.plot([50, 50], [0, 1], "k-", alpha=0.5)
    ax.plot([500, 500], [0, 1], "k-", alpha=0.5)
for ax in [axes[0], axes[1], axes[3]]:
    ax.plot([5000, 5000], [0, 1], "k-", alpha=0.5)
axes[2].plot([2500, 2500], [0, 1], "k-", alpha=0.5)

convterms = [convterms_trr, convterms_epg, convterms_mpFP, convterms_mpPO]
names = ['Single-agent risk-reward', 'Two-agent risk-reward',
         'Zero-sum fixed point', 'Zero-sum periodic orbit']
for i, ax in enumerate(axes):
    ia.plot_convterms(convterms[i],
                    keys=['DRisa', 'DTisas', 'DQisa', 'DMaxQisa', 'DTDisa'],
                    colors=['red', 'blue', 'orange', 'skyblue', 'green'],
                    labels=[f'$\Delta R^i(s,a)$', f"$\Delta T^i(s,a,s')$", 
                    f'$\Delta Q^i(s,a)$', f'$\Delta maxQ^i(s,a)$',
                    f'$\Delta \delta^i(s,a)$'],
                    ax=ax)
    
    ax.set_ylim(-0.00,0.59); ax.set_xlim(-50, 5500+50)
    #ax.set_yticks([0, 0.1, ])
    ax.annotate(names[i], xy=(0.5, 0.9), xycoords="axes fraction",
                va="top", ha="center", fontsize='large',
                bbox=dict(boxstyle='round', fc='white'))
for ax in axes[:-1]:   
    ax.set_xticklabels([])

axes[-1].set_xlabel(f"Batch size $K$")
h, l = axes[-1].get_legend_handles_labels()
axes[-1].get_legend().remove()
axes[0].legend(bbox_to_anchor=(1.0, 1.07), loc='upper left')

axes[2].annotate('Relative error', xy=(-0.15, 1.1), rotation=90,
                 xycoords="axes fraction",
                 va="center", ha="center" )

figname = figfolder+"ALL_Convergence.png"
# plt.savefig(figname, dpi=300)



# %%
