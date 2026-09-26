import json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
d=json.load(open('results/numbers.json'))['eval']
INK,INK2,GRID,SURF='#0b0b0b','#52514e','#e4e3df','#fcfcfb'
GRAY,BLUE,ORANGE,AQUA='#8a8984','#2a78d6','#eb6834','#1baf7a'
plt.rcParams.update({'font.size':10,'axes.edgecolor':GRID,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'text.color':INK})
fig,(a,b)=plt.subplots(1,2,figsize=(12,4.6),dpi=100,gridspec_kw=dict(width_ratios=[1,1.35]),facecolor=SURF)
labels=['Exp1  real only','Exp2  geometric only','Exp3  Point-E only','Exp4  real + geometric','Exp5  real + Point-E','Exp6  real + geo + Point-E']
cols=[GRAY,BLUE,ORANGE,BLUE,ORANGE,AQUA]
acc=[d[str(i)]['acc']*100 for i in range(1,7)]
y=np.arange(6)[::-1]
a.barh(y,acc,color=cols,height=0.62)
for yi,v in zip(y,acc): a.text(v+1,yi,f'{v:.1f}',va='center',color=INK,fontsize=10)
a.axvline(acc[0],color=INK2,lw=1,ls=(0,(3,3)))
a.set_yticks(y,labels); a.set_xlim(0,85); a.set_xlabel('Test accuracy (%)')
a.set_title('A. Overall test accuracy (N = 910, 1 seed)',loc='left',fontsize=11)
for s in ['top','right']: a.spines[s].set_visible(False)
a.grid(axis='x',color=GRID,lw=0.8); a.set_axisbelow(True); a.set_facecolor(SURF)
# panel B: per-class delta vs baseline
bl=d['1']['percls']; cls=list(bl)
def delta(k): p=d[k]['percls']; return np.array([(p[c][0]/p[c][1]-bl[c][0]/bl[c][1])*100 for c in cls])
g,pe=delta('4'),delta('5'); order=np.argsort(g-pe)[::-1]
x=np.arange(40)
b.axhline(0,color=INK2,lw=1)
b.scatter(x,g[order],s=36,color=BLUE,edgecolor=SURF,lw=1.5,zorder=3,label=f'real + geometric (Exp4)  mean {g.mean():+.1f}')
b.scatter(x,pe[order],s=36,color=ORANGE,edgecolor=SURF,lw=1.5,zorder=3,label=f'real + Point-E (Exp5)  mean {pe.mean():+.1f}')
b.set_xticks(x,[cls[i].replace('_',' ') for i in order],rotation=90,fontsize=7.5)
b.set_ylabel('Δ accuracy vs real only (pts)'); b.set_xlim(-1,40); b.set_ylim(-60,100)
b.set_title('B. Per-class change vs baseline (20–25 test samples per class)',loc='left',fontsize=11)
for s in ['top','right']: b.spines[s].set_visible(False)
b.grid(axis='y',color=GRID,lw=0.8); b.set_axisbelow(True); b.set_facecolor(SURF)
b.legend(frameon=False,loc='upper left',fontsize=9)
fig.suptitle('ModelNet40, 25 training samples per class, PointNet',x=0.01,ha='left',fontsize=12,fontweight='bold')
fig.tight_layout(); fig.savefig('assets/hero.png',facecolor=SURF)
