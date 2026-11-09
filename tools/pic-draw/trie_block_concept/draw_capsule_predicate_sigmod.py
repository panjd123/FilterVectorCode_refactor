"""Compact paper figure: oblique layers and corresponding planar detail.

The palette and geometry follow the supplied reference: black elliptical
vertices, fine gray edges, two slanted planes and a cyan illustrative path.
Capsule membership is encoded by faint colored regions. Gray dashed vertical
lines denote correspondence, never a cross-layer search transition.

All four views use the same 13 vectors in six groups of 1--3 vectors. The
overview omits vector names; the detail uses lowercase v_i beside each vertex.
Graph geometry and selected edges are schematic, not an actual index dump.
"""

from pathlib import Path
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon


plt.rcParams.update({
    "font.family":"Liberation Serif", "mathtext.fontset":"stix",
    "font.size":8, "svg.fonttype":"none", "pdf.fonttype":42,
    "savefig.facecolor":"white",
})
OUT=Path(__file__).resolve().parent
INK="#20252C"
GRAY="#8D969F"
EDGE="#6D7680"
CYAN="#20BDE4"
COL={1:"#5D9CB5",2:"#C9A064",3:"#A28CBC"}
FILL={1:"#EFF7FA",2:"#FCF7ED",3:"#F6F2FA"}
LABEL={1:"{1}",2:"{1,2}",3:"{1,2,4}",4:"{1,3}",5:"{1,3,5}",6:"{1,6}"}
IDS={1:[1,2,3],2:[4,5],3:[6,7,8],4:[9,10],5:[11],6:[12,13]}
MEMBERS={1:[1,6],2:[2,3],3:[4,5]}
OWNER={g:c for c,groups in MEMBERS.items() for g in groups}
BASE_LINKS=[(1,5),(3,9),(2,12),(4,7),(10,11)]
CAP_LINKS=[(1,5),(3,9)]
CAP_LOCAL={1:[(1,2),(2,3),(12,13),(1,12),(2,13),(3,12)],
           2:[(4,5),(5,6),(7,8),(4,7),(5,8),(6,7)],
           3:[(9,10),(9,11),(10,11)]}
PATH=[(1,5),(5,8)]

fig,ax=plt.subplots(figsize=(7.16,3.938))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax.set(xlim=(0,16),ylim=(0,8.8),aspect="equal")
ax.axis("off")


def txt(x,y,s,size=8,color=INK,ha="center",**kwargs):
    ax.text(x,y,s,fontsize=size,color=color,ha=ha,va="center",zorder=30,**kwargs)


def edge(p,q,color=EDGE,lw=.55,arrow=False,rad=0,z=5,shrink=1.8):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>" if arrow else "-",
        mutation_scale=6.4,connectionstyle=f"arc3,rad={rad}",
        linewidth=lw,color=color,shrinkA=shrink,shrinkB=shrink,zorder=z))


def dot(p,small=False,label=None,below=False,fill="white"):
    w,h=(.14,.067) if small else (.13,.085)
    ax.add_patch(Ellipse(p,w,h,facecolor=INK,edgecolor=INK,linewidth=.25,zorder=15))
    if label is not None:
        dy=-.18 if below else .16
        txt(p[0]+.055,p[1]+dy,rf"$v_{{{label}}}$",size=7.1,
            bbox={"facecolor":fill,"edgecolor":"none","pad":.12})


def region(cx,cy,w,h,c,rounding=.2,dashed=False):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,
        boxstyle=f"round,pad=.008,rounding_size={rounding}",
        facecolor=FILL[c],edgecolor=COL[c],linewidth=.62,
        linestyle=(0,(2,2)) if dashed else "solid",zorder=2))


def layer_heading(x,y,n,title,right_note=None):
    txt(x,y,f"Level {n}",size=8.5,ha="left")
    txt(x+1.27,y,title,size=9.0,ha="left",weight="bold")
    if right_note:
        txt(15.60,y,right_note,size=6.7,color=GRAY,ha="right")


txt(.35,8.42,"(a) Layered index",size=10,ha="left",weight="bold")
txt(5.75,8.42,"(b) Corresponding planar views",size=10,ha="left",weight="bold")

# Left: a small oblique rendering with the reference's black ellipse vertices.
PY={0:1.55,1:5.02}
def ob(u,v,level):
    return (.35+3.70*u+1.02*v,PY[level]+1.58*v)

for level in (0,1):
    ax.add_patch(Polygon([ob(0,0,level),ob(1,0,level),ob(1,1,level),ob(0,1,level)],
        closed=True,facecolor="white",edgecolor="#4A5159",linewidth=.78,zorder=1))

GP={1:(.50,.77),2:(.16,.56),3:(.16,.16),4:(.84,.56),5:(.84,.16),6:(.50,.20)}
OFF={1:[(0,0)],2:[(-.052,-.015),(.052,.025)],
     3:[(-.060,-.025),(0,.065),(.060,-.025)]}
OP={level:{} for level in (0,1)}
for level in (0,1):
    for g,(u,v) in GP.items():
        for i,(du,dv) in zip(IDS[g],OFF[len(IDS[g])]):
            OP[level][i]=ob(u+du,v+dv,level)

for c,u0,u1 in [(2,.025,.295),(1,.355,.645),(3,.705,.975)]:
    ax.add_patch(Polygon([ob(u0,.065,1),ob(u1,.065,1),ob(u1,.93,1),ob(u0,.93,1)],
        closed=True,facecolor=FILL[c],edgecolor=COL[c],linewidth=.45,zorder=2))
    txt(*ob((u0+u1)/2,.99,1),rf"$C_{c}$",size=7.5,color=COL[c],
        bbox={"facecolor":"white","edgecolor":"none","pad":.3})

for g,(u,v) in GP.items():
    hull=[ob(u+.105*math.cos(t),v+.13*math.sin(t),0)
          for t in [j*math.pi/32 for j in range(64)]]
    ax.add_patch(Polygon(hull,closed=True,facecolor=FILL[OWNER[g]],
        edgecolor=COL[OWNER[g]],linewidth=.42,zorder=2))

for i in (8,13,11):
    a,b=OP[1][i],OP[0][i]
    ax.plot([a[0],b[0]],[a[1],b[1]],color="#B9BFC5",lw=.55,
            ls=(0,(3,3)),zorder=3)

for g,ids in IDS.items():
    for i,j in zip(ids[:-1],ids[1:]):
        edge(OP[0][i],OP[0][j],lw=.48,shrink=.5)
    if len(ids)==3:
        edge(OP[0][ids[0]],OP[0][ids[-1]],lw=.48,shrink=.5)
for i,j in BASE_LINKS:
    edge(OP[0][i],OP[0][j],arrow=True,lw=.55)
for pairs in CAP_LOCAL.values():
    for i,j in pairs:
        edge(OP[1][i],OP[1][j],lw=.48,shrink=.5)
for i,j in CAP_LINKS:
    edge(OP[1][i],OP[1][j],arrow=True,lw=.55)
for i,j in PATH:
    edge(OP[1][i],OP[1][j],color=CYAN,lw=1.25,arrow=True,z=8)
for level in (0,1):
    for p in OP[level].values():
        dot(p,small=True)

for level,y,title in [(1,7.02,"Capsule layer"),(0,3.57,"Predicate tree layer")]:
    txt(.43,y,f"Level {level}",size=8.6,ha="left")
    txt(1.70,y,title,size=9,ha="left",weight="bold",
        bbox={"facecolor":"white","edgecolor":"none","pad":1})
txt(2.7,.98,"Unlabeled nodes show the same vectors.",size=6.8,color=GRAY)

# Quiet view-expansion arrows in the central gutter.
for y in (5.70,2.25):
    edge((5.06,y),(5.57,y),color="#A7AFB8",lw=.6,arrow=True)

# Right: planar capsule graphs. Original group membership remains visible in
# the formulas and vector identities, while the background identifies a capsule.
layer_heading(5.76,7.80,1,"Capsule layer","Groups merged; connectivity rebuilt")
CC={1:(10.70,6.62),2:(7.68,5.65),3:(13.72,5.65)}
CP={}
for c,(cx,cy) in CC.items():
    region(cx,cy,2.62,1.55,c,rounding=.30)
    a,b=MEMBERS[c]
    prefix={1:"{1}",2:"{1,2}",3:"{1,3}"}[c]
    txt(cx,cy+.55,rf"$C_{c}$ : {prefix}",size=8.6,color=COL[c])
    txt(cx,cy+.29,rf"$G_{a}\cup G_{b}$",size=7.0,color=GRAY)
    ids=[i for g in MEMBERS[c] for i in IDS[g]]
    layout=([(-.76,-.09),(0,-.09),(.76,-.09),(-.42,-.48),(.42,-.48)]
            if len(ids)==5 else [(-.53,-.12),(.53,-.12),(0,-.50)])
    CP.update({i:(cx+dx,cy+dy) for i,(dx,dy) in zip(ids,layout)})
    for i,j in CAP_LOCAL[c]:
        edge(CP[i],CP[j],lw=.55,shrink=.5)

for i,j in CAP_LINKS:
    edge(CP[i],CP[j],arrow=True,lw=.72,rad=.015)
for i,j in PATH:
    edge(CP[i],CP[j],color=CYAN,lw=1.35,arrow=True,z=8,rad=.015)
for c,groups in MEMBERS.items():
    for g in groups:
        for i in IDS[g]:
            dot(CP[i],label=i,fill=FILL[c])

# Base groups use dashed contours to distinguish exact predicate groups from
# solid capsule boundaries. Every vector identity is carried through exactly.
layer_heading(5.76,4.35,0,"Predicate tree layer","One label set per group")
GC={1:(10.70,3.35),2:(7.68,2.84),3:(7.68,1.14),
    4:(13.72,2.84),5:(13.72,1.14),6:(10.70,1.38)}
BP={}
for g,(cx,cy) in GC.items():
    c=OWNER[g]
    region(cx,cy,2.30,1.10,c,rounding=.15,dashed=True)
    txt(cx,cy+.32,rf"$G_{g}$ : {LABEL[g]}",size=8.2)
    offsets={1:[(0,-.08)],2:[(-.57,-.08),(.57,-.08)],
             3:[(-.67,-.11),(0,.015),(.67,-.11)]}[len(IDS[g])]
    pts=[(cx+dx,cy+dy) for dx,dy in offsets]
    BP.update(dict(zip(IDS[g],pts)))
    for p,q in zip(pts[:-1],pts[1:]):
        edge(p,q,lw=.55,shrink=.5)
    if len(pts)==3:
        edge(pts[0],pts[-1],lw=.55,shrink=.5)
    for p,i in zip(pts,IDS[g]):
        dot(p,label=i,below=True,fill=FILL[c])

# Arrows summarize inter-group vector connections at the group boundaries.
for p,q in [((9.52,3.24),(8.86,2.98)),((11.88,3.24),(12.54,2.98)),
            ((10.70,2.78),(10.70,1.95)),((7.68,2.27),(7.68,1.71)),
            ((13.72,2.27),(13.72,1.71))]:
    edge(p,q,arrow=True,lw=.65)

# One unobtrusive shared legend; no path connects the two materialized levels.
dot((.65,.28))
txt(.84,.28,r"Vector $v_i$",size=7,ha="left")
for x,c in [(3.0,1),(4.7,2),(6.4,3)]:
    ax.add_patch(FancyBboxPatch((x-.10,.20),.25,.16,
        boxstyle="round,pad=0,rounding_size=.03",facecolor=FILL[c],
        edgecolor=COL[c],linewidth=.55))
    txt(x+.26,.28,rf"Capsule $C_{c}$",size=7,ha="left")
edge((8.2,.28),(8.75,.28),color=CYAN,lw=1.2,arrow=True)
txt(8.90,.28,"Illustrative path",size=7,ha="left")
ax.plot([11.48,12.04],[.28,.28],color="#AEB7C1",lw=.55,ls=(0,(3,3)))
txt(12.22,.28,"Layer correspondence",size=7,ha="left")

assert set(CP)==set(BP)==set(OP[0])==set(OP[1])==set(range(1,14))
for c,groups in MEMBERS.items():
    inventory={i for g in groups for i in IDS[g]}
    assert all(i in inventory and j in inventory for i,j in CAP_LOCAL[c])
assert PATH[0] in CAP_LINKS and PATH[1] in CAP_LOCAL[2]

for ext in ("svg","pdf","png"):
    path=OUT/f"capsule_predicate_sigmod.{ext}"
    fig.savefig(path,dpi=360)
    print(path)
plt.close(fig)
