"""Publication schematic: compact oblique overview + corresponding planar detail.

The right view uses the same group, block, vector IDs and colors at both levels.
Left overview nodes are representative glyphs, not a complete vector inventory.
Dashed vertical correspondences are not search transitions. Red route segments
remain inside one materialized layer. Positions and vector edges are schematic.
Base arrows between group boundaries summarize the prefix-Trie vector links.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon


plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 8.5,
    "mathtext.fontset": "stix", "svg.fonttype": "none", "pdf.fonttype": 42,
})
OUT = Path(__file__).resolve().parent
INK = "#2A3546"
GRAY = "#748196"
RED = "#E24B53"
COLORS = {"A":"#3789CE", "B":"#E69A32", "C":"#9270CF"}
FILLS = {"A":"#EAF4FE", "B":"#FFF4E3", "C":"#F2ECFC"}
NODE = {1:"#69B2EB", 2:"#F6C15B", 3:"#EE9C55",
        4:"#B497E1", 5:"#9471CD", 6:"#5CC1B9"}
LABELS = {1:"{a}", 2:"{a,b}", 3:"{a,b,d}",
          4:"{a,c}", 5:"{a,c,e}", 6:"{a,f}"}
OWNER = {1:"A",6:"A",2:"B",3:"B",4:"C",5:"C"}
MEMBERS = {"A":(1,6), "B":(2,3), "C":(4,5)}
VECTOR_IDS = {g:list(range(3*g-2,3*g+1)) for g in LABELS}

fig, ax = plt.subplots(figsize=(7.6,4.21))
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
ax.set(xlim=(0,16.6),ylim=(0,9.2),aspect="equal")
ax.axis("off")


def text(x,y,s,size=8.5,color=INK,ha="center",**kw):
    ax.text(x,y,s,fontsize=size,color=color,ha=ha,va="center",zorder=20,**kw)


def edge(p,q,color=INK,lw=.85,arrow=False,rad=0,z=6,**kw):
    ax.add_patch(FancyArrowPatch(
        p,q,arrowstyle="-|>" if arrow else "-",mutation_scale=7,
        connectionstyle=f"arc3,rad={rad}",linewidth=lw,color=color,
        shrinkA=2.4,shrinkB=2.4,zorder=z,**kw))


def frame(cx,cy,w,h,key,white=False,lw=.8):
    ax.add_patch(FancyBboxPatch(
        (cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.015,rounding_size=.11",
        facecolor="white" if white else FILLS[key],edgecolor=COLORS[key],
        linewidth=lw,zorder=2))


def dot(p,g,number=None,small=False,entry=False):
    if small:
        ax.add_patch(Ellipse(p,.16,.085,facecolor=NODE[g],edgecolor=INK,
                            linewidth=.55,zorder=10))
        return
    if entry:
        ax.add_patch(Circle(p,.16,facecolor="#FFF2B7",edgecolor="#DAAF38",
                            linewidth=.8,zorder=9))
    ax.add_patch(Circle(p,.115,facecolor=NODE[g],edgecolor="white",
                        linewidth=.6,zorder=10))
    if number is not None:
        text(*p,str(number),size=5.8,color="#19334A")


def section(x,y,n,title,subtitle=None):
    ax.add_patch(Circle((x,y),.17,facecolor=INK,edgecolor="none",zorder=12))
    text(x,y,str(n),size=7,color="white",weight="bold")
    text(x+.30,y,title,size=9,ha="left",weight="bold",
         bbox={"facecolor":"white","edgecolor":"none","pad":1.4})
    if subtitle:
        text(16.05,y,subtitle,size=6.7,ha="right",color=GRAY)


# Matched panel headings and deliberately quiet secondary text.
text(.5,8.75,"(a) Oblique overview",size=11,ha="left",weight="bold")
text(.5,8.30,"Two layers, one vector collection",size=7.5,ha="left",color=GRAY)
text(6.65,8.75,"(b) Corresponding planar views",size=11,ha="left",weight="bold")
ax.plot([6.05,6.05],[.95,8.1],color="#E6EAF0",lw=.65,zorder=0)

# LEFT: only enough structure to communicate the hierarchy.
BASE_Y = {0:1.75,1:5.40}
def ob(u,v,layer):
    return (.50+4.25*u+.65*v,BASE_Y[layer]+1.45*v)

OV = {1:(.5,.79),6:(.5,.20),2:(.16,.58),3:(.16,.16),
      4:(.84,.58),5:(.84,.16)}
for layer in (0,1):
    ax.add_patch(Polygon([ob(0,0,layer),ob(1,0,layer),ob(1,1,layer),ob(0,1,layer)],
                         closed=True,facecolor="#FBFCFE",edgecolor="#596678",
                         linewidth=.9,zorder=1))

for key,u0,u1 in [("B",.025,.30),("A",.355,.645),("C",.70,.975)]:
    ax.add_patch(Polygon([ob(u0,.06,1),ob(u1,.06,1),ob(u1,.94,1),ob(u0,.94,1)],
                         closed=True,facecolor=FILLS[key],edgecolor=COLORS[key],
                         linewidth=.65,zorder=2))
    text(*ob((u0+u1)/2,1.075,1),rf"$B_{key}$",size=8,color=COLORS[key])

for g in (3,6,5):
    p,q=ob(*OV[g],1),ob(*OV[g],0)
    ax.plot([p[0],q[0]],[p[1],q[1]],color="#A6B3C3",lw=.65,
            ls=(0,(3,3)),zorder=3)

for layer in (0,1):
    for a,b in [(1,2),(1,4),(1,6),(2,3),(4,5)]:
        color=COLORS[OWNER[a]] if layer and OWNER[a]==OWNER[b] else INK
        edge(ob(*OV[a],layer),ob(*OV[b],layer),color=color,lw=.75,
             arrow=not(layer and OWNER[a]==OWNER[b]))
    if layer:
        edge(ob(*OV[1],1),ob(*OV[2],1),color=RED,lw=1.1,arrow=True,z=8)
    for g,(u,v) in OV.items():
        dot(ob(u,v,layer),g,small=True)
        if layer==0:
            dx,dy,ha={1:(0,.16,"center"),2:(-.12,.20,"right"),
                      4:(.16,.18,"left")}.get(g,(.17,-.055,"left"))
            x,y=ob(u,v,layer)
            text(x+dx,y+dy,rf"$G_{g}$",size=7,ha=ha,color=GRAY,
                 bbox={"facecolor":"#FBFCFE","edgecolor":"none","pad":.15})

section(.66,7.35,1,"Block layer")
section(.66,3.95,0,"Trie group layer")
text(2.95,1.20,"Representative nodes",size=6.8,color=GRAY)

# These outward arrows mean expanding the illustration, not graph traversal.
for y in (6.05,2.70):
    edge((5.48,y),(6.47,y),color="#9EABBB",lw=.75,arrow=True,z=2)

# RIGHT TOP: all vectors in a block, with fine groups merged into one graph.
section(6.80,7.95,1,"Block graph","Groups merged; connectivity rebuilt")
BC = {"A":(11.25,6.65),"B":(8.1,5.65),"C":(14.4,5.65)}
BLOCK_POINTS={}
for key,(cx,cy) in BC.items():
    frame(cx,cy,2.65,1.58,key)
    prefix={"A":"{a}","B":"{a,b}","C":"{a,c}"}[key]
    text(cx,cy+.53,rf"$B_{key}$  :  {prefix}",size=9,color=COLORS[key],weight="bold")
    a,b=MEMBERS[key]
    text(cx,cy+.22,rf"$G_{a}\ \cup\ G_{b}$",size=7.6,color=GRAY)
    pts=[(cx+dx,cy+dy) for dy in (-.12,-.52) for dx in (-.80,0,.80)]
    BLOCK_POINTS[key]=pts
    for i,j in [(0,1),(1,2),(3,4),(4,5),(0,3),(1,4),(2,5),(1,3),(2,4)]:
        edge(pts[i],pts[j],color=COLORS[key],lw=.7)
    for g,part in [(a,pts[:3]),(b,pts[3:])]:
        for p,n in zip(part,VECTOR_IDS[g]):
            dot(p,g,n)

edge(BLOCK_POINTS["A"][0],BLOCK_POINTS["B"][2],arrow=True,lw=.9,rad=.04)
edge(BLOCK_POINTS["A"][2],BLOCK_POINTS["C"][0],arrow=True,lw=.9,rad=-.04)
edge(BLOCK_POINTS["A"][0],BLOCK_POINTS["B"][2],color=RED,lw=1.25,arrow=True,rad=.04,z=8)
edge(BLOCK_POINTS["B"][2],BLOCK_POINTS["B"][4],color=RED,lw=1.25,arrow=True,z=8)
dot(BLOCK_POINTS["A"][0],1,1,entry=True)

# RIGHT BOTTOM: each exact-label group has its own vector graph. Keeping the
# same vector numbers makes the top-to-bottom correspondence explicit.
section(6.80,4.22,0,"Trie group graph","One label set per group")
GC={1:(11.25,3.17),6:(11.25,1.35),2:(8.1,2.72),3:(8.1,1.13),
    4:(14.4,2.72),5:(14.4,1.13)}
for g,(cx,cy) in GC.items():
    key=OWNER[g]
    frame(cx,cy,2.26,1.10,key,lw=.65)
    text(cx,cy+.31,rf"$G_{g}$  {LABELS[g]}",size=8.2)
    pts=[(cx-.65,cy-.22),(cx,cy-.02),(cx+.65,cy-.22)]
    for i,j in [(0,1),(1,2),(0,2)]:
        edge(pts[i],pts[j],color=COLORS[key],lw=.7)
    for p,n in zip(pts,VECTOR_IDS[g]):
        dot(p,g,n)

for p,q in [((10.09,3.10),(9.25,2.80)),((12.41,3.10),(13.25,2.80)),
            ((11.25,2.59),(11.25,1.93)),((8.1,2.14),(8.1,1.71)),
            ((14.4,2.14),(14.4,1.71))]:
    edge(p,q,arrow=True,lw=.8)

# A single, compact legend serves both views.
legend_y=.30
for x,key in [(1.0,"A"),(3.1,"B"),(5.2,"C")]:
    ax.add_patch(Circle((x,legend_y),.075,facecolor=COLORS[key],edgecolor="none"))
    text(x+.18,legend_y,rf"Block $B_{key}$",size=7,ha="left")
edge((7.05,legend_y),(7.65,legend_y),color=RED,lw=1.15,arrow=True)
text(7.83,legend_y,"Example route",size=7,ha="left")
ax.plot([10.60,11.20],[legend_y,legend_y],color="#A6B3C3",lw=.7,ls=(0,(3,3)))
text(11.40,legend_y,"Cross-layer correspondence",size=7,ha="left")

for ext in ("svg","pdf","png"):
    path=OUT/f"trie_block_overview_detail.{ext}"
    fig.savefig(path,dpi=300,facecolor="white")
    print(path)
plt.close(fig)
