"""Two tilted graph planes, with colorful vector nodes and block regions.

Geometry is schematic. The upper plane has one materialized block layer;
the lower plane has exact-label groups connected by prefix-Trie topology.
Dashed vertical correspondences do not denote cross-layer search edges.
The example red route stays within the block layer. Dot counts and vector
connections are illustrative rather than a persisted index dump.
"""

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon


plt.rcParams.update({
    "font.family": "Liberation Serif",
    "font.size": 11,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "mathtext.fontset": "stix",
})
OUT = Path(__file__).resolve().parent
INK = "#253140"
RED = "#E53D45"
COLORS = {"A": "#3985D5", "B": "#ED9936", "C": "#8A65CE"}
LIGHT = {"A": "#E4F1FF", "B": "#FFF0D8", "C": "#F0E7FF"}
SHADES = ["#58A8EF", "#F5B64E", "#EE893C", "#AD83E0", "#8060C3", "#40B8B0"]
LEVELS = {0: .88, 1: 4.40}

fig, ax = plt.subplots(figsize=(10.8, 6.65))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set(xlim=(0, 12), ylim=(0, 7.39), aspect="equal")
ax.axis("off")


def xy(u, v, layer):
    return (2.05 + 7.9*u + 1.35*v, LEVELS[layer] + 2.05*v)


def txt(x, y, value, size=11, color=INK, **kw):
    ax.text(x, y, value, fontsize=size, color=color,
            ha=kw.pop("ha", "center"), va="center", **kw)


def edge(p, q, color=INK, width=1, rad=0, directed=False, zorder=5, **kw):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>" if directed else "-",
        mutation_scale=10, connectionstyle=f"arc3,rad={rad}",
        linewidth=width, color=color, shrinkA=3, shrinkB=3,
        zorder=zorder, **kw))


def node(p, fill, zorder=8, entry=False):
    if entry:
        ax.add_patch(Ellipse(p, .31, .19, facecolor="#FFF0A3",
                            edgecolor="#D5A121", linewidth=1.15, zorder=zorder))
    ax.add_patch(Ellipse(p, .205, .115,
                        facecolor=fill, edgecolor=INK, linewidth=.9,
                        zorder=zorder+1))


def region(u0, v0, u1, v1, layer, key):
    ax.add_patch(Polygon([xy(u0,v0,layer), xy(u1,v0,layer),
                          xy(u1,v1,layer), xy(u0,v1,layer)],
                         closed=True, facecolor=LIGHT[key],
                         edgecolor=COLORS[key], linewidth=.9, zorder=2))


def group_oval(u, v, key):
    pts = [xy(u+.105*math.cos(t),v+.108*math.sin(t),0)
           for t in [i*2*math.pi/80 for i in range(80)]]
    ax.add_patch(Polygon(pts, closed=True, facecolor=LIGHT[key],
                         edgecolor=COLORS[key], linewidth=.8, zorder=3))


GROUPS = {
    1: (.50, .77, "A", "{a}"),
    2: (.16, .58, "B", "{a,b}"),
    3: (.16, .17, "B", "{a,b,d}"),
    4: (.84, .58, "C", "{a,c}"),
    5: (.84, .17, "C", "{a,c,e}"),
    6: (.50, .23, "A", "{a,f}"),
}
OFFSETS = [(-.047,-.028),(.0,.038),(.047,-.025)]
POINTS = {
    layer: {g: [xy(u+du,v+dv,layer) for du,dv in OFFSETS]
            for g,(u,v,_,_) in GROUPS.items()}
    for layer in (0,1)
}

# Separate flat parallelograms evoke the supplied reference without perspective
# distortion of text or a raster-only rendering pipeline.
for layer in (0,1):
    front_v = -.10 if layer == 0 else 0
    ax.add_patch(Polygon([xy(0,front_v,layer),xy(1,front_v,layer),
                          xy(1,1,layer),xy(0,1,layer)],
                         closed=True, facecolor="#FBFCFF", edgecolor=INK,
                         linewidth=1.45, zorder=1))

# Coarse colored regions replace fine group boundaries at the upper plane.
region(.355,.075,.645,.96,1,"A")
region(.025,.055,.295,.84,1,"B")
region(.705,.055,.975,.84,1,"C")
for g,(u,v,key,labels) in GROUPS.items():
    group_oval(u,v,key)

# Same-vector correspondences, styled independently of directed graph edges.
for g in (3,6,5):
    p, q = POINTS[1][g][0], POINTS[0][g][0]
    ax.plot([p[0],q[0]],[p[1],q[1]],color="#8191A2",lw=.85,
            ls=(0,(4,3)),alpha=.7,zorder=4)

for g,(_,_,key,labels) in GROUPS.items():
    fine = POINTS[0][g]
    for i,j in [(0,1),(1,2),(0,2)]:
        edge(fine[i],fine[j],color=COLORS[key],width=1,zorder=5)
    for p in fine:
        node(p,SHADES[g-1])
    u,v,_,_ = GROUPS[g]
    label_v = v-.15 if g in (3,5,6) else v+.15
    x,y = xy(u,label_v,0)
    txt(x,y,rf"$G_{g}$  {labels}",size=11,zorder=10)

# Nearest-terminal prefix edges summarize the base vector connections.
for g,h,i,j,rad in [(1,2,0,2,.04),(1,4,2,0,-.04),
                     (1,6,1,1,0),(2,3,1,1,0),(4,5,1,1,0)]:
    edge(POINTS[0][g][i],POINTS[0][h][j],width=1.05,
         directed=True,rad=rad,zorder=6)

# Block-local graphs include new connections between original label groups.
for key, pair in [("A",(1,6)),("B",(2,3)),("C",(4,5))]:
    a,b = pair
    pts = POINTS[1][a]+POINTS[1][b]
    for i,j in [(0,1),(1,2),(0,2),(3,4),(4,5),(3,5),
                (0,3),(1,4),(2,5),(2,4)]:
        edge(pts[i],pts[j],color=COLORS[key],width=1.05,zorder=5)
    for g in pair:
        for p in POINTS[1][g]:
            node(p,SHADES[g-1])

edge(POINTS[1][1][0],POINTS[1][2][2],directed=True,width=1.2,rad=.04)
edge(POINTS[1][1][2],POINTS[1][4][0],directed=True,width=1.2,rad=-.04)

for u,v,key,prefix in [(.50,.925,"A","{a}"),
                         (.16,.90,"B","{a,b}"),(.84,.90,"C","{a,c}")]:
    x,y = xy(u,v,1)
    txt(x,y,rf"$B_{key}$ : {prefix}",size=12,color=COLORS[key],
        weight="bold",zorder=10)

# An example path on the upper plane only. The same yellow entry convention
# as the reference distinguishes a chosen seed from ordinary vector nodes.
entry = POINTS[1][1][0]
node(entry,SHADES[0],entry=True,zorder=11)
edge(entry,POINTS[1][2][2],color=RED,width=1.8,directed=True,rad=.04,zorder=12)
edge(POINTS[1][2][2],POINTS[1][3][1],color=RED,width=1.8,directed=True,zorder=12)
x,y = entry
txt(x-.38,y+.43,"Entry vector",size=11,ha="right",zorder=13)
edge((x-.31,y+.37),(x-.045,y+.09),color="#637081",width=.8,zorder=10)

txt(.35,5.55,"Layer 1",size=14,ha="left",weight="bold")
txt(.35,5.25,"Block graph",size=11,ha="left",color="#5F6D7D")
txt(.35,2.05,"Layer 0",size=14,ha="left",weight="bold")
txt(.35,1.75,"Trie group graph",size=11,ha="left",color="#5F6D7D")
txt(6.8,7.02,"Trie–Block Index",size=18,weight="bold")

# Compact legend outside the planes.
node((2.5,.34),SHADES[0])
txt(2.72,.34,"Vector",size=10,ha="left")
edge((3.8,.34),(4.3,.34),color=RED,width=1.7,directed=True)
txt(4.4,.34,"Example route",size=10,ha="left")
ax.plot([6.28,6.78],[.34,.34],lw=.9,color="#8191A2",ls=(0,(4,3)))
txt(6.9,.34,"Same-vector correspondence",size=10,ha="left")

for ext in ("svg","pdf","png"):
    path = OUT / f"trie_block_stacked.{ext}"
    fig.savefig(path,dpi=220,facecolor="white")
    print(path)
plt.close(fig)
