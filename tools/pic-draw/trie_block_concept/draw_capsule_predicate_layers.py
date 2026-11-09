"""Capsule / Predicate tree diagram with overview and detailed planar views.

The overview shows small vector graphs without individual names. Both views use
the same vectors and 1--3 vectors per group. The detailed view adds numeric
predicates and lowercase v_i labels beside each vector. Geometric
positions and proximity edges are illustrative, not an index dump. Dashed
overview connectors show membership, not cross-layer search transitions.
"""

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon


plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 10,
    "mathtext.fontset": "stix", "svg.fonttype": "none", "pdf.fonttype": 42,
})
OUT = Path(__file__).resolve().parent
INK = "#26364A"
MUTED = "#8491A2"
LINK = "#526780"
CAP_COLOR = {1:"#378FD1",2:"#E6A03A",3:"#9774CB"}
CAP_FILL = {1:"#ECF5FD",2:"#FFF5E5",3:"#F4EEFC"}
GROUP_COLOR = {1:"#5BADE3",2:"#F1BA4E",3:"#EB9253",
               4:"#B399DC",5:"#8B6BC5",6:"#51B9AF"}
GROUP_FILL = {1:"#EFF7FE",2:"#FFF8EB",3:"#FFF3E9",
              4:"#F7F2FD",5:"#F2EDFA",6:"#EDF8F6"}
LABELS = {1:"{1}",2:"{1,2}",3:"{1,2,4}",
          4:"{1,3}",5:"{1,3,5}",6:"{1,6}"}
COUNTS = {1:3,2:2,3:3,4:2,5:1,6:2}
MEMBERS = {1:(1,6),2:(2,3),3:(4,5)}
OWNER = {g:c for c,groups in MEMBERS.items() for g in groups}
VECTOR_IDS = {}
next_id = 1
for g in range(1,7):
    VECTOR_IDS[g] = list(range(next_id,next_id+COUNTS[g]))
    next_id += COUNTS[g]

fig,ax=plt.subplots(figsize=(11.7,7.08))
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
ax.set(xlim=(0,19.5),ylim=(0,11.8),aspect="equal")
ax.axis("off")


def text(x,y,value,size=10,color=INK,ha="center",**kw):
    ax.text(x,y,value,fontsize=size,color=color,ha=ha,va="center",zorder=30,**kw)


def edge(p,q,color=LINK,lw=.95,directed=False,rad=0,z=5,shrink=3):
    ax.add_patch(FancyArrowPatch(
        p,q,arrowstyle="-|>" if directed else "-",
        mutation_scale=10,connectionstyle=f"arc3,rad={rad}",
        linewidth=lw,color=color,shrinkA=shrink,shrinkB=shrink,zorder=z))


def rounded_box(cx,cy,w,h,fill,stroke,rounding=.17,lw=1):
    ax.add_patch(FancyBboxPatch(
        (cx-w/2,cy-h/2),w,h,
        boxstyle=f"round,pad=.012,rounding_size={rounding}",
        facecolor=fill,edgecolor=stroke,linewidth=lw,zorder=2))


def vector(p,g,i,fill):
    ax.add_patch(Circle(p,.062,facecolor=GROUP_COLOR[g],
                        edgecolor="white",linewidth=.55,zorder=12))
    # Names sit beside the vector glyph rather than inside the node.
    text(p[0]+.075,p[1]+.16,rf"$v_{{{i}}}$",size=8.9,color=INK,
         bbox={"facecolor":fill,"edgecolor":"none","pad":.22})


def heading(x,y,title,level,note=None):
    ax.add_patch(Circle((x,y),.19,facecolor=INK,edgecolor="none",zorder=31))
    ax.text(x,y,str(level),fontsize=9,color="white",ha="center",va="center",
            weight="bold",zorder=32)
    text(x+.34,y,title,size=12,ha="left",weight="bold",
         bbox={"facecolor":"white","edgecolor":"none","pad":2})
    if note:
        text(18.85,y,note,size=8.1,color=MUTED,ha="right")


text(.55,11.22,"(a) Structural overview",size=14,ha="left",weight="bold")
text(6.85,11.22,"(b) Detailed planar views",size=14,ha="left",weight="bold")
ax.plot([6.15,6.15],[1.0,10.60],color="#E5EAF0",lw=.75)

# LEFT: the same small graphs without individual vector labels. Lower ellipses
# separate groups, while the upper plane merges them into capsule regions.
PLANE_Y={0:2.15,1:6.55}
def ob(u,v,level):
    return (.60+4.35*u+.65*v,PLANE_Y[level]+1.55*v)

for level in (0,1):
    ax.add_patch(Polygon([ob(0,0,level),ob(1,0,level),
                          ob(1,1,level),ob(0,1,level)],
                         closed=True,facecolor="#FBFCFE",edgecolor="#718095",
                         linewidth=1.0,zorder=1))

for c,u0,u1 in [(2,.025,.30),(1,.355,.645),(3,.70,.975)]:
    ax.add_patch(Polygon([ob(u0,.08,1),ob(u1,.08,1),
                          ob(u1,.93,1),ob(u0,.93,1)],
                         closed=True,facecolor=CAP_FILL[c],edgecolor=CAP_COLOR[c],
                         linewidth=.8,zorder=2))
    text(*ob((u0+u1)/2,1.07,1),rf"$C_{c}$",size=10,color=CAP_COLOR[c])

GP={1:(.5,.78),2:(.16,.57),3:(.16,.16),
    4:(.84,.57),5:(.84,.16),6:(.5,.21)}
OV_OFFSETS={1:[(0,0)],2:[(-.05,-.01),(.05,.025)],
            3:[(-.055,-.025),(0,.06),(.055,-.025)]}
OV_POINTS={
    level:{g:[ob(u+du,v+dv,level) for du,dv in OV_OFFSETS[COUNTS[g]]]
           for g,(u,v) in GP.items()} for level in (0,1)
}
for g,(u,v) in GP.items():
    hull=[ob(u+.105*math.cos(t),v+.125*math.sin(t),0)
          for t in [j*2*math.pi/64 for j in range(64)]]
    ax.add_patch(Polygon(hull,closed=True,facecolor=GROUP_FILL[g],
                         edgecolor=GROUP_COLOR[g],linewidth=.65,zorder=2))
for g in (6,3,5):
    p,q=OV_POINTS[1][g][-1],OV_POINTS[0][g][-1]
    ax.plot([p[0],q[0]],[p[1],q[1]],lw=.8,color="#B1BDCC",
            ls=(0,(3,4)),zorder=3)
for level in (0,1):
    for g,pts in OV_POINTS[level].items():
        for i in range(len(pts)-1):
            edge(pts[i],pts[i+1],color=GROUP_COLOR[g],lw=.8,shrink=1)
        if len(pts)==3:
            edge(pts[0],pts[-1],color=GROUP_COLOR[g],lw=.8,shrink=1)
        for p in pts:
            ax.add_patch(Ellipse(p,.12,.075,facecolor=GROUP_COLOR[g],
                                edgecolor="white",linewidth=.45,zorder=10))
    # Connections between groups become ordinary capsule-local edges in L1.
    for g,h,i,j in [(1,2,0,-1),(1,4,-1,0),(1,6,1,0),
                    (2,3,0,1),(4,5,-1,0)]:
        local=level==1 and OWNER[g]==OWNER[h]
        edge(OV_POINTS[level][g][i],OV_POINTS[level][h][j],
             color=CAP_COLOR[OWNER[g]] if local else LINK,
             directed=not local,lw=.8,shrink=2)

heading(.78,8.90,"Capsule layer",1)
heading(.78,4.46,"Predicate tree layer",0)
text(3.0,1.47,"Dashed lines indicate membership",size=8.0,color=MUTED)
for y in (7.30,2.95):
    edge((5.65,y),(6.65,y),color="#A5B3C3",lw=.9,directed=True)

# RIGHT TOP: numeric capsule IDs and predicate prefixes, vector-level graphs.
heading(7.00,10.30,"Capsule layer",1,"Groups merged; connectivity rebuilt")
CC={1:(13.25,8.70),2:(9.35,7.45),3:(17.15,7.45)}
CAP_PTS={}
CAP_LIST={}


def capsule_positions(n,cx,cy):
    rows={3:[2,1],4:[2,2],5:[3,2],6:[3,3]}[n]
    ys=[cy-.20,cy-.76]
    out=[]
    for count,y in zip(rows,ys):
        xs=[cx+(j-(count-1)/2)*1.02 for j in range(count)]
        out.extend((x,y) for x in xs)
    return out,rows


for c,(cx,cy) in CC.items():
    rounded_box(cx,cy,3.28,2.16,CAP_FILL[c],CAP_COLOR[c],rounding=.28,lw=1.15)
    prefix={1:"{1}",2:"{1,2}",3:"{1,3}"}[c]
    text(cx,cy+.73,rf"$C_{c}$ : {prefix}",size=11.5,color=CAP_COLOR[c],weight="bold")
    a,b=MEMBERS[c]
    text(cx,cy+.39,rf"$G_{a}\ \cup\ G_{b}$",size=9.5,color=MUTED)
    inventory=[(g,i) for g in MEMBERS[c] for i in VECTOR_IDS[g]]
    pts,rows=capsule_positions(len(inventory),cx,cy)
    CAP_PTS[c]=pts
    CAP_LIST[c]=inventory
    # A sparse proximity graph: each row is connected, with bridges to the
    # next row. Connectivity spans original groups in each capsule.
    row_ids=[]
    start=0
    for count in rows:
        row_ids.append(list(range(start,start+count)))
        start+=count
    pairs=set()
    for row in row_ids:
        pairs.update(zip(row[:-1],row[1:]))
    for upper,lower in zip(row_ids,row_ids[1:]):
        for j,k in enumerate(lower):
            pairs.add((upper[min(j,len(upper)-1)],k))
        pairs.add((upper[-1],lower[0]))
    for i,j in sorted(pairs):
        edge(pts[i],pts[j],color=CAP_COLOR[c],lw=.9,shrink=1)
    for p,(g,i) in zip(pts,inventory):
        vector(p,g,i,CAP_FILL[c])

# Inter-capsule arcs exit below the headers, leaving text and vector names
# unobstructed. They summarize the directed parent-to-child connections.
edge((11.59,8.40),(11.01,7.75),directed=True,lw=1.1,rad=.07)
edge((14.91,8.40),(15.49,7.75),directed=True,lw=1.1,rad=-.07)

# RIGHT BOTTOM: exact numeric predicates, variable-size vector groups.
heading(7.00,5.65,"Predicate tree layer",0,"One label set per group")
GC={1:(13.25,4.30),6:(13.25,1.73),2:(9.35,3.79),3:(9.35,1.44),
    4:(17.15,3.79),5:(17.15,1.44)}
GROUP_LAYOUT={
    1:[(0,-.30)],
    2:[(-.65,-.30),(.65,-.30)],
    3:[(-.90,-.35),(0,-.05),(.90,-.35)],
}
for g,(cx,cy) in GC.items():
    rounded_box(cx,cy,3.05,1.74,GROUP_FILL[g],GROUP_COLOR[g],rounding=.18,lw=.9)
    text(cx,cy+.57,rf"$G_{g}$ : {LABELS[g]}",size=10.5)
    pts=[(cx+dx,cy+dy) for dx,dy in GROUP_LAYOUT[COUNTS[g]]]
    # Only 1--3 vectors per group, with names preserved in the upper layer.
    pairs={(i,i+1) for i in range(len(pts)-1)}
    if len(pts)>2:
        pairs.add((0,len(pts)-1))
    for i,j in sorted(pairs):
        edge(pts[i],pts[j],color=GROUP_COLOR[g],lw=.85,shrink=1)
    for p,i in zip(pts,VECTOR_IDS[g]):
        vector(p,g,i,GROUP_FILL[g])

for p,q in [((11.70,4.20),(10.90,3.93)),((14.80,4.20),(15.60,3.93)),
            ((13.25,3.40),(13.25,2.63)),((9.35,2.89),(9.35,2.34)),
            ((17.15,2.89),(17.15,2.34))]:
    edge(p,q,directed=True,lw=1.0)

# A quiet shared legend. Shade differences identify original groups; the three
# capsule background colors preserve ownership across the entire figure.
for x,c in [(1.05,1),(4.1,2),(7.15,3)]:
    ax.add_patch(Circle((x,.33),.077,facecolor=CAP_COLOR[c],edgecolor="none"))
    text(x+.22,.33,rf"Capsule $C_{c}$",size=9,ha="left")
ax.add_patch(Circle((10.30,.33),.062,facecolor=INK,edgecolor="none"))
text(10.50,.33,r"$v_i$  Vector",size=9,ha="left")
edge((13.35,.33),(13.95,.33),directed=True,lw=1)
text(14.12,.33,"Inter-group / capsule link",size=9,ha="left")

for ext in ("svg","pdf","png"):
    path=OUT/f"capsule_predicate_layers.{ext}"
    fig.savefig(path,dpi=260,facecolor="white")
    print(path)
print("group_sizes:",COUNTS)
print("total_vectors:",sum(COUNTS.values()))
plt.close(fig)
