"""Draw a schematic of the Trie base graph and one block overlay.

Run with python3; outputs are written beside this script. Vector locations and
edges are illustrative, not an actual index dump. Every group contains vectors
with one exact label set. Colored regions show disjoint direct ownership within
one materialized block layer, including a parent block whose descendants belong
to separate child blocks. Dashed vertical lines denote membership only.

An example partition realizing these regions uses T=3 and group masses
G1=4, G2=2, G3=2, G4=2, G5=2, G6=3. Bottom-up uncovered mass creates roots
{a,b}, {a,c}, then {a}. Drawn dot counts are illustrative, not those masses.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


plt.rcParams.update({
    "font.family": "Liberation Sans",
    "font.size": 9,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT = Path(__file__).resolve().parent
INK = "#27313B"
MUTED = "#68717B"
GRAY = "#89939D"
COLORS = {"A": "#3778AC", "B": "#BC742D", "C": "#388477"}
FILLS = {"A": "#EFF5FA", "B": "#FCF3E8", "C": "#EEF7F3"}

fig, ax = plt.subplots(figsize=(7.2, 5.50))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set(xlim=(0, 7.2), ylim=(-0.85, 4.65))
ax.axis("off")


def text(x, y, label, size=9, color=INK, **kw):
    return ax.text(x, y, label, fontsize=size, color=color,
                   ha=kw.pop("ha", "center"), va="center", **kw)


def line(p, q, color=GRAY, lw=0.8, **kw):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, **kw)


def arrow(p, q, rad=0, color=INK, lw=1.05, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>",
                                mutation_scale=8, linewidth=lw,
                                connectionstyle=f"arc3,rad={rad}",
                                color=color, **kw))


def box(x, y, w, h, key, alpha=1.0, lw=0.9, fill=None):
    ax.add_patch(FancyBboxPatch(
        (x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.055",
        facecolor=fill or FILLS[key], edgecolor=COLORS[key],
        linewidth=lw, alpha=alpha,
    ))


def dots(points, key, edges):
    for i, j in edges:
        line(points[i], points[j], color=COLORS[key], lw=0.75, zorder=3)
    ax.scatter(*zip(*points), s=12, facecolor=COLORS[key],
               edgecolor="white", linewidth=0.35, zorder=5)


def block(x, y, key, prefix):
    box(x, y, 1.88, 0.72, key)
    text(x, y+0.21, rf"$B_{key}$ : {prefix}", size=9.5)
    pts = [(x-.62, y-.10), (x-.32, y+.01), (x-.10, y-.22),
           (x+.15, y-.03), (x+.43, y-.22), (x+.66, y-.04)]
    dots(pts, key, [(0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(3,5),(4,5)])
    return pts


def group(x, y, key, i, labels):
    box(x, y, 1.5, .67, key, fill="white", lw=.85)
    text(x, y+.15, rf"$G_{i}$  {labels}", size=9)
    pts = [(x-.42,y-.12), (x,y-.03), (x+.42,y-.15)]
    dots(pts, key, [(0,1),(1,2),(0,2)])
    return pts


# The same three colors identify direct ownership in both panels.
text(.22, 4.46, "(a) Block layer", size=10.5, ha="left", weight="bold")
text(6.98, 4.46, "One proximity graph per block", size=9, ha="right", color=MUTED)

ba = block(3.6, 3.95, "A", "{a}")
bb = block(1.45, 2.94, "B", "{a,b}")
bc = block(5.75, 2.94, "C", "{a,c}")
arrow(ba[0], (bb[3][0], 3.31), rad=.10)
arrow(ba[-1], (bc[2][0], 3.31), rad=-.10)
text(3.6, 3.31, "Directed links to child blocks", size=8.5, color=MUTED,
     bbox={"facecolor":"white", "edgecolor":"none", "pad":2}, zorder=9)

# Membership projections are deliberately gray and dashed: these lines are
# correspondences between representations, not edges for layer promotion.
for x, y0, y1 in [(3.6,3.57,2.03), (1.45,2.56,1.49), (5.75,2.56,1.49)]:
    line((x,y0),(x,y1), color="#ADB5BD", lw=.8, ls=(0,(3,3)), zorder=0)

text(.22, 2.18, "(b) Trie-organized group layer", size=10.5, ha="left", weight="bold",
     bbox={"facecolor":"white", "edgecolor":"none", "pad":2}, zorder=9)

# A parent block excludes all groups owned by either child block. Keeping
# these regions disjoint conveys this invariant without a nested container.
box(3.6, .78, 1.90, 2.46, "A", lw=.85)
box(1.45, .50, 1.94, 1.94, "B", lw=.85)
box(5.75, .50, 1.94, 1.94, "C", lw=.85)
text(4.29, 1.86, r"$B_A$", size=9, color=COLORS["A"])
text(2.15, 1.33, r"$B_B$", size=9, color=COLORS["B"])
text(6.45, 1.33, r"$B_C$", size=9, color=COLORS["C"])

g1 = group(3.6, 1.45, "A", 1, "{a}")
g2 = group(1.45, .93, "B", 2, "{a,b}")
g3 = group(1.45, -.02, "B", 3, "{a,b,d}")
g4 = group(5.75, .93, "C", 4, "{a,c}")
g5 = group(5.75, -.02, "C", 5, "{a,c,e}")
g6 = group(3.6, -.02, "A", 6, "{a,f}")

# Arrows summarize vector-level cross-group links along nearest-terminal
# prefix relations. Fine proximity graphs remain inside individual groups.
arrow(g1[0], (2.22,.97), rad=.07)
arrow(g1[-1], (4.98,.97), rad=-.07)
arrow((3.6,1.08), (3.6,.34))
arrow((1.45,.58), (1.45,.34))
arrow((5.75,.58), (5.75,.34))

# Compact shared legend; redundant border labels keep ownership legible
# when printed in grayscale.
ax.scatter([.33],[-.69],s=12,color=INK)
text(.44,-.69,"vector",size=8,ha="left")
line((1.12,-.69),(1.37,-.69),color=INK)
text(1.47,-.69,"local edge",size=8,ha="left")
arrow((2.47,-.69),(2.82,-.69),lw=.9)
text(2.92,-.69,"cross-group / block link",size=8,ha="left")
line((4.91,-.69),(5.26,-.69),color="#ADB5BD",ls=(0,(3,3)))
text(5.36,-.69,"membership only",size=8,ha="left")

for ext in ("svg", "pdf", "png"):
    path = OUT / f"trie_block_concept.{ext}"
    fig.savefig(path, dpi=220, facecolor="white")
    print(path)
plt.close(fig)
