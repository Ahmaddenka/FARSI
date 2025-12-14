import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from pathlib import Path


BASE_DIR = Path("graph_file")
OUT_DIR = Path("output")

# ====== بارگذاری ======
G = nx.read_graphml(str(BASE_DIR / "all_network.graphml"))
# ====== تشخیص سطح هر نود ======
LEVEL_KEY = "level"
CAP = {"bsm":"BSM","control":"Control","soe":"SOE","group":"Group","agent":"Agent"}

def get_level(n: str) -> str:
    # 1) اگر attribute سطح موجود است، همان را استفاده کن
    lvl = G.nodes[n].get(LEVEL_KEY, None)
    if isinstance(lvl, str):
        lvl = CAP.get(lvl.lower(), lvl)
        if lvl in {"BSM","Control","SOE","Group","Agent"}:
            return lvl

    # 2) fallback هیوریستیکی فقط اگر attribute نبود
    s = n.lower()
    parts = s.split("_")
    if any(p.isdigit() for p in parts): return "Agent"
    if len(parts) >= 3: return "Group"
    if len(parts) == 2: return "SOE"
    return "Other"



levels = {n: get_level(n) for n in G.nodes()}

# ====== تنظیمات بصری ======
color_map = {
    "BSM": "#ff6666",       # قرمز
    "Control": "#ffcc00",   # زرد
    "SOE": "#66ccff",       # آبی
    "Group": "#99ff99",     # سبز
    "Agent": "#cccccc",     # خاکستری روشن
    "Other": "#999999"
}

# محور y برای لایه‌ها (Control عمودی خواهد بود)
y_pos = {"BSM": 5, "SOE": 4, "Group": 2, "Agent": 1}

# دامنه‌ی لرزش عمودی (برای شکستن خط صاف هر لایه)
jitter = {"BSM":0.20, "SOE":0.20, "Group":0.15, "Agent":0.10}

# اگر True باشد یال‌های درون‌سطحی خیلی کم‌رنگ/ناپیدا می‌شوند
FADE_SAME_LEVEL_EDGES = True
HIDE_SAME_LEVEL_EDGES = False   # اگر True شود، یال‌های درون‌سطحی اصلاً رسم نمی‌شوند

# قوس برای یال‌های هم‌سطح (به کاهش خط صاف کمک می‌کند)
ARC_FOR_SAME_LEVEL = True
ARC_RAD = 0.12

# ====== موقعیت‌دهی ======
random.seed(42)
np.random.seed(42)

pos = {}

# 1) لایه‌های غیر از Control: x تصادفی، y حول خط لایه با jitter پایدار
for n in G.nodes():
    lvl = levels[n]
    if lvl == "Control":
        continue
    base_y = y_pos.get(lvl, 0.0)
    # x تصادفی در بازه‌ی [0, 100]
    x = random.random() * 100.0
    # jitter پایدار بر اساس hash نود
    r = (hash(n) & 0xffff) / 0xffff
    y = base_y + (r - 0.5) * 2 * jitter.get(lvl, 0.1)
    pos[n] = (x, y)

# 2) چیدمان عمودی Control در سمت چپ: y ≈ میانگین y مقصدهای خروجی هر کنترل
x_control = -10.0
controls = [n for n in G.nodes() if levels[n] == "Control"]
for c in controls:
    targets = list(G.successors(c))
    if targets:
        y_targets = [pos[t][1] if t in pos else 3.0 for t in targets]
        y = sum(y_targets) / len(y_targets)
    else:
        y = 3.0
    y += (random.random() - 0.5) * 0.25  # اندکی نویز برای جلوگیری از هم‌افتادگی
    pos[c] = (x_control, y)

# ====== تفکیک یال‌ها ======
edges_same = [(u, v) for (u, v) in G.edges() if levels.get(u) == levels.get(v)]
edges_cross = [(u, v) for (u, v) in G.edges() if levels.get(u) != levels.get(v)]

# ====== رسم ======
plt.figure(figsize=(30, 20))

# نودها
node_colors = [color_map.get(levels[n], "#999999") for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=15, node_color=node_colors, alpha=0.85, linewidths=0)

# یال‌های بین‌سطحی (صاف)
nx.draw_networkx_edges(
    G, pos, edgelist=edges_cross, arrows=True,
    width=0.08, alpha=0.22, arrowsize=6, min_source_margin=0, min_target_margin=0
)

# یال‌های هم‌سطح (قوسی یا صاف، و کم‌رنگ)
if not HIDE_SAME_LEVEL_EDGES and edges_same:
    edge_kwargs_same = dict(
        arrows= True, width=0.05,
        alpha=0.08 if FADE_SAME_LEVEL_EDGES else 0.2,
        arrowsize=5, min_source_margin=0, min_target_margin=0
    )
    if ARC_FOR_SAME_LEVEL:
        edge_kwargs_same["connectionstyle"] = f"arc3,rad={ARC_RAD}"
    nx.draw_networkx_edges(G, pos, edgelist=edges_same, **edge_kwargs_same)

# ====== برچسب لایه‌ها ======
# برچسب افقی برای BSM/SOE/Group/Agent (سمت راست)
for lvl, y in y_pos.items():
    plt.text(102, y, lvl, fontsize=14, fontweight='bold', va='center')

# برچسب عمودی برای Control
mid_y = (min(y_pos.values()) + max(y_pos.values())) / 2
plt.text(x_control - 2.0, mid_y, "Control", fontsize=14, fontweight='bold',
         va='center', ha='center', rotation=90)

plt.title("Multilevel Organizational Network", fontsize=16, fontweight='bold')
plt.axis("off")

# حاشیه‌ها برای دیده‌شدن ستون کنترل و برچسب‌ها
plt.xlim(x_control - 6, 112)
plt.ylim(0, 6)
plt.savefig(OUT_DIR / "multilevel_network.png", dpi=600, bbox_inches="tight")
#plt.savefig("multilevel_network.svg", bbox_inches="tight") 
plt.show()
