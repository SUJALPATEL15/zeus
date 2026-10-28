# Z.E.U.S. – Zone Evaluation and Utility System

**Adani University | Faculty of Engineering Sciences and Technology (FEST)**  
**Department of Computer Science and Engineering (AI-ML)**  
**Course Code:** ECSCT24301 - Foundation of AI  

### Team Members:
1. **Patel Meet** (Enrollment No: 2410111108)
2. **Patel Sujal** (Enrollment No: 2410111124)
3. **Somani Swayam** (Enrollment No: 2410111163)

---

## ⚡ Executive Overview

**Z.E.U.S.** (Zone Evaluation and Utility System) is an autonomous tactical decision-making and pathfinding engine modeled after competitive Battle Royale scenarios (such as BGMI / PUBG Erangel). It implements a 3-stage intelligent In-Game Leader (IGL) pipeline:

1. **Stage 1: Probabilistic Inference (Unit 3)** — Bayesian Networks forecast future safe zones based on landmass, compound cover, and historical shift distributions:
   $$P(Z_{t+1}=c \mid L, Z_t) \propto P(c \mid L) \cdot \mathcal{N}(c; Z_t, \sigma^2) \cdot \mathbf{1}[\|c - Z_t\| \le R_t - R_{t+1}]$$
2. **Stage 2: Expert Systems (Unit 4)** — Forward-chaining rule engine calculating multi-attribute tactical utility to select macro-strategies: defending a compound (**HOLD**), intercepting rotators at a choke point (**GATEKEEP**), or executing a flank (**ROTATE**).
3. **Stage 3: Autonomous Navigation & Heuristic Search (Units 1 & 2)** — Threat-Aware $A^*$ search with an admissible Chebyshev heuristic ($h(n) = 0.4 \cdot \text{Chebyshev}(n, g)$) and step cost function $c(n) = \max(0.4, 1 + 3\cdot\text{threat} - 0.6\cdot\text{cover})$.

---

## 🖥️ Interactive Web Dashboard & HUD

Launch the complete tactical command center:
```bash
streamlit run app.py
```

### Dashboard Modes & Features:
- 🎯 **Tactical Radar (IGL Command Center)**: Interactive Erangel radar canvas with dual-zone controls (Blue Playzone $Z_t$ & Yellow Safe Zone $Z_{t+1}$), squad deployment, hostile camper recon, animated tactical route replay, gatekeep field-of-fire sector cone (angles & bearings), weak-side flank paths, and vehicle/sprint transit margins.
- 🔮 **Stage 1: Bayesian Forecaster**: 3D interactive posterior surface plot, prior landmass distribution heatmap, and conjugate inverse-gamma variance ($\sigma$) updates.
- 🧠 **Stage 2: Expert Rules Engine**: Interactive working memory simulator, live forward-chaining rule execution trace ($R_1 \dots R_7$), and real-time utility breakdown bar chart ($U_{\text{HOLD}}, U_{\text{GATEKEEP}}, U_{\text{ROTATE}}$).
- ⚡ **Stage 3: Threat-Aware A\* Search**: Admissibility ($h \le h^*$) and consistency ($h(n) \le c + h(n')$) verification, naive Manhattan counterexample analysis (4,825 violations), and 3-way live comparison against Baseline $A^*$ and BFS.
- 📊 **Empirical Benchmarks & Scalability**: Interactive benchmark visualizations from `results/` across 3 topologies (Urban, Open, Bridge) and 3 grid scales ($50\times50$, $100\times100$, $250\times250$) demonstrating a **+300% survival rate gain**.
- 📜 **Academic Project Dossier**: Complete PEAS specification, state space formalization, transition model, and error analysis directly aligned with `FAI.pdf`.

---

## 📁 Repository Structure

| File / Folder | Purpose |
|---|---|
| [`app.py`](file:///c:/Users/Meet%20Patel/OneDrive/Desktop/Z.E.U.S/app.py) | Full-featured esports HUD & 6-tab academic command dashboard |
| [`zeus.py`](file:///c:/Users/Meet%20Patel/OneDrive/Desktop/Z.E.U.S/zeus.py) | Core algorithms: Bayesian forecaster, rule base, BFS, Manhattan $A^*$, Threat-Aware $A^*$ |
| [`test_heuristics.py`](file:///c:/Users/Meet%20Patel/OneDrive/Desktop/Z.E.U.S/test_heuristics.py) | Admissibility & consistency unit tests + naive Manhattan counterexample |
| [`test_dual_zones.py`](file:///c:/Users/Meet%20Patel/OneDrive/Desktop/Z.E.U.S/test_dual_zones.py) | Dual-zone verification tests for ROTATE vs GATEKEEP decision priority |
| [`benchmark.py`](file:///c:/Users/Meet%20Patel/OneDrive/Desktop/Z.E.U.S/benchmark.py) | Automated test harness simulating 100 game phases across 3 topologies |
| [`results/`](file:///c:/Users/Meet%20Patel/OneDrive/Desktop/Z.E.U.S/results) | Quantitative CSV summaries, benchmark logs, and generated comparative plots |
| [`FAI.pdf`](file:///c:/Users/Meet%20Patel/OneDrive/Desktop/Z.E.U.S/FAI.pdf) | Foundation of AI project term paper (Adani University) |

---

## 🚀 Quickstart

1. **Verify Unit Tests**:
   ```bash
   python test_heuristics.py
   python test_dual_zones.py
   ```
2. **Run Benchmarks**:
   ```bash
   python benchmark.py
   ```
3. **Launch UI**:
   ```bash
   streamlit run app.py
   ```
