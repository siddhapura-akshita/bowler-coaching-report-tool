<div align="center">

# 🏏 T20 Fast-Bowler Coaching Tool

### *Turn 5 million deliveries into a coaching conversation.*

[![Made with Python](https://img.shields.io/badge/Made%20with-Python%203.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20on-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Powered by Pandas](https://img.shields.io/badge/Powered%20by-Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![ML](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-bowling%20a%20maiden-brightgreen)](#)

*Pick a bowler. Pick a cohort. Get a scouting report no spreadsheet could give you.*

</div>

---

Point it at ball-by-ball T20 data and it tells you what *kind* of bowler someone is, whether he's being used in the right phase, how consistent his overs are, and whether he's getting better or worse over time. Then it writes the coaching report for you. 📋

---

## 🎬 The 30-second pitch

You've got a fast bowler. Is he a *Powerplay strike weapon*, a *Middle-overs strangler*, or a *Death-overs specialist*? Is he actually being **bowled** in the phase his skills suit? Is he leaking runs to lefties? Trending up or quietly falling off?

A scorecard won't tell you. **This will.**

```
   clean ball-by-ball table  ──►  five views  ──►  a coaching report
   (tidy, fast, correct)         (interactive)      (download & hand over)
```

---

## 🧭 The five tabs (a.k.a. the whole scouting toolkit)

<table>
<tr>
<td width="33%" valign="top">

### 🫆 Performance Fingerprint
Economy, strike rate, dot-%, boundary-% — shown as **percentiles vs his peers**, not raw numbers. High percentile = good, *everywhere*, so you never have to remember which way is up.

</td>
<td width="33%" valign="top">

### 🎭 Role Identification
**k-means** learns three bowling archetypes from the data, then checks the awkward question: is he *bowled* in the phase he's actually good at?

</td>
<td width="33%" valign="top">

### 🎯 Execution & Matchups
How tightly his overs cluster, how he takes his wickets, where the runs leak, and which oppositions have his number.

</td>
</tr>
<tr>
<td width="33%" valign="top">

### 📈 Development Trajectory
Rolling trends per metric, an **improvement index**, and a trend-vs-latest-step read on every stat. Getting better, or coasting?

</td>
<td width="33%" valign="top">

### 📋 Coaching Report
Everything above, distilled into a prioritised plan of action — and an **export button** so you can hand it over.

</td>
<td width="33%" valign="top">

### ✨ ...all filterable
Slice by **years** and **countries** in the sidebar. Your selected bowler sticks around when you change filters — it only resets if he drops out of the cohort.

</td>
</tr>
</table>

---

## 🚀 Quick start

<details open>
<summary><b>1. Install the kit</b></summary>

```bash
pip install -r requirements.txt
```
</details>

<details>
<summary><b>2. Bowl the first over</b> 🎳</summary>

```bash
streamlit run app.py
```

Then open the URL it prints, pick a bowler, and start scouting.
</details>

---

## 🧪 The rules, done properly

This isn't "plausible-looking" cricket math — the edge cases are handled so the numbers are actually *correct*:

| Concept | How it's counted |
|---|---|
| **Legal ball** | Not a wide, not a no-ball |
| **Runs charged to bowler** | Batter runs + wides + no-balls (byes / leg-byes / penalty are *not* his fault) |
| **Bowler's wicket** | bowled, caught, lbw, stumped, caught & bowled, hit wicket — run-outs & retirements don't count |
| **Dot ball** | A *legal* delivery with zero runs scored |
| **Boundary conceded** | Four or six off the bat (boundary byes excluded) |
| **Phases** | Powerplay (overs 1–6), Middle (7–15), Death (16–20) |

---

## 🛠️ Under the hood

```
🏏 Sports Analytics Tool/
├── app.py                          # the whole Streamlit app (5 tabs, single file)
├── t20_deliveries_clean.parquet    # cleaned parquet
└──  requirements.txt                # streamlit · pandas · numpy · plotly · sklearn · scipy …


```

**Stack:** Streamlit · pandas · NumPy · Plotly · scikit-learn (k-means) · SciPy · matplotlib (PDF export)

> 💡 SciPy and matplotlib are *optional* — the app degrades gracefully and tells you in the UI if they're missing.

---

<div align="center">

*Filters live behind an "Apply" button, so nothing recomputes while you're still deciding. You're welcome.* 😌

**Now go find your next death-overs specialist.** 🔥

</div>