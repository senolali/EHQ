"""
EHQ-3000 Figure Generator
===========================
EHQ v1'de kullanilan src/visualizer.py'nin portu (300 DPI, PDF+PNG,
akademik stil -- makale icin hazir).

DEGISIKLIKLER (v1 -> v2):
  - fig_category_ehq2: FEQ/PCQ/HNQ sabit kodlanmisti; artik CCQ dahil
    herhangi bir kategori listesiyle calisir.
  - "EHQ vs CQ" figuru (v1 fig5) KALDIRILDI: eski 7-modelin sabit CQ
    taban degerleri (Senol et al. 2026) yeni 20 modelin buyuk
    cogunluguyla isim uyusmadigi icin anlamsiz olurdu (bkz.
    exporter.py'deki ayni gerekce). CQ karsilastirmasi elde guncel
    veri olunca ayri bir adimda yapilmali.
  - Model renk paleti 7'den 20 modele genisletildi (matplotlib tab20).
"""

import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger("ehq_visualizer")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         300,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "savefig.bbox":       "tight",
    "savefig.dpi":        300,
})

try:
    _TAB20 = matplotlib.colormaps["tab20"].colors        # matplotlib >= 3.7
except AttributeError:
    _TAB20 = plt.get_cmap("tab20").colors                # matplotlib < 3.7
MODEL_COLORS = ["#%02x%02x%02x" % tuple(int(c * 255) for c in rgb) for rgb in _TAB20]

CATEGORY_COLORS = {
    "FEQ": "#E74C3C",
    "PCQ": "#2E75B6",
    "HNQ": "#27AE60",
    "CCQ": "#8E44AD",
}


def _get_ordered_models(all_results):
    models = [(n, d) for n, d in all_results.items()
              if n not in ("__correlation__",)
              and isinstance(d, dict) and "EHQ" in d]
    return sorted(models, key=lambda x: x[1].get("EHQ", 0), reverse=True)


def _save(fig, path, fmt="pdf"):
    full = str(path).replace(".pdf", f".{fmt}")
    fig.savefig(full, format=fmt, bbox_inches="tight", dpi=300)
    logger.info("Kaydedildi: %s", full)
    return full


# ─────────────────────────────────────────────────────────────
# FİGÜR 1: Genel EHQ Skor Tablosu
# ─────────────────────────────────────────────────────────────

def fig_overall_scores(all_results, out_dir):
    models = _get_ordered_models(all_results)
    if not models:
        return
    names = [n for n, _ in models]
    ehq1  = [d.get("EHQ1", 0) for _, d in models]
    ehq2  = [d.get("EHQ2", 0) for _, d in models]
    ehq3  = [d.get("EHQ3", 0) for _, d in models]
    ehq   = [d.get("EHQ",  0) for _, d in models]

    x = np.arange(len(names))
    w = 0.18
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.55), 5))

    bars = [
        ax.bar(x - 1.5*w, ehq1, w, label="EHQ₁ (Epistemic Restraint)", color="#4472C4", alpha=0.85),
        ax.bar(x - 0.5*w, ehq2, w, label="EHQ₂ (Hallucination Resistance)", color="#ED7D31", alpha=0.85),
        ax.bar(x + 0.5*w, ehq3, w, label="EHQ₃ (Confidence Alignment)", color="#70AD47", alpha=0.85),
        ax.bar(x + 1.5*w, ehq,  w, label="EHQ (Composite)", color="#1F3864", alpha=0.90),
    ]
    for group in bars:
        for bar in group:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.012, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score [0, 1]")
    ax.set_title("Figure 1. Overall EHQ Scores Across All Models")
    ax.legend(loc="upper right", framealpha=0.9, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    paths = [_save(fig, Path(out_dir) / "fig1_overall_scores.pdf", fmt) for fmt in ("pdf", "png")]
    plt.close(fig)
    return paths


# ─────────────────────────────────────────────────────────────
# FİGÜR 2: Radar Plot
# ─────────────────────────────────────────────────────────────

def fig_radar(all_results, out_dir):
    models = _get_ordered_models(all_results)
    if not models:
        return

    dims = ["EHQ1", "EHQ2", "EHQ3"]
    labels = ["EHQ₁\n(Restraint)", "EHQ₂\n(Hallucination\nResistance)", "EHQ₃\n(Calibration)"]
    N = len(dims)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    for idx, (name, d) in enumerate(models):
        vals = [d.get(dim, 0) for dim in dims]
        vals += vals[:1]
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        ax.plot(angles, vals, "o-", linewidth=1.5, color=color, label=name, alpha=0.85)
        ax.fill(angles, vals, alpha=0.04, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=7)
    ax.set_title("Figure 2. EHQ Dimensional Profiles", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.45, -0.10), framealpha=0.9, fontsize=7, ncol=1)

    plt.tight_layout()
    paths = [_save(fig, Path(out_dir) / "fig2_radar.pdf", fmt) for fmt in ("pdf", "png")]
    plt.close(fig)
    return paths


# ─────────────────────────────────────────────────────────────
# FİGÜR 3: Kategori Bazında EHQ2
# ─────────────────────────────────────────────────────────────

def fig_category_ehq2(all_results, out_dir, categories):
    models = _get_ordered_models(all_results)
    if not models:
        return

    names = [n for n, _ in models]
    n_cat = len(categories)
    x = np.arange(len(names))
    w = 0.8 / n_cat
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.55), 5))

    for i, cat in enumerate(categories):
        vals = [d.get("EHQ2_by_category", {}).get(cat, 0) or 0 for _, d in models]
        offset = (i - (n_cat - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, label=cat,
                      color=CATEGORY_COLORS.get(cat, "#888888"), alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.010, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("EHQ₂ — Hallucination Resistance")
    ax.set_title(f"Figure 3. EHQ₂ by Question Category ({', '.join(categories)})")
    ax.legend(framealpha=0.9)
    ax.axhline(y=0.80, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    paths = [_save(fig, Path(out_dir) / "fig3_category_ehq2.pdf", fmt) for fmt in ("pdf", "png")]
    plt.close(fig)
    return paths


# ─────────────────────────────────────────────────────────────
# FİGÜR 4: Response Distribution
# ─────────────────────────────────────────────────────────────

def fig_response_distribution(all_results, out_dir):
    models = _get_ordered_models(all_results)
    if not models:
        return

    names = [n for n, _ in models]
    types = ["ABSTAIN", "HEDGE", "CONFIDENT_CORRECT", "CONFIDENT_WRONG"]
    colors = ["#2E75B6", "#70AD47", "#FFC000", "#E74C3C"]
    labels_map = {
        "ABSTAIN": "Abstain",
        "HEDGE": "Hedge",
        "CONFIDENT_CORRECT": "Confident (Correct)",
        "CONFIDENT_WRONG": "Confident (Wrong / Hallucination)",
    }

    data = {t: [] for t in types}
    for _, d in models:
        dist = d.get("response_distribution", {})
        for t in types:
            data[t].append(dist.get(t, {}).get("pct", 0))

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.55), 5))

    bottom = np.zeros(len(names))
    for t, color in zip(types, colors):
        vals = np.array(data[t])
        ax.bar(x, vals, bottom=bottom, label=labels_map[t], color=color,
               alpha=0.88, edgecolor="white", linewidth=0.5)
        for i, v in enumerate(vals):
            if v >= 5:
                ax.text(x[i], bottom[i] + v/2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Percentage of Responses (%)")
    ax.set_title("Figure 4. Response Type Distribution Across Models")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    paths = [_save(fig, Path(out_dir) / "fig4_response_dist.pdf", fmt) for fmt in ("pdf", "png")]
    plt.close(fig)
    return paths


# ─────────────────────────────────────────────────────────────
# FİGÜR 5: Kalibrasyon (confidence vs accuracy scatter)
# ─────────────────────────────────────────────────────────────

def fig_calibration(all_results, out_dir):
    models = _get_ordered_models(all_results)
    if not models:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Perfect calibration")

    for idx, (name, d) in enumerate(models):
        raw = d.get("raw_results", [])
        conf_res = [r for r in raw
                    if r.get("response_type") in ("CONFIDENT_CORRECT", "CONFIDENT_WRONG")]
        if not conf_res:
            continue
        avg_conf = sum(r["confidence"] for r in conf_res) / len(conf_res)
        n_corr = sum(1 for r in conf_res if r.get("response_type") == "CONFIDENT_CORRECT")
        avg_acc = n_corr / len(conf_res)

        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        ax.scatter(avg_conf, avg_acc, color=color, s=90, zorder=5)
        ax.annotate(name, (avg_conf, avg_acc), textcoords="offset points",
                    xytext=(6, 4), fontsize=7, color=color)
        ax.annotate("", xy=(avg_conf, avg_acc), xytext=(avg_conf, avg_conf),
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.4, lw=1.0))

    ax.set_xlabel("Average Confidence (expressed)")
    ax.set_ylabel("Average Accuracy (actual)")
    ax.set_title("Figure 5. Confidence–Accuracy Calibration\n(CONFIDENT responses only)")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.04, color="red")
    ax.text(0.80, 0.20, "Overconfident\nregion", fontsize=8, color="red", alpha=0.6, ha="center")
    ax.legend(fontsize=8)

    plt.tight_layout()
    paths = [_save(fig, Path(out_dir) / "fig5_calibration.pdf", fmt) for fmt in ("pdf", "png")]
    plt.close(fig)
    return paths


# ─────────────────────────────────────────────────────────────
# ANA FONKSIYON
# ─────────────────────────────────────────────────────────────

def generate_all_figures(all_results: dict, figures_dir: str, categories: list) -> list:
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    funcs = [
        ("Fig1 Overall Scores", lambda r, d: fig_overall_scores(r, d)),
        ("Fig2 Radar", lambda r, d: fig_radar(r, d)),
        ("Fig3 Category EHQ2", lambda r, d: fig_category_ehq2(r, d, categories)),
        ("Fig4 Response Distribution", lambda r, d: fig_response_distribution(r, d)),
        ("Fig5 Calibration", lambda r, d: fig_calibration(r, d)),
    ]

    generated = []
    for name, func in funcs:
        try:
            result = func(all_results, figures_dir)
            if result:
                generated.extend(result)
            logger.info("✓ %s", name)
        except Exception as e:
            logger.error("✗ %s: %s", name, e, exc_info=True)

    logger.info("Toplam %d dosya üretildi: %s", len(generated), figures_dir)
    return generated
