"""Generate capstone presentation slides as a PDF."""

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import datetime

# ── Slide configuration ────────────────────────────────────────────────────
STUDENT_NAME = "Saibal Banerjee"
BITS_ID      = "2026A1C100038"
PROJECT      = "NYC Taxi Demand & Fare Forecasting"
COURSE       = "C1 AIEMLOps Capstone"
GITHUB_URL   = "github.com/saibalban2-tech/taxi-forecasting-student"
DATE         = datetime.date.today().strftime("%B %Y")

# Colour palette
DARK_BG   = "#0D1B2A"   # slide background
ACCENT    = "#00C6FF"   # cyan accent
ACCENT2   = "#FF6B35"   # orange accent
WHITE     = "#FFFFFF"
LIGHT     = "#B0C4DE"   # muted text
CARD_BG   = "#1A2B3C"   # card background

SLIDE_W, SLIDE_H = 16, 9   # inches (16:9)

# ── Helper utilities ────────────────────────────────────────────────────────

def new_slide(bg=DARK_BG):
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H))
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, SLIDE_W)
    ax.set_ylim(0, SLIDE_H)
    ax.axis("off")
    ax.set_facecolor(bg)
    return fig, ax

def accent_bar(ax, y=8.55, color=ACCENT):
    ax.axhline(y, xmin=0.04, xmax=0.96, color=color, linewidth=2.5, alpha=0.9)

def slide_title(ax, title, y=8.1, color=ACCENT, size=28):
    ax.text(0.5 * SLIDE_W, y, title,
            ha="center", va="center", color=color,
            fontsize=size, fontweight="bold",
            fontfamily="DejaVu Sans")

def body_text(ax, text, x, y, size=14, color=WHITE, ha="left", va="top", alpha=1.0):
    ax.text(x, y, text, ha=ha, va=va, color=color,
            fontsize=size, fontfamily="DejaVu Sans", alpha=alpha,
            linespacing=1.5)

def card(ax, x, y, w, h, color=CARD_BG, radius=0.3):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=color, edgecolor=ACCENT, linewidth=0.8, alpha=0.9)
    ax.add_patch(box)

def footer(ax, text=GITHUB_URL):
    ax.text(0.5 * SLIDE_W, 0.18, text,
            ha="center", va="bottom", color=LIGHT, fontsize=10,
            alpha=0.7, fontfamily="DejaVu Sans")

def phase_badge(ax, x, y, number, label, color=ACCENT):
    circle = plt.Circle((x, y), 0.38, color=color, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, str(number), ha="center", va="center",
            color=DARK_BG, fontsize=13, fontweight="bold", zorder=4)
    ax.text(x + 0.55, y, label, ha="left", va="center",
            color=WHITE, fontsize=12, fontfamily="DejaVu Sans")


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title / Identity
# ═══════════════════════════════════════════════════════════════════════════
def slide_title_intro(pdf):
    fig, ax = new_slide()

    # Decorative gradient bars on left
    for i, alpha in enumerate([0.8, 0.5, 0.3]):
        ax.axvline(0.55 + i * 0.12, ymin=0.1, ymax=0.9,
                   color=ACCENT, linewidth=3, alpha=alpha)

    # Top accent line
    accent_bar(ax, y=8.6, color=ACCENT)

    # Course label
    ax.text(SLIDE_W * 0.5, 8.1,
            COURSE.upper(),
            ha="center", va="center", color=ACCENT,
            fontsize=13, fontweight="bold",
            fontfamily="DejaVu Sans")

    # Project title
    ax.text(SLIDE_W * 0.5, 6.8, PROJECT,
            ha="center", va="center", color=WHITE,
            fontsize=32, fontweight="bold", fontfamily="DejaVu Sans")

    # Subtitle
    ax.text(SLIDE_W * 0.5, 6.0,
            "End-to-End MLOps: Data → Training → Serving → CI/CD →\n"
            "Monitoring → Responsible AI → Dashboard",
            ha="center", va="center", color=LIGHT,
            fontsize=13, fontfamily="DejaVu Sans", linespacing=1.6)

    # Identity card
    card(ax, x=3.8, y=3.2, w=8.4, h=2.1)
    ax.text(SLIDE_W * 0.5, 5.0, "Student Identity",
            ha="center", va="center", color=ACCENT,
            fontsize=11, fontweight="bold", fontfamily="DejaVu Sans")
    ax.text(SLIDE_W * 0.5, 4.5, STUDENT_NAME,
            ha="center", va="center", color=WHITE,
            fontsize=22, fontweight="bold", fontfamily="DejaVu Sans")
    ax.text(SLIDE_W * 0.5, 4.0, f"BITS ID: {BITS_ID}",
            ha="center", va="center", color=ACCENT2,
            fontsize=16, fontfamily="DejaVu Sans")
    ax.text(SLIDE_W * 0.5, 3.5, DATE,
            ha="center", va="center", color=LIGHT,
            fontsize=12, fontfamily="DejaVu Sans")

    # GitHub
    footer(ax, f"github.com/saibalban2-tech/taxi-forecasting-student")

    # Bottom accent line
    accent_bar(ax, y=0.4, color=ACCENT2)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Agenda / Phase Overview
# ═══════════════════════════════════════════════════════════════════════════
def slide_agenda(pdf):
    fig, ax = new_slide()
    accent_bar(ax)
    slide_title(ax, "Capstone Phases Overview")

    phases = [
        ("1", "Data Pipeline",       "Ingest → Clean → Aggregate → Feature Engineering"),
        ("2", "Model Training",      "Optuna HPO · LightGBM · MLflow Tracking & Registry"),
        ("3", "Model Serving",       "FastAPI · Fare · Demand · Batch Forecast · SHAP Explain"),
        ("4", "CI/CD & Docker",      "GitHub Actions · Lint · Tests · Docker Build & Run"),
        ("5", "Monitoring & Drift",  "PSI · KS Test · DriftReport · Performance Baseline"),
        ("6", "Responsible AI",      "SHAP Explainability · Fairness Analysis · Model Cards"),
        ("7", "Dashboard",           "Streamlit · Live API · Drift Panel · Explainability UI"),
        ("8", "Golden Path",         "run_all.sh · Ansible E2E Verification · Docker Deploy"),
    ]

    colors = [ACCENT, ACCENT, ACCENT, ACCENT2, ACCENT2, ACCENT2, ACCENT, ACCENT2]

    col1_x, col2_x = 0.7, 8.5
    y_start = 7.4
    row_h = 0.82

    for i, (num, label, desc) in enumerate(phases):
        col_x = col1_x if i < 4 else col2_x
        row   = i if i < 4 else i - 4
        y     = y_start - row * row_h

        card(ax, x=col_x - 0.1, y=y - 0.35, w=7.0, h=0.7, color=CARD_BG)
        phase_badge(ax, x=col_x + 0.4, y=y, number=num, label="", color=colors[i])
        ax.text(col_x + 1.0, y + 0.08, label,
                ha="left", va="center", color=WHITE,
                fontsize=12, fontweight="bold", fontfamily="DejaVu Sans")
        ax.text(col_x + 1.0, y - 0.22, desc,
                ha="left", va="center", color=LIGHT,
                fontsize=9.5, fontfamily="DejaVu Sans")

    footer(ax)
    accent_bar(ax, y=0.4, color=ACCENT2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Phase 1 & 2: Data + Training
# ═══════════════════════════════════════════════════════════════════════════
def slide_data_training(pdf):
    fig, ax = new_slide()
    accent_bar(ax)
    slide_title(ax, "Phase 1 & 2 — Data Pipeline & Model Training")

    # Phase 1 card
    card(ax, x=0.6, y=1.1, w=6.9, h=6.6)
    ax.text(1.1, 7.5, "Phase 1 · Data Pipeline", color=ACCENT,
            fontsize=14, fontweight="bold", fontfamily="DejaVu Sans")
    steps = [
        ("Ingest",     "TLC Parquet + Zone CSV → data/raw/"),
        ("Validate",   "Schema checks, cleaning bounds, CleaningReport"),
        ("Aggregate",  "Dense hourly grid: 253 zones × 744 hours"),
        ("Features",   "Demand: lags 1/2/3/24/168h + rolling stats"),
        ("           ","Fare: distance, duration, zones, calendar"),
        ("Split",      "80/10/10 chronological (no leakage)"),
        ("Tests",      "102 unit tests — all green"),
    ]
    for j, (k, v) in enumerate(steps):
        y = 7.0 - j * 0.76
        ax.text(1.1, y, f"▸ {k}", color=ACCENT2, fontsize=11,
                fontweight="bold", fontfamily="DejaVu Sans")
        ax.text(3.1, y, v, color=WHITE, fontsize=11,
                fontfamily="DejaVu Sans")

    # Phase 2 card
    card(ax, x=8.0, y=1.1, w=7.4, h=6.6)
    ax.text(8.5, 7.5, "Phase 2 · Model Training", color=ACCENT,
            fontsize=14, fontweight="bold", fontfamily="DejaVu Sans")
    training = [
        ("Algorithm",  "LightGBM  (beat XGBoost in both problems)"),
        ("HPO",        "Optuna TPE · 25 trials · val RMSE criterion"),
        ("Tracking",   "MLflow · params, metrics, artefacts logged"),
        ("Registry",   "taxi-demand-forecaster v1"),
        ("         ",  "taxi-fare-predictor v1"),
        ("Demand",     "test R² = 0.971   RMSE = 8.59 pickups/hr"),
        ("Fare",       "test R² = 0.955   RMSE = $3.47"),
        ("Explain",    "SHAP summary plots + model cards generated"),
    ]
    for j, (k, v) in enumerate(training):
        y = 7.0 - j * 0.76
        ax.text(8.5, y, f"▸ {k}", color=ACCENT2, fontsize=11,
                fontweight="bold", fontfamily="DejaVu Sans")
        ax.text(10.8, y, v, color=WHITE, fontsize=11,
                fontfamily="DejaVu Sans")

    footer(ax)
    accent_bar(ax, y=0.4, color=ACCENT2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Phase 3 & 4: Serving + CI/CD
# ═══════════════════════════════════════════════════════════════════════════
def slide_serving_cicd(pdf):
    fig, ax = new_slide()
    accent_bar(ax)
    slide_title(ax, "Phase 3 & 4 — Model Serving & CI/CD Infrastructure")

    # Phase 3 card
    card(ax, x=0.6, y=1.1, w=6.9, h=6.6)
    ax.text(1.1, 7.5, "Phase 3 · FastAPI Serving", color=ACCENT,
            fontsize=14, fontweight="bold", fontfamily="DejaVu Sans")
    endpoints = [
        ("GET  /health",          "Model status · history_hours · zones"),
        ("POST /predict/fare",    "Single-trip fare prediction"),
        ("POST /predict/demand",  "Zone-hour pickup count"),
        ("POST /forecast/demand", "Recursive multi-hour batch forecast"),
        ("POST /explain/fare",    "SHAP feature contributions"),
        ("POST /explain/demand",  "SHAP feature contributions"),
        ("GET  /metrics",         "Prometheus instrumentation"),
    ]
    for j, (ep, desc) in enumerate(endpoints):
        y = 7.0 - j * 0.78
        ax.text(1.1, y, ep, color=ACCENT2, fontsize=10,
                fontweight="bold", fontfamily="monospace")
        ax.text(5.3, y, desc, color=LIGHT, fontsize=10,
                fontfamily="DejaVu Sans")

    # Phase 4 card
    card(ax, x=8.0, y=1.1, w=7.4, h=6.6)
    ax.text(8.5, 7.5, "Phase 4 · CI/CD & Docker", color=ACCENT,
            fontsize=14, fontweight="bold", fontfamily="DejaVu Sans")
    cicd = [
        ("GitHub Actions", ".github/workflows/ci.yml"),
        ("Job: lint",      "ruff — zero warnings"),
        ("Job: unit-test", "pytest · ≥75% coverage gate"),
        ("Job: integ-test","smoke tests · no AWS required"),
        ("Job: docker-api","Dockerfile build verified"),
        ("Job: docker-dash","Dockerfile.dashboard verified"),
        ("Docker Image",   "taxi-forecasting-api:latest"),
        ("Container",      "/health → status: ok · both models loaded"),
    ]
    for j, (k, v) in enumerate(cicd):
        y = 7.0 - j * 0.76
        ax.text(8.5, y, f"▸ {k}", color=ACCENT2, fontsize=11,
                fontweight="bold", fontfamily="DejaVu Sans")
        ax.text(11.3, y, v, color=WHITE, fontsize=11,
                fontfamily="DejaVu Sans")

    footer(ax)
    accent_bar(ax, y=0.4, color=ACCENT2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Phase 5 & 6: Monitoring + Responsible AI
# ═══════════════════════════════════════════════════════════════════════════
def slide_monitoring_rai(pdf):
    fig, ax = new_slide()
    accent_bar(ax)
    slide_title(ax, "Phase 5 & 6 — Monitoring & Responsible AI")

    # Phase 5
    card(ax, x=0.6, y=1.1, w=6.9, h=6.6)
    ax.text(1.1, 7.5, "Phase 5 · Monitoring & Drift", color=ACCENT,
            fontsize=14, fontweight="bold", fontfamily="DejaVu Sans")
    mon = [
        ("Capture",   "JSON prediction logs per request (firehose-ready)"),
        ("PSI",       "Population Stability Index — quantile-binned"),
        ("KS Test",   "Kolmogorov-Smirnov two-sample per feature"),
        ("Alert",     "drifted=True when PSI ≥ 0.2"),
        ("Demand",    "demand_drift_metrics.json — 2 features drifted"),
        ("Fare",      "fare_drift_metrics.json   — 2 features drifted"),
        ("Perf",      "Live RMSE vs baseline · 20% tolerance gate"),
    ]
    for j, (k, v) in enumerate(mon):
        y = 7.0 - j * 0.78
        ax.text(1.1, y, f"▸ {k}", color=ACCENT2, fontsize=11,
                fontweight="bold", fontfamily="DejaVu Sans")
        ax.text(3.2, y, v, color=WHITE, fontsize=11,
                fontfamily="DejaVu Sans")

    # Phase 6
    card(ax, x=8.0, y=1.1, w=7.4, h=6.6)
    ax.text(8.5, 7.5, "Phase 6 · Responsible AI", color=ACCENT,
            fontsize=14, fontweight="bold", fontfamily="DejaVu Sans")
    rai = [
        ("SHAP",      "TreeExplainer — global + per-instance"),
        ("Fairness",  "5 subgroup dimensions per model"),
        ("Dims",      "Borough · Time-of-day · Weekend"),
        ("         ", "Airport zone · Passenger count"),
        ("Flagged",   "3 of 5 dimensions flagged (disparity > 1.5×)"),
        ("Cards",     "docs/model_cards/demand.md + fare.md"),
        ("Artefacts", "SHAP PNGs · fairness JSON committed"),
    ]
    for j, (k, v) in enumerate(rai):
        y = 7.0 - j * 0.78
        ax.text(8.5, y, f"▸ {k}", color=ACCENT2, fontsize=11,
                fontweight="bold", fontfamily="DejaVu Sans")
        ax.text(10.6, y, v, color=WHITE, fontsize=11,
                fontfamily="DejaVu Sans")

    footer(ax)
    accent_bar(ax, y=0.4, color=ACCENT2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 6 — Phase 7 & 8: Dashboard + Golden Path
# ═══════════════════════════════════════════════════════════════════════════
def slide_dashboard_golden(pdf):
    fig, ax = new_slide()
    accent_bar(ax)
    slide_title(ax, "Phase 7 & 8 — Dashboard & Golden Path")

    # Phase 7
    card(ax, x=0.6, y=1.1, w=6.9, h=6.6)
    ax.text(1.1, 7.5, "Phase 7 · Streamlit Dashboard", color=ACCENT,
            fontsize=14, fontweight="bold", fontfamily="DejaVu Sans")
    dash = [
        ("URL",        "http://localhost:8501"),
        ("Overview",   "Project summary + model metrics"),
        ("Demand",     "Zone selector · 24h forecast chart"),
        ("Fare",       "Live fare estimate with inputs"),
        ("Explain",    "SHAP bar chart — per-instance contributions"),
        ("Resp. AI",   "Fairness report + disparity table"),
        ("Monitoring", "Drift heatmap · KPIs · performance baseline"),
    ]
    for j, (k, v) in enumerate(dash):
        y = 7.0 - j * 0.78
        ax.text(1.1, y, f"▸ {k}", color=ACCENT2, fontsize=11,
                fontweight="bold", fontfamily="DejaVu Sans")
        ax.text(3.5, y, v, color=WHITE, fontsize=11,
                fontfamily="DejaVu Sans")

    # Phase 8
    card(ax, x=8.0, y=1.1, w=7.4, h=6.6)
    ax.text(8.5, 7.5, "Phase 8 · Golden Path", color=ACCENT,
            fontsize=14, fontweight="bold", fontfamily="DejaVu Sans")
    golden = [
        ("Entry point","./run_all.sh"),
        ("Stages",     "Pipeline → Train → Serve → Drift → RAI"),
        ("Portable",   "make_mlflow_portable.py → mlflow.container.db"),
        ("Docker",     "docker build → docker run → /health: ok"),
        ("Ansible",    "ansible-playbook verify_e2e.yml"),
        ("Result",     "All 8 phases: PASS"),
        ("Repo",       GITHUB_URL),
    ]
    for j, (k, v) in enumerate(golden):
        y = 7.0 - j * 0.78
        ax.text(8.5, y, f"▸ {k}", color=ACCENT2, fontsize=11,
                fontweight="bold", fontfamily="DejaVu Sans")
        ax.text(10.7, y, v, color=WHITE, fontsize=11,
                fontfamily="DejaVu Sans")

    footer(ax)
    accent_bar(ax, y=0.4, color=ACCENT2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# SLIDE 7 — Summary / Thank You
# ═══════════════════════════════════════════════════════════════════════════
def slide_summary(pdf):
    fig, ax = new_slide()
    accent_bar(ax, y=8.6)

    ax.text(SLIDE_W * 0.5, 8.1, COURSE.upper(),
            ha="center", va="center", color=ACCENT,
            fontsize=12, fontweight="bold", fontfamily="DejaVu Sans")

    ax.text(SLIDE_W * 0.5, 7.1, "All 8 Phases Completed",
            ha="center", va="center", color=WHITE,
            fontsize=34, fontweight="bold", fontfamily="DejaVu Sans")

    # Results grid
    results = [
        ("Phase 1", "Data Pipeline",      "102 tests green"),
        ("Phase 2", "Model Training",     "R²=0.971 / 0.955"),
        ("Phase 3", "Model Serving",      "5 live endpoints"),
        ("Phase 4", "CI/CD & Docker",     "GitHub Actions ✓"),
        ("Phase 5", "Monitoring",         "Drift reports ✓"),
        ("Phase 6", "Responsible AI",     "SHAP + Fairness ✓"),
        ("Phase 7", "Dashboard",          ":8501 live ✓"),
        ("Phase 8", "Golden Path",        "Ansible PASS ✓"),
    ]

    cols = [(0.7, 4), (8.3, 12)]
    for i, (ph, label, result) in enumerate(results):
        col_idx = 0 if i < 4 else 1
        cx, rx = cols[col_idx]
        y = 6.1 - (i % 4) * 1.05

        card(ax, x=cx - 0.1, y=y - 0.38, w=7.1, h=0.74, color=CARD_BG)
        ax.text(cx + 0.3, y, ph, color=ACCENT2,
                fontsize=11, fontweight="bold",
                ha="left", va="center", fontfamily="DejaVu Sans")
        ax.text(cx + 1.8, y, label, color=WHITE,
                fontsize=12, ha="left", va="center",
                fontfamily="DejaVu Sans")
        ax.text(rx - 0.2, y, result, color=ACCENT,
                fontsize=11, fontweight="bold",
                ha="right", va="center", fontfamily="DejaVu Sans")

    # Student name bottom
    card(ax, x=3.5, y=0.55, w=9.0, h=0.85)
    ax.text(SLIDE_W * 0.5, 1.1, f"{STUDENT_NAME}  ·  {BITS_ID}",
            ha="center", va="center", color=WHITE,
            fontsize=14, fontfamily="DejaVu Sans")
    ax.text(SLIDE_W * 0.5, 0.72, GITHUB_URL,
            ha="center", va="center", color=ACCENT,
            fontsize=11, fontfamily="DejaVu Sans")

    accent_bar(ax, y=0.38, color=ACCENT2)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Generate PDF ────────────────────────────────────────────────────────────
output_path = "docs/capstone_slides.pdf"

import os
os.makedirs("docs", exist_ok=True)

with PdfPages(output_path) as pdf:
    slide_title_intro(pdf)
    slide_agenda(pdf)
    slide_data_training(pdf)
    slide_serving_cicd(pdf)
    slide_monitoring_rai(pdf)
    slide_dashboard_golden(pdf)
    slide_summary(pdf)

    d = pdf.infodict()
    d["Title"]   = f"{PROJECT} — Capstone Presentation"
    d["Author"]  = STUDENT_NAME
    d["Subject"] = COURSE

print(f"Slides saved to: {output_path}  (7 slides)")
