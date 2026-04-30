"""Generate system architecture diagram PNG for PakSentinel FastAPI."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 15))
ax.set_xlim(0, 14)
ax.set_ylim(0, 15)
ax.axis("off")

C_CLIENT = "#4A90E2"
C_API = "#2ECC71"
C_MIDDLEWARE = "#F5B041"
C_RATE = "#EB984E"
C_ENDPOINT = "#85C1E9"
C_LIFESPAN = "#A569BD"
C_ARTIFACTS = "#E74C3C"
C_ARTIFACT_BOX = "#FADBD8"
C_TEXT = "#1B2631"


def box(x, y, w, h, color, text, fontsize=11, weight="normal", text_color=C_TEXT, edge=None):
    edge = edge or color
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            linewidth=1.5, edgecolor=edge, facecolor=color, alpha=0.92,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, weight=weight, wrap=True)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=22, linewidth=2.2, color="#2C3E50",
    ))


# Title
ax.text(7, 14.5, "PakSentinel — FastAPI Inference System Architecture",
        ha="center", fontsize=16, weight="bold", color=C_TEXT)

# Client
box(4.5, 13.1, 5, 0.9, C_CLIENT, "Client (HTTP)", fontsize=13, weight="bold", text_color="white")

# FastAPI container outer
ax.add_patch(FancyBboxPatch(
    (0.4, 3.0), 13.2, 9.4,
    boxstyle="round,pad=0.05,rounding_size=0.25",
    linewidth=2.2, edgecolor=C_API, facecolor="#EAFAF1", alpha=0.55,
))
ax.text(7, 12.05, "FastAPI Application", ha="center", fontsize=14,
        weight="bold", color="#1E8449")

# Middleware
box(0.9, 11.0, 12.2, 0.85, C_MIDDLEWARE,
    "RequestLoggingMiddleware  (console + rotating file, X-Processing-Time header)",
    fontsize=11, weight="bold", text_color="white")

# Rate limiter
box(0.9, 9.95, 12.2, 0.85, C_RATE,
    "Rate Limiter (slowapi)   |   /classify: 100/min   |   /classify/batch: 10/min",
    fontsize=11, weight="bold", text_color="white")

# Endpoints heading
ax.text(7, 9.55, "Endpoints", ha="center", fontsize=12, weight="bold", color=C_TEXT)

endpoints = [
    ("GET  /health",            "Model name, version, stage, F1, load timestamp"),
    ("POST /preprocess",        "NLP pipeline steps (tokens, removed stopwords, time)"),
    ("POST /classify",          "Single prediction + class probs + top features"),
    ("POST /classify/batch",    "Up to 100 texts, full batch < 500 ms"),
    ("POST /retrieve/similar",  "Top-k similar fact-checked claims (cosine)"),
    ("GET  /model/performance", "Live metrics & version history from MLflow"),
]
y = 9.0
for ep, desc in endpoints:
    box(0.9, y - 0.55, 4.4, 0.6, C_ENDPOINT, ep, fontsize=10.5, weight="bold")
    box(5.4, y - 0.55, 7.7, 0.6, "#FFFFFF", desc, fontsize=10, edge="#85C1E9")
    y -= 0.7

# Lifespan
box(0.9, 3.2, 12.2, 0.7, C_LIFESPAN,
    "Lifespan Context Manager  →  Load model, vectorizer, encoder once at startup",
    fontsize=11, weight="bold", text_color="white")

# Artifacts container
ax.add_patch(FancyBboxPatch(
    (0.4, 0.4), 13.2, 2.3,
    boxstyle="round,pad=0.05,rounding_size=0.2",
    linewidth=2, edgecolor=C_ARTIFACTS, facecolor="#FDEDEC", alpha=0.7,
))
ax.text(7, 2.4, "Model Artifacts (disk)", ha="center", fontsize=13,
        weight="bold", color="#922B21")

artifacts = [
    "best_model.pkl  (L2 Logistic Regression)",
    "tfidf_vectorizer.pkl",
    "label_encoder.pkl",
    "tfidf_matrix.pkl  (similarity search)",
    "metrics.pkl",
]
ax_y = 1.85
for a in artifacts:
    box(0.9, ax_y - 0.3, 12.2, 0.32, C_ARTIFACT_BOX, a, fontsize=10, edge="#E74C3C")
    ax_y -= 0.34

# Arrows: Client -> FastAPI
arrow(7, 13.08, 7, 12.45)
# FastAPI -> Artifacts
arrow(7, 3.0, 7, 2.72)

plt.tight_layout()
out = "reports/figures/system_architecture.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print("Saved:", out)
