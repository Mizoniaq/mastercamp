from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.guardrails import WARNING_TEXT, apply_safety_guardrails
from src.inference import robust_predict, vlm_predict
from src.database import fetch_recent_runs, log_prediction

SAMPLE_DIR = ROOT / "data" / "sample_images"
REAL_CASES_CSV = ROOT / "data" / "real" / "real_cases.csv"
RESULTS_DIR = ROOT / "eval" / "results"
MEDGEMMA_MODEL = "google/medgemma-4b-it"

# class -> display label, badge colour and accent colour for the confidence bar.
CLASS_META = {
    "normal": {"label": "Normal", "color": "green", "emoji": "🟢", "bar": "#16a34a"},
    "suspected_opacity": {"label": "Opacité suspectée", "color": "orange", "emoji": "🟠", "bar": "#ea580c"},
    "uncertain": {"label": "Incertain", "color": "gray", "emoji": "⚪", "bar": "#64748b"},
}
QUALITY_META = {"good": "Bonne", "limited": "Limitée", "poor": "Faible"}

MODE_LABELS = {"Baseline": "baseline", "Amélioré": "improved"}
ENGINE_LABELS = {"Jouet (rapide)": "toy", "MedGemma (réel)": "medgemma"}

SShort = {  # short, readable engine names for the result line
    "toy-imgfeat-baseline": "Jouet · baseline",
    "toy-imgfeat-improved": "Jouet · amélioré",
    MEDGEMMA_MODEL: "MedGemma 4B",
}

PAGE_CSS = """
<style>
  #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1140px; }
  h1, h2, h3 { letter-spacing: -0.01em; }
  [data-testid="stMetricValue"] { font-size: 1.15rem; }
</style>
"""


# ── Données ────────────────────────────────────────────────────────────────
def load_catalog() -> pd.DataFrame:
    csv_path = ROOT / "data" / "synthetic_cases.csv"
    return pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()


def list_sample_images() -> list[Path]:
    return sorted(SAMPLE_DIR.glob("*.png")) if SAMPLE_DIR.exists() else []


def list_real_images() -> list[Path]:
    if not REAL_CASES_CSV.exists():
        return []
    df = pd.read_csv(REAL_CASES_CSV)
    return [Path(p) for p in df["image_path"].tolist() if Path(p).exists()][:60]


def image_options() -> dict[str, Path]:
    """Ordered {display label -> path} for the example picker."""
    opts: dict[str, Path] = {}
    for p in list_sample_images():
        opts[f"Synthétique · {p.name}"] = p
    for p in list_real_images():
        opts[f"Réelle · {p.parent.name}/{p.name}"] = p
    return opts


def save_upload(uploaded) -> tuple[Path, str]:
    suffix = Path(uploaded.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
    return Path(tmp.name), uploaded.name


def _confidence(pred: dict) -> float:
    try:
        return max(0.0, min(float(pred.get("confidence", 0.0)), 1.0))
    except (TypeError, ValueError):
        return 0.0


# ── Rendu ──────────────────────────────────────────────────────────────────
def render_header() -> None:
    left, right = st.columns([3, 1])
    with left:
        st.markdown("## 🩻 RadiAssist")
        st.caption("Assistant radiologue virtuel · prototype pédagogique EFREI — analyse de radiographie thoracique frontale")
    with right:
        st.markdown("")
        st.badge("Prototype · non clinique", color="red")
    st.warning(f"**Usage non clinique.** {WARNING_TEXT}", icon="⚠️")


def render_prediction_card(pred: dict) -> None:
    cls = pred.get("predicted_class", "uncertain")
    meta = CLASS_META.get(cls, CLASS_META["uncertain"])
    pct = int(round(_confidence(pred) * 100))
    model = str(pred.get("model_name", "—"))

    st.markdown(f"#### {meta['emoji']} {meta['label']}")
    st.badge(f"classe : {cls}", color=meta["color"])
    st.progress(pct / 100, text=f"Confiance : {pct}%")

    c1, c2, c3 = st.columns(3)
    c1.metric("Qualité image", QUALITY_META.get(pred.get("image_quality", "good"), "—"))
    c2.metric("Latence", f"{pred.get('latency_ms', '—')} ms")
    c3.metric("Moteur", SShort.get(model, model))

    col_obs, col_just = st.columns(2)
    with col_obs:
        st.markdown("**Observations visuelles**")
        evidence = pred.get("visual_evidence") or ["Aucune observation disponible."]
        for item in evidence:
            st.markdown(f"- {item}")
    with col_just:
        st.markdown("**Justification**")
        st.write(pred.get("justification", "—"))

    limitations = pred.get("limitations") or []
    if limitations:
        st.markdown("**Limites**")
        st.markdown(" ".join(f"`{item}`" for item in limitations))

    guardrail_errors = pred.get("guardrail_errors") or []
    if guardrail_errors:
        st.warning("Garde-fous déclenchés : " + ", ".join(map(str, guardrail_errors)), icon="🛡️")


def predict(image_path: Path, mode: str, engine: str) -> dict:
    if engine == "medgemma":
        with st.spinner("MedGemma analyse la radiographie… (~30 s ; chargement du modèle au 1er appel)"):
            try:
                return apply_safety_guardrails(vlm_predict(image_path, mode=mode, model_id=MEDGEMMA_MODEL))
            except Exception as exc:
                st.error(f"MedGemma indisponible ({exc}). Repli sur le moteur jouet.")
    return apply_safety_guardrails(robust_predict(image_path, mode=mode))


# ── Onglet Prédiction ────────────────────────────────────────────────────────
def tab_prediction() -> None:
    st.subheader("Analyse d'une radiographie")

    c_mode, c_engine = st.columns(2)
    with c_mode:
        mode_label = st.segmented_control(
            "Mode d'analyse", list(MODE_LABELS), default="Baseline",
            help="Baseline : règle simple. Amélioré : ajoute la règle d'incertitude (qualité + seuil de confiance).",
        )
    with c_engine:
        engine_label = st.segmented_control(
            "Moteur", list(ENGINE_LABELS), default="Jouet (rapide)",
            help="Jouet : classifieur d'image local instantané. MedGemma : vrai modèle médical (~30 s, GPU + HF_TOKEN).",
        )
    mode = MODE_LABELS.get(mode_label or "Baseline", "baseline")
    engine = ENGINE_LABELS.get(engine_label or "Jouet (rapide)", "toy")
    if engine == "medgemma":
        st.info("Moteur **MedGemma réel** : ~30 s par analyse, GPU et licence requis.", icon="🧠")

    st.divider()
    col_src, col_up = st.columns(2)
    with col_src:
        options = image_options()
        choice = st.selectbox("Choisir un cas d'exemple", list(options), index=0 if options else None)
    with col_up:
        uploaded = st.file_uploader("…ou importer une radiographie", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        image_path, image_name = save_upload(uploaded)
    elif options and choice:
        image_path, image_name = options[choice], choice
    else:
        st.info("Aucune image disponible.")
        return

    st.divider()
    col_img, col_res = st.columns([1, 1], gap="large")
    with col_img:
        st.markdown("**Image analysée**")
        st.image(Image.open(image_path), caption=image_name, width="stretch")
    with col_res:
        st.markdown("**Résultat**")
        pred = predict(image_path, mode, engine)
        render_prediction_card(pred)
        log_prediction(case_id=Path(image_name).stem, image_path=str(image_path), prediction=pred)
        with st.expander("Sortie JSON structurée"):
            st.json(pred)


# ── Onglet Exploration ───────────────────────────────────────────────────────
def tab_exploration(catalog: pd.DataFrame) -> None:
    st.subheader("Catalogue des cas synthétiques")
    if catalog.empty:
        st.warning("Aucun catalogue trouvé dans `data/synthetic_cases.csv`.")
        return

    f1, f2, f3 = st.columns(3)
    labels = f1.multiselect("Label", sorted(catalog["label"].unique()), default=sorted(catalog["label"].unique()))
    quals = f2.multiselect("Qualité", sorted(catalog["quality"].unique()), default=sorted(catalog["quality"].unique()))
    splits = f3.multiselect("Split", sorted(catalog["split"].unique()), default=sorted(catalog["split"].unique()))
    df = catalog[catalog["label"].isin(labels) & catalog["quality"].isin(quals) & catalog["split"].isin(splits)]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cas", len(df))
    m2.metric("Labels", df["label"].nunique())
    m3.metric("Qualités", df["quality"].nunique())
    m4.metric("Splits", df["split"].nunique())

    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("⬇️ Télécharger (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "cas_filtres.csv", "text/csv")
    with st.expander("Statistiques descriptives"):
        st.dataframe(df.describe(include="all").astype(str), width="stretch")


# ── Onglet Visualisations ────────────────────────────────────────────────────
def tab_visualisations(catalog: pd.DataFrame) -> None:
    st.subheader("Visualisations du jeu de données")
    if catalog.empty:
        st.warning("Aucune donnée à visualiser.")
        return

    palette = {"normal": "#16a34a", "suspected_opacity": "#ea580c", "uncertain": "#64748b"}
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Répartition des classes**")
        counts = catalog["label"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar(counts.index, counts.values, color=[palette.get(k, "#94a3b8") for k in counts.index])
        ax.set_ylabel("Nombre de cas"); ax.tick_params(axis="x", rotation=15)
        fig.tight_layout(); st.pyplot(fig)
    with c2:
        st.markdown("**Répartition des qualités**")
        q = catalog["quality"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.pie(q.values, labels=q.index, autopct="%1.0f%%", colors=["#60a5fa", "#fbbf24", "#f87171"])
        fig.tight_layout(); st.pyplot(fig)

    st.markdown("**Labels par split**")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    pd.crosstab(catalog["split"], catalog["label"]).plot(
        kind="bar", stacked=True, ax=ax, color=[palette.get(k, "#94a3b8") for k in sorted(catalog["label"].unique())]
    )
    ax.set_ylabel("Nombre de cas"); ax.tick_params(axis="x", rotation=0); ax.legend(title="Label")
    fig.tight_layout(); st.pyplot(fig)


# ── Onglet Tableau de bord ───────────────────────────────────────────────────
def _show_summary(title: str, path: Path, cols: list[str]) -> None:
    if not path.exists():
        return
    st.markdown(f"**{title}**")
    df = pd.read_csv(path)
    keep = [c for c in cols if c in df.columns]
    st.dataframe(df[keep], width="stretch", hide_index=True)


def tab_dashboard() -> None:
    st.subheader("Comparaison baseline vs amélioration")

    summary_path = RESULTS_DIR / "before_after_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        st.dataframe(summary, width="stretch", hide_index=True)
        metric_cols = [c for c in ("accuracy", "macro_f1", "sensitivity", "specificity", "uncertain_rate") if c in summary.columns]
        if metric_cols and "mode" in summary.columns:
            st.bar_chart(summary.set_index("mode")[metric_cols])
    else:
        st.info("Lancez `python eval/run_evaluation.py --mode toy` pour générer les résultats.")

    vlm_cols = ["mode", "accuracy", "macro_f1", "sensitivity", "specificity", "uncertain_rate", "overclaim_rate", "json_valid_rate"]
    st.divider()
    _show_summary("MedGemma — jeu synthétique", RESULTS_DIR / "vlm_before_after_summary.csv", vlm_cols)
    _show_summary("MedGemma — vraies radios (Kaggle)", RESULTS_DIR / "real_vlm" / "vlm_before_after_summary.csv", vlm_cols)

    conf_b, conf_i = RESULTS_DIR / "baseline_confusion.csv", RESULTS_DIR / "improved_confusion.csv"
    if conf_b.exists() and conf_i.exists():
        st.divider()
        st.markdown("**Matrices de confusion**")
        cols = st.columns(2)
        for col, name, path in [(cols[0], "Baseline", conf_b), (cols[1], "Amélioration", conf_i)]:
            with col:
                cm = pd.read_csv(path).set_index("true_label")
                cm.columns = [c.replace("pred_", "") for c in cm.columns]
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                ax.set_title(name); ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
                fig.tight_layout(); st.pyplot(fig)

    sweep = RESULTS_DIR / "threshold_sweep.csv"
    if sweep.exists():
        st.divider()
        st.markdown("**Sensibilité au seuil de qualité**")
        st.caption("Trop bas : aucun bénéfice · trop haut : sur-abstention. Seuil retenu : 13 (zone sûre).")
        df = pd.read_csv(sweep).set_index("limited_contrast_std")
        st.line_chart(df[["accuracy", "macro_f1", "uncertain_rate"]])

    st.divider()
    st.markdown("**Journal des prédictions (SQLite)**")
    st.caption("Chaque prédiction de la démo est tracée : image, modèle, prompt, classe, confiance, latence.")
    runs = fetch_recent_runs(limit=50)
    if runs:
        st.dataframe(pd.DataFrame(runs), width="stretch", hide_index=True)
    else:
        st.info("Aucune prédiction encore journalisée. Lancez une analyse dans l'onglet « Prédiction ».")


def main() -> None:
    st.set_page_config(page_title="RadiAssist", page_icon="🩻", layout="wide")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    render_header()
    catalog = load_catalog()
    t1, t2, t3, t4 = st.tabs(["🔬 Prédiction", "📁 Données", "📊 Visualisations", "📈 Tableau de bord"])
    with t1:
        tab_prediction()
    with t2:
        tab_exploration(catalog)
    with t3:
        tab_visualisations(catalog)
    with t4:
        tab_dashboard()


if __name__ == "__main__":
    main()
