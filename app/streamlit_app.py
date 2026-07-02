from __future__ import annotations

import html
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
from src.inference import robust_predict
from src.database import fetch_recent_runs, log_prediction

SAMPLE_DIR = ROOT / "data" / "sample_images"
RESULTS_DIR = ROOT / "eval" / "results"
FEATURED_SAMPLES = [
    ("Normal", "CXR_SYN_001_normal.png"),
    ("Opacité", "CXR_SYN_002_suspected_opacity.png"),
    ("Incertain", "CXR_SYN_003_uncertain.png"),
]

CLASS_META = {
    "normal": {"label": "Normal", "tone": "success", "bar": "#16a34a"},
    "suspected_opacity": {"label": "Opacité suspectée", "tone": "warn", "bar": "#d97706"},
    "uncertain": {"label": "Incertain", "tone": "muted", "bar": "#64748b"},
}

QUALITY_META = {
    "good": "Bonne",
    "limited": "Limitée",
    "poor": "Faible",
}


def load_catalog() -> pd.DataFrame:
    """Charge le fichier CSV du catalogue."""
    csv_path = ROOT / "data" / "synthetic_cases.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)

APP_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  :root {
    --bg: #f6f7f9;
    --surface: #ffffff;
    --border: #e4e7ec;
    --text: #111827;
    --muted: #667085;
    --primary: #1d4ed8;
    --primary-soft: #eff4ff;
    --warn-bg: #fffaeb;
    --warn-border: #fedf89;
    --warn-text: #93370d;
  }

  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }

  .stApp {
    background: var(--bg);
    color: var(--text);
  }

  .block-container {
    padding-top: 0;
    padding-bottom: 2rem;
    max-width: 1080px;
  }

  header[data-testid="stHeader"],
  #MainMenu,
  footer,
  .stDeployButton,
  section[data-testid="stSidebar"],
  button[data-testid="stSidebarCollapsedControl"],
  button[data-testid="collapsedControl"] {
    display: none !important;
  }

  .ra-shell,
  .ra-topbar,
  .ra-banner,
  .ra-card,
  .ra-result,
  .ra-empty,
  .ra-source {
    box-sizing: border-box;
    font-family: 'Inter', sans-serif;
  }

  .ra-topbar-wrap {
    margin: 0 -1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }

  .ra-topbar {
    max-width: 1080px;
    margin: 0 auto;
    padding: 1.25rem 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
  }

  .ra-topbar__brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    min-width: 0;
  }

  .ra-topbar__mark {
    width: 2.5rem;
    height: 2.5rem;
    border-radius: 0.5rem;
    background: var(--primary);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    flex-shrink: 0;
  }

  .ra-topbar__title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.2;
  }

  .ra-topbar__subtitle {
    margin-top: 0.15rem;
    font-size: 0.85rem;
    color: var(--muted);
  }

  .ra-topbar__session {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.35rem 0.65rem 0.35rem 0.45rem;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: #fafafa;
    white-space: nowrap;
  }

  .ra-topbar__status {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    background: #ecfdf3;
    color: #067647;
    font-size: 0.72rem;
    font-weight: 600;
  }

  .ra-topbar__dot {
    width: 0.45rem;
    height: 0.45rem;
    border-radius: 999px;
    background: #12b76a;
  }

  .ra-topbar__user {
    font-size: 0.875rem;
    color: var(--text);
    font-weight: 500;
  }

  .ra-banner {
    margin-bottom: 1.25rem;
    padding: 0.75rem 0.9rem;
    border: 1px solid var(--warn-border);
    border-radius: 0.65rem;
    background: var(--warn-bg);
    color: var(--warn-text);
    font-size: 0.84rem;
    line-height: 1.5;
  }

  .ra-banner strong { font-weight: 600; }

  .ra-page-title {
    margin: 0 0 0.25rem;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text);
  }

  .ra-page-subtitle {
    margin: 0 0 1.25rem;
    font-size: 0.875rem;
    color: var(--muted);
  }

  .ra-toolbar-card {
    margin-bottom: 1rem;
    padding: 0.15rem 0;
  }

  .ra-toolbar-card .ra-toolbar__label {
    margin-bottom: 0.65rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted);
  }

  div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--border) !important;
    border-radius: 0.75rem !important;
    background: var(--surface) !important;
    padding: 0.85rem 1rem 1rem !important;
    margin-bottom: 1rem !important;
  }

  .ra-toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    align-items: end;
    margin-bottom: 1rem;
    padding: 0.9rem 1rem;
    border: 1px solid var(--border);
    border-radius: 0.75rem;
    background: var(--surface);
  }

  .ra-toolbar__group {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    min-width: 0;
  }

  .ra-toolbar__label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted);
  }

  .ra-card {
    margin-bottom: 0.75rem;
    padding: 1rem 1.05rem;
    border: 1px solid var(--border);
    border-radius: 0.75rem;
    background: var(--surface);
  }

  .ra-card__title {
    margin: 0 0 0.25rem;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text);
  }

  .ra-card__text {
    margin: 0;
    font-size: 0.84rem;
    line-height: 1.55;
    color: var(--muted);
  }

  div[data-testid="stFileUploader"] {
    margin-top: 0.5rem;
    border: 2px dashed #cbd5e1;
    border-radius: 0.75rem;
    background: #fcfcfd;
    padding: 2rem 1.5rem;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  div[data-testid="stFileUploader"]:hover {
    border-color: var(--primary);
    background: var(--primary-soft);
  }

  div[data-testid="stFileUploader"] section {
    padding: 0.5rem 0.75rem !important;
    border: none !important;
    background: transparent !important;
    cursor: pointer !important;
  }

  div[data-testid="stFileUploader"] section > div {
    text-align: center !important;
  }

  div[data-testid="stFileUploader"] span {
    font-size: 0.9rem !important;
    color: var(--text) !important;
    font-weight: 500 !important;
  }

  div[data-testid="stFileUploader"]::before {
    content: "📤";
    display: block;
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
    text-align: center;
  }

  div[data-testid="stFileUploader"]::after {
    content: "Cliquez ou glissez-déposez votre radiographie ici";
    display: block;
    margin-top: 0.5rem;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text);
    text-align: center;
  }

  div[data-testid="stFileUploader"] button {
    display: none !important;
  }

  div[data-testid="stFileUploader"] small {
    display: none !important;
  }

  .ra-empty {
    padding: 2rem 1rem;
    border: 1px dashed var(--border);
    border-radius: 0.75rem;
    background: var(--surface);
    text-align: center;
    color: var(--muted);
    font-size: 0.875rem;
  }

  .ra-source {
    margin-bottom: 1rem;
    padding: 0.6rem 0.85rem;
    border: 1px solid #bfdbfe;
    border-radius: 0.65rem;
    background: var(--primary-soft);
    color: #1e40af;
    font-size: 0.84rem;
  }

  .ra-source code {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.8rem;
  }

  .ra-result {
    padding: 1rem;
    border: 1px solid var(--border);
    border-radius: 0.75rem;
    background: var(--surface);
  }

  .ra-result__head {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 0.75rem;
    align-items: flex-start;
  }

  .ra-result__eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--muted);
  }

  .ra-result__class {
    margin-top: 0.35rem;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text);
  }

  .ra-result__code {
    margin-top: 0.15rem;
    font-size: 0.78rem;
    color: var(--muted);
  }

  .ra-result__meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
  }

  .ra-tag {
    display: inline-flex;
    align-items: center;
    padding: 0.22rem 0.55rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    border: 1px solid var(--border);
    background: #f9fafb;
    color: #475467;
  }

  .ra-tag--success { color: #067647; background: #ecfdf3; border-color: #abefc6; }
  .ra-tag--warn { color: #b54708; background: #fffaeb; border-color: #fedf89; }
  .ra-tag--muted { color: #475467; background: #f2f4f7; border-color: #e4e7ec; }

  .ra-confidence {
    margin-top: 0.85rem;
  }

  .ra-confidence__head {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.35rem;
    font-size: 0.78rem;
    color: var(--muted);
  }

  .ra-confidence__value {
    font-weight: 600;
    color: var(--text);
  }

  .ra-confidence__track {
    height: 0.4rem;
    border-radius: 999px;
    background: #eef2f6;
    overflow: hidden;
  }

  .ra-confidence__fill {
    height: 100%;
    border-radius: 999px;
  }

  .ra-result__grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-top: 1rem;
  }

  @media (max-width: 768px) {
    .ra-result__grid { grid-template-columns: 1fr; }
    .ra-topbar__session { max-width: 100%; }
  }

  .ra-panel {
    padding: 0.85rem;
    border: 1px solid var(--border);
    border-radius: 0.65rem;
    background: #fcfcfd;
  }

  .ra-panel__title {
    margin-bottom: 0.4rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text);
  }

  .ra-panel__text,
  .ra-panel li {
    font-size: 0.84rem;
    line-height: 1.55;
    color: #475467;
  }

  .ra-panel ul {
    margin: 0;
    padding-left: 1rem;
  }

  .ra-alert {
    margin-top: 0.75rem;
    padding: 0.6rem 0.75rem;
    border: 1px solid #fecdca;
    border-radius: 0.55rem;
    background: #fef3f2;
    color: #b42318;
    font-size: 0.78rem;
  }

  .stRadio > div {
    gap: 0.35rem !important;
  }

  .stRadio label {
    background: #f9fafb !important;
    border: 1px solid var(--border) !important;
    border-radius: 0.5rem !important;
    padding: 0.35rem 0.75rem !important;
    font-size: 0.84rem !important;
    color: var(--text) !important;
  }

  .stRadio label span {
    color: var(--text) !important;
  }

  h3, h4 {
    color: var(--text) !important;
    font-weight: 600 !important;
  }

  button[kind="primary"],
  button[kind="secondary"] {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
  }

  button[kind="primary"]:hover,
  button[kind="secondary"]:hover {
    background: #f9fafb !important;
    border-color: var(--primary) !important;
  }

  button p {
    color: var(--text) !important;
  }
</style>
"""


def inject_ui() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def render_topbar() -> None:
    st.markdown(
        """
        <div class="ra-topbar-wrap">
          <div class="ra-topbar">
            <div class="ra-topbar__brand">
              <div class="ra-topbar__mark">RA</div>
              <div>
                <div class="ra-topbar__title">RadiAssist</div>
                <div class="ra-topbar__subtitle">Prototype EFREI · analyse radiographique</div>
              </div>
            </div>
            <div class="ra-topbar__session">
              <span class="ra-topbar__status">
                <span class="ra-topbar__dot"></span>
                Connectée
              </span>
              <span class="ra-topbar__user">Dr Claire Martin · Radiologie</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_warning_banner() -> None:
    st.markdown(
        f"""
        <div class="ra-banner">
          <strong>Usage non clinique.</strong> {html.escape(WARNING_TEXT)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def list_sample_images() -> list[Path]:
    if not SAMPLE_DIR.exists():
        return []
    return sorted(SAMPLE_DIR.glob("*.png"))


def resolve_active_image(uploaded, sample_path: Path | None) -> tuple[Path | None, str | None]:
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            return Path(tmp.name), uploaded.name
    if sample_path is not None and sample_path.exists():
        return sample_path, sample_path.name
    return None, None


def confidence_bar(confidence: float, color: str) -> str:
    pct = max(0, min(int(confidence * 100), 100))
    return f"""
    <div class="ra-confidence">
      <div class="ra-confidence__head">
        <span>Confiance</span>
        <span class="ra-confidence__value">{pct}%</span>
      </div>
      <div class="ra-confidence__track">
        <div class="ra-confidence__fill" style="width:{pct}%; background:{color};"></div>
      </div>
    </div>
    """


def render_prediction_card(pred: dict) -> None:
    meta = CLASS_META.get(pred["predicted_class"], CLASS_META["uncertain"])
    quality_label = QUALITY_META.get(pred.get("image_quality", "good"), "—")
    evidence = pred.get("visual_evidence") or []
    limitations = pred.get("limitations") or []
    guardrail_errors = pred.get("guardrail_errors") or []

    evidence_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in evidence) or (
        "<li>Aucune observation disponible.</li>"
    )
    limits_html = "".join(f'<span class="ra-tag">{html.escape(str(item))}</span>' for item in limitations)

    guardrail_block = ""
    if guardrail_errors:
        guardrail_block = (
            f'<div class="ra-alert">Garde-fous déclenchés : {html.escape(", ".join(guardrail_errors))}</div>'
        )

    st.markdown(
        f"""
        <div class="ra-result">
          <div class="ra-result__head">
            <div>
              <div class="ra-result__eyebrow">Résultat expérimental</div>
              <div class="ra-result__class">{html.escape(meta["label"])}</div>
              <div class="ra-result__code">Classe : {html.escape(pred["predicted_class"])}</div>
            </div>
            <div class="ra-result__meta">
              <span class="ra-tag ra-tag--{meta["tone"]}">{html.escape(meta["label"])}</span>
              <span class="ra-tag">Qualité {html.escape(quality_label)}</span>
              <span class="ra-tag">{html.escape(str(pred.get("latency_ms", "—")))} ms</span>
              <span class="ra-tag">{html.escape(str(pred.get("model_name", "—")))}</span>
            </div>
          </div>
          {confidence_bar(float(pred["confidence"]), meta["bar"])}
          <div class="ra-result__grid">
            <div class="ra-panel">
              <div class="ra-panel__title">Observations</div>
              <ul>{evidence_html}</ul>
            </div>
            <div class="ra-panel">
              <div class="ra-panel__title">Justification</div>
              <p class="ra-panel__text">{html.escape(str(pred.get("justification", "")))}</p>
            </div>
          </div>
          <div class="ra-panel" style="margin-top:0.75rem;">
            <div class="ra-panel__title">Limites</div>
            <div class="ra-result__meta">{limits_html}</div>
          </div>
          {guardrail_block}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_toolbar() -> str:
    with st.container(border=True):
        st.markdown('<div class="ra-toolbar__label">Paramètres d\'analyse</div>', unsafe_allow_html=True)

        col_mode, col_samples, col_catalog = st.columns([1.1, 1.2, 1.4])

        with col_mode:
            st.caption("Mode")
            mode = st.radio(
                "Mode",
                options=["baseline", "improved"],
                horizontal=True,
                format_func=lambda value: "Baseline" if value == "baseline" else "Amélioré",
                label_visibility="collapsed",
            )

        with col_samples:
            st.caption("Exemples rapides")
            sample_cols = st.columns(3)
            for idx, (label, filename) in enumerate(FEATURED_SAMPLES):
                sample_path = SAMPLE_DIR / filename
                if sample_cols[idx].button(label, use_container_width=True, disabled=not sample_path.exists()):
                    st.session_state.selected_sample = str(sample_path)
                    st.rerun()

        with col_catalog:
            st.caption("Catalogue")
            samples = list_sample_images()
            if samples:
                sample_names = [path.name for path in samples]
                current_name = (
                    Path(st.session_state.selected_sample).name
                    if st.session_state.selected_sample
                    else sample_names[0]
                )
                default_index = sample_names.index(current_name) if current_name in sample_names else 0
                picked_name = st.selectbox(
                    "Cas synthétique",
                    sample_names,
                    index=default_index,
                    label_visibility="collapsed",
                )
                if st.button("Charger", use_container_width=True):
                    st.session_state.selected_sample = str(SAMPLE_DIR / picked_name)
                    st.rerun()

    if st.session_state.selected_sample:
        name = Path(st.session_state.selected_sample).name
        if st.button(f"Retirer · {name}"):
            st.session_state.selected_sample = None
            st.rerun()

    return mode


def render_dashboard_tab() -> None:
    """Affiche la comparaison baseline vs amélioration et les logs SQLite."""
    st.header("Comparaison baseline vs amélioration")

    summary_path = RESULTS_DIR / "before_after_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        st.dataframe(summary, use_container_width=True, hide_index=True)

        metric_cols = [c for c in ("accuracy", "macro_f1", "sensitivity", "specificity", "uncertain_rate") if c in summary.columns]
        if metric_cols and "mode" in summary.columns:
            chart_df = summary.set_index("mode")[metric_cols]
            st.bar_chart(chart_df)
    else:
        st.info(
            "Aucun résultat d'évaluation trouvé. Générez-les avec :\n\n"
            "`python eval/run_evaluation.py --mode toy`"
        )

    # Matrices de confusion baseline vs amélioration
    conf_baseline = RESULTS_DIR / "baseline_confusion.csv"
    conf_improved = RESULTS_DIR / "improved_confusion.csv"
    if conf_baseline.exists() and conf_improved.exists():
        st.subheader("Matrices de confusion")
        col_a, col_b = st.columns(2)
        for col, title, path in [
            (col_a, "Baseline", conf_baseline),
            (col_b, "Amélioration", conf_improved),
        ]:
            with col:
                st.caption(title)
                cm = pd.read_csv(path).set_index("true_label")
                cm.columns = [c.replace("pred_", "") for c in cm.columns]
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
                st.pyplot(fig)

    # Analyse de sensibilité au seuil
    sweep_path = RESULTS_DIR / "threshold_sweep.csv"
    if sweep_path.exists():
        st.subheader("Sensibilité au seuil de qualité")
        st.caption("Trop bas : aucun bénéfice · trop haut : sur-abstention. Le seuil retenu (13) est dans la zone sûre.")
        sweep = pd.read_csv(sweep_path).set_index("limited_contrast_std")
        st.line_chart(sweep[["accuracy", "macro_f1", "uncertain_rate"]])

    st.header("Journal des prédictions (SQLite)")
    st.caption("Chaque prédiction de la démo est tracée : image, modèle, prompt, classe, confiance, latence.")
    runs = fetch_recent_runs(limit=50)
    if runs:
        st.dataframe(pd.DataFrame(runs), use_container_width=True, hide_index=True)
    else:
        st.info("Aucune prédiction encore journalisée. Lancez une analyse dans l'onglet « Prédiction ».")


def main() -> None:
    st.set_page_config(
        page_title="RadiAssist",
        page_icon="🩻",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = None

    inject_ui()
    render_topbar()
    render_warning_banner()

    catalog = load_catalog()

    # ── CRÉATION D'ONGLETS POUR EXPLORER LES DONNÉES ────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Prédiction", "Exploration des données", "Visualisations", "Tableau de bord & logs"]
    )

    with tab1:
        st.markdown('<h1 class="ra-page-title">Analyse radiographique</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="ra-page-subtitle">Radiographie thoracique frontale · sortie JSON structurée</p>',
            unsafe_allow_html=True,
        )

        mode = render_toolbar()

        st.markdown(
            """
            <div class="ra-card">
              <p class="ra-card__title">Importer une radiographie</p>
              <p class="ra-card__text">PNG, JPG ou JPEG · une image à la fois · données non conservées.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Importer une radiographie",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )

        sample_path = Path(st.session_state.selected_sample) if st.session_state.selected_sample else None
        image_path, image_name = resolve_active_image(uploaded, sample_path)

        if image_path is None:
            st.markdown(
                '<div class="ra-empty">Chargez une image ou selectionnez un cas synthetique pour lancer l\'analyse.</div>',
                unsafe_allow_html=True,
            )
        else:
            source_label = "Import manuel" if uploaded is not None else "Cas synthétique"
            st.markdown(
                f"""
                <div class="ra-source">
                  Source : <strong>{html.escape(source_label)}</strong>
                  · <code>{html.escape(image_name or "")}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

            image_col, result_col = st.columns([1.05, 1], gap="large")
            with image_col:
                st.markdown("#### Image")
                st.image(Image.open(image_path), caption=image_name, use_container_width=True)

            with result_col:
                st.markdown("#### Résultat")
                pred = apply_safety_guardrails(robust_predict(image_path, mode=mode))
                render_prediction_card(pred)

                # Trace de la prédiction (contrat de journalisation).
                log_prediction(
                    case_id=Path(image_name or "upload").stem,
                    image_path=str(image_path),
                    prediction=pred,
                )

                with st.expander("JSON structuré"):
                    st.code(json.dumps(pred, indent=2, ensure_ascii=False), language="json")

    # ── ONGLET 2 : EXPLORATION DES DONNÉES DU CSV ──────────────────────────────
    with tab2:
        st.header("Catalogue complet des cas")

        if catalog.empty:
            st.warning("Aucun catalogue trouvé dans data/synthetic_cases.csv")
        else:
            # Filtres interactifs
            col_filtre1, col_filtre2, col_filtre3 = st.columns(3)
            with col_filtre1:
                label_filter = st.multiselect(
                    "Filtrer par label",
                    options=catalog["label"].unique(),
                    default=catalog["label"].unique()
                )
            with col_filtre2:
                quality_filter = st.multiselect(
                    "Filtrer par qualité",
                    options=catalog["quality"].unique(),
                    default=catalog["quality"].unique()
                )
            with col_filtre3:
                split_filter = st.multiselect(
                    "Filtrer par split",
                    options=catalog["split"].unique(),
                    default=catalog["split"].unique()
                )

            # Application des filtres
            filtered_df = catalog[
                (catalog["label"].isin(label_filter)) &
                (catalog["quality"].isin(quality_filter)) &
                (catalog["split"].isin(split_filter))
            ]

            # Métriques
            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
            with col_met1:
                st.metric("Total des cas", len(filtered_df))
            with col_met2:
                st.metric("Labels uniques", filtered_df["label"].nunique())
            with col_met3:
                st.metric("Qualités uniques", filtered_df["quality"].nunique())
            with col_met4:
                st.metric("Splits uniques", filtered_df["split"].nunique())

            # Affichage du tableau
            st.subheader("Tableau des données")
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "case_id": "ID du cas",
                    "image_path": "Chemin de l'image",
                    "label": st.column_config.TextColumn("Label", help="Classification de l'image"),
                    "quality": "Qualité",
                    "split": "Split",
                    "notes": st.column_config.TextColumn("Notes", width="large")
                }
            )

            # Statistiques descriptives
            with st.expander("Statistiques descriptives"):
                st.dataframe(filtered_df.describe(include='all'), use_container_width=True)

    # ── ONGLET 3 : VISUALISATIONS ──────────────────────────────────────────────
    with tab3:
        st.header("Visualisations des données")

        if catalog.empty:
            st.warning("Aucune donnée à visualiser")
        else:
            col_viz1, col_viz2 = st.columns(2)

            with col_viz1:
                st.subheader("Distribution des labels")
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                catalog["label"].value_counts().plot(kind="bar", ax=ax1, color=['green', 'orange', 'red'])
                ax1.set_title("Distribution des classes")
                ax1.set_xlabel("Label")
                ax1.set_ylabel("Nombre de cas")
                ax1.tick_params(axis='x', rotation=0)
                st.pyplot(fig1)

            with col_viz2:
                st.subheader("Distribution des qualités")
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                catalog["quality"].value_counts().plot(kind="pie", ax=ax2, autopct='%1.1f%%')
                ax2.set_title("Répartition des qualités")
                st.pyplot(fig2)

            # Graphiques supplémentaires
            st.subheader("Distribution des splits")
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            pd.crosstab(catalog["split"], catalog["label"]).plot(kind="bar", stacked=True, ax=ax3)
            ax3.set_title("Répartition des labels par split")
            ax3.set_xlabel("Split")
            ax3.set_ylabel("Nombre de cas")
            ax3.legend(title="Label")
            ax3.tick_params(axis='x', rotation=0)
            st.pyplot(fig3)

            # Matrice de corrélation (si colonnes numériques)
            with st.expander("Analyse des corrélations (colonnes numériques)"):
                numeric_cols = catalog.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 0:
                    fig4, ax4 = plt.subplots(figsize=(10, 8))
                    sns.heatmap(catalog[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax4)
                    st.pyplot(fig4)
                else:
                    st.info("Aucune colonne numérique disponible pour la matrice de corrélation")

    # ── ONGLET 4 : TABLEAU DE BORD & LOGS ──────────────────────────────────────
    with tab4:
        render_dashboard_tab()

    # ── BONUS : EXPORT DES DONNÉES FILTRÉES ────────────────────────────────────
    if not catalog.empty:
        st.sidebar.header("Exporter les données")
        if st.sidebar.button("Télécharger les données filtrées (CSV)"):
            if 'filtered_df' in locals():
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.sidebar.download_button(
                    label="Cliquez pour télécharger",
                    data=csv,
                    file_name='donnees_filtrees.csv',
                    mime='text/csv'
                )

        st.sidebar.info("Les données du catalogue sont chargées et affichées dans les onglets 'Exploration des données' et 'Visualisations'")


if __name__ == "__main__":
    main()
