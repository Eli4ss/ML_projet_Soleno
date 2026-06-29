"""Contenu des pages (partagé entre app.py et pages/)."""
from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    CASCADE_APPROACH_HELP,
    CASCADE_APPROACH_LABELS,
    CASCADE_FINAL_TARGETS,
    CASCADE_MECHANICAL_TARGETS,
    CASCADE_THERMAL_TARGETS,
    DEPLOYMENT_STATUS_LABELS,
    FILTER_COLUMNS,
    MVP_TARGETS,
    PATHS,
    PIPELINE_STEPS,
    PREDICTION_LOGS_DIR,
    TARGETS,
)
from src.data_loader import (
    dataset_summary,
    load_cleaned,
    load_feature_decision,
    load_feature_engineered,
    load_metrics_bundle,
    load_registry,
    load_target_distribution_summary,
    load_target_outlier_summary,
    load_target_quality_flags_summary,
    load_target_valid,
    load_unified_evaluation,
)
from src.explainability import compute_live_importance, get_importance_table
from src.lot_workflow import enrich_batch_predictions, export_column_rename, predict_lot_with_metadata
from src.prediction import (
    predict_batch,
    predict_cascade_single,
    cascade_approaches_available,
    resolve_model_path,
    target_unit,
)
from src.prediction_logs import append_prediction_log
from src.preprocessing import prepare_uploaded_df
from src.quality_check import check_lot_quality, load_input_rules
from src.target_readiness import load_target_readiness
from src.ui_mode import is_expert, render_mode_sidebar
from src.utils import file_status, show_missing_file
from src.outlier_viz import LOG_SCALE_SUGGESTED, apply_view_filter, column_summary_stats, top_extreme_rows
from src.visualization import (
    bar_metrics,
    correlation_heatmap,
    deployment_status_chart,
    distribution_dual_histogram,
    ecdf_comparison,
    parity_plot,
    reliability_comparison_bars,
    scatter_colored,
    status_breakdown_bar,
    violin_boxplot_by_group,
)


def _apply_context_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Re-apply the active sidebar context-filter selectbox values to any dataframe."""
    for _label, col in FILTER_COLUMNS.items():
        if col in df.columns:
            sel = st.session_state.get(f"filter_{col}", "Tous")
            if sel and sel != "Tous":
                df = df[df[col].astype(str) == sel]
    return df


def _get_viz_df(
    col: str,
    fallback: pd.DataFrame,
    full_size: int,
) -> tuple[pd.DataFrame, str]:
    """Return the best dataframe for visualising *col* and a source caption.

    For target columns the validated Phase-2 dataset is preferred
    (invalid rows already removed, non-null rows only). For feature columns
    the caller-supplied *fallback* dataframe is returned unchanged.
    """
    if col in TARGETS:
        vdf = load_target_valid(col)
        if vdf is not None and not vdf.empty:
            vdf = _apply_context_filters(vdf)
            caption = (
                f"Source : **données validées phase 2** — "
                f"{len(vdf):,} lots avec `{col}` non-null et physiquement valide "
                f"(sur {full_size:,} lots NAV au total)"
            )
            return vdf, caption
    caption = f"Source : dataset feature-engineered complet — {len(fallback):,} lots"
    return fallback, caption


def _render_model_readiness_section() -> None:
    st.subheader("Fiabilité des modèles par cible")
    st.caption(
        "Toutes les cibles n'ont pas le même niveau de confiance — certains tests sont rares dans le dataset NAV. "
        "Utilisez cette table pour décider quelles prédictions méritent une vérification labo."
    )
    ready = load_target_readiness()
    if ready.empty:
        st.info("Table de disponibilité non disponible — valeurs par défaut du MVP.")
        return
    display = ready.copy()
    if "deployment_label" in display.columns:
        display["Statut déploiement"] = display["deployment_label"]
    display["target"] = display["target"].map(
        lambda t: TARGETS.get(t, {}).get("label", t)
    )
    rename_map = {
        "status": "Statut scientifique",
        "usage_recommended": "Usage recommandé",
        "confidence_prior": "Confiance a priori",
        "notes": "Notes",
        "Statut déploiement": "Statut déploiement",
    }
    renamed = display.rename(columns=rename_map)
    cols = [c for c in [
        "target", "Statut déploiement", "Statut scientifique",
        "Usage recommandé", "Confiance a priori", "Notes",
    ] if c in renamed.columns]
    st.dataframe(renamed[cols], use_container_width=True, hide_index=True)


def render_home() -> None:
    render_mode_sidebar()
    st.title("Soleno NAV — Prédiction des résines recyclées")
    st.markdown(
        "**Objectif :** prédire les propriétés finales d'un lot de résine recyclée "
        "dès la **réception** — avant les tests laboratoire complets — pour permettre "
        "un tri, classement et orientation rapide par l'équipe R&D."
    )

    st.info(
        "**Comment ça marche ?** À partir de mesures simples réalisées à la réception "
        "(MI, HLMI, densité, DSC…), les modèles ML estiment les propriétés qui nécessitent "
        "normalement des tests longs et coûteux (OIT, NCLS, UCLS, flexion…). "
        "Consultez la page **Fiabilité des modèles** pour savoir quelles cibles sont déployables "
        "et dans quelles conditions (nouveau fournisseur, grade, site).",
        icon="ℹ️",
    )

    df = load_feature_engineered()
    summary = dataset_summary(df)
    reg = load_registry()

    st.markdown("---")
    st.subheader("Dataset NAV")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Lots de résine",
        f"{summary['rows']:,}" if summary["rows"] else "—",
        help="Nombre total de lots dans le dataset NAV (Business Central)",
    )
    c2.metric(
        "Variables disponibles",
        summary["cols"] or "—",
        help="Features brutes + features dérivées (FRR, log_mi, charge_total…)",
    )
    c3.metric(
        "Modèles enregistrés",
        len(reg) if not reg.empty else "—",
        help="Modèles .joblib entraînés et sauvegardés dans outputs/models/",
    )
    c4.metric(
        "Cibles prédictibles",
        len(MVP_TARGETS),
        help="OIT, NCLS, UCLS, IZOD, Traction, Allongement, Flexion",
    )

    # Cibles avec disponibilité
    st.markdown("---")
    st.subheader("Cibles prédites et disponibilité des données")
    st.caption(
        "Chaque cible n'est pas mesurée sur tous les lots — le taux de disponibilité "
        "influence directement la qualité du modèle."
    )
    target_cols = st.columns(len(MVP_TARGETS))
    for col, t in zip(target_cols, MVP_TARGETS):
        info = TARGETS.get(t, {})
        labeled = summary.get("targets_labeled", {}).get(t, "—")
        col.metric(
            label=info.get("label", t),
            value=f"{labeled}",
            help=f"Unité : {info.get('unit','—')} — {labeled} lots avec mesure disponible",
        )

    # Approches de modélisation
    st.markdown("---")
    st.subheader("Approches de modélisation disponibles")

    cascade_avail = cascade_approaches_available()
    model_rows = [
        {
            "Approche": "Modèle A",
            "Entrées": "Matière (MI, HLMI, densité, DSC, charges)",
            "Usage": "Prédiction rapide dès réception, sans info fournisseur",
            "Statut": "✅ Disponible",
        },
        {
            "Approche": "Modèle B",
            "Entrées": "Matière + Contexte (fournisseur, site, grade, date)",
            "Usage": "Plus précis si le fournisseur est connu",
            "Statut": "✅ Disponible",
        },
        {
            "Approche": CASCADE_APPROACH_LABELS["cascade"],
            "Entrées": "Matière → OIT → Méca → NCLS/UCLS (intermédiaires mesurés)",
            "Usage": "Optimiste — utile si mesures labo partielles disponibles",
            "Statut": "✅ Disponible" if "cascade" in cascade_avail else "⚠️ Lancer run_phase4.py",
        },
        {
            "Approche": CASCADE_APPROACH_LABELS["pspp"],
            "Entrées": "Chaîne entièrement prédite (évaluation OOF — sans fuite)",
            "Usage": "Inférence sans mesures labo intermédiaires — erreurs propagées",
            "Statut": "✅ Disponible" if "pspp" in cascade_avail else "⚠️ Lancer run_phase4.py",
        },
    ]
    st.dataframe(pd.DataFrame(model_rows), hide_index=True, use_container_width=True)

    if not cascade_avail:
        st.warning(
            "Les modèles Cascade et PSPP ne sont pas encore entraînés. "
            "Exécutez `python run_phase4.py` pour les activer.",
            icon="⚠️",
        )

    # Pipeline
    st.markdown("---")
    st.subheader("Statut du pipeline")
    st.caption("Chaque phase génère des fichiers de sortie détectés automatiquement.")
    status = file_status()
    phase_map = {
        "Audit NAV": "feature_engineered",
        "Nettoyage & standardisation": "cleaned",
        "Feature engineering": "feature_engineered",
        "Datasets par cible": "feature_engineered",
        "Baselines ML": "baseline_reg",
        "Deep Learning": "dl_reg",
        "Évaluation standard": "baseline_reg",
        "Validation des cibles (phase 2)": "registry_phase2",
        "Modélisation après correction": "ml_after_cleaning",
        "Validation robuste (phase 3)": "robust_validation",
        "Cascade / PSPP (phase 4)": "cascade_all_results",
        "Évaluation unifiée (phase 5)": "deployment_status",
    }
    cols_pipe = st.columns(2)
    for i, step in enumerate(PIPELINE_STEPS):
        key = phase_map.get(step)
        ok = status.get(key, False) if key else False
        icon = "✅" if ok else "⬜"
        cols_pipe[i % 2].markdown(f"{icon} {step}")

    with st.expander("Détail des fichiers détectés"):
        for k, ok in sorted(status.items()):
            st.write(f"{'✅' if ok else '❌'} `{k}` → `{PATHS.get(k, '')}`")

    st.markdown("---")
    _render_model_readiness_section()

    if not status.get("feature_engineered"):
        st.error(
            "Dataset enrichi absent — exécutez `python run_pipeline.py` "
            "à la racine du projet pour démarrer.",
            icon="🚨",
        )

    st.caption(f"Journal des prédictions : `{PREDICTION_LOGS_DIR / 'prediction_history.csv'}`")


def render_data_explorer() -> None:
    render_mode_sidebar()
    st.title("Explorateur de données")
    st.markdown(
        "Explorez le dataset NAV pour comprendre la distribution des propriétés, "
        "identifier les lots atypiques et analyser les corrélations entre variables."
    )

    choice = st.radio(
        "Source des données",
        ["Données enrichies", "Données nettoyées"],
        horizontal=True,
        help=(
            "**Enrichies** : dataset complet avec toutes les features dérivées (FRR, log_mi…) — recommandé.\n\n"
            "**Nettoyées** : données après standardisation des colonnes et détection d'outliers, "
            "avant feature engineering."
        ),
    )
    df = load_feature_engineered() if choice == "Données enrichies" else load_cleaned()

    if df is None or df.empty:
        show_missing_file("dataset", PATHS["feature_engineered"])
        return

    c1, c2 = st.columns(2)
    c1.metric("Lots disponibles", f"{len(df):,}", help="Nombre total de lignes dans le dataset")
    c2.metric("Variables", len(df.columns), help="Features brutes + features dérivées")

    tab1, tab2, tab3 = st.tabs(["Aperçu", "Qualité des données", "Visualisations"])

    with tab1:
        st.caption(
            "Les 200 premières lignes du dataset. Les colonnes en gris clair "
            "sont des features dérivées calculées automatiquement (FRR, log_mi, charge_total…)."
        )
        st.dataframe(df.head(200), use_container_width=True)
        with st.expander("Liste complète des colonnes et types"):
            st.dataframe(pd.DataFrame({"colonne": df.columns, "dtype": df.dtypes.astype(str).values}))

    with tab2:
        st.info(
            "Les cibles (OIT, NCLS, UCLS…) ont des taux de manquants élevés car tous les tests "
            "ne sont pas réalisés sur chaque lot. C'est **normal** : chaque modèle est entraîné "
            "uniquement sur les lots où la mesure est disponible.",
            icon="ℹ️",
        )
        miss = df.isna().sum().sort_values(ascending=False)
        st.subheader("Valeurs manquantes par colonne")
        st.dataframe(
            pd.DataFrame({"colonne": miss.index, "manquantes": miss.values, "taux_%": (miss / len(df) * 100).round(1)}),
            use_container_width=True,
            height=320,
        )

        flag_cols = [c for c in ("is_outlier_feature", "is_suspicious_value", "is_duplicate_lot") if c in df.columns]
        if flag_cols:
            st.subheader("Flags qualité pipeline (étape 3)")
            rows = []
            for fc in flag_cols:
                n = int(df[fc].fillna(False).astype(bool).sum())
                rows.append({"flag": fc, "lots_concernés": n, "taux_%": round(100 * n / len(df), 2)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        outlier_sum = load_target_outlier_summary()
        dist_sum = load_target_distribution_summary()
        if not outlier_sum.empty:
            st.subheader("Outliers par cible (phase 2)")
            st.caption("Comptages IQR, z-score robuste et hors plage physique — voir `outputs/14_target_validation/`.")
            display_o = outlier_sum.rename(
                columns={
                    "target_name": "Cible",
                    "n_iqr": "Outliers IQR",
                    "n_robust_z": "Outliers z robuste",
                    "n_below_min": "Sous min phys.",
                    "n_above_max": "Au-dessus max phys.",
                }
            )
            st.dataframe(display_o, use_container_width=True, hide_index=True)

        if not dist_sum.empty:
            with st.expander("Statistiques de distribution par cible (min, max, médiane, P95…)"):
                show_cols = [
                    c
                    for c in (
                        "target_name", "available_count", "min", "max", "median",
                        "p5", "p95", "n_iqr_outliers", "skewness",
                    )
                    if c in dist_sum.columns
                ]
                st.dataframe(dist_sum[show_cols], use_container_width=True, hide_index=True)

        qflags = load_target_quality_flags_summary()
        if not qflags.empty:
            with st.expander("Synthèse flags qualité cibles (phase 2)"):
                st.dataframe(qflags, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("#### Filtres contexte")
        st.caption(
            "Ces filtres s'appliquent à **tous les graphes** de cet onglet. "
            "Utilisez-les pour comparer un fournisseur ou un grade spécifique."
        )
        fdf = df.copy()
        for label, col in FILTER_COLUMNS.items():
            if col in fdf.columns:
                opts = ["Tous"] + sorted(fdf[col].dropna().astype(str).unique().tolist())[:200]
                sel = st.selectbox(label.capitalize(), opts, key=f"filter_{col}")
                if sel != "Tous":
                    fdf = fdf[fdf[col].astype(str) == sel]

        st.markdown("#### Options d'affichage")
        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            view_mode = st.selectbox(
                "Jeu affiché",
                [
                    ("all", "Toutes les données"),
                    ("zoom_pct", "Zoom P5–P95"),
                    ("no_pipeline_outliers", "Sans outliers pipeline"),
                    ("physical_valid", "Plage physique valide"),
                ],
                format_func=lambda x: x[1],
                key="viz_view_mode",
            )[0]
        with vc2:
            corr_method = st.selectbox("Corrélation", ["pearson", "spearman"], key="viz_corr_method")
        with vc3:
            winsor_corr = st.checkbox("Winsoriser corrélations (P1–P99)", value=True, key="viz_winsor")

        num_cols = fdf.select_dtypes(include="number").columns.tolist()
        target_num = [c for c in MVP_TARGETS if c in num_cols]

        st.markdown("---")
        st.markdown("#### Distribution & ECDF")
        st.caption(
            "**Histogramme** : montre la forme de la distribution. "
            "**ECDF** : indique quelle proportion de lots est en dessous d'une valeur donnée — "
            "utile pour définir des seuils de tri. "
            "Pour les cibles (OIT, NCLS…), les données validées Phase 2 sont utilisées automatiquement "
            "(outliers physiques exclus)."
        )
        dc1, dc2 = st.columns(2)
        with dc1:
            hist_col = st.selectbox(
                "Variable",
                num_cols,
                index=num_cols.index("oit_min") if "oit_min" in num_cols else (num_cols.index("mi") if "mi" in num_cols else 0),
                key="viz_hist_col",
            )
        with dc2:
            use_log = st.checkbox(
                "Échelle log1p",
                value=hist_col in LOG_SCALE_SUGGESTED,
                key="viz_log_scale",
                help="Recommandé pour OIT, IZOD et cibles très asymétriques.",
            )

        base_df, src_caption = _get_viz_df(hist_col, fdf, len(df))
        st.caption(src_caption)

        plot_df = apply_view_filter(base_df, hist_col, view_mode)
        stats = column_summary_stats(base_df[hist_col], hist_col)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Valeurs", f"{stats.get('n', 0):,}")
        m2.metric("Médiane", f"{stats.get('median', 0):.3g}" if stats.get("n") else "—")
        m3.metric("Max", f"{stats.get('max', 0):.3g}" if stats.get("n") else "—")
        m4.metric("Outliers IQR", stats.get("iqr_outliers", "—"))

        fig_h = distribution_dual_histogram(plot_df, hist_col, log_scale=use_log)
        if fig_h:
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("Pas assez de données pour l'histogramme.")

        ec1, ec2 = st.columns([2, 1])
        with ec1:
            fig_e = ecdf_comparison(plot_df, hist_col)
            if fig_e:
                st.plotly_chart(fig_e, use_container_width=True)
        with ec2:
            fig_sb = status_breakdown_bar(base_df, hist_col)
            if fig_sb:
                st.plotly_chart(fig_sb, use_container_width=True)

        with st.expander("Top 10 valeurs extrêmes (haut et bas)"):
            t1, t2 = st.columns(2)
            with t1:
                st.caption("Plus hautes valeurs")
                st.dataframe(top_extreme_rows(base_df, hist_col, n=10, ascending=False), hide_index=True)
            with t2:
                st.caption("Plus basses valeurs")
                st.dataframe(top_extreme_rows(base_df, hist_col, n=10, ascending=True), hide_index=True)

        st.markdown("---")
        st.markdown("#### Comparaison par groupe")
        st.caption(
            "**Lecture du violon :** la largeur représente la densité de lots à cette valeur. "
            "Un violon large en bas = beaucoup de lots avec de faibles valeurs. "
            "Un boxplot étroit avec une médiane haute = fournisseur ou grade de qualité homogène."
        )
        gc1, gc2 = st.columns(2)
        with gc1:
            box_val = st.selectbox(
                "Variable",
                target_num or num_cols[:1],
                format_func=lambda t: TARGETS.get(t, {}).get("label", t),
                key="viz_box_val",
            )
        with gc2:
            box_grp = st.selectbox(
                "Grouper par",
                [c for c in FILTER_COLUMNS.values() if c in fdf.columns],
                key="viz_box_grp",
            )
        box_base_df, box_caption = _get_viz_df(box_val, fdf, len(df))
        st.caption(box_caption)
        box_df = apply_view_filter(box_base_df, box_val, view_mode)
        fig_v = violin_boxplot_by_group(box_df, box_val, box_grp)
        if fig_v:
            st.plotly_chart(fig_v, use_container_width=True)
        else:
            st.info("Données insuffisantes pour le violon / boxplot.")

        st.markdown("---")
        st.markdown("#### Nuage de points")
        st.caption(
            "Explorez les relations entre deux variables. "
            "**Conseil :** essayez MI vs OIT, densité vs flexion, ou FRR vs NCLS pour voir les corrélations clés. "
            "Les points colorés signalent des valeurs atypiques à investiguer."
        )
        sc1, sc2 = st.columns(2)
        with sc1:
            x_sc = st.selectbox("Axe X", num_cols, key="sx")
        with sc2:
            y_sc = st.selectbox("Axe Y", num_cols, key="sy", index=min(1, len(num_cols) - 1))
        hover = [c for c in ("numero", "supplier_code", "description_fr", "location") if c in fdf.columns]
        fig_s = scatter_colored(fdf, x_sc, y_sc, hover_cols=hover)
        if fig_s:
            st.plotly_chart(fig_s, use_container_width=True)
            st.caption(
                "Couleurs : bleu = normal, orange = hors plage plausible, rouge = suspect, "
                "violet = outlier IQR, jaune = flag pipeline. Axes zoomés P1–P99."
            )

        st.markdown("---")
        st.markdown("#### Matrice de corrélation")
        st.caption(
            "Valeurs de -1 à +1. **+1** : les deux variables augmentent ensemble. "
            "**-1** : quand l'une monte, l'autre descend. **0** : pas de relation linéaire. "
            "La méthode **Spearman** est plus robuste aux outliers que Pearson."
        )
        default_corr = [c for c in ["mi", "hlmi", "density_g_cm3", "oit_min", "izod", "flexion", "carbon_black", "ash"] if c in num_cols]
        corr_cols = st.multiselect("Variables à inclure", num_cols, default=default_corr or num_cols[:8], key="viz_corr_cols")
        if len(corr_cols) >= 2:
            corr_df = apply_view_filter(fdf, None, view_mode)
            fig_c = correlation_heatmap(
                corr_df,
                corr_cols,
                method=corr_method,
                winsorize=winsor_corr,
            )
            if fig_c:
                st.plotly_chart(fig_c, use_container_width=True)


def _model_selector_with_cascade() -> tuple[str, str]:
    """Retourne (model_family, model_version_or_approach).

    model_family : "classic" | "cascade"
    second value  : "A" | "B" | "cascade" | "pspp"
    """
    cascade_avail = cascade_approaches_available()
    options = ["A", "B"]
    labels = {
        "A": "Modèle A — Matière",
        "B": "Modèle B — Matière + Contexte",
    }
    if "cascade" in cascade_avail:
        options.append("cascade")
        labels["cascade"] = "Cascade — Blocs Thermique→Méca→Perf"
    if "pspp" in cascade_avail:
        options.append("pspp")
        labels["pspp"] = CASCADE_APPROACH_LABELS["pspp"]

    helps = {
        "A": "Utilise uniquement les propriétés physico-chimiques de la matière (MI, HLMI, densité, DSC, charges). Rapide, sans données contextuelles.",
        "B": "Ajoute le fournisseur, le site, le grade et l'année de réception. Plus précis si ces informations sont disponibles.",
        "cascade": "Prédit en 3 étapes : d'abord OIT (thermique), puis traction/flexion/IZOD (mécanique), puis NCLS/UCLS (performance). Chaque bloc utilise les résultats du précédent.",
        "pspp": CASCADE_APPROACH_HELP["pspp"],
    }

    choice = st.radio(
        "Approche de modélisation",
        options,
        horizontal=True,
        format_func=lambda x: labels.get(x, x),
    )
    st.caption(helps.get(choice, ""))

    family = "cascade" if choice in ("cascade", "pspp") else "classic"
    return family, choice


def _render_cascade_single_result(
    preds: dict,
    approach: str,
    measured_inputs: dict | None = None,
) -> None:
    """Affiche les prédictions cascade par bloc.

    Les valeurs provenant de mesures réelles sont distinguées visuellement
    des valeurs prédites par le modèle.
    """
    measured_inputs = measured_inputs or {}
    bloc_defs = [
        ("Bloc 1 — Structure thermique", CASCADE_THERMAL_TARGETS),
        ("Bloc 2 — Structure mécanique", CASCADE_MECHANICAL_TARGETS),
        ("Bloc 3 — Performance finale", CASCADE_FINAL_TARGETS),
    ]
    approach_label = CASCADE_APPROACH_LABELS.get(approach, approach)
    st.markdown(f"#### Résultats — {approach_label}")
    st.caption("**Mesuré** = valeur laboratoire fournie · **Prédit** = estimation du modèle")

    cols = st.columns(3)
    for col, (bloc_title, bloc_targets) in zip(cols, bloc_defs):
        with col:
            st.markdown(f"**{bloc_title}**")
            any_val = False
            for t in bloc_targets:
                val = preds.get(t)
                label = TARGETS.get(t, {}).get("label", t)
                unit = TARGETS.get(t, {}).get("unit", "")
                is_measured = t in measured_inputs and measured_inputs[t] is not None
                if val is not None and val == val:
                    tag = " ✓ mesuré" if is_measured else " ~ prédit"
                    st.metric(
                        label=f"{label} ({unit}){tag}",
                        value=f"{val:.2f}",
                    )
                    any_val = True
                else:
                    st.metric(label=f"{label} ({unit})", value="—")
            if not any_val:
                st.caption("Modèle non disponible pour ce bloc.")

    # Vue tableau
    rows = []
    for bloc_title, bloc_targets in bloc_defs:
        for t in bloc_targets:
            val = preds.get(t)
            is_measured = t in measured_inputs and measured_inputs[t] is not None
            rows.append({
                "Bloc": bloc_title.split("—")[1].strip(),
                "Cible": TARGETS.get(t, {}).get("label", t),
                "Valeur": f"{val:.3f}" if val is not None and val == val else "—",
                "Unité": TARGETS.get(t, {}).get("unit", ""),
                "Source": "Mesuré" if is_measured else "Prédit",
            })
    if rows:
        with st.expander("Vue tableau complète", expanded=False):
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Graphique barres avec couleur par source
    valid_rows = [r for r in rows if r["Valeur"] != "—"]
    if valid_rows:
        bar_data = pd.DataFrame(valid_rows)
        bar_data["Valeur_num"] = pd.to_numeric(bar_data["Valeur"], errors="coerce")
        fig = px.bar(
            bar_data.dropna(subset=["Valeur_num"]),
            x="Cible",
            y="Valeur_num",
            color="Source",
            text="Valeur",
            title=f"Prédictions — {approach_label}",
            color_discrete_map={"Mesuré": "#2ca02c", "Prédit": "#636EFA"},
            labels={"Valeur_num": "Valeur"},
        )
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        fig.update_layout(showlegend=True, height=380)
        st.plotly_chart(fig, use_container_width=True)


def render_single_prediction() -> None:
    render_mode_sidebar()
    st.title("Prédiction unitaire")
    st.markdown(
        "Saisissez les mesures disponibles pour un lot, choisissez un modèle "
        "et obtenez instantanément des prédictions pour les propriétés d'usage."
    )

    st.markdown("---")
    family, approach = _model_selector_with_cascade()

    # Pour les modèles classiques, options supplémentaires
    compare_ab = False
    if family == "classic":
        compare_ab = st.checkbox(
            "Comparer modèle A et modèle B",
            value=False,
            help=(
                "Affiche les prédictions des deux modèles côte à côte. "
                "Un écart important peut indiquer que le contexte fournisseur "
                "joue un rôle significatif pour ce lot."
            ),
        )
        targets_sel = st.multiselect(
            "Cibles à prédire",
            MVP_TARGETS,
            default=["oit_min", "izod", "flexion"],
            format_func=lambda t: TARGETS.get(t, {}).get("label", t),
            help="Sélectionnez uniquement les propriétés dont vous avez besoin. Les modèles sont indépendants par cible.",
        )
    else:
        st.info(
            f"**{CASCADE_APPROACH_LABELS.get(approach, approach)}** : "
            "toutes les cibles sont prédites en séquence — OIT d'abord, puis propriétés mécaniques, "
            "puis NCLS/UCLS en utilisant les résultats des blocs précédents.",
            icon="🔗",
        )
        targets_sel = MVP_TARGETS

    st.markdown("---")
    st.markdown("#### Paramètres matière")
    st.caption(
        "Renseignez les valeurs mesurées à la réception. "
        "Les champs vides ou à zéro sont acceptés mais réduisent la précision."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        mi = st.number_input(
            "MI — Melt Index (g/10min)",
            0.0, value=0.5, step=0.01,
            help="Indice de fluidité à faible cisaillement (190°C / 2.16 kg). "
                 "Plage typique PE recyclé : 0.1 – 5 g/10min.",
        )
        hlmi = st.number_input(
            "HLMI (g/10min)",
            0.0, value=4.0, step=0.1,
            help="High Load Melt Index (190°C / 21.6 kg). "
                 "Le rapport HLMI/MI (FRR) caractérise la distribution des masses molaires.",
        )
        density = st.number_input(
            "Densité (g/cm³)",
            0.5, 1.5, 0.95, 0.001,
            help="Densité du granulé. Plage PE : 0.920 – 0.965 g/cm³.",
        )
    with c2:
        carbon = st.number_input(
            "Noir de carbone %",
            0.0, value=0.0,
            help="Teneur en noir de carbone. Le noir de carbone améliore la résistance UV.",
        )
        ash = st.number_input(
            "Cendres %",
            0.0, value=0.0,
            help="Résidu après incinération — indicateur de charges inorganiques (silice, talc…).",
        )
        pp = st.number_input(
            "PP %",
            0.0, value=0.0,
            help="Pourcentage de polypropylène — contaminant fréquent dans les PE recyclés.",
        )
    with c3:
        onset = st.number_input(
            "DSC Onset (°C)",
            value=0.0,
            help="Température de début de fusion DSC. Indicateur de la structure cristalline.",
        )
        peak = st.number_input(
            "DSC Peak (°C)",
            value=0.0,
            help="Température de pic de fusion DSC. Liée au type de PE (HDPE ~130–135°C).",
        )
        delta_h = st.number_input(
            "Delta H (J/g)",
            value=0.0,
            help="Enthalpie de fusion DSC. Corrélée au taux de cristallinité de la résine.",
        )
        recycled = st.selectbox(
            "Recyclé / Vierge",
            ["RECYCLE", "VIERGE", ""],
            help="Type de matière. Influence la dégradation thermique et mécanique.",
        )

    input_row = {
        "mi": mi, "hlmi": hlmi,
        "density_g_cm3": density, "density_plaque_g_cm3": density,
        "carbon_black": carbon, "ash": ash, "pp": pp,
        "onset": onset, "peak": peak, "delta_h": delta_h,
        "recycled_virgin": recycled,
        "fluidity_g_10min": mi,
    }

    if family == "classic" and approach == "B":
        st.markdown("---")
        st.markdown("#### Contexte opérationnel (modèle B)")
        st.caption(
            "Ces champs permettent au modèle B d'exploiter les spécificités connues "
            "du fournisseur et du site de production, en plus des propriétés physico-chimiques."
        )
        b1, b2 = st.columns(2)
        with b1:
            input_row["supplier_code"] = st.text_input(
                "Code fournisseur",
                help="Code interne NAV du fournisseur (ex: FR001). Laisser vide si inconnu.",
            )
            input_row["location"] = st.text_input(
                "Site / emplacement",
                help="Site de stockage ou de production (ex: Rouen, Gent…).",
            )
        with b2:
            input_row["description_fr"] = st.text_input(
                "Grade / description produit",
                help="Description commerciale du grade (ex: HDPE BM, Recyclé PEBD…).",
            )
            input_row["reception_year"] = st.number_input(
                "Année de réception",
                2010, 2030, 2023,
                help="L'année de réception capture les tendances temporelles du marché des recyclés.",
            )

    # ---- Mesures intermédiaires (cascade seulement) ----
    if family == "cascade":
        st.markdown("---")
        with st.expander(
            "Mesures intermédiaires disponibles — OIT, Traction, Flexion… (optionnel)",
            expanded=False,
        ):
            st.info(
                "**Comment ça fonctionne ?** \n\n"
                "Le modèle Cascade prédit d'abord OIT (Bloc 1), puis les propriétés mécaniques (Bloc 2), "
                "puis NCLS/UCLS (Bloc 3) en enchaînant les blocs. "
                "Si vous avez déjà des **mesures labo réelles** pour certaines propriétés intermédiaires, "
                "renseignez-les ici : elles remplaceront les prédictions du modèle, "
                "ce qui **améliore significativement** la précision des blocs suivants.",
                icon="💡",
            )
            st.markdown("**Bloc 1 — Structure thermique** *(utilisé par Blocs 2 et 3)*")
            t1 = st.columns(len(CASCADE_THERMAL_TARGETS))
            for col, t in zip(t1, CASCADE_THERMAL_TARGETS):
                label = TARGETS.get(t, {}).get("label", t)
                unit = TARGETS.get(t, {}).get("unit", "")
                val = col.number_input(
                    f"{label} ({unit}) — mesuré",
                    value=None,
                    placeholder="laisser vide = prédit",
                    key=f"inter_{t}",
                )
                if val is not None:
                    input_row[t] = val

            st.markdown("**Bloc 2 — Structure mécanique** *(utilisé par Bloc 3)*")
            t2 = st.columns(len(CASCADE_MECHANICAL_TARGETS))
            for col, t in zip(t2, CASCADE_MECHANICAL_TARGETS):
                label = TARGETS.get(t, {}).get("label", t)
                unit = TARGETS.get(t, {}).get("unit", "")
                val = col.number_input(
                    f"{label} ({unit}) — mesuré",
                    value=None,
                    placeholder="laisser vide = prédit",
                    key=f"inter_{t}",
                )
                if val is not None:
                    input_row[t] = val

    st.markdown("---")
    lot_id = st.text_input(
        "Identifiant lot (optionnel)",
        value="lot_manuel",
        help="Référence libre pour tracer la prédiction dans le journal. Ex: lot NAV #42153.",
    )

    quality_preview = check_lot_quality(input_row, approach if family == "classic" else "A")
    if is_expert():
        with st.expander("Contrôle qualité (aperçu avant prédiction)", expanded=True):
            st.caption(
                "Vérification des valeurs saisies par rapport aux plages historiques NAV. "
                "Les alertes n'empêchent pas la prédiction mais signalent un lot inhabituel."
            )
            st.markdown(quality_preview.summary_text())
            if quality_preview.out_of_range:
                for msg in quality_preview.out_of_range:
                    st.warning(f"Attention : {msg}")
            if quality_preview.missing_features:
                st.info(f"Manquants : {', '.join(quality_preview.missing_features)}")

    if st.button("Prédire", type="primary"):
        from src.feature_engineering import add_derived_features
        full_row = add_derived_features(pd.DataFrame([input_row])).iloc[0].to_dict()

        # ---- Cascade / PSPP path ----
        if family == "cascade":
            with st.spinner(f"Prédiction {CASCADE_APPROACH_LABELS.get(approach, approach)} en cours…"):
                res = predict_cascade_single(full_row, approach=approach)

            if res["status"] == "OK":
                # Identifie quelles cibles intermédiaires ont été fournies
                intermediate_all = CASCADE_THERMAL_TARGETS + CASCADE_MECHANICAL_TARGETS
                measured_inputs = {t: input_row.get(t) for t in intermediate_all if t in input_row}
                _render_cascade_single_result(res["predictions"], approach, measured_inputs)

                for t, val in res["predictions"].items():
                    if val is not None and val == val:
                        append_prediction_log(
                            lot_id=lot_id, target=t,
                            model_version=approach,
                            features=input_row,
                            prediction=val,
                            confidence_score=0.0,
                            confidence_level="cascade",
                            recommendation="",
                            data_quality_level=quality_preview.quality_level,
                            prediction_status="ok",
                            mode="single",
                        )
            elif res["status"] == "model_not_available":
                st.warning(res.get("message", "Modèles cascade non disponibles."))
                st.info("Exécutez `python run_phase4.py` pour entraîner les modèles cascade.")
            else:
                st.error(f"Erreur : {res.get('error', 'inconnue')}")

        # ---- Modèles classiques A / B ----
        else:
            results, global_rec, quality = predict_lot_with_metadata(
                full_row, targets_sel, approach, compare_ab=compare_ab
            )

            st.markdown("#### Qualité des données d'entrée")
            st.caption(
                "Évaluation de la complétude et de la plausibilité des features fournies. "
                "Une qualité élevée = prédictions plus fiables."
            )
            if quality.quality_level == "élevée":
                st.success(quality.summary_text(), icon="✅")
            elif quality.quality_level == "moyenne":
                st.warning(quality.summary_text(), icon="⚠️")
            else:
                st.error(quality.summary_text(), icon="❌")

            st.markdown("#### Recommandation")
            st.info(global_rec, icon="🔎")

            display = results.copy()
            display["Cible"] = display["target"].map(lambda t: TARGETS.get(t, {}).get("label", t))
            display["Prédiction"] = display.apply(
                lambda r: f"{r['prediction']:.4f} {target_unit(r['target'])}"
                if r["prediction"] is not None and r["prediction_status"] == "ok"
                else "—",
                axis=1,
            )
            display["Confiance"] = display["confidence_level"]

            cols_show = ["Cible", "Prédiction", "Confiance", "recommendation"]
            if compare_ab:
                cols_show = ["Cible", "pred_model_a", "pred_model_b", "gap_a_b", "ab_comment", "Confiance", "recommendation"]
            st.dataframe(display[cols_show].rename(columns={"recommendation": "Recommandation"}), hide_index=True)

            if is_expert():
                with st.expander("Détails techniques"):
                    st.dataframe(results, use_container_width=True)
                    st.markdown("**Features dérivées**")
                    for k in ("frr", "fluidity_g_10min", "charge_total"):
                        if k in full_row:
                            st.write(f"- {k} : {full_row.get(k)}")
                    for target in targets_sel:
                        path = resolve_model_path(target, approach)
                        st.caption(f"{target} — modèle : `{path}`")
                    try:
                        from src.pipeline_bridge import get_ood_reference
                        ood_mod = get_ood_reference()
                        ood = ood_mod.check_ood(
                            full_row,
                            targets_sel[0] if targets_sel else "izod",
                            approach,
                            ood_mod.load_reference(),
                            ood_mod.load_risky_suppliers_by_target(),
                        )
                        st.warning(f"OOD trust : {ood.trust_score:.0%} — {ood.summary}")
                        for a in ood.alerts[:8]:
                            st.caption(f"• {a.message}")
                    except Exception as ex:
                        st.caption(f"OOD non disponible : {ex}")

                if compare_ab:
                    st.markdown("#### Comparaison A vs B")
                    ab = results[["target", "pred_model_a", "pred_model_b", "gap_a_b", "ab_comment"]].copy()
                    ab.columns = ["Cible", "Modèle A", "Modèle B", "Écart", "Commentaire"]
                    st.dataframe(ab, hide_index=True)

            for _, row in results.iterrows():
                append_prediction_log(
                    lot_id=lot_id, target=row["target"],
                    model_version=approach,
                    features=input_row,
                    prediction=row["prediction"],
                    confidence_score=row["confidence_score"],
                    confidence_level=row["confidence_level"],
                    recommendation=row["recommendation"],
                    data_quality_level=quality.quality_level,
                    prediction_status=row["prediction_status"],
                    mode="single",
                )


def _export_buttons(df_export: pd.DataFrame, prefix: str) -> None:
    """Boutons d'export CSV et Excel réutilisables."""
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_bytes = df_export.to_csv(index=False).encode("utf-8-sig")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Exporter CSV",
            csv_bytes,
            f"{prefix}_{ts}.csv",
            "text/csv",
            use_container_width=True,
        )
    with c2:
        buf = io.BytesIO()
        try:
            df_export.to_excel(buf, index=False, engine="openpyxl")
            st.download_button(
                "Exporter Excel",
                buf.getvalue(),
                f"{prefix}_{ts}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception:
            st.caption("Export Excel : `pip install openpyxl`.")


def render_batch_prediction() -> None:
    render_mode_sidebar()
    st.title("Prédiction par lot")
    st.markdown(
        "Chargez un fichier CSV ou Excel contenant plusieurs lots. "
        "Le modèle prédit toutes les propriétés pour chaque ligne, "
        "avec export des résultats."
    )

    with st.expander("Format attendu du fichier", expanded=False):
        st.markdown(
            "Le fichier doit contenir au minimum les colonnes suivantes (noms flexibles) :\n\n"
            "| Colonne | Description | Obligatoire |\n"
            "|---|---|---|\n"
            "| `mi` ou `MI` | Melt Index g/10min | Oui |\n"
            "| `hlmi` ou `HLMI` | High Load MI g/10min | Recommandé |\n"
            "| `density` ou `densité` | Densité g/cm³ | Oui |\n"
            "| `carbon_black` | Noir de carbone % | Optionnel |\n"
            "| `ash` | Cendres % | Optionnel |\n"
            "| `onset`, `peak`, `delta_h` | DSC | Optionnel |\n"
            "| `supplier_code` | Code fournisseur (modèle B) | Optionnel |\n\n"
            "Des colonnes additionnelles sont ignorées. "
            "Une colonne `id` ou `lot_id` permet d'identifier les lignes dans les résultats."
        )

    uploaded = st.file_uploader(
        "Chargez votre fichier (CSV ou Excel)",
        type=["csv", "xlsx", "xls"],
        help="Taille max recommandée : 50 000 lignes. Au-delà, la prédiction peut prendre plusieurs minutes.",
    )
    if not uploaded:
        st.info(
            "Aucun fichier chargé. Utilisez le bouton ci-dessus pour démarrer. "
            "Vous pouvez aussi exporter un lot depuis NAV/Business Central directement en CSV.",
            icon="📂",
        )
        return

    if uploaded.name.endswith(".csv"):
        raw = pd.read_csv(uploaded)
    else:
        raw = pd.read_excel(uploaded)

    st.subheader(f"Aperçu — {len(raw):,} lots chargés")
    st.dataframe(raw.head(20), use_container_width=True)

    df, missing_core = prepare_uploaded_df(raw)
    if missing_core:
        st.warning(
            f"Colonnes cœur manquantes : {', '.join(missing_core)}\n\n"
            "Les prédictions peuvent être moins précises ou indisponibles pour certaines cibles.",
            icon="⚠️",
        )

    if is_expert():
        rules = load_input_rules()
        if not rules.empty:
            with st.expander("Plages plausibles des entrées (input_validity_rules.csv)"):
                st.caption("Valeurs hors de ces plages déclenchent des alertes qualité par lot.")
                st.dataframe(rules, use_container_width=True)

    st.markdown("---")
    # ---- Sélecteur de modèle (classique ou cascade) ----
    family, approach = _model_selector_with_cascade()

    targets_sel = MVP_TARGETS
    if family == "classic":
        targets_sel = st.multiselect(
            "Cibles à prédire",
            MVP_TARGETS,
            default=MVP_TARGETS[:4],
            format_func=lambda t: TARGETS.get(t, {}).get("label", t),
            help="Réduire la sélection accélère le calcul pour les grands lots.",
        )
    else:
        st.info(
            f"**{CASCADE_APPROACH_LABELS.get(approach, approach)}** : "
            "toutes les cibles sont prédites en séquence (OIT → Méca → NCLS/UCLS). "
            "Chaque bloc utilise les prédictions du bloc précédent.",
            icon="🔗",
        )

    if st.button("Lancer les prédictions", type="primary"):

        # ========== BRANCH CASCADE / PSPP ==========
        if family == "cascade":
            from src.prediction import predict_cascade_batch
            with st.spinner(f"Prédiction {CASCADE_APPROACH_LABELS.get(approach, approach)} sur {len(df)} lots…"):
                result = predict_cascade_batch(df, approach=approach)

            cascade_cols = [c for c in result.columns if c.startswith(f"{approach}_")]
            all_targets_predicted = CASCADE_THERMAL_TARGETS + CASCADE_MECHANICAL_TARGETS + CASCADE_FINAL_TARGETS

            # Résumé métriques
            st.markdown("#### Résumé — moyennes prédites sur le lot")
            st.caption(
                "Valeurs moyennes sur l'ensemble des lots du fichier. "
                "Consultez les tableaux par bloc ci-dessous pour le détail par lot."
            )
            n_pred = sum(1 for t in all_targets_predicted if f"{approach}_{t}" in result.columns and result[f"{approach}_{t}"].notna().any())
            mcols = st.columns(len(all_targets_predicted))
            for col, t in zip(mcols, all_targets_predicted):
                col_name = f"{approach}_{t}"
                label = TARGETS.get(t, {}).get("label", t)
                unit = TARGETS.get(t, {}).get("unit", "")
                if col_name in result.columns and result[col_name].notna().any():
                    mean_val = result[col_name].mean()
                    col.metric(
                        label=f"{label} ({unit})",
                        value=f"{mean_val:.2f}",
                        help=f"Moyenne prédite pour '{t}' sur {result[col_name].notna().sum()} lots",
                    )
                else:
                    col.metric(label=f"{label} ({unit})", value="—")

            # Tableau résultats par bloc
            for bloc_title, bloc_targets in [
                ("Bloc 1 — Structure thermique", CASCADE_THERMAL_TARGETS),
                ("Bloc 2 — Structure mécanique", CASCADE_MECHANICAL_TARGETS),
                ("Bloc 3 — Performance finale", CASCADE_FINAL_TARGETS),
            ]:
                bloc_cols = [f"{approach}_{t}" for t in bloc_targets if f"{approach}_{t}" in result.columns]
                if not bloc_cols:
                    continue
                with st.expander(bloc_title, expanded=True):
                    id_cols = [c for c in ("lot_id", "id", "sample_id") if c in result.columns]
                    st.dataframe(
                        result[id_cols + bloc_cols].rename(
                            columns={f"{approach}_{t}": f"{TARGETS.get(t,{}).get('label',t)} ({TARGETS.get(t,{}).get('unit','')})"
                                     for t in bloc_targets}
                        ),
                        use_container_width=True,
                    )

            # Graphique distributions
            valid_cascade_cols = [c for c in cascade_cols if result[c].notna().any()]
            if valid_cascade_cols:
                with st.expander("Distributions des prédictions", expanded=False):
                    for col_name in valid_cascade_cols:
                        t = col_name.replace(f"{approach}_", "")
                        label = TARGETS.get(t, {}).get("label", t)
                        fig = px.histogram(
                            result[result[col_name].notna()],
                            x=col_name,
                            nbins=30,
                            title=f"Distribution — {label}",
                            labels={col_name: label},
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # Export
            st.markdown("#### Export")
            id_cols = [c for c in ("lot_id", "id", "sample_id") if c in result.columns]
            export_df = result[id_cols + cascade_cols].copy()
            export_df.columns = (
                id_cols
                + [f"{TARGETS.get(c.replace(f'{approach}_',''),{}).get('label', c.replace(f'{approach}_',''))} ({TARGETS.get(c.replace(f'{approach}_',''),{}).get('unit','')})"
                   for c in cascade_cols]
            )
            _export_buttons(export_df, f"predictions_{approach}")

            # Log
            for idx, row in result.iterrows():
                lot = str(row.get("lot_id", row.get("id", idx)))
                for t in all_targets_predicted:
                    col_name = f"{approach}_{t}"
                    val = row.get(col_name)
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        continue
                    append_prediction_log(
                        lot_id=lot, target=t,
                        model_version=approach,
                        features={k: row.get(k) for k in df.columns if k in row},
                        prediction=val,
                        confidence_score=0.0,
                        confidence_level="cascade",
                        recommendation="",
                        data_quality_level="batch",
                        prediction_status="ok",
                        mode="batch",
                    )

        # ========== BRANCH CLASSIQUE A / B ==========
        else:
            result = predict_batch(df, targets_sel, approach)
            result = enrich_batch_predictions(result, targets_sel, approach)
            result["feature_version"] = "04_NAV_feature_engineered + dérivées MVP"

            st.markdown("#### Résumé")
            n_ok = (result["prediction_status"] == "ok").sum()
            st.metric("Lots avec au moins une prédiction OK", int(n_ok))

            show_cols = [c for c in result.columns if c.startswith(("predicted_", "confidence_", "recommendation", "prediction_status", "model_version", "generated_at"))]
            id_cols = [c for c in ("lot_id", "id", "sample_id") if c in result.columns]
            st.dataframe(result[id_cols + show_cols] if id_cols else result[show_cols], use_container_width=True)

            export_df = export_column_rename(result, targets_sel)
            _export_buttons(export_df, f"predictions_nav_{approach}")

            if is_expert():
                st.dataframe(result, use_container_width=True)

            for idx, row in result.iterrows():
                lot = str(row.get("lot_id", row.get("id", idx)))
                for target in targets_sel:
                    pred = row.get(f"predicted_{target}")
                    if pred is None or (isinstance(pred, float) and pd.isna(pred)):
                        continue
                    append_prediction_log(
                        lot_id=lot, target=target,
                        model_version=approach,
                        features={k: row.get(k) for k in df.columns if k in row},
                        prediction=pred,
                        confidence_score=row.get(f"confidence_{target}", 0),
                        confidence_level=row.get(f"confidence_level_{target}", ""),
                        recommendation=row.get(f"recommendation_{target}", ""),
                        data_quality_level="batch",
                        prediction_status=row.get(f"status_{target}", "ok"),
                        mode="batch",
                    )

        st.success(f"Journal mis à jour : `{PREDICTION_LOGS_DIR / 'prediction_history.csv'}`")


def render_explainability() -> None:
    render_mode_sidebar()
    if not is_expert():
        st.info(
            "La page Explicabilité est réservée au **mode Expert** (activez-le dans la barre latérale).",
            icon="🔒",
        )
        return
    st.title("Explicabilité des modèles")
    st.markdown(
        "Comprendre **pourquoi** le modèle fait une prédiction — quelles features influencent le plus le résultat."
    )
    st.caption(
        "L'importance est calculée par permutation sur le dataset NAV ou récupérée depuis les fichiers de sortie du pipeline. "
        "Une feature avec une haute importance = une grande influence sur les prédictions de cette cible."
    )
    target = st.selectbox(
        "Cible à analyser",
        MVP_TARGETS,
        format_func=lambda t: TARGETS.get(t, {}).get("label", t),
        help="Chaque cible a son propre modèle et ses propres drivers.",
    )
    model_version = st.radio(
        "Modèle",
        ["A", "B"],
        horizontal=True,
        help="A = Matière uniquement. B = Matière + contexte (fournisseur, site…).",
    )

    imp = get_importance_table(target, model_version)
    if imp.empty:
        st.info("Calcul d'importance à la volée sur un échantillon NAV — peut prendre quelques secondes…", icon="⏳")
        df = load_feature_engineered()
        imp = compute_live_importance(df, target, model_version) if df is not None else pd.DataFrame()

    if imp.empty:
        st.warning("Importance non disponible pour cette cible. Vérifiez que le modèle est entraîné.")
        return

    st.plotly_chart(
        px.bar(
            imp.head(15),
            x="importance",
            y="feature",
            orientation="h",
            title=f"Top 15 features — {TARGETS.get(target, {}).get('label', target)} (modèle {model_version})",
            labels={"importance": "Importance", "feature": "Variable"},
        ).update_layout(yaxis={"categoryorder": "total ascending"}),
        use_container_width=True,
    )

    if model_version == "A":
        st.info(
            "**Modèle A** : les drivers dominants sont liés à la **matière** "
            "(rhéologie : MI/HLMI/FRR, densité, charges, thermique DSC). "
            "Ces features sont mesurables à la réception sans information fournisseur.",
            icon="🔬",
        )
    else:
        st.info(
            "**Modèle B** : en plus des features matière, le contexte opérationnel "
            "(fournisseur, site, grade, année) peut capturer des effets de process non mesurés. "
            "Comparez avec le modèle A pour isoler la contribution du contexte.",
            icon="🏭",
        )

    imp_b = get_importance_table(target, "B" if model_version == "A" else "A")
    if not imp_b.empty:
        with st.expander("Comparaison A vs B — top 10 features"):
            st.caption(
                "Features présentes dans un modèle mais absentes de l'autre apparaissent avec une valeur vide. "
                "Un rang différent pour la même feature révèle l'impact du contexte opérationnel."
            )
            st.dataframe(
                pd.merge(
                    imp.head(10).rename(columns={"importance": f"imp_{model_version}"}),
                    imp_b.head(10).rename(columns={"importance": f"imp_{'B' if model_version == 'A' else 'A'}"}),
                    on="feature",
                    how="outer",
                ),
                use_container_width=True,
            )


def render_model_comparison() -> None:
    render_mode_sidebar()
    st.title("Comparaison des modèles ML")
    st.markdown(
        "Analysez les performances (R², MAE) de chaque algorithme et phase du pipeline "
        "pour choisir le meilleur modèle par cible."
    )
    st.caption(
        "Ces métriques sont calculées par **validation croisée** (K-Fold) sur le dataset NAV — "
        "elles reflètent la capacité de généralisation et non la performance sur les données d'entraînement."
    )
    metrics = load_metrics_bundle()

    source = st.selectbox(
        "Source des métriques",
        [k for k in metrics.keys()] or ["aucune disponible"],
        format_func=lambda k: {
            "baseline_reg": "Référence ML — Phase 1 (baselines)",
            "dl_reg": "Apprentissage profond",
            "ml_after_cleaning": "ML après correction — Phase 2",
            "before_after": "Avant / après correction (impact du nettoyage)",
            "robust_validation": "Validation robuste — Phase 3",
            "ab_robust": "Comparaison A vs B robuste",
        }.get(k, k),
        help=(
            "Chaque source correspond à une étape du pipeline. "
            "**Phase 3 (validation robuste)** est la source la plus fiable pour les décisions."
        ),
    )

    if source not in metrics:
        st.warning("Aucun fichier de métriques trouvé.")
        return

    df = metrics[source].copy()
    targets = sorted(df["target"].dropna().unique()) if "target" in df.columns else []
    target_f = st.selectbox("Cible", ["Toutes"] + list(targets), key="mc_target")

    if target_f != "Toutes" and "target" in df.columns:
        df = df[df["target"] == target_f]

    if "model_version" in df.columns:
        mv = st.multiselect("Modèle A/B", df["model_version"].unique(), default=list(df["model_version"].unique()))
        df = df[df["model_version"].isin(mv)]

    st.dataframe(df.head(100), use_container_width=True)

    if "r2" in df.columns and "algorithm" in df.columns:
        agg = df.groupby("algorithm", as_index=False)["r2"].mean().sort_values("r2", ascending=False)
        fig = bar_metrics(agg, "algorithm", "r2", "R² moyen par algorithme")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    if "mae" in df.columns and "target" in df.columns:
        agg2 = df.groupby("target", as_index=False)["mae"].mean()
        fig2 = bar_metrics(agg2, "target", "mae", "MAE moyen par cible")
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)

    _render_model_readiness_section()

    st.subheader("Interprétation")
    st.markdown(
        "- **Référence décisionnelle** : validation robuste phase 3 (`group_supplier` > `kfold_random`)\n"
        "- **Cibles les plus stables** : souvent `oit_min`, `flexion`, `pct_elongation` en domaine connu\n"
        "- **Cibles fragiles** : `ncls`, `izod` — effondrement fréquent en GroupKFold fournisseur\n"
        "- **UCLS** : effectif trop faible — ne pas déployer sans nouvelles données\n"
        "- **GroupKFold** : écart MAE random vs fournisseur → prudence sur nouveaux fournisseurs"
    )


# ---------------------------------------------------------------------------
# Page Fiabilité modèles (phase 5)
# ---------------------------------------------------------------------------

def render_model_reliability() -> None:
    render_mode_sidebar()
    st.title("Fiabilité des modèles — référence scientifique")
    st.markdown(
        "Cette page consolide la **validation robuste (phase 3)** et le **statut de déploiement (phase 5)**. "
        "Pour des résines recyclées, la généralisation fournisseur/grade prime sur le R² en KFold aléatoire."
    )

    unified = load_unified_evaluation()
    deployment = unified.get("deployment_status", pd.DataFrame())
    recommended = unified.get("recommended_model", pd.DataFrame())
    metrics_long = unified.get("master_metrics_long", pd.DataFrame())
    pred_cv = unified.get("robust_predictions", pd.DataFrame())

    if deployment.empty:
        st.warning(
            "Tables phase 5 non disponibles. Exécutez : `python run_scientific_evaluation.py` "
            "(après `run_phase2.py` et `run_phase3.py`)."
        )
        metrics = load_metrics_bundle()
        if "robust_validation" in metrics:
            st.info("Affichage fallback — métriques phase 3 brutes.")
            st.dataframe(metrics["robust_validation"], use_container_width=True)
        return

    st.success("Évaluation unifiée chargée — protocole scientifique phase 5 actif.")

    tab_overview, tab_protocol, tab_target, tab_parity = st.tabs([
        "Vue d'ensemble",
        "Protocoles CV",
        "Par cible",
        "Diagnostics",
    ])

    with tab_overview:
        fig_status = deployment_status_chart(recommended if not recommended.empty else deployment)
        if fig_status:
            st.plotly_chart(fig_status, use_container_width=True)

        if not recommended.empty:
            show = recommended.copy()
            show["target_label"] = show["target"].map(lambda t: TARGETS.get(t, {}).get("label", t))
            cols = [
                "target_label", "model_version", "tier", "deployment_label",
                "r2_random", "r2_group_supplier", "mae_degradation_ratio", "status_reason",
            ]
            st.dataframe(
                show[[c for c in cols if c in show.columns]].rename(columns={
                    "target_label": "Cible",
                    "model_version": "Modèle",
                    "tier": "Tier",
                    "deployment_label": "Statut",
                    "r2_random": "R² random",
                    "r2_group_supplier": "R² group fournisseur",
                    "mae_degradation_ratio": "Ratio MAE group/random",
                    "status_reason": "Justification",
                }),
                use_container_width=True,
                hide_index=True,
            )

    with tab_protocol:
        st.caption(
            "Ordre de lecture : **group_supplier** (nouveau fournisseur) → grade → site → temporel → random."
        )
        if not metrics_long.empty:
            scheme_agg = (
                metrics_long.groupby(["validation_scheme", "validation_label"], as_index=False)["r2_mean"]
                .mean(numeric_only=True)
                .sort_values("validation_scheme")
            )
            fig = bar_metrics(scheme_agg, "validation_label", "r2_mean", "R² moyen par protocole (toutes cibles)")
            if fig:
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)

            heatmap_df = metrics_long.pivot_table(
                index="target",
                columns="validation_scheme",
                values="r2_mean",
                aggfunc="first",
            )
            if not heatmap_df.empty:
                st.markdown("#### Heatmap R² — cible × protocole")
                st.dataframe(heatmap_df.style.format("{:.3f}", na_rep="—"), use_container_width=True)

    with tab_target:
        target = st.selectbox(
            "Cible",
            sorted(metrics_long["target"].dropna().unique()) if not metrics_long.empty else MVP_TARGETS,
            format_func=lambda t: TARGETS.get(t, {}).get("label", t),
            key="reliability_target",
        )
        if not metrics_long.empty:
            c1, c2 = st.columns(2)
            with c1:
                fig_r2 = reliability_comparison_bars(metrics_long, target, "r2_mean")
                if fig_r2:
                    st.plotly_chart(fig_r2, use_container_width=True)
            with c2:
                fig_mae = reliability_comparison_bars(metrics_long, target, "mae_mean")
                if fig_mae:
                    st.plotly_chart(fig_mae, use_container_width=True)

        if not recommended.empty:
            row = recommended[recommended["target"].astype(str) == target]
            if not row.empty:
                r = row.iloc[0]
                st.info(
                    f"**{TARGETS.get(target, {}).get('label', target)}** — "
                    f"{r.get('deployment_label', '')}. "
                    f"{r.get('polymer_rationale', '')} "
                    f"({r.get('status_reason', '')})"
                )

    with tab_parity:
        target_p = st.selectbox(
            "Cible (parity plot)",
            MVP_TARGETS,
            format_func=lambda t: TARGETS.get(t, {}).get("label", t),
            key="parity_target",
        )
        scheme = st.selectbox(
            "Schéma CV",
            ["kfold_random", "group_supplier", "temporal"],
            format_func=lambda s: {
                "kfold_random": "KFold aléatoire",
                "group_supplier": "GroupKFold fournisseur",
                "temporal": "Temporel",
            }.get(s, s),
            key="parity_scheme",
        )
        if not pred_cv.empty:
            sub_pred = pred_cv[
                (pred_cv["target"].astype(str) == target_p)
                & (pred_cv["validation_scheme"].astype(str) == scheme)
            ]
            fig_p = parity_plot(sub_pred, target_p)
            if fig_p:
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                st.info("Pas assez de prédictions CV pour cette combinaison.")
        else:
            st.info("Prédictions CV non disponibles — exécutez run_phase3.py.")


# ---------------------------------------------------------------------------
# Page Cascade / PSPP
# ---------------------------------------------------------------------------

def _load_cascade_csv(key: str) -> pd.DataFrame:
    path = PATHS.get(key)
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _cascade_architecture_diagram() -> None:
    st.markdown(
        """
```
Bloc 1 — Thermique        Bloc 2 — Mécanique        Bloc 3 — Performance
─────────────────────     ──────────────────────     ──────────────────────
Matière (MI, HLMI,        Matière                    Matière
  Densité, CB, DSC…)  →  + OIT (Bloc 1)         →  + OIT
  → OIT                    → Traction                + Traction / Flexion
                            → Flexion                + Izod / Allongement
                            → Izod                   → NCLS
                            → % Allongement          → UCLS
```
"""
    )


def render_cascade_comparison() -> None:
    render_mode_sidebar()
    st.title("Comparaison des approches — Cascade / PSPP")

    st.markdown(
        "Cette page compare les **3 stratégies de modélisation** disponibles dans le système "
        "pour prédire les propriétés finales des résines PE recyclées."
    )

    st.info(
        "**Pourquoi une architecture en cascade ?** \n\n"
        "Les propriétés des résines recyclées suivent une logique physique : "
        "la composition de la matière détermine sa stabilité thermique (OIT), "
        "qui à son tour influence ses propriétés mécaniques (traction, flexion, IZOD), "
        "qui enfin conditionnent sa durabilité en service (NCLS, UCLS).\n\n"
        "**Important :** la cascade *lab-assistée* utilise les vraies mesures intermédiaires en évaluation — "
        "elle est optimiste. **PSPP (OOF)** est la référence pour l'inférence sans mesures labo.",
        icon="💡",
    )

    st.markdown(
        """
| Approche | Logique | Avantage | Limite |
|---|---|---|---|
| **Direct** | Matière → cible | Simple, référence production | Ignore la chaîne physique |
| **Cascade lab-assisté** | Blocs avec intermédiaires **mesurés** | Bon pour R&D avec essais partiels | Surestime la prod sans labo |
| **PSPP (OOF)** | Chaîne prédite sans fuite | Référence honnête inférence | Plus lent à entraîner |
"""
    )

    # ---- Disponibilité des modèles ----
    available = cascade_approaches_available()
    if not available:
        st.warning(
            "Aucun modèle de cascade disponible. "
            "Exécutez d'abord : `python run_phase4.py`"
        )
        st.info(
            "**Architecture cible :**\n\n"
            "Bloc 1 (Thermique) → Bloc 2 (Mécanique) → Bloc 3 (Performance finale)"
        )
        _cascade_architecture_diagram()
        return

    st.success(f"Approches disponibles : {', '.join([CASCADE_APPROACH_LABELS.get(a, a) for a in available])}")

    # ---- Onglets ----
    tab_arch, tab_r2, tab_mae, tab_gain, tab_pred = st.tabs([
        "Architecture",
        "R² par approche",
        "MAE par approche",
        "Gain cascade",
        "Prédiction live",
    ])

    # --- Tab Architecture ---
    with tab_arch:
        st.subheader("Structure en cascade — Vue d'ensemble")
        _cascade_architecture_diagram()

        st.markdown("---")
        st.markdown("#### Détail des blocs")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Bloc 1 — Thermique**")
            st.caption("Entrées : Matière (MI, HLMI, densité, DSC, charges)")
            for t in CASCADE_THERMAL_TARGETS:
                lbl = TARGETS.get(t, {}).get("label", t)
                unit = TARGETS.get(t, {}).get("unit", "")
                st.markdown(f"- **{lbl}** `{t}` ({unit})")
        with col2:
            st.markdown("**Bloc 2 — Mécanique**")
            st.caption("Entrées : Matière + sorties Bloc 1")
            for t in CASCADE_MECHANICAL_TARGETS:
                lbl = TARGETS.get(t, {}).get("label", t)
                unit = TARGETS.get(t, {}).get("unit", "")
                st.markdown(f"- **{lbl}** `{t}` ({unit})")
        with col3:
            st.markdown("**Bloc 3 — Performance finale**")
            st.caption("Entrées : Matière + sorties Blocs 1 et 2")
            for t in CASCADE_FINAL_TARGETS:
                lbl = TARGETS.get(t, {}).get("label", t)
                unit = TARGETS.get(t, {}).get("unit", "")
                st.markdown(f"- **{lbl}** `{t}` ({unit})")

        st.markdown("---")
        st.markdown("#### Différence entraînement vs. inférence")
        st.info(
            "**Pendant l'entraînement** : chaque bloc utilise les **valeurs mesurées** des propriétés "
            "intermédiaires (vraies OIT, vraies tractions…), ce qui maximise la qualité des modèles.\n\n"
            "**Pendant l'inférence** (sans mesures labo) : le bloc 1 prédit OIT, "
            "puis le bloc 2 reçoit OIT **prédit** (pas mesuré), puis le bloc 3 reçoit "
            "traction/flexion **prédits**. Cette propagation d'erreurs peut légèrement dégrader NCLS/UCLS.\n\n"
            "**Solution** : si vous avez des mesures labo pour OIT, traction ou flexion, "
            "injectez-les dans l'onglet « Prédiction live » pour court-circuiter les blocs intermédiaires.",
            icon="⚠️",
        )

    # --- Tab R² ---
    with tab_r2:
        st.subheader("R² par cible et approche")
        st.caption(
            "**R² (coefficient de détermination)** : mesure la part de variance expliquée par le modèle. "
            "**1.0** = prédiction parfaite. **0** = équivalent à prédire la moyenne. "
            "**Négatif** = le modèle est moins bon que la moyenne — problème de généralisation. "
            "Un R² > 0.7 est généralement considéré bon pour des données industrielles bruitées."
        )
        pivot_r2 = _load_cascade_csv("cascade_pivot_r2")
        if not pivot_r2.empty:
            pivot_r2 = pivot_r2.set_index(pivot_r2.columns[0]) if pivot_r2.columns[0] not in ("target",) else pivot_r2.set_index("target")
            approaches_in_pivot = [c for c in pivot_r2.columns if c in ("direct", "cascade", "pspp")]
            if approaches_in_pivot:
                melted_r2 = pivot_r2[approaches_in_pivot].reset_index().melt(
                    id_vars=pivot_r2.index.name or "target",
                    var_name="approche_key",
                    value_name="R²",
                )
                melted_r2["Approche"] = melted_r2["approche_key"].map(
                    lambda k: CASCADE_APPROACH_LABELS.get(k, k)
                )
                fig = px.bar(
                    melted_r2,
                    x=pivot_r2.index.name or "target",
                    y="R²",
                    color="Approche",
                    barmode="group",
                    title="R² comparé par cible et approche",
                    color_discrete_map={
                        CASCADE_APPROACH_LABELS["direct"]: "#636EFA",
                        CASCADE_APPROACH_LABELS["cascade"]: "#EF553B",
                        CASCADE_APPROACH_LABELS["pspp"]: "#00CC96",
                    },
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pivot_r2.style.format("{:.3f}", na_rep="—"), use_container_width=True)
        else:
            st.info("Données R² non disponibles — exécutez run_phase4.py.")

    # --- Tab MAE ---
    with tab_mae:
        st.subheader("MAE par cible et approche")
        st.caption(
            "**MAE (Mean Absolute Error)** : erreur absolue moyenne en unités réelles de la cible. "
            "Un MAE de 5 sur OIT (minutes) signifie que le modèle se trompe en moyenne de 5 minutes. "
            "Contrairement au R², le MAE est interprétable directement. "
            "**Plus bas = mieux.** Comparez entre approches pour la même cible."
        )
        pivot_mae = _load_cascade_csv("cascade_pivot_mae")
        if not pivot_mae.empty:
            pivot_mae = pivot_mae.set_index(pivot_mae.columns[0])
            approaches_in_pivot = [c for c in pivot_mae.columns if c in ("direct", "cascade", "pspp")]
            if approaches_in_pivot:
                melted_mae = pivot_mae[approaches_in_pivot].reset_index().melt(
                    id_vars=pivot_mae.index.name or "target",
                    var_name="approche_key",
                    value_name="MAE",
                )
                melted_mae["Approche"] = melted_mae["approche_key"].map(
                    lambda k: CASCADE_APPROACH_LABELS.get(k, k)
                )
                fig = px.bar(
                    melted_mae,
                    x=pivot_mae.index.name or "target",
                    y="MAE",
                    color="Approche",
                    barmode="group",
                    title="MAE comparé par cible et approche (plus bas = mieux)",
                    color_discrete_map={
                        CASCADE_APPROACH_LABELS["direct"]: "#636EFA",
                        CASCADE_APPROACH_LABELS["cascade"]: "#EF553B",
                        CASCADE_APPROACH_LABELS["pspp"]: "#00CC96",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pivot_mae.style.format("{:.3f}", na_rep="—"), use_container_width=True)
        else:
            st.info("Données MAE non disponibles — exécutez run_phase4.py.")

    # --- Tab Gain ---
    with tab_gain:
        st.subheader("Gain de la cascade sur les cibles finales (NCLS / UCLS)")
        st.caption(
            "Gain en R² de la Cascade et du PSPP par rapport au modèle Direct (baseline). "
            "Un gain positif = la cascade fait mieux que prédire directement depuis la matière. "
            "Les cibles **NCLS** et **UCLS** bénéficient le plus de l'approche cascade "
            "car elles sont physiquement liées aux propriétés thermiques et mécaniques intermédiaires."
        )
        gain = _load_cascade_csv("cascade_gain")
        winner = _load_cascade_csv("cascade_winner")

        if not gain.empty:
            for _, row in gain.iterrows():
                target = row.get("target", "")
                gain_casc = row.get("gain_cascade_vs_direct", 0)
                gain_pspp = row.get("gain_pspp_vs_direct", 0)
                icon_c = "+" if gain_casc >= 0 else ""
                icon_p = "+" if gain_pspp >= 0 else ""
                st.metric(
                    label=f"{TARGETS.get(target, {}).get('label', target)} — gain R² vs Direct",
                    value=f"Cascade: {icon_c}{gain_casc:.3f}",
                    delta=f"{CASCADE_APPROACH_LABELS['pspp']}: {icon_p}{gain_pspp:.3f}",
                    delta_color="normal",
                )

            st.markdown("---")
            st.markdown(
                "**Lecture** : un gain positif signifie que l'injection des propriétés "
                "intermédiaires (OIT, traction…) améliore la prédiction de NCLS/UCLS "
                "par rapport au modèle direct."
            )

        if not winner.empty:
            st.subheader("Approche gagnante par cible")
            if "approach" in winner.columns:
                winner["Approche"] = winner["approach"].map(CASCADE_APPROACH_LABELS).fillna(winner["approach"])
            cols_show = [c for c in ["target", "Approche", "algorithm", "r2", "mae"] if c in winner.columns]
            st.dataframe(winner[cols_show], use_container_width=True)

        if gain.empty and winner.empty:
            st.info("Données de comparaison non disponibles — exécutez run_phase4.py.")

    # --- Tab Prédiction live ---
    with tab_pred:
        st.subheader("Prédiction live — comparer les approches sur un lot")
        if not available:
            st.warning(
                "Exécutez `python run_phase4.py` pour activer les modèles Cascade et PSPP.",
                icon="⚠️",
            )
        else:
            st.caption(
                "Saisissez les mesures d'un lot pour voir côte à côte les prédictions "
                "de chaque approche disponible. Cela permet d'estimer si la cascade "
                "converge avec le modèle direct ou non."
            )
            with st.form("cascade_live_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    mi = st.number_input(
                        "MI (g/10min)",
                        min_value=0.0, value=0.3, step=0.05,
                        help="Melt Index — indice de fluidité standard.",
                    )
                    hlmi = st.number_input(
                        "HLMI (g/10min)",
                        min_value=0.0, value=8.0, step=0.5,
                        help="High Load Melt Index. Le ratio HLMI/MI (FRR) décrit la distribution des masses.",
                    )
                    density = st.number_input(
                        "Densité (g/cm³)",
                        min_value=0.900, max_value=0.980, value=0.951, step=0.001, format="%.3f",
                        help="Densité du granulé. Plage PE HDPE typique : 0.940–0.965.",
                    )
                with col2:
                    carbon_black = st.number_input(
                        "Noir de carbone (%)",
                        min_value=0.0, max_value=5.0, value=2.5, step=0.1,
                        help="Teneur en noir de carbone — améliore la résistance UV.",
                    )
                    ash = st.number_input(
                        "Cendres (%)",
                        min_value=0.0, max_value=5.0, value=0.1, step=0.05,
                        help="Charges inorganiques résiduelles (silice, talc…).",
                    )
                    pp = st.number_input(
                        "PP (%)",
                        min_value=0.0, max_value=20.0, value=0.0, step=0.5,
                        help="Contamination en polypropylène — dégrade les propriétés mécaniques.",
                    )
                with col3:
                    onset = st.number_input(
                        "DSC Onset (°C)",
                        min_value=100.0, max_value=250.0, value=194.0, step=0.5,
                        help="Température de début de fusion DSC.",
                    )
                    peak = st.number_input(
                        "DSC Peak (°C)",
                        min_value=100.0, max_value=260.0, value=212.0, step=0.5,
                        help="Température de pic de fusion DSC. HDPE : ~130–135°C.",
                    )
                    delta_h = st.number_input(
                        "Delta H (J/g)",
                        min_value=0.0, max_value=300.0, value=168.0, step=1.0,
                        help="Enthalpie de fusion — liée au taux de cristallinité.",
                    )

                submitted = st.form_submit_button(
                    "Prédire avec toutes les approches",
                    type="primary",
                    help="Lance en parallèle les modèles Direct, Cascade et PSPP (si disponibles).",
                )

            if submitted:
                input_row = {
                    "mi": mi, "hlmi": hlmi,
                    "density_g_cm3": density, "density_plaque_g_cm3": density,
                    "carbon_black": carbon_black, "ash": ash, "pp": pp,
                    "onset": onset, "peak": peak, "delta_h": delta_h,
                    "fluidity_g_10min": mi,
                }

                results_table: list[dict] = []
                for approach in available:
                    res = predict_cascade_single(input_row, approach=approach)
                    if res["status"] == "OK":
                        row_d = {"Approche": CASCADE_APPROACH_LABELS.get(approach, approach)}
                        for target, val in res["predictions"].items():
                            label = TARGETS.get(target, {}).get("label", target)
                            unit = TARGETS.get(target, {}).get("unit", "")
                            row_d[f"{label} ({unit})"] = round(val, 2) if val == val else "—"
                        results_table.append(row_d)

                if results_table:
                    import pandas as _pd
                    df_res = _pd.DataFrame(results_table).set_index("Approche")
                    st.dataframe(df_res, use_container_width=True)
                    st.caption(
                        "**Lecture :** des résultats proches entre Direct et Cascade indiquent "
                        "que les propriétés intermédiaires n'apportent pas d'information "
                        "supplémentaire pour ce lot. Un écart significatif suggère que "
                        "la structure thermique ou mécanique joue un rôle clé."
                    )
                else:
                    st.warning("Aucune prédiction disponible pour les approches sélectionnées.")
