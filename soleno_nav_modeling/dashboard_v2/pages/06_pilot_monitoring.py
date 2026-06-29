"""Page 6 — Suivi du pilote (niveau Pilote contrôlé)."""
import _bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import show_plotly
from components.layout import page_header, render_expert_toggle_sidebar
from config import TARGETS
from v2_config.settings import PILOT_JOURNAL_FILE
from services.pilot_journal import (
    delete_pilot_predictions,
    delete_test_predictions,
    is_test_prediction,
    load_pilot_journal,
    pilot_summary_stats,
    update_lab_feedback,
)

render_expert_toggle_sidebar()
page_header(
    "Suivi du pilote",
    "Validation terrain — comparaison estimation / laboratoire et apprentissage organisationnel.",
    "pilot",
)

df = load_pilot_journal()
stats = pilot_summary_stats(df)

st.markdown("### Questions du pilote")
st.markdown(
    """
- Le modèle fonctionne-t-il sur les **nouveaux lots** ?
- Fonctionne-t-il sur des **fournisseurs jamais vus** ?
- Fonctionne-t-il **dans le temps** ?
- Quelles propriétés sont **utiles** aux utilisateurs ?
- Dans quels cas les erreurs deviennent-elles **importantes** ?
- Quel niveau d'**incertitude** est acceptable pour Soleno ?
    """
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Prédictions réalisées", stats["n_predictions"])
c2.metric("Résultats labo reçus", stats["n_lab_received"])
c3.metric("En attente labo", stats["n_pending"])
c4.metric("MAE terrain", f"{stats['mae_field']:.3g}" if stats["mae_field"] else "—")
c5.metric("Erreur relative moy. %", f"{stats['mean_rel_error']:.1f}" if stats["mean_rel_error"] else "—")

if df.empty:
    st.info(
        f"Journal pilote vide. Les prédictions de la page « Prédiction pilote » "
        f"sont enregistrées dans `{PILOT_JOURNAL_FILE}`."
    )
else:
    st.subheader("Gestion du journal")
    test_mask = df.apply(is_test_prediction, axis=1)
    n_test = int(test_mask.sum())
    if n_test:
        st.caption(f"{n_test} prédiction(s) identifiée(s) comme test (lot_pilote, lot_manuel, TEST-…).")
        c_del1, c_del2 = st.columns(2)
        with c_del1:
            if st.button(f"Supprimer les {n_test} prédiction(s) de test", type="secondary"):
                removed = delete_test_predictions()
                st.success(f"{removed} entrée(s) supprimée(s).")
                st.rerun()
        with c_del2:
            sel_del = st.multiselect(
                "Supprimer des entrées sélectionnées",
                df["prediction_id"].astype(str).tolist(),
                format_func=lambda pid: (
                    f"{pid} — {df.loc[df['prediction_id'].astype(str) == pid, 'lot_id'].iloc[0]}"
                ),
                key="v2_pilot_delete_ids",
            )
            if st.button("Supprimer la sélection", disabled=not sel_del):
                removed = delete_pilot_predictions(sel_del)
                st.success(f"{removed} entrée(s) supprimée(s).")
                st.rerun()
    else:
        sel_del = st.multiselect(
            "Supprimer des entrées du journal",
            df["prediction_id"].astype(str).tolist(),
            format_func=lambda pid: (
                f"{pid} — {df.loc[df['prediction_id'].astype(str) == pid, 'lot_id'].iloc[0]}"
            ),
            key="v2_pilot_delete_ids",
        )
        if st.button("Supprimer la sélection", disabled=not sel_del):
            removed = delete_pilot_predictions(sel_del)
            st.success(f"{removed} entrée(s) supprimée(s).")
            st.rerun()

    st.subheader("Journal des prédictions")
    show = df.copy()
    show["Test ?"] = test_mask.map({True: "oui", False: "—"})
    if "target" in show.columns:
        show["Propriété"] = show["target"].map(lambda t: TARGETS.get(t, {}).get("label", t))
    st.dataframe(show, use_container_width=True, hide_index=True)

    lab_df = df[pd.to_numeric(df["lab_result"], errors="coerce").notna()]
    if not lab_df.empty:
        st.subheader("Erreur par fournisseur")
        if "supplier_code" in lab_df.columns:
            by_sup = lab_df.groupby("supplier_code")["abs_error"].mean().reset_index()
            by_sup.columns = ["Fournisseur", "MAE"]
            by_sup = by_sup[by_sup["Fournisseur"].astype(str).str.len() > 0]
            if not by_sup.empty:
                show_plotly(
                    px.bar(by_sup.head(15), x="Fournisseur", y="MAE", title="MAE terrain par fournisseur")
                )

        st.subheader("Erreur par propriété")
        by_t = lab_df.groupby("target")["rel_error_pct"].mean().reset_index()
        by_t["label"] = by_t["target"].map(lambda t: TARGETS.get(t, {}).get("label", t))
        if not by_t.empty:
            show_plotly(px.bar(by_t, x="label", y="rel_error_pct", title="Erreur relative moyenne (%)"))

        st.subheader("Prédictions les plus éloignées")
        worst = lab_df.nlargest(10, "abs_error")[
            ["lot_id", "target", "predicted_value", "lab_result", "abs_error", "user_comment"]
        ]
        st.dataframe(worst, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Retour laboratoire et décision réelle")
pending = df[pd.to_numeric(df["lab_result"], errors="coerce").isna()] if not df.empty else pd.DataFrame()
if pending.empty and df.empty:
    st.caption("Aucune prédiction en attente.")
elif pending.empty:
    st.success("Toutes les prédictions ont un résultat laboratoire.")
else:
    pid = st.selectbox(
        "Prédiction à compléter",
        pending["prediction_id"].astype(str).tolist(),
        format_func=lambda x: f"{x} — lot {pending.loc[pending['prediction_id'].astype(str)==x, 'lot_id'].iloc[0]}",
    )
    lab_val = st.number_input("Résultat laboratoire", value=0.0)
    comment = st.text_area("Commentaire utilisateur")
    decision = st.selectbox(
        "Décision réelle prise par l'équipe",
        ["", "Essai complémentaire planifié", "Lot utilisé avec prudence", "Lot écarté", "Autre"],
    )
    if st.button("Enregistrer le retour"):
        if update_lab_feedback(pid, lab_val, user_comment=comment, actual_decision=decision):
            st.success("Retour enregistré.")
            st.rerun()
        else:
            st.error("Échec de mise à jour.")
