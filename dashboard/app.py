"""Analyst dashboard (Streamlit): alert queue, case workflow, feedback labels, KPIs, drift status, live scorer."""
import os
from datetime import datetime, timezone
import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
KEY = os.getenv("API_KEY", "")
H = {"X-API-Key": KEY} if KEY else {}
STATUSES = ["new", "investigating", "confirmed_fraud", "false_positive"]

st.set_page_config(page_title="FraudShield analyst console", layout="wide")


def get(path, **params):
    r = requests.get(f"{API}{path}", params=params, headers=H, timeout=15)
    r.raise_for_status()
    return r.json()


def call(method, path, **kw):
    r = requests.request(method, f"{API}{path}", headers=H, timeout=30, **kw)
    if not r.ok:
        st.error(f"{r.status_code}: {r.text[:200]}")
        return None
    return r.json()


st.title("FraudShield analyst console")
analyst = st.sidebar.text_input("Your name", value="analyst1")
status_f = st.sidebar.selectbox("Status", ["(all)"] + STATUSES, index=1)
min_score = st.sidebar.slider("Minimum score", 0.0, 1.0, 0.0, 0.01)
if st.sidebar.button("Refresh"):
    st.rerun()

try:
    stats = get("/stats")
except Exception as e:
    st.error(f"Cannot reach API at {API}: {e}")
    st.stop()

tab_alerts, tab_kpi, tab_drift, tab_live = st.tabs(["Alert queue", "KPIs", "Model health", "Score a transaction"])

with tab_alerts:
    alerts = get("/alerts", status=None if status_f == "(all)" else status_f, min_score=min_score, limit=300)
    if not alerts:
        st.info("No alerts match these filters. Start the producer (or `make replay`) to generate traffic.")
    else:
        df = pd.DataFrame(alerts)
        df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        view = df[["id", "time", "user_id", "amount", "score", "xgb_score", "graph_score", "cluster_size", "status", "assignee"]]
        sel = st.dataframe(view, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")
        rows = sel.selection.rows if sel and sel.selection else []
        if rows:
            a = alerts[rows[0]]
            st.subheader(f"Case #{a['id']} - {a['txn_id']} ({a['user_id']})")
            c1, c2, c3 = st.columns(3)
            c1.metric("Final score", f"{a['score']:.3f}")
            c2.metric("XGBoost", f"{a['xgb_score']:.3f}")
            c3.metric("Graph (GNN)", "n/a" if a["graph_score"] is None else f"{a['graph_score']:.3f}",
                      help=f"Accounts linked via shared devices/cards: {a['cluster_size']}")
            if a.get("cluster_size") and a["cluster_size"] > 1:
                try:
                    ring = get(f"/rings/{a['user_id']}")
                    with st.expander(f"Fraud-ring candidate: {ring['size']} accounts linked by shared devices/cards"):
                        st.write(", ".join(ring["members"]) + (" ..." if ring["size"] > len(ring["members"]) else ""))
                except Exception:
                    st.caption("Linked-account list unavailable (run the graph step).")
            st.markdown("**Why it was flagged**")
            for r in a["reasons"] or []:
                st.write(f"- `{r['feature']}` = {r['value']} (contribution {r['contribution']:+.2f})")
            note = st.text_area("Case note", value=a["note"], key=f"note{a['id']}")
            b1, b2, b3 = st.columns(3)
            for col, label, status in ((b1, "Start investigating", "investigating"), (b2, "Confirm fraud", "confirmed_fraud"),
                                       (b3, "Mark false positive", "false_positive")):
                if col.button(label, key=f"{status}{a['id']}"):
                    if call("PATCH", f"/alerts/{a['id']}", json={"status": status, "assignee": analyst, "note": note}):
                        st.success(f"Case #{a['id']} -> {status}. Label saved for retraining.")
                        st.rerun()

with tab_kpi:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Transactions scored", f"{stats['scored']:,}")
    k2.metric("Alerts", f"{stats['alerts']:,}", f"{stats['alert_rate']:.2%} of traffic")
    k3.metric("Reviewed", stats["reviewed"])
    k4.metric("Alert precision (reviewed)", "n/a" if stats["alert_precision"] is None else f"{stats['alert_precision']:.0%}")
    if stats["by_status"]:
        st.bar_chart(pd.Series(stats["by_status"], name="alerts"))

with tab_drift:
    try:
        d = get("/drift/latest")
        if d.get("status") == "insufficient_data":
            st.info(f"Need {d['min_rows']} scored transactions (have {d['n_current']}).")
        else:
            st.metric("Features drifted", f"{d['share_drifted']:.0%}", "DRIFT" if d["dataset_drift"] else "stable")
            st.dataframe(pd.DataFrame(d["features"]).T.sort_values("psi", ascending=False), use_container_width=True)
            if d.get("retrain_triggered"):
                st.warning("Retraining triggered: " + "; ".join(d["retrain_reasons"]))
            st.markdown(f"[Open full Evidently report]({API}/drift/report.html)")
    except requests.HTTPError:
        st.info("No drift report yet. Run `make drift`.")

with tab_live:
    with st.form("live"):
        c = st.columns(4)
        user = c[0].text_input("user_id", "U000001")
        amount = c[1].number_input("amount", 0.0, 1e7, 5000.0)
        typ = c[2].selectbox("type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        c2 = st.columns(4)
        dev = c2[0].text_input("device_id", "NEW-DEVICE-1")
        ctry = c2[1].text_input("country", "NG")
        commit = c2[2].checkbox("Update user state", value=False)
        if st.form_submit_button("Score"):
            res = call("POST", f"/score?commit={str(commit).lower()}", json=dict(
                txn_id=f"MANUAL-{int(datetime.now(timezone.utc).timestamp()*1000)}", user_id=user,
                timestamp=datetime.now(timezone.utc).timestamp(), amount=amount, type=typ, device_id=dev, country=ctry))
            if res:
                st.json(res)
