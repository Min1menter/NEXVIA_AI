import { useEffect, useState } from "react";
import { decide, getStudy, overlayUrl, scanUrl } from "../api.js";

export default function Viewer({ uid, role, onChanged, onAuthError }) {
  const [study, setStudy] = useState(null);
  const [scan, setScan] = useState(null);
  const [overlay, setOverlay] = useState(null);
  const [label, setLabel] = useState(null);
  const [show, setShow] = useState(true);
  const [opacity, setOpacity] = useState(0.8);
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const fail = (e) => (e.status === 401 ? onAuthError() : setError(e.message));

  // Load study metadata + scan once per study (the parent re-mounts this via key={uid}).
  useEffect(() => {
    let alive = true, url = null;
    (async () => {
      try {
        const s = await getStudy(uid);
        if (!alive) return;
        setStudy(s);
        setLabel(s.overlays.includes(s.top_label) ? s.top_label : s.overlays[0] || null);
        if (s.status !== "error") { url = await scanUrl(uid); if (alive) setScan(url); }
      } catch (e) { if (alive) fail(e); }
    })();
    return () => { alive = false; if (url) URL.revokeObjectURL(url); };
  }, [uid]);

  // Load whichever overlay is selected.
  useEffect(() => {
    if (!label) return;
    let alive = true, url = null;
    overlayUrl(uid, label).then((u) => { url = u; if (alive) setOverlay(u); }).catch((e) => alive && fail(e));
    return () => { alive = false; if (url) URL.revokeObjectURL(url); };
  }, [uid, label]);

  async function submit(action) {
    setBusy(true); setError("");
    try {
      await decide(uid, action, reason.trim());
      setStudy(await getStudy(uid));
      setRejecting(false);
      onChanged();
    } catch (e) { fail(e); }
    finally { setBusy(false); }
  }

  if (!study) return <p className="muted">{error || "Loading..."}</p>;

  const p = study.predictions || {};
  const isSeg = p.kind === "segmentation";
  const scores = Object.entries(p.scores || {}).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const canDecide = role === "radiologist" && study.status === "pending_review";

  return (
    <div className="viewer">
      <div className="imgwrap">
        {scan && <img className="scan" src={scan} alt="scan" />}
        {scan && overlay && show && <img className="ovl" src={overlay} alt="AI overlay" style={{ opacity }} />}
        <div className="wm">EDUCATION ONLY</div>
      </div>

      <div className="side">
        <h3>{study.top_label || "No result"}</h3>
        {study.top_prob != null && (
          <div>
            <div className="muted small">Model confidence (uncalibrated score)</div>
            <div className="bar"><div style={{ width: `${study.top_prob * 100}%` }} /></div>
            <strong>{(study.top_prob * 100).toFixed(1)}%</strong>
          </div>
        )}
        <div className="muted small">Model: {study.model_version || "-"}</div>
        {study.error && <p className="error">Analysis failed: {study.error}</p>}

        {study.overlays.length > 0 && (
          <div className="controls">
            <label><input type="checkbox" checked={show} onChange={(e) => setShow(e.target.checked)} /> Show AI overlay</label>
            {study.overlays.length > 1 && (
              <label>Overlay for
                <select value={label || ""} onChange={(e) => setLabel(e.target.value)}>
                  {study.overlays.map((l) => <option key={l}>{l}</option>)}
                </select>
              </label>
            )}
            <label>Opacity
              <input type="range" min="0.1" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(+e.target.value)} />
            </label>
          </div>
        )}

        {isSeg ? (
          <table className="scores"><tbody>
            <tr><td>Spleen volume</td><td>{p.volume_ml?.toFixed(0)} mL</td></tr>
            <tr><td>Mean voxel confidence</td><td>{(p.mean_confidence * 100).toFixed(1)}%</td></tr>
            <tr><td>Slice shown</td><td>{(p.slice_index ?? 0) + 1} / {p.n_slices}</td></tr>
          </tbody></table>
        ) : (
          <table className="scores"><tbody>
            {scores.map(([k, v]) => <tr key={k}><td>{k}</td><td>{(v * 100).toFixed(1)}%</td></tr>)}
          </tbody></table>
        )}

        <div className="decision">
          {study.status !== "pending_review" && (
            <p className={"badge big " + study.status}>
              {study.status === "approved" ? "Approved" : study.status === "rejected" ? "Rejected" : study.status}
              {study.reviewed_by && ` by ${study.reviewed_by}`}
            </p>
          )}
          {canDecide && !rejecting && (
            <div className="buttons">
              <button className="approve" disabled={busy} onClick={() => submit("approve")}>Approve AI result</button>
              <button className="reject" disabled={busy} onClick={() => setRejecting(true)}>Reject</button>
            </div>
          )}
          {canDecide && rejecting && (
            <div>
              <textarea placeholder="Reason for rejecting (required)" value={reason} onChange={(e) => setReason(e.target.value)} rows={3} />
              <div className="buttons">
                <button className="reject" disabled={busy || !reason.trim()} onClick={() => submit("reject")}>Confirm reject</button>
                <button className="ghost" onClick={() => setRejecting(false)}>Cancel</button>
              </div>
            </div>
          )}
          {role !== "radiologist" && study.status === "pending_review" && <p className="muted small">Only radiologists can approve or reject.</p>}
          {error && <p className="error">{error}</p>}
        </div>
      </div>
    </div>
  );
}
