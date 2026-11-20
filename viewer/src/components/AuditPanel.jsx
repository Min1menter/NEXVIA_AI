import { useEffect, useState } from "react";
import { auditList, auditVerify } from "../api.js";

export default function AuditPanel({ onAuthError }) {
  const [rows, setRows] = useState([]);
  const [verify, setVerify] = useState(null);
  const [error, setError] = useState("");

  const fail = (e) => (e.status === 401 ? onAuthError() : setError(e.message));
  useEffect(() => { auditList(100).then(setRows).catch(fail); }, []);

  async function runVerify() {
    setError("");
    try { setVerify(await auditVerify()); } catch (e) { fail(e); }
  }

  return (
    <main className="audit">
      <div className="row">
        <h2>Audit log</h2>
        <button onClick={runVerify}>Verify hash chain</button>
      </div>
      {verify && (
        <p className={verify.ok ? "ok" : "error"}>
          {verify.ok
            ? `Chain intact: ${verify.checked} entries verified. Head ${verify.head_hash.slice(0, 16)}...`
            : `TAMPERING DETECTED at entry #${verify.first_bad_id} (after ${verify.checked} entries checked).`}
        </p>
      )}
      {error && <p className="error">{error}</p>}
      <table className="log">
        <thead><tr><th>#</th><th>Time (UTC)</th><th>Actor</th><th>Action</th><th>Study</th><th>Reason</th><th>Hash</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id}>
              <td>{r.id}</td><td>{r.ts}</td><td>{r.actor}</td><td>{r.action}</td>
              <td title={r.study_uid}>{r.study_uid ? "..." + r.study_uid.slice(-8) : ""}</td>
              <td>{r.reason}</td><td className="mono">{r.hash.slice(0, 10)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  );
}
