const LABEL = { pending_review: "Pending", approved: "Approved", rejected: "Rejected", error: "Error" };

export default function StudyList({ studies, selected, onSelect }) {
  if (!studies.length) {
    return <aside className="list"><p className="muted pad">No studies yet. Upload DICOM to Orthanc and wait a few seconds.</p></aside>;
  }
  return (
    <aside className="list">
      {studies.map((s) => (
        <button key={s.study_uid} className={"item" + (selected === s.study_uid ? " sel" : "")} onClick={() => onSelect(s.study_uid)}>
          <div className="row">
            <span className="chip">{s.modality || "?"}</span>
            <span className={"badge " + s.status}>{LABEL[s.status] || s.status}</span>
          </div>
          <div className="title">{s.top_label || "(no result)"}{s.top_prob != null && ` - ${(s.top_prob * 100).toFixed(0)}%`}</div>
          <div className="muted small">{s.patient_id || "unknown patient"} - {new Date(s.created_at).toLocaleString()}</div>
        </button>
      ))}
    </aside>
  );
}
