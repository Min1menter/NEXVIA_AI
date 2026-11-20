import { useCallback, useEffect, useState } from "react";
import { hasToken, listStudies, setToken } from "./api.js";
import Login from "./components/Login.jsx";
import StudyList from "./components/StudyList.jsx";
import Viewer from "./components/Viewer.jsx";
import AuditPanel from "./components/AuditPanel.jsx";

export default function App() {
  const [user, setUser] = useState(() => {
    try { return hasToken() ? JSON.parse(sessionStorage.getItem("user")) : null; } catch { return null; }
  });
  const [studies, setStudies] = useState([]);
  const [selected, setSelected] = useState(null);
  const [tab, setTab] = useState("worklist");

  const logout = useCallback(() => {
    setToken(null);
    sessionStorage.removeItem("user");
    setUser(null);
    setSelected(null);
  }, []);

  const refresh = useCallback(async () => {
    try { setStudies(await listStudies()); }
    catch (e) { if (e.status === 401) logout(); }
  }, [logout]);

  useEffect(() => {
    if (!user) return;
    refresh();
    const id = setInterval(refresh, 5000); // new AI results show up without a reload
    return () => clearInterval(id);
  }, [user, refresh]);

  function onLogin(u) {
    sessionStorage.setItem("user", JSON.stringify(u));
    setUser(u);
  }

  return (
    <div className="app">
      <div className="banner" role="alert">
        FOR EDUCATION ONLY - NOT FOR CLINICAL USE. AI output may be wrong; never use it for patient care.
      </div>
      {!user ? (
        <Login onLogin={onLogin} />
      ) : (
        <>
          <header className="topbar">
            <strong>MedImaging AI</strong>
            <nav>
              <button className={tab === "worklist" ? "tab active" : "tab"} onClick={() => setTab("worklist")}>Worklist</button>
              <button className={tab === "audit" ? "tab active" : "tab"} onClick={() => setTab("audit")}>Audit log</button>
            </nav>
            <span className="who">{user.username} ({user.role})</span>
            <button className="ghost" onClick={logout}>Sign out</button>
          </header>
          {tab === "worklist" ? (
            <main className="layout">
              <StudyList studies={studies} selected={selected} onSelect={setSelected} />
              <section className="pane">
                {selected ? (
                  <Viewer key={selected} uid={selected} role={user.role} onChanged={refresh} onAuthError={logout} />
                ) : (
                  <p className="muted">Select a study from the worklist.</p>
                )}
              </section>
            </main>
          ) : (
            <AuditPanel onAuthError={logout} />
          )}
        </>
      )}
    </div>
  );
}
