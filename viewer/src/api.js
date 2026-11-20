// Thin API client. The JWT lives in sessionStorage (cleared when the tab closes).
const BASE = "/api";
let token = sessionStorage.getItem("token");

export const hasToken = () => !!token;
export function setToken(t) {
  token = t;
  if (t) sessionStorage.setItem("token", t);
  else sessionStorage.removeItem("token");
}

async function errorFrom(res) {
  let msg = res.statusText || `HTTP ${res.status}`;
  try {
    const j = await res.json();
    if (typeof j.detail === "string") msg = j.detail;
    else if (j.detail) msg = JSON.stringify(j.detail);
  } catch { /* non-JSON body */ }
  const err = new Error(msg);
  err.status = res.status;
  return err;
}

async function request(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(BASE + path, { ...opts, headers });
  if (!res.ok) throw await errorFrom(res);
  return res;
}

export async function login(username, password) {
  // The API uses OAuth2PasswordRequestForm -> form-encoded body.
  const res = await request("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ username, password }),
  });
  const data = await res.json();
  setToken(data.access_token);
  return { username: data.username, role: data.role };
}

export const listStudies = async () => (await request("/studies")).json();
export const getStudy = async (uid) => (await request(`/studies/${encodeURIComponent(uid)}`)).json();
export const decide = async (uid, action, reason) =>
  (await request(`/studies/${encodeURIComponent(uid)}/decision`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, reason: reason || "" }),
  })).json();
export const auditList = async (limit = 50) => (await request(`/audit?limit=${limit}`)).json();
export const auditVerify = async () => (await request("/audit/verify")).json();

// <img> cannot send an Authorization header, so fetch the PNG and hand back a blob: URL.
export async function imageBlobUrl(path) {
  const res = await request(path);
  return URL.createObjectURL(await res.blob());
}
export const scanUrl = (uid) => imageBlobUrl(`/studies/${encodeURIComponent(uid)}/image`);
export const overlayUrl = (uid, label) =>
  imageBlobUrl(`/studies/${encodeURIComponent(uid)}/overlay/${encodeURIComponent(label)}`);
