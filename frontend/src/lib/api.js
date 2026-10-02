import { useCallback, useEffect, useRef, useState } from "react";

const BASE = "/api";

export async function apiGet(path) {
  const res = await fetch(BASE + path);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`GET ${path} → ${res.status} ${text.slice(0, 200)}`);
  }
  return res.json();
}

export async function apiPost(path, body) {
  const res = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`POST ${path} → ${res.status} ${text.slice(0, 200)}`);
  }
  return res.json();
}

/** Fetch `path` (a string, or a function returning one) and re-fetch when `deps` change. */
export function useApi(path, deps = []) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const seq = useRef(0);
  const resolvedPath = typeof path === "function" ? path() : path;

  const reload = useCallback(() => {
    const mine = ++seq.current;
    setState((s) => ({ ...s, loading: true, error: null }));
    apiGet(resolvedPath)
      .then((data) => {
        if (mine === seq.current) setState({ data, error: null, loading: false });
      })
      .catch((error) => {
        if (mine === seq.current) setState({ data: null, error, loading: false });
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resolvedPath]);

  useEffect(() => {
    if (resolvedPath == null) return;
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { ...state, reload };
}
