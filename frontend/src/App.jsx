import { createContext, useContext, useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { NAV } from "./lib/constants.js";
import { apiGet } from "./lib/api.js";

import Overview from "./pages/Overview.jsx";
import LiveFeed from "./pages/LiveFeed.jsx";
import Investigation from "./pages/Investigation.jsx";
import FalsePositiveLab from "./pages/FalsePositiveLab.jsx";
import ReviewQueue from "./pages/ReviewQueue.jsx";
import SelfAudit from "./pages/SelfAudit.jsx";
import CalibrationLab from "./pages/CalibrationLab.jsx";
import PolicyLab from "./pages/PolicyLab.jsx";
import FraudCycle from "./pages/FraudCycle.jsx";
import Architecture from "./pages/Architecture.jsx";

const DatasetCtx = createContext({ dataset: "synthetic", setDataset: () => {} });
export const useDataset = () => useContext(DatasetCtx);

function ApiStatus() {
  const [s, setS] = useState({ state: "checking", info: null });
  useEffect(() => {
    let alive = true;
    const ping = () =>
      apiGet("/kpis?dataset=synthetic")
        .then((k) => alive && setS({ state: "up", info: `${k.n_transactions} txns · ${k.model_version}` }))
        .catch((e) => alive && setS({ state: "down", info: String(e.message || e).slice(0, 80) }));
    ping();
    const id = setInterval(ping, 15000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);
  const dot = { checking: "bg-slate-300", up: "bg-emerald-500", down: "bg-rose-500" }[s.state];
  const label = { checking: "checking API…", up: "API connected", down: "API offline" }[s.state];
  return (
    <span className="flex items-center gap-1.5 text-xs text-slate-500" title={s.info || ""}>
      <span className={`h-2 w-2 rounded-full ${dot}`} />
      {label}
    </span>
  );
}

export default function App() {
  const [dataset, setDataset] = useState("synthetic");

  return (
    <DatasetCtx.Provider value={{ dataset, setDataset }}>
      <div className="flex min-h-screen">
        <aside className="hidden w-60 shrink-0 border-r border-slate-200 bg-white lg:block">
          <div className="border-b border-slate-100 px-4 py-4">
            <div className="text-sm font-semibold text-slate-900">SafeTriage</div>
            <div className="text-xs text-slate-500">Fraud triage console</div>
          </div>
          <nav className="flex flex-col gap-0.5 p-2">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                end={n.end}
                className={({ isActive }) =>
                  `rounded px-3 py-2 text-sm ${
                    isActive ? "bg-indigo-50 font-medium text-indigo-700" : "text-slate-600 hover:bg-slate-50"
                  }`
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          <header className="flex items-center justify-between border-b border-slate-200 bg-white px-5 py-3">
            <div className="flex items-center gap-4">
              <div className="text-sm text-slate-500">
                A false-positive-aware human-AI fraud triage &amp; self-auditing prototype
              </div>
              <ApiStatus />
            </div>
            <label className="flex items-center gap-2 text-xs text-slate-600">
              Dataset
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                className="rounded border border-slate-300 bg-white px-2 py-1 text-xs font-medium text-slate-800"
              >
                <option value="synthetic">synthetic (behavioural)</option>
                <option value="ieee">ieee-cis (reduced)</option>
              </select>
            </label>
          </header>

          <main className="mx-auto w-full max-w-6xl flex-1 p-5">
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/feed" element={<LiveFeed />} />
              <Route path="/investigation" element={<Investigation />} />
              <Route path="/investigation/:id" element={<Investigation />} />
              <Route path="/fp-lab" element={<FalsePositiveLab />} />
              <Route path="/review-queue" element={<ReviewQueue />} />
              <Route path="/self-audit" element={<SelfAudit />} />
              <Route path="/calibration" element={<CalibrationLab />} />
              <Route path="/policy-lab" element={<PolicyLab />} />
              <Route path="/cycle" element={<FraudCycle />} />
              <Route path="/architecture" element={<Architecture />} />
            </Routes>
          </main>
        </div>
      </div>
    </DatasetCtx.Provider>
  );
}
