import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useDataset } from "../App.jsx";
import { apiGet, apiPost } from "../lib/api.js";
import { fixed, pct, titleCase, shortId } from "../lib/format.js";
import { SectionCard, StatTile, DecisionBadge, Loading, ErrorState, Banner, KeyValue } from "../components/ui.jsx";

const FEEDBACK = [
  ["CONFIRM_FRAUD", "Confirm fraud", "bg-rose-600"],
  ["MARK_LEGITIMATE", "Mark legitimate", "bg-emerald-600"],
  ["NEED_MORE_EVIDENCE", "Need more evidence", "bg-amber-600"],
];

function ShapBar({ f }) {
  const mag = Math.min(1, Math.abs(f.shap_value) / 2);
  const pos = f.shap_value > 0;
  return (
    <div className="flex items-center gap-2 text-xs">
      <div className="w-44 shrink-0 truncate text-slate-600" title={f.feature}>
        {f.friendly}
      </div>
      <div className="relative h-3 flex-1 rounded bg-slate-100">
        <div
          className={`absolute top-0 h-3 rounded ${pos ? "left-1/2 bg-rose-400" : "right-1/2 bg-emerald-400"}`}
          style={{ width: `${mag * 50}%` }}
        />
        <div className="absolute left-1/2 top-0 h-3 w-px bg-slate-300" />
      </div>
      <div className={`w-14 text-right tabular-nums ${pos ? "text-rose-600" : "text-emerald-600"}`}>
        {f.shap_value > 0 ? "+" : ""}
        {fixed(f.shap_value, 2)}
      </div>
    </div>
  );
}

export default function Investigation() {
  const { id } = useParams();
  const nav = useNavigate();
  const { dataset } = useDataset();
  const [stored, setStored] = useState(null);
  const [score, setScore] = useState(null);
  const [state, setState] = useState({ loading: !!id, error: null });
  const [pickList, setPickList] = useState([]);
  const [fb, setFb] = useState({ sending: null, done: null, error: null });

  useEffect(() => {
    if (!id) {
      apiGet(`/transactions?dataset=${dataset}&decision=STEP_UP_REVIEW&limit=25`)
        .then((d) => setPickList(d.items || []))
        .catch(() => setPickList([]));
      return;
    }
    setState({ loading: true, error: null });
    setFb({ sending: null, done: null, error: null });
    Promise.all([apiGet(`/transactions/${id}`), apiPost("/score", { transaction_id: id })])
      .then(([s, sc]) => {
        setStored(s);
        setScore(sc);
        setState({ loading: false, error: null });
      })
      .catch((error) => setState({ loading: false, error }));
  }, [id, dataset]);

  if (!id) {
    return (
      <div className="space-y-5">
        <h1 className="text-lg font-semibold text-slate-900">Transaction Investigation</h1>
        <SectionCard title="Pick a review case" subtitle="Or open one from the Live Feed / Review Queue">
          {pickList.length === 0 ? (
            <Loading />
          ) : (
            <ul className="divide-y divide-slate-100 text-sm">
              {pickList.map((r) => (
                <li key={r.transaction_id}>
                  <button
                    onClick={() => nav(`/investigation/${r.transaction_id}`)}
                    className="flex w-full items-center justify-between py-2 hover:text-indigo-700"
                  >
                    <span className="font-mono text-xs">{shortId(r.transaction_id)}</span>
                    <span className="tabular-nums text-slate-500">p={fixed(r.calibrated_probability, 4)}</span>
                    <DecisionBadge decision={r.decision} />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </SectionCard>
      </div>
    );
  }

  if (state.loading) return <Loading />;
  if (state.error) return <ErrorState error={state.error} onRetry={() => nav(0)} />;

  const raw = stored?.raw || {};
  const gt = stored?.is_fraud_label;
  const comps = score.behavioral_components || {};
  const graph = score.graph_proxy;

  const sendFeedback = (decisionValue) => {
    setFb({ sending: decisionValue, done: null, error: null });
    apiPost("/feedback", { transaction_id: id, analyst_decision: decisionValue, reason_tag: "investigation_ui" })
      .then((r) => setFb({ sending: null, done: r.stored, error: null }))
      .catch((error) => setFb({ sending: null, done: null, error }));
  };

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-slate-900">
            Investigation · <span className="font-mono">{shortId(id)}</span>
          </h1>
          <p className="text-sm text-slate-500">
            dataset <code>{score.dataset}</code> · {score.latency_ms} ms · model <code>{score.model_version}</code>
          </p>
        </div>
        <DecisionBadge decision={score.decision} />
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile label="Raw XGB p(fraud)" value={fixed(score.fraud_probability, 4)} />
        <StatTile label="Calibrated p(fraud)" value={fixed(score.calibrated_probability, 4)} tone="bad" />
        <StatTile label="Behavioural consistency" value={fixed(score.behavioral_consistency_score, 3)} tone="good" />
        <StatTile label="Disagreement uncertainty" value={fixed(score.uncertainty, 3)} tone="warn" />
      </div>

      <SectionCard title="Policy decision" subtitle={`rule: ${score.policy_rule} · reason code: ${score.fraud_reason_code}`}>
        <p className="text-sm text-slate-700">{score.decision_explanation}</p>
        <div className="mt-2 text-xs text-slate-500">
          predicted typology: <strong>{titleCase(score.predicted_typology) || "—"}</strong>
          {gt != null && (
            <>
              {"  ·  "}ground truth:{" "}
              <strong className={gt ? "text-rose-600" : "text-emerald-600"}>{gt ? "fraud" : "legitimate"}</strong>
            </>
          )}
        </div>
      </SectionCard>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <SectionCard title="Model explanation (SHAP)" subtitle="Top feature contributions to the XGBoost margin — model level, not the decision">
          {score.top_features?.length ? (
            <div className="space-y-2">
              {score.top_features.map((f) => (
                <ShapBar key={f.feature} f={f} />
              ))}
              <div className="mt-2 flex justify-between text-[11px] text-slate-400">
                <span>← lowers fraud risk</span>
                <span>raises fraud risk →</span>
              </div>
            </div>
          ) : (
            <p className="text-sm text-slate-400">No SHAP available for this dataset.</p>
          )}
        </SectionCard>

        <SectionCard title="Behavioural consistency components" subtitle="Computed independently of the fraud score">
          <div className="space-y-1.5">
            {Object.entries(comps).map(([k, v]) => (
              <div key={k} className="flex items-center gap-2 text-xs">
                <div className="w-40 shrink-0 text-slate-600">{titleCase(k)}</div>
                <div className="h-2 flex-1 rounded bg-slate-100">
                  <div className="h-2 rounded bg-emerald-400" style={{ width: `${Math.max(0, Math.min(1, v)) * 100}%` }} />
                </div>
                <div className="w-10 text-right tabular-nums text-slate-500">{fixed(v, 2)}</div>
              </div>
            ))}
          </div>
          {score.is_behavioral_proxy && (
            <Banner tone="proxy" >Reduced mode — components are C*/D* proxies (IEEE-CIS has no behavioural fields).</Banner>
          )}
        </SectionCard>
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <SectionCard title="Evidence" subtitle="From SHAP direction">
          <div className="space-y-3 text-sm">
            <div>
              <div className="text-xs font-semibold uppercase tracking-wide text-rose-600">Supporting (toward fraud)</div>
              <ul className="mt-1 list-disc pl-5 text-slate-700">
                {(score.supporting_evidence || []).map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
                {!score.supporting_evidence?.length && <li className="list-none text-slate-400">none</li>}
              </ul>
            </div>
            <div>
              <div className="text-xs font-semibold uppercase tracking-wide text-emerald-600">Counter (against fraud)</div>
              <ul className="mt-1 list-disc pl-5 text-slate-700">
                {(score.counter_evidence || []).map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
                {!score.counter_evidence?.length && <li className="list-none text-slate-400">none</li>}
              </ul>
            </div>
          </div>
        </SectionCard>

        <SectionCard title="Next best evidence" subtitle="Deterministic template recommendation — no LLM">
          {score.next_best_evidence?.length ? (
            <ol className="space-y-2 text-sm">
              {score.next_best_evidence.map((e, i) => (
                <li key={i} className="rounded border border-slate-200 p-2">
                  <div className="font-medium text-slate-800">{e.evidence}</div>
                  <div className="text-xs text-slate-500">{e.rationale}</div>
                  <div className="mt-0.5 text-[11px] text-slate-400">source: {e.source}</div>
                </li>
              ))}
            </ol>
          ) : (
            <p className="text-sm text-slate-400">No recommendations (AUTO_CLEAR).</p>
          )}
        </SectionCard>
      </div>

      {graph && (
        <SectionCard title="Graph proxy" subtitle={graph.note}>
          <GraphView graph={graph} />
        </SectionCard>
      )}

      <SectionCard title="Transaction fields" subtitle="Stored raw record">
        <KeyValue
          items={Object.entries(raw)
            .filter(([k]) => !k.startsWith("_") && k !== "isFraud" && k !== "fraud_typology")
            .slice(0, 40)
            .map(([k, v]) => [k, typeof v === "number" ? (Number.isInteger(v) ? v : fixed(v, 3)) : String(v)])}
        />
      </SectionCard>

      <SectionCard title="Human feedback" subtitle="Stored for future recalibration (not applied live)">
        <div className="flex flex-wrap gap-2">
          {FEEDBACK.map(([val, label, color]) => (
            <button
              key={val}
              disabled={fb.sending}
              onClick={() => sendFeedback(val)}
              className={`rounded px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50 ${color}`}
            >
              {fb.sending === val ? "Saving…" : label}
            </button>
          ))}
        </div>
        {fb.done && (
          <p className="mt-2 text-xs text-emerald-700">
            Saved feedback #{fb.done.id}: {fb.done.analyst_decision} (original decision {fb.done.original_policy_decision}).
          </p>
        )}
        {fb.error && <p className="mt-2 text-xs text-rose-600">{String(fb.error.message || fb.error)}</p>}
      </SectionCard>
    </div>
  );
}

function GraphView({ graph }) {
  const nodes = graph.nodes || [];
  const edges = graph.edges || [];
  const W = 640;
  const H = 60 + Math.ceil(nodes.length / 4) * 90;
  const pos = {};
  nodes.forEach((n, i) => {
    pos[n.id] = { x: 80 + (i % 4) * 160, y: 50 + Math.floor(i / 4) * 90 };
  });
  const colorOf = (risk) =>
    ({ subject: "#4f46e5", flagged: "#e11d48", mule: "#b91c1c", intermediary: "#f59e0b", new: "#d97706", known: "#64748b" }[risk] || "#64748b");

  return (
    <div className="thin-scroll overflow-x-auto">
      <svg width={W} height={H} className="min-w-full">
        {edges.map((e, i) => {
          const a = pos[e.source];
          const b = pos[e.target];
          if (!a || !b) return null;
          return (
            <g key={i}>
              <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#cbd5e1" strokeWidth="1.5" />
              <text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 - 4} fontSize="9" fill="#94a3b8" textAnchor="middle">
                {e.label}
              </text>
            </g>
          );
        })}
        {nodes.map((n) => {
          const p = pos[n.id];
          return (
            <g key={n.id}>
              <circle cx={p.x} cy={p.y} r="14" fill={colorOf(n.risk)} opacity="0.9" />
              <text x={p.x} y={p.y + 28} fontSize="10" fill="#475569" textAnchor="middle">
                {n.label?.slice(0, 22)}
              </text>
              <text x={p.x} y={p.y + 3} fontSize="8" fill="#fff" textAnchor="middle">
                {n.type?.[0]?.toUpperCase()}
              </text>
            </g>
          );
        })}
      </svg>
      <p className="text-xs text-slate-400">
        Illustrative only — {graph.shared_device_with_flagged_account ? "shares a device with a flagged account; " : ""}
        {graph.hop_distance_to_known_mule != null ? `${graph.hop_distance_to_known_mule} hop(s) to a known mule.` : "no mule-network proximity."}
      </p>
    </div>
  );
}
