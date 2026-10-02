import { useDataset } from "../App.jsx";
import { useApi } from "../lib/api.js";
import { SectionCard, Loading } from "../components/ui.jsx";

const LIVE = [
  "Dataset processing (IEEE-CIS + synthetic behavioural)",
  "XGBoost fraud classifier",
  "Random Forest (disagreement proxy only)",
  "Probability calibration — sigmoid/isotonic, Brier + ECE",
  "Behavioural-consistency scoring (6 components)",
  "Model-disagreement uncertainty proxy",
  "Policy engine — thresholds, capacity, cost",
  "SHAP explanations + deterministic reason codes",
  "FastAPI backend (9 endpoints)",
  "SQLite persistence",
  "Analyst-feedback storage",
  "Self-audit sampling (both blind-spot populations)",
  "Baseline vs SafeTriage evaluation (held-out test)",
  "Capacity / cost simulation",
];
const SIMULATED = [
  "Live transaction replay (records scored from a static file)",
  "Graph visualisation (illustrative structure, no graph DB / GraphSAGE)",
  "Next-best-evidence (fixed templates, no ranking model)",
  "Analyst-assistant explanation text (template sentences, no LLM)",
  "Analyst resolution of review cases in evaluation (ground-truth proxy)",
];
const FUTURE = [
  "Kafka / Redpanda streaming",
  "Redis feature store",
  "Graph database + GraphSAGE",
  "LLM analyst assistant",
  "Live feedback-driven recalibration",
  "SAP HANA / AI Core / XSUAA",
  "Drift monitoring · verified audit labels · security & privacy hardening",
];

function Column({ title, tone, items }) {
  const tones = {
    live: "border-emerald-200 bg-emerald-50/50",
    sim: "border-amber-200 bg-amber-50/50",
    future: "border-slate-200 bg-slate-50",
  };
  const dot = { live: "bg-emerald-500", sim: "bg-amber-500", future: "bg-slate-400" };
  return (
    <div className={`rounded-lg border p-4 ${tones[tone]}`}>
      <h3 className="text-sm font-semibold text-slate-800">{title}</h3>
      <ul className="mt-2 space-y-1.5 text-xs text-slate-700">
        {items.map((i) => (
          <li key={i} className="flex gap-2">
            <span className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${dot[tone]}`} />
            <span>{i}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function Architecture() {
  const { dataset } = useDataset();
  const { data, loading } = useApi(() => `/kpis?dataset=${dataset}`, [dataset]);

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold text-slate-900">Architecture</h1>
        <p className="text-sm text-slate-500">
          What actually runs vs what is stubbed vs what is future work. Runs fully locally — no external services, no API
          keys.
        </p>
      </div>

      <SectionCard title="Pipeline">
        <div className="flex flex-wrap items-center gap-2 text-xs font-medium">
          {["PREDICTION", "INTERPRETATION", "DECISION", "HUMAN FEEDBACK", "AUDIT"].map((s, i, arr) => (
            <span key={s} className="flex items-center gap-2">
              <span className="rounded bg-indigo-600 px-2.5 py-1 text-white">{s}</span>
              {i < arr.length - 1 && <span className="text-slate-400">→</span>}
            </span>
          ))}
        </div>
        <p className="mt-2 text-xs text-slate-500">
          The calibrated probability is one input to the policy engine — never the block/allow decision itself.
        </p>
        {loading ? (
          <Loading label="…" />
        ) : (
          data && (
            <p className="mt-2 text-xs text-slate-400">
              active: model <code>{data.model_version}</code> · policy <code>{data.policy_version}</code>
            </p>
          )
        )}
      </SectionCard>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Column title="LIVE" tone="live" items={LIVE} />
        <Column title="SIMULATED" tone="sim" items={SIMULATED} />
        <Column title="FUTURE" tone="future" items={FUTURE} />
      </div>
    </div>
  );
}
