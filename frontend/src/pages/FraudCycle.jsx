import { useDataset } from "../App.jsx";
import { useApi } from "../lib/api.js";
import { num, pct } from "../lib/format.js";
import { SectionCard, Loading, ErrorState } from "../components/ui.jsx";

const STEPS = [
  ["1", "Ingest", "Transaction arrives with account history, device, beneficiary, velocity and security context."],
  ["2", "Predict", "XGBoost produces a raw fraud score; Random Forest scores in parallel for disagreement."],
  ["3", "Calibrate", "Raw score → calibrated probability. Only the calibrated value is used downstream."],
  ["4", "Interpret", "Behavioural consistency (independent of the score) + model-disagreement proxy + SHAP + reason code."],
  ["5", "Decide", "Policy engine combines all signals → AUTO_CLEAR / STEP_UP_REVIEW / SOFT_BLOCK / ESCALATE."],
  ["6", "Capacity check", "Projected review volume vs analyst budget; lowest-value reviews concede to AUTO_CLEAR."],
  ["7", "Act", "Clear, request evidence, hold, or escalate — with a plain-language decision explanation."],
  ["8", "Human feedback", "Analyst verdict + reason tag stored against the original model + policy output."],
  ["9", "Self-audit", "Sample auto-cleared and blocked/escalated populations for hidden FN / FP (proxy labels)."],
  ["10", "Recalibrate (future)", "Feed verified verdicts + audit findings back into calibration and thresholds."],
];

export default function FraudCycle() {
  const { dataset } = useDataset();
  const kpis = useApi(() => `/kpis?dataset=${dataset}`, [dataset]);
  const audit = useApi(() => `/audit/sample?dataset=${dataset}`, [dataset]);
  const fb = useApi(`/feedback`, []);

  const k = kpis.data;
  const ap = audit.data?.summary?.populations || {};

  const liveCount = {
    "2": k && `${num(k.n_transactions)} scored`,
    "3": k && `mean cal p ${k.mean_calibrated_probability?.toFixed(4)}`,
    "4": k && `mean consistency ${k.mean_behavioural_consistency?.toFixed(3)}`,
    "5": k &&
      `${pct(k.auto_clear_rate, 0)} clear · ${pct(k.review_rate, 0)} review · ${pct(
        k.soft_block_rate + k.escalation_rate,
        0
      )} hard`,
    "6": k && `${num(k.capacity_concessions)} concessions`,
    "8": fb.data && `${num(fb.data.count)} feedback rows`,
    "9": audit.data && `AUTO_CLEAR n=${num(ap.AUTO_CLEAR?.population_size ?? 0)} · BLOCK/ESC n=${num(ap.BLOCK_ESCALATE?.population_size ?? 0)}`,
  };

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold text-slate-900">Fraud Detection Cycle</h1>
        <p className="text-sm text-slate-500">
          The closed loop: <strong>prediction → interpretation → decision → human feedback → audit</strong>. Live counts
          are for dataset <code>{dataset}</code>.
        </p>
      </div>

      {(kpis.loading || audit.loading) && <Loading />}
      {kpis.error && <ErrorState error={kpis.error} onRetry={kpis.reload} />}

      <SectionCard title="Ten steps">
        <ol className="relative space-y-3 border-l-2 border-slate-200 pl-6">
          {STEPS.map(([n, title, body]) => (
            <li key={n} className="relative">
              <span className="absolute -left-[31px] flex h-6 w-6 items-center justify-center rounded-full bg-indigo-600 text-xs font-semibold text-white">
                {n}
              </span>
              <div className="text-sm font-semibold text-slate-800">{title}</div>
              <div className="text-xs text-slate-600">{body}</div>
              {liveCount[n] && <div className="mt-0.5 text-[11px] font-medium text-indigo-700">{liveCount[n]}</div>}
            </li>
          ))}
        </ol>
        <p className="mt-4 text-xs text-slate-400">
          Step 10 is architecture-only in this MVP: feedback and audit findings are captured but not yet fed back into a
          live retrain (see Architecture → FUTURE).
        </p>
      </SectionCard>
    </div>
  );
}
