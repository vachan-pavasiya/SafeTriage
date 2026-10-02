import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, CartesianGrid, Legend } from "recharts";
import { useDataset } from "../App.jsx";
import { useApi } from "../lib/api.js";
import { CHART } from "../lib/constants.js";
import { pct, num, fixed, money, titleCase, shortId } from "../lib/format.js";
import { SectionCard, StatTile, Loading, ErrorState, Banner, DecisionBadge } from "../components/ui.jsx";

const METRICS = [
  ["precision", "Precision", pct],
  ["recall", "Recall", pct],
  ["false_positive_rate", "False positive rate", (v) => pct(v, 2)],
  ["false_discovery_rate", "False discovery rate", pct],
  ["fp", "False positives (count)", num],
  ["fn", "Missed fraud (count)", num],
  ["block_rate", "Hard-block rate", pct],
  ["review_rate", "Review rate", pct],
  ["auto_clear_rate", "Auto-clear rate", pct],
  ["analyst_workload_per_1k", "Analyst workload / 1k", (v) => fixed(v, 1)],
  ["fraud_caught_per_1k_reviewed", "Fraud caught / 1k reviewed", (v) => (v == null ? "n/a" : fixed(v, 1))],
];

export default function FalsePositiveLab() {
  const { dataset } = useDataset();
  const { data, loading, error, reload } = useApi(() => `/evaluation?dataset=${dataset}`, [dataset]);

  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} onRetry={reload} />;
  const e = data;
  const b = e.baseline;
  const s = e.safetriage;

  const chartData = [
    { k: "Recall", baseline: b.recall, safetriage: s.recall },
    { k: "Precision", baseline: b.precision, safetriage: s.precision },
    { k: "FP rate", baseline: b.false_positive_rate, safetriage: s.false_positive_rate },
    { k: "Hard-block rate", baseline: b.block_rate, safetriage: s.block_rate },
  ];

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold text-slate-900">False Positive Lab</h1>
        <p className="text-sm text-slate-500">
          Probability-threshold-only baseline vs full SafeTriage — held-out test split, {num(e.test_n)} transactions,{" "}
          {num(e.test_fraud)} fraud.{" "}
          {e.reduced_safetriage && <span className="text-violet-700">reduced mode (IEEE-CIS)</span>}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile
          label="False positives"
          value={`${num(b.fp)} → ${num(s.fp)}`}
          tone="good"
          hint={e.deltas.false_positive_reduction_pct != null ? `${e.deltas.false_positive_reduction_pct}% fewer` : ""}
        />
        <StatTile label="Recall" value={`${pct(b.recall)} → ${pct(s.recall)}`} tone={e.deltas.recall < -0.02 ? "warn" : "default"} hint={`${(e.deltas.recall * 100).toFixed(1)} pts`} />
        <StatTile label="Analyst workload / 1k" value={`0 → ${fixed(s.analyst_workload_per_1k, 1)}`} tone="warn" />
        <StatTile
          label="Expected cost"
          value={`${money(e.cost.baseline.total_expected_cost)} → ${money(e.cost.safetriage.total_expected_cost)}`}
          tone={e.deltas.expected_cost < 0 ? "good" : "bad"}
        />
      </div>

      <SectionCard title="Baseline vs SafeTriage" subtitle="Rates on the test split">
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 8, right: 12, left: 4, bottom: 4 }}>
              <CartesianGrid stroke={CHART.grid} vertical={false} />
              <XAxis dataKey="k" tick={{ fontSize: 11, fill: CHART.axis }} />
              <YAxis tick={{ fontSize: 11, fill: CHART.axis }} tickFormatter={(v) => pct(v, 0)} />
              <Tooltip formatter={(v) => pct(v, 2)} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="baseline" name="Baseline" fill={CHART.series[4]} radius={[3, 3, 0, 0]} />
              <Bar dataKey="safetriage" name="SafeTriage" fill={CHART.series[0]} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </SectionCard>

      <SectionCard title="Full metric table" subtitle={`Baseline threshold ${fixed(b.threshold, 4)} (frozen from validation)`}>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs uppercase tracking-wide text-slate-500">
                <th className="py-2 text-left font-medium">Metric</th>
                <th className="py-2 text-right font-medium">Baseline</th>
                <th className="py-2 text-right font-medium">SafeTriage</th>
              </tr>
            </thead>
            <tbody>
              {METRICS.map(([key, label, fmt]) => (
                <tr key={key} className="border-t border-slate-100">
                  <td className="py-1.5 text-slate-600">{label}</td>
                  <td className="py-1.5 text-right tabular-nums">{fmt(b[key])}</td>
                  <td className="py-1.5 text-right font-medium tabular-nums">{fmt(s[key])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <ul className="mt-3 space-y-1 text-xs text-slate-500">
          {e.notes.map((n, i) => (
            <li key={i}>· {n}</li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="Demo cases" subtitle="Real test transactions matching each pattern, re-scored in full">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          {Object.entries(e.demo_cases).map(([key, c]) => (
            <div key={key} className="rounded-lg border border-slate-200 p-3">
              <div className="text-xs font-semibold text-slate-500">{titleCase(key.replace(/^[ABC]_/, ""))}</div>
              <div className="mt-1 flex items-center gap-2">
                <DecisionBadge decision={c.decision} />
                <span className="font-mono text-[11px] text-slate-400">{shortId(c.transaction_id)}</span>
              </div>
              <p className="mt-1 text-xs text-slate-600">{c._headline}</p>
              <div className="mt-1 text-[11px] text-slate-400">
                cal p={fixed(c.calibrated_probability, 4)} · consistency {fixed(c.behavioral_consistency_score, 2)} · GT{" "}
                {c._ground_truth_isFraud ? "fraud" : "legit"}
                {c._ground_truth_typology && c._ground_truth_typology !== "NONE" ? ` · ${c._ground_truth_typology}` : ""}
              </div>
              <div className="mt-1 text-[11px] text-slate-500">reason: {c.fraud_reason_code}</div>
              {c.top_features?.length > 0 && (
                <div className="mt-1 text-[11px] text-slate-400">
                  top: {c.top_features.slice(0, 3).map((f) => f.friendly).join(", ")}
                </div>
              )}
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}
