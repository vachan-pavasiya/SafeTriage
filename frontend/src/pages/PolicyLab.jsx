import { useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { useDataset } from "../App.jsx";
import { useApi } from "../lib/api.js";
import { CHART } from "../lib/constants.js";
import { pct, num, fixed, money } from "../lib/format.js";
import { SectionCard, StatTile, Loading, ErrorState } from "../components/ui.jsx";

const COSTS = [
  ["fraud_loss_cost", "Fraud loss", 10, 300, 5],
  ["false_positive_cost", "False positive (hard block)", 0, 60, 1],
  ["review_cost", "Manual review", 0, 10, 0.5],
  ["customer_friction_cost", "Step-up friction", 0, 20, 0.5],
];

export default function PolicyLab() {
  const { dataset } = useDataset();
  const [costs, setCosts] = useState({});
  const [capIdx, setCapIdx] = useState(6); // grid index closest to the config default (80/1k)

  const qs = useMemo(() => {
    const p = new URLSearchParams({ dataset });
    for (const [k, v] of Object.entries(costs)) if (v != null) p.set(k, v);
    return p.toString();
  }, [dataset, costs]);

  const { data, loading, error, reload } = useApi(() => `/policy/simulate?${qs}`, [qs]);

  if (loading && !data) return <Loading />;
  if (error) return <ErrorState error={error} onRetry={reload} />;

  const pts = data.points || [];
  const cp = data.cost_params || {};
  const sel = pts[Math.min(capIdx, pts.length - 1)] || {};
  const maxCost = Math.max(...pts.map((p) => p.expected_cost), 1);

  const chart = pts.map((p) => ({
    cap: p.analyst_capacity_per_1k,
    recall: p.recall,
    fpr: p.false_positive_rate,
    review: p.review_rate,
    costN: p.expected_cost / maxCost,
  }));

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold text-slate-900">Policy Lab</h1>
        <p className="text-sm text-slate-500">
          Capacity + cost sweep on the <strong>{data.split}</strong> split ({num(data.n)} txns, {num(data.fraud)} fraud),
          dataset <code>{data.dataset}</code>. Cost sliders recompute expected cost from stored confusion counts.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        <SectionCard title="Cost parameters" className="lg:col-span-1">
          <div className="space-y-4">
            {COSTS.map(([key, label, min, max, step]) => {
              const val = costs[key] ?? cp[key];
              return (
                <label key={key} className="block text-xs">
                  <div className="flex justify-between text-slate-600">
                    <span>{label}</span>
                    <span className="tabular-nums font-medium">{val}</span>
                  </div>
                  <input
                    type="range"
                    min={min}
                    max={max}
                    step={step}
                    value={val}
                    onChange={(e) => setCosts((c) => ({ ...c, [key]: Number(e.target.value) }))}
                    className="mt-1 w-full accent-indigo-600"
                  />
                </label>
              );
            })}
            <button
              onClick={() => setCosts({})}
              className="w-full rounded border border-slate-300 bg-white py-1 text-xs font-medium text-slate-600 hover:bg-slate-50"
            >
              reset to config defaults
            </button>
          </div>
        </SectionCard>

        <SectionCard title="Selected operating point" className="lg:col-span-2">
          <label className="block text-xs text-slate-600">
            Analyst capacity: <span className="font-semibold">{sel.analyst_capacity_per_1k}</span> reviews / 1,000 txns
            <input
              type="range"
              min={0}
              max={pts.length - 1}
              step={1}
              value={Math.min(capIdx, pts.length - 1)}
              onChange={(e) => setCapIdx(Number(e.target.value))}
              className="mt-1 w-full accent-indigo-600"
            />
          </label>
          <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-3">
            <StatTile label="Recall" value={pct(sel.recall)} tone="default" />
            <StatTile label="False positives" value={num(sel.false_positives)} tone="bad" hint={pct(sel.false_positive_rate, 2)} />
            <StatTile label="Review volume" value={num(sel.review_volume)} tone="warn" hint={`${sel.analyst_workload_per_1k}/1k`} />
            <StatTile label="Auto-clear rate" value={pct(sel.auto_clear_rate)} tone="good" />
            <StatTile label="Capacity concessions" value={num(sel.capacity_concessions)} />
            <StatTile label="Expected cost" value={money(sel.expected_cost)} />
          </div>
        </SectionCard>
      </div>

      <SectionCard title="Sweep" subtitle="How recall / false-positive rate / review rate / expected cost move with capacity">
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chart} margin={{ top: 8, right: 16, left: 4, bottom: 4 }}>
              <CartesianGrid stroke={CHART.grid} />
              <XAxis dataKey="cap" tick={{ fontSize: 11, fill: CHART.axis }} label={{ value: "analyst capacity / 1k", position: "insideBottom", offset: -2, fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: CHART.axis }} tickFormatter={(v) => v.toFixed(1)} />
              <Tooltip
                formatter={(v, n) => (n === "expected cost (norm)" ? fixed(v, 2) : pct(v, 1))}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <ReferenceLine x={sel.analyst_capacity_per_1k} stroke="#4f46e5" strokeDasharray="3 3" />
              <Line type="monotone" dataKey="recall" name="recall" stroke={CHART.series[2]} dot={false} />
              <Line type="monotone" dataKey="fpr" name="false-positive rate" stroke={CHART.series[4]} dot={false} />
              <Line type="monotone" dataKey="review" name="review rate" stroke={CHART.series[3]} dot={false} />
              <Line type="monotone" dataKey="costN" name="expected cost (norm)" stroke={CHART.series[5]} strokeDasharray="5 3" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        {data.cost_overrides_applied && (
          <p className="mt-1 text-xs text-slate-400">cost overrides applied: {JSON.stringify(data.cost_overrides_applied)}</p>
        )}
      </SectionCard>
    </div>
  );
}
