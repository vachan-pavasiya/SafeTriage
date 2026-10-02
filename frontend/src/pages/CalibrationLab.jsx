import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { useDataset } from "../App.jsx";
import { useApi } from "../lib/api.js";
import { CHART } from "../lib/constants.js";
import { fixed, num } from "../lib/format.js";
import { SectionCard, StatTile, Loading, ErrorState, Banner } from "../components/ui.jsx";

export default function CalibrationLab() {
  const { dataset } = useDataset();
  const { data, loading, error, reload } = useApi(() => `/calibration?dataset=${dataset}`, [dataset]);

  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} onRetry={reload} />;
  const c = data;

  const merge = {};
  (c.reliability_raw || []).forEach((b) => {
    merge[b.mean_predicted] = { x: b.mean_predicted, raw: b.observed_frequency };
  });
  (c.reliability_calibrated || []).forEach((b) => {
    merge[b.mean_predicted] = { ...(merge[b.mean_predicted] || { x: b.mean_predicted }), calibrated: b.observed_frequency };
  });
  const curve = Object.values(merge).sort((a, b) => a.x - b.x);

  const mm = c.model_metrics || {};
  const cv = mm.cv_train_val || {};
  const dis = mm.disagreement_proxy_norm || {};

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold text-slate-900">Calibration + Uncertainty Lab</h1>
        <p className="text-sm text-slate-500">
          dataset <code>{c.dataset}</code> · method <code>{c.method}</code> · reported on the {c.split} split · {num(c.n)} rows
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile label="Brier (raw)" value={fixed(c.brier_raw, 4)} />
        <StatTile label="Brier (calibrated)" value={fixed(c.brier_calibrated, 4)} tone={c.brier_calibrated <= c.brier_raw ? "good" : "warn"} />
        <StatTile label="ECE (raw)" value={fixed(c.ece_raw, 4)} />
        <StatTile label="ECE (calibrated)" value={fixed(c.ece_calibrated, 4)} tone={c.ece_calibrated <= c.ece_raw ? "good" : "warn"} />
      </div>

      <SectionCard title="Reliability curve" subtitle="Mean predicted probability vs observed fraud frequency — dashed line is perfect calibration">
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={curve} margin={{ top: 8, right: 16, left: 4, bottom: 4 }}>
              <CartesianGrid stroke={CHART.grid} />
              <XAxis type="number" dataKey="x" domain={[0, 1]} tick={{ fontSize: 11, fill: CHART.axis }} tickFormatter={(v) => v.toFixed(1)} label={{ value: "mean predicted", position: "insideBottom", offset: -2, fontSize: 11 }} />
              <YAxis type="number" domain={[0, 1]} tick={{ fontSize: 11, fill: CHART.axis }} tickFormatter={(v) => v.toFixed(1)} label={{ value: "observed", angle: -90, position: "insideLeft", fontSize: 11 }} />
              <Tooltip formatter={(v) => fixed(v, 3)} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#94a3b8" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="raw" name="raw XGBoost" stroke={CHART.series[4]} dot={{ r: 3 }} connectNulls />
              <Line type="monotone" dataKey="calibrated" name="calibrated" stroke={CHART.series[0]} dot={{ r: 3 }} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </SectionCard>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <SectionCard title="Discrimination (cross-validated)" subtitle="5-fold CV over train+val — never the test split">
          <div className="grid grid-cols-2 gap-3">
            <StatTile label="XGBoost ROC-AUC" value={fixed(cv.xgb_roc_auc, 3)} />
            <StatTile label="XGBoost PR-AUC" value={fixed(cv.xgb_pr_auc, 3)} />
            <StatTile label="Random Forest ROC-AUC" value={fixed(cv.rf_roc_auc, 3)} />
            <StatTile label="CV fraud rows" value={num(cv.n_fraud)} hint={cv.subsampled_from ? `subsampled from ${num(cv.subsampled_from)}` : "full pool"} />
          </div>
        </SectionCard>

        <SectionCard title="Model-disagreement uncertainty proxy" subtitle="|calibrated XGB − RF|, normalised by the validation gap distribution">
          <Banner tone="warn">
            This is a <strong>disagreement proxy</strong>, not epistemic uncertainty. Two models agreeing does not mean the
            prediction is right.
          </Banner>
          <div className="mt-3 grid grid-cols-2 gap-3">
            <StatTile label="gap p05" value={fixed(dis.p05, 4)} />
            <StatTile label="gap p95" value={fixed(dis.p95, 4)} />
            <StatTile label="gap mean" value={fixed(dis.mean, 4)} />
            <StatTile label="gap max" value={fixed(dis.max, 4)} />
          </div>
        </SectionCard>
      </div>
    </div>
  );
}
