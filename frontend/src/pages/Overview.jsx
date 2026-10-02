import { BarChart, Bar, XAxis, YAxis, Cell, ResponsiveContainer, Tooltip, CartesianGrid } from "recharts";
import { useDataset } from "../App.jsx";
import { useApi } from "../lib/api.js";
import { DECISION_ORDER, DECISION_STYLE, CHART } from "../lib/constants.js";
import { pct, num, fixed } from "../lib/format.js";
import { SectionCard, StatTile, Loading, ErrorState, Banner } from "../components/ui.jsx";

export default function Overview() {
  const { dataset } = useDataset();
  const kpis = useApi(() => `/kpis?dataset=${dataset}`, [dataset]);
  const audit = useApi(() => `/audit/sample?dataset=${dataset}`, [dataset]);

  if (kpis.loading) return <Loading />;
  if (kpis.error) return <ErrorState error={kpis.error} onRetry={kpis.reload} />;
  const k = kpis.data;

  const dist = DECISION_ORDER.map((d) => ({
    decision: d,
    label: DECISION_STYLE[d].label,
    count: k.decision_distribution[d] || 0,
    fill: DECISION_STYLE[d].bar,
  }));

  const ap = audit.data?.summary?.populations || {};
  const acStrat = ap.AUTO_CLEAR?.samples?.risk_stratified;
  const beStrat = ap.BLOCK_ESCALATE?.samples?.risk_stratified;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold text-slate-900">Overview</h1>
        <p className="text-sm text-slate-500">
          {num(k.n_transactions)} scored transactions · dataset <code>{k.dataset}</code> · model{" "}
          <code>{k.model_version}</code> · policy <code>{k.policy_version}</code>
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile label="Transactions" value={num(k.n_transactions)} hint={`${pct(k.fraud_prevalence, 2)} labelled fraud`} />
        <StatTile label="Auto-clear rate" value={pct(k.auto_clear_rate)} tone="good" />
        <StatTile label="Review rate" value={pct(k.review_rate)} tone="warn" hint={`${k.analyst_workload_per_1k}/1k workload`} />
        <StatTile
          label="Hard actions"
          value={pct(k.soft_block_rate + k.escalation_rate)}
          tone="bad"
          hint={`soft-block ${pct(k.soft_block_rate)} · escalate ${pct(k.escalation_rate)}`}
        />
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <SectionCard title="Decision distribution" subtitle="How the policy engine routed every scored transaction">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={dist} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
                <CartesianGrid stroke={CHART.grid} vertical={false} />
                <XAxis dataKey="label" tick={{ fontSize: 11, fill: CHART.axis }} />
                <YAxis tick={{ fontSize: 11, fill: CHART.axis }} allowDecimals={false} />
                <Tooltip formatter={(v) => num(v)} />
                <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                  {dist.map((d) => (
                    <Cell key={d.decision} fill={d.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </SectionCard>

        <SectionCard
          title="Self-audit proxies"
          subtitle="Blind-spot sampling on the held-out test split"
          right={
            audit.data && (
              <span className="text-xs text-slate-400">run {audit.data.run_id?.split("_").pop()}</span>
            )
          }
        >
          {audit.loading && <Loading label="Loading audit…" />}
          {audit.error && <ErrorState error={audit.error} onRetry={audit.reload} />}
          {audit.data && (
            <div className="space-y-3">
              <Banner tone="proxy">
                Outcomes use ground-truth <code>isFraud</code> as an <strong>audit proxy</strong> (no verified
                human re-review).
              </Banner>
              <div className="grid grid-cols-2 gap-3">
                <StatTile
                  label="Auto-clear: missed fraud"
                  value={acStrat ? pct(acStrat.proxy_missed_fraud_rate, 2) : "—"}
                  hint={acStrat ? `${acStrat.proxy_missed_fraud_count}/${acStrat.n_sampled} stratified sample` : "no sample"}
                  tone="warn"
                />
                <StatTile
                  label="Blocked/escalated: false positives"
                  value={beStrat ? pct(beStrat.proxy_false_positive_rate, 2) : "—"}
                  hint={
                    beStrat
                      ? `${beStrat.proxy_false_positive_count}/${beStrat.n_sampled} stratified sample`
                      : `population size ${ap.BLOCK_ESCALATE?.population_size ?? 0}`
                  }
                  tone="bad"
                />
              </div>
              <div className="text-xs text-slate-500">
                populations sampled: AUTO_CLEAR (n={ap.AUTO_CLEAR?.population_size ?? 0}), BLOCK_ESCALATE (n=
                {ap.BLOCK_ESCALATE?.population_size ?? 0})
              </div>
            </div>
          )}
        </SectionCard>
      </div>

      <SectionCard title="Score summary" subtitle="Means across all stored transactions">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <StatTile label="Mean calibrated p(fraud)" value={fixed(k.mean_calibrated_probability, 4)} />
          <StatTile label="Mean behavioural consistency" value={fixed(k.mean_behavioural_consistency, 3)} />
          <StatTile label="Mean disagreement uncertainty" value={fixed(k.mean_uncertainty, 3)} />
          <StatTile label="Capacity concessions" value={num(k.capacity_concessions)} hint="reviews downgraded past budget" />
        </div>
      </SectionCard>
    </div>
  );
}
