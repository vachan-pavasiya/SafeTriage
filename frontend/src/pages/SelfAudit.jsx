import { useDataset } from "../App.jsx";
import { useApi } from "../lib/api.js";
import { pct, num, fixed, shortId } from "../lib/format.js";
import { SectionCard, StatTile, Loading, ErrorState, Banner, DataTable, DecisionBadge } from "../components/ui.jsx";

function PopulationCard({ title, subtitle, pop, kind }) {
  if (!pop) return null;
  const strat = pop.samples?.risk_stratified;
  const ctrl = pop.samples?.random_control;
  const rateKey = kind === "fn" ? "proxy_missed_fraud_rate" : "proxy_false_positive_rate";
  const cntKey = kind === "fn" ? "proxy_missed_fraud_count" : "proxy_false_positive_count";
  return (
    <SectionCard title={title} subtitle={subtitle}>
      <div className="grid grid-cols-2 gap-3">
        <StatTile
          label="Risk-stratified sample"
          value={strat ? pct(strat[rateKey], 2) : "—"}
          hint={strat ? `${strat[cntKey]}/${strat.n_sampled}` : "no sample"}
          tone={kind === "fn" ? "warn" : "bad"}
        />
        <StatTile
          label="Random control sample"
          value={ctrl ? pct(ctrl[rateKey], 2) : "—"}
          hint={ctrl ? `${ctrl[cntKey]}/${ctrl.n_sampled}` : "no sample"}
        />
      </div>
      <p className="mt-2 text-xs text-slate-500">population size on test split: {num(pop.population_size)}</p>
    </SectionCard>
  );
}

export default function SelfAudit() {
  const { dataset } = useDataset();
  const { data, loading, error, reload } = useApi(() => `/audit/sample?dataset=${dataset}`, [dataset]);

  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} onRetry={reload} />;

  const sum = data.summary || {};
  const pops = sum.populations || {};
  const samples = data.samples || [];

  const cols = [
    { key: "population", header: "Population", render: (r) => <span className="text-xs">{r.population}</span> },
    { key: "sample_type", header: "Sample", render: (r) => <span className="text-xs text-slate-500">{r.sample_type}</span> },
    { key: "transaction_id", header: "Txn", render: (r) => <span className="font-mono text-xs">{shortId(r.transaction_id)}</span> },
    { key: "calibrated_probability", header: "cal p", align: "right", render: (r) => fixed(r.calibrated_probability, 4) },
    { key: "decision", header: "Decision", render: (r) => <DecisionBadge decision={r.decision} /> },
    {
      key: "audit_proxy_label",
      header: "Proxy outcome",
      render: (r) =>
        r.audit_proxy_label ? <span className="text-rose-600 text-xs">fraud</span> : <span className="text-emerald-600 text-xs">legit</span>,
    },
  ];

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold text-slate-900">Self-Audit</h1>
        <p className="text-sm text-slate-500">
          Run <code>{data.run_id}</code> · dataset <code>{data.dataset}</code> · test split n={num(sum.test_n)}
        </p>
      </div>

      <Banner tone="proxy">{sum.note}</Banner>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <PopulationCard
          title="AUTO_CLEAR — hidden false negatives"
          subtitle="Fraud we may have waved through"
          pop={pops.AUTO_CLEAR}
          kind="fn"
        />
        <PopulationCard
          title="BLOCK / ESCALATE — hidden false positives"
          subtitle="Legitimate customers we may have blocked"
          pop={pops.BLOCK_ESCALATE}
          kind="fp"
        />
      </div>

      <SectionCard title={`Sampled transactions (${samples.length})`} subtitle="Risk-stratified + random control, drawn from the held-out test split only">
        <DataTable columns={cols} rows={samples.map((s) => ({ ...s, _key: s.id }))} />
      </SectionCard>
    </div>
  );
}
