import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useDataset } from "../App.jsx";
import { useApi } from "../lib/api.js";
import { DECISION_ORDER } from "../lib/constants.js";
import { fixed, titleCase, shortId } from "../lib/format.js";
import { SectionCard, DataTable, DecisionBadge, Loading, ErrorState } from "../components/ui.jsx";

const PAGE = 40;

export default function LiveFeed() {
  const { dataset } = useDataset();
  const [decision, setDecision] = useState("");
  const [sort, setSort] = useState({ key: "calibrated_probability", desc: true });
  const [shown, setShown] = useState(PAGE);

  const q = `/transactions?dataset=${dataset}&limit=400&order_by=${sort.key}&desc=${sort.desc}${
    decision ? `&decision=${decision}` : ""
  }`;
  const { data, loading, error, reload } = useApi(() => q, [q]);

  const rows = useMemo(() => {
    const items = (data?.items || []).map((r) => ({ ...r, _key: r.transaction_id }));
    return items.slice(0, shown);
  }, [data, shown]);

  const nav = useNavigate();
  const onSort = (key) => {
    setSort((s) => ({ key, desc: s.key === key ? !s.desc : true }));
    setShown(PAGE);
  };

  const columns = [
    { key: "transaction_id", header: "Txn", render: (r) => <span className="font-mono text-xs">{shortId(r.transaction_id)}</span> },
    { key: "split", header: "Split", render: (r) => <span className="text-xs text-slate-500">{r.split || "—"}</span> },
    { key: "calibrated_probability", header: "p(fraud) cal", sortable: true, align: "right", render: (r) => fixed(r.calibrated_probability, 4) },
    { key: "behavioural_consistency_score", header: "Consistency", sortable: true, align: "right", render: (r) => fixed(r.behavioural_consistency_score, 3) },
    { key: "uncertainty", header: "Disagree", sortable: true, align: "right", render: (r) => fixed(r.uncertainty, 3) },
    { key: "decision", header: "Decision", render: (r) => <DecisionBadge decision={r.decision} /> },
    { key: "predicted_typology", header: "Typology", render: (r) => <span className="text-xs">{titleCase(r.predicted_typology) || "—"}</span> },
    { key: "fraud_reason_code", header: "Reason", render: (r) => <span className="text-xs text-slate-500">{r.fraud_reason_code || "—"}</span> },
    { key: "is_fraud_label", header: "GT", align: "right", render: (r) => (r.is_fraud_label ? <span className="text-rose-600">fraud</span> : <span className="text-slate-400">legit</span>) },
  ];

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold text-slate-900">Live Transaction Feed</h1>
        <p className="text-sm text-slate-500">
          Replay of scored transactions from <code>{dataset}</code>. Click a row to investigate.
        </p>
      </div>

      <SectionCard
        title={`${data?.items?.length ?? 0} loaded · showing ${rows.length}`}
        right={
          <div className="flex items-center gap-2">
            <select
              value={decision}
              onChange={(e) => {
                setDecision(e.target.value);
                setShown(PAGE);
              }}
              className="rounded border border-slate-300 bg-white px-2 py-1 text-xs"
            >
              <option value="">all decisions</option>
              {DECISION_ORDER.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
            <button
              onClick={() => setShown(PAGE)}
              className="rounded border border-slate-300 bg-white px-2 py-1 text-xs font-medium hover:bg-slate-50"
            >
              ⟲ Replay
            </button>
          </div>
        }
      >
        {loading && <Loading />}
        {error && <ErrorState error={error} onRetry={reload} />}
        {data && (
          <>
            <DataTable
              columns={columns}
              rows={rows}
              sort={sort}
              onSort={onSort}
              onRowClick={(r) => nav(`/investigation/${r.transaction_id}`)}
            />
            {rows.length < (data.items?.length || 0) && (
              <button
                onClick={() => setShown((s) => s + PAGE)}
                className="mt-3 w-full rounded border border-slate-300 bg-white py-2 text-sm font-medium text-slate-600 hover:bg-slate-50"
              >
                Play next {PAGE}
              </button>
            )}
          </>
        )}
      </SectionCard>
    </div>
  );
}
