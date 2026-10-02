import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useDataset } from "../App.jsx";
import { apiPost, useApi } from "../lib/api.js";
import { fixed, shortId } from "../lib/format.js";
import { SectionCard, DecisionBadge, Loading, ErrorState, Banner } from "../components/ui.jsx";

export default function ReviewQueue() {
  const { dataset } = useDataset();
  const { data, loading, error, reload } = useApi(
    () => `/transactions?dataset=${dataset}&decision=STEP_UP_REVIEW&limit=200&order_by=calibrated_probability&desc=true`,
    [dataset]
  );
  const [openId, setOpenId] = useState(null);
  const [detail, setDetail] = useState({});
  const [fbMsg, setFbMsg] = useState({});
  const nav = useNavigate();

  useEffect(() => {
    setOpenId(null);
    setDetail({});
  }, [dataset]);

  const toggle = (id) => {
    setOpenId((cur) => (cur === id ? null : id));
    if (!detail[id]) {
      apiPost("/score", { transaction_id: id })
        .then((sc) => setDetail((d) => ({ ...d, [id]: sc })))
        .catch(() => setDetail((d) => ({ ...d, [id]: { error: true } })));
    }
  };

  const sendFeedback = (id, decision) => {
    setFbMsg((m) => ({ ...m, [id]: "saving" }));
    apiPost("/feedback", { transaction_id: id, analyst_decision: decision, reason_tag: "review_queue" })
      .then((r) => setFbMsg((m) => ({ ...m, [id]: `saved #${r.stored.id}` })))
      .catch(() => setFbMsg((m) => ({ ...m, [id]: "error" })));
  };

  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} onRetry={reload} />;
  const items = data.items || [];

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold text-slate-900">Review Queue</h1>
        <p className="text-sm text-slate-500">
          {items.length} STEP_UP_REVIEW cases in <code>{dataset}</code>, highest calibrated risk first.
        </p>
      </div>

      <Banner>
        These transactions were <strong>not blocked</strong> — they carry soft friction and an evidence request while an
        analyst decides. That reclassification is the false-positive-reduction mechanism.
      </Banner>

      <SectionCard title="Queue">
        {items.length === 0 ? (
          <p className="py-6 text-center text-sm text-slate-400">Queue is empty for this dataset.</p>
        ) : (
          <ul className="divide-y divide-slate-100">
            {items.map((r) => {
              const d = detail[r.transaction_id];
              const open = openId === r.transaction_id;
              return (
                <li key={r.transaction_id} className="py-2">
                  <div className="flex items-center gap-3 text-sm">
                    <button onClick={() => toggle(r.transaction_id)} className="font-mono text-xs text-indigo-700 hover:underline">
                      {shortId(r.transaction_id)}
                    </button>
                    <span className="tabular-nums text-slate-500">p={fixed(r.calibrated_probability, 4)}</span>
                    <span className="tabular-nums text-slate-400">cons={fixed(r.behavioural_consistency_score, 2)}</span>
                    <span className="tabular-nums text-slate-400">unc={fixed(r.uncertainty, 2)}</span>
                    <span className="text-xs text-slate-500">{r.fraud_reason_code}</span>
                    <span className="ml-auto text-xs">
                      GT {r.is_fraud_label ? <span className="text-rose-600">fraud</span> : <span className="text-slate-400">legit</span>}
                    </span>
                  </div>

                  {open && (
                    <div className="mt-2 rounded border border-slate-200 bg-slate-50 p-3 text-sm">
                      {!d && <Loading label="Scoring…" />}
                      {d?.error && <p className="text-rose-600 text-xs">Could not score.</p>}
                      {d && !d.error && (
                        <>
                          <p className="text-slate-700">{d.decision_explanation}</p>
                          <div className="mt-2">
                            <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Next best evidence</div>
                            <ol className="mt-1 list-decimal pl-5 text-xs text-slate-600">
                              {(d.next_best_evidence || []).map((e, i) => (
                                <li key={i}>
                                  <span className="font-medium text-slate-700">{e.evidence}</span> — {e.rationale}
                                </li>
                              ))}
                            </ol>
                          </div>
                          <div className="mt-3 flex flex-wrap items-center gap-2">
                            <button onClick={() => sendFeedback(r.transaction_id, "CONFIRM_FRAUD")} className="rounded bg-rose-600 px-2.5 py-1 text-xs font-medium text-white">
                              Confirm fraud
                            </button>
                            <button onClick={() => sendFeedback(r.transaction_id, "MARK_LEGITIMATE")} className="rounded bg-emerald-600 px-2.5 py-1 text-xs font-medium text-white">
                              Mark legitimate
                            </button>
                            <button onClick={() => sendFeedback(r.transaction_id, "NEED_MORE_EVIDENCE")} className="rounded bg-amber-600 px-2.5 py-1 text-xs font-medium text-white">
                              Need more evidence
                            </button>
                            <button onClick={() => nav(`/investigation/${r.transaction_id}`)} className="rounded border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium">
                              Full investigation →
                            </button>
                            {fbMsg[r.transaction_id] && (
                              <span className="text-xs text-emerald-700">{fbMsg[r.transaction_id]}</span>
                            )}
                          </div>
                        </>
                      )}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </SectionCard>
    </div>
  );
}
