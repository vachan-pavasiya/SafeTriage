import { DECISION_STYLE } from "../lib/constants.js";

export function SectionCard({ title, subtitle, right, children, className = "" }) {
  return (
    <section className={`rounded-lg border border-slate-200 bg-white ${className}`}>
      {(title || right) && (
        <header className="flex items-start justify-between gap-3 border-b border-slate-100 px-4 py-3">
          <div>
            {title && <h2 className="text-sm font-semibold text-slate-800">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function StatTile({ label, value, hint, tone = "default" }) {
  const tones = {
    default: "text-slate-900",
    good: "text-emerald-700",
    warn: "text-amber-700",
    bad: "text-rose-700",
  };
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-4 py-3">
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mt-1 text-2xl font-semibold tabular-nums ${tones[tone]}`}>{value}</div>
      {hint && <div className="mt-0.5 text-xs text-slate-400">{hint}</div>}
    </div>
  );
}

export function DecisionBadge({ decision }) {
  const s = DECISION_STYLE[decision] || { label: decision, chip: "bg-slate-100 text-slate-600 ring-slate-500/20" };
  return (
    <span className={`inline-flex items-center rounded px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${s.chip}`}>
      {s.label}
    </span>
  );
}

export function Loading({ label = "Loading…" }) {
  return (
    <div className="flex items-center gap-2 py-10 text-sm text-slate-500">
      <span className="h-3 w-3 animate-pulse rounded-full bg-indigo-400" />
      {label}
    </div>
  );
}

export function ErrorState({ error, onRetry }) {
  return (
    <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">
      <div className="font-semibold">Could not load data</div>
      <pre className="mt-1 whitespace-pre-wrap break-words text-xs text-rose-700">{String(error?.message || error)}</pre>
      <p className="mt-2 text-xs text-rose-600">
        Is the API running? <code className="rounded bg-white px-1">python -m uvicorn api.main:app --reload</code>
      </p>
      {onRetry && (
        <button onClick={onRetry} className="mt-2 rounded border border-rose-300 bg-white px-2 py-1 text-xs font-medium hover:bg-rose-100">
          Retry
        </button>
      )}
    </div>
  );
}

export function Banner({ children, tone = "info" }) {
  const tones = {
    info: "border-sky-200 bg-sky-50 text-sky-800",
    warn: "border-amber-200 bg-amber-50 text-amber-800",
    proxy: "border-violet-200 bg-violet-50 text-violet-800",
  };
  return <div className={`rounded-md border px-3 py-2 text-xs ${tones[tone]}`}>{children}</div>;
}

export function DataTable({ columns, rows, onRowClick, sort, onSort, empty = "No rows." }) {
  if (!rows?.length) return <div className="py-8 text-center text-sm text-slate-400">{empty}</div>;
  return (
    <div className="thin-scroll max-h-[70vh] overflow-auto rounded border border-slate-200">
      <table className="grid-table w-full text-left text-sm">
        <thead>
          <tr className="text-xs uppercase tracking-wide text-slate-500">
            {columns.map((c) => (
              <th
                key={c.key}
                onClick={c.sortable && onSort ? () => onSort(c.key) : undefined}
                className={`whitespace-nowrap border-b border-slate-200 px-3 py-2 font-medium ${
                  c.sortable && onSort ? "cursor-pointer select-none hover:text-slate-800" : ""
                } ${c.align === "right" ? "text-right" : ""}`}
              >
                {c.header}
                {sort?.key === c.key ? (sort.desc ? " ▾" : " ▴") : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={r._key ?? i}
              onClick={onRowClick ? () => onRowClick(r) : undefined}
              className={`border-b border-slate-100 last:border-0 ${
                onRowClick ? "cursor-pointer hover:bg-slate-50" : ""
              }`}
            >
              {columns.map((c) => (
                <td key={c.key} className={`whitespace-nowrap px-3 py-2 ${c.align === "right" ? "text-right tabular-nums" : ""}`}>
                  {c.render ? c.render(r) : r[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function KeyValue({ items }) {
  return (
    <dl className="grid grid-cols-1 gap-x-6 gap-y-1 sm:grid-cols-2">
      {items.map(([k, v]) => (
        <div key={k} className="flex justify-between gap-4 border-b border-slate-100 py-1 text-sm last:border-0">
          <dt className="text-slate-500">{k}</dt>
          <dd className="text-right font-medium tabular-nums text-slate-800">{v}</dd>
        </div>
      ))}
    </dl>
  );
}
