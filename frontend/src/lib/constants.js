// Semantic colours for the four policy actions. Restrained, not neon.
export const DECISION_STYLE = {
  AUTO_CLEAR: { label: "Auto-clear", chip: "bg-emerald-50 text-emerald-700 ring-emerald-600/20", bar: "#059669" },
  STEP_UP_REVIEW: { label: "Step-up review", chip: "bg-amber-50 text-amber-700 ring-amber-600/20", bar: "#d97706" },
  SOFT_BLOCK: { label: "Soft block", chip: "bg-orange-50 text-orange-700 ring-orange-600/20", bar: "#ea580c" },
  ESCALATE: { label: "Escalate", chip: "bg-rose-50 text-rose-700 ring-rose-600/20", bar: "#e11d48" },
};

export const DECISION_ORDER = ["AUTO_CLEAR", "STEP_UP_REVIEW", "SOFT_BLOCK", "ESCALATE"];

export const NAV = [
  { to: "/", label: "Overview", end: true },
  { to: "/feed", label: "Live Transaction Feed" },
  { to: "/investigation", label: "Investigation" },
  { to: "/fp-lab", label: "False Positive Lab" },
  { to: "/review-queue", label: "Review Queue" },
  { to: "/self-audit", label: "Self-Audit" },
  { to: "/calibration", label: "Calibration + Uncertainty" },
  { to: "/policy-lab", label: "Policy Lab" },
  { to: "/cycle", label: "Fraud Detection Cycle" },
  { to: "/architecture", label: "Architecture" },
];

export const CHART = {
  axis: "#64748b",
  grid: "#e2e8f0",
  series: ["#4f46e5", "#0891b2", "#059669", "#d97706", "#e11d48", "#7c3aed"],
};
