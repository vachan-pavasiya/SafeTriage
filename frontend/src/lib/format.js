export const pct = (x, digits = 1) =>
  x == null || Number.isNaN(x) ? "—" : `${(x * 100).toFixed(digits)}%`;

export const num = (x, digits = 0) =>
  x == null || Number.isNaN(x) ? "—" : Number(x).toLocaleString(undefined, { maximumFractionDigits: digits });

export const fixed = (x, digits = 3) =>
  x == null || Number.isNaN(x) ? "—" : Number(x).toFixed(digits);

export const money = (x) =>
  x == null || Number.isNaN(x) ? "—" : Number(x).toLocaleString(undefined, { maximumFractionDigits: 0 });

export const titleCase = (s) =>
  String(s || "")
    .replaceAll("_", " ")
    .toLowerCase()
    .replace(/\b\w/g, (c) => c.toUpperCase());

export const shortId = (s) => String(s || "").replace(/^([a-z]+)-/, "");
