import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The dashboard talks to the FastAPI backend at :8000 via a /api proxy so the
// same build works in dev and when served statically behind a reverse proxy.
// The proxy is declared for BOTH `vite` (dev) and `vite preview` (built app) so
// `npm run dev` and `npm run build && npm run preview` both reach the API.
const apiProxy = {
  "/api": {
    target: process.env.SAFETRIAGE_API || "http://127.0.0.1:8000",
    changeOrigin: true,
    rewrite: (p) => p.replace(/^\/api/, ""),
  },
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, proxy: apiProxy },
  preview: { port: 4173, proxy: apiProxy },
});
