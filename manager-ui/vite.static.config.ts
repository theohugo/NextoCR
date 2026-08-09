import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  root: resolve(__dirname, "static-src"),
  base: "./",
  plugins: [react()],
  build: {
    outDir: resolve(__dirname, "static"),
    emptyOutDir: true,
    assetsDir: "assets",
  },
});
