import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// GitHub Pages serves the site under /<repository>/; SITE_BASE=/ suits a custom domain.
export default defineConfig({
  base: process.env.SITE_BASE ?? "/NextTransformer/",
  plugins: [react()],
  // Records are bundled from the repository's results/ directory, one level up.
  server: { fs: { allow: [".."] } },
  // The bundle carries every published record; it is one page, so no code splitting.
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 1500 },
});
