import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 8090,
    proxy: {
      "/api": "http://localhost:8088",
    },
  },
});
