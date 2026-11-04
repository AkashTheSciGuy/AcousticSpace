import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  // Development-only proxy: browser calls /api on Vite's origin,
  // and Vite forwards them to FastAPI without opening CORS access.
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
