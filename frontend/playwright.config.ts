import { defineConfig } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import path from 'node:path'

const python = process.env.ACOUSTICSPACE_TEST_PYTHON || 'python'
const fixtureDir = path.resolve('.e2e')
execFileSync(python, ['../backend/scripts/create_e2e_fixtures.py', '--output', fixtureDir])

export default defineConfig({
  testDir: './e2e',
  workers: 1,
  use: {
    baseURL: 'http://127.0.0.1:5173', headless: true,
    launchOptions: { executablePath: process.env.ACOUSTICSPACE_CHROMIUM },
  },
  webServer: [
    {
      command: `${JSON.stringify(python)} -m uvicorn app.main:app --host 127.0.0.1 --port 8000`,
      cwd: '../backend',
      url: 'http://127.0.0.1:8000/health',
      env: {
        ACOUSTICSPACE_BASELINE_MODEL: path.join(fixtureDir, 'baseline.joblib'),
        ACOUSTICSPACE_ACOUSTIC_MODEL: path.join(fixtureDir, 'acoustic.joblib'),
        ACOUSTICSPACE_COMPARISON_REPORT: path.join(fixtureDir, 'comparison.json'),
      },
    },
    { command: 'npm run dev -- --host 127.0.0.1', url: 'http://127.0.0.1:5173' },
  ],
})
