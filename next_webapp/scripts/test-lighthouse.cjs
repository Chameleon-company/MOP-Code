const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const root = path.resolve(__dirname, '..');
const cli = require.resolve('@lhci/cli/src/cli.js');
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mop-lhci-test-'));
const report = {
  requestedUrl: 'http://localhost:3001/en',
  finalUrl: 'http://localhost:3001/en',
  categories: { performance: { score: 0.9 } },
  audits: {
    'largest-contentful-paint': { score: 1, numericValue: 2000 },
    'total-blocking-time': { score: 1, numericValue: 100 },
    'cumulative-layout-shift': { score: 1, numericValue: 0.01 },
    'http-status-code': { score: 1 },
  },
};
function check(label, input, expected) {
  fs.writeFileSync(path.join(dir, 'report.json'), JSON.stringify(input));
  const result = spawnSync(process.execPath, [cli, 'assert', `--config=${path.join(root, 'lighthouserc.cjs')}`, `--lhr=${path.join(dir, 'report.json')}`], { cwd: dir, encoding: 'utf8' });
  assert.equal(result.status, expected, `${label}: ${result.stdout}\n${result.stderr}`);
  console.log(`PASS: ${label} (exit ${expected})`);
}
try {
  check('healthy report passes', report, 0);
  const slow = structuredClone(report);
  slow.categories.performance.score = 0.4;
  check('performance regression fails', slow, 1);
  const latePaint = structuredClone(report);
  latePaint.audits['largest-contentful-paint'].numericValue = 5000;
  check('loading regression fails', latePaint, 1);
  const blocked = structuredClone(report);
  blocked.audits['total-blocking-time'].numericValue = 800;
  check('blocking regression fails', blocked, 1);
  const shifted = structuredClone(report);
  shifted.audits['cumulative-layout-shift'].numericValue = 0.3;
  check('layout regression fails', shifted, 1);
  const errorPage = structuredClone(report);
  errorPage.audits['http-status-code'].score = 0;
  check('HTTP error page fails', errorPage, 1);
} finally {
  fs.rmSync(dir, { recursive: true, force: true });
}
