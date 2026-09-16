# Sprint 3: Lighthouse CI

This originally shipped with a repository-root `.github/workflows/lighthouse.yml`
GitHub Actions workflow, gated on PRs targeting `staging`. That workflow has been
removed: it never actually ran (zero executions since it merged) because this
repo's PR flow exclusively targets `master`, not `staging` — no PR has ever been
opened against `staging`. The tooling below (`lighthouserc.cjs`, the npm scripts)
is unaffected and fully functional on its own; only the automated CI trigger is
gone. See "Re-adding CI enforcement" below if reviving it.

## Audits and budgets

Lighthouse uses its default mobile simulation, three runs per URL, and median
aggregation for each budget. The pages are `/en`, `/en/about`, and
`/en/privacypolicy` (home, About, and Privacy Policy).

| Metric | Required budget |
| --- | --- |
| Performance score | At least 70/100 |
| Largest Contentful Paint | At most 4,000 ms |
| Total Blocking Time | At most 600 ms |
| Cumulative Layout Shift | At most 0.1 |
| HTTP status audit | Must pass |

These are initial absolute budgets, not comparisons with the previous commit.
Regressions that remain within the budgets will not fail. Review real GitHub
runner measurements before tightening budgets in `lighthouserc.cjs`; do not lower
them simply to hide a failing PR. Other Lighthouse categories remain visible in
reports but do not block this performance task.

The build and audits use `scripts/lighthouse-env.cjs` to supply non-secret local
service settings. No production credentials or GitHub write token are needed,
including for fork PRs. Database-backed home sections may show empty/error states;
this measures the public frontend shell, not a populated deployed staging site.
Auditing the PR build ensures the proposed changes are measured before merging.

## Run locally

With Node 22 and Google Chrome installed, from `next_webapp`:

```sh
npm ci
npm run test:lighthouse
npm run build:lighthouse
npm run lighthouse
```

If Chrome is not detected on macOS:

```sh
CHROME_PATH='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' npm run lighthouse
```

Port 3001 must be free. Lighthouse starts and stops the production server itself.
Open HTML reports in `lighthouse-reports/`; raw runs and assertion results are in
`.lighthouseci/`. Both directories are ignored by Git. A budget failure exits
nonzero. Build/startup/browser failures also fail the workflow.

## Re-adding CI enforcement

There is no GitHub Actions workflow running these audits automatically right
now — only the local commands above. To wire this back up correctly:

1. Add a workflow (e.g. `.github/workflows/lighthouse.yml`) triggered on
   `pull_request: branches: [master]` — not `staging`, since that's the branch
   PRs in this repo actually target. Have it run `npm run build:lighthouse`
   then `npm run lighthouse` from `next_webapp`, and upload
   `lighthouse-reports/` and `.lighthouseci/` as artifacts.
2. Open a real PR to `master` so the workflow runs at least once — GitHub only
   lets you select a status check for branch protection after it has reported
   once.
3. A repository administrator then adds the check's job name as a **required
   status check** in `master`'s branch protection rule/ruleset. This can't be
   enforced by the YAML file alone.

Implementation follows the [official Lighthouse CI configuration documentation](https://googlechrome.github.io/lighthouse-ci/docs/configuration.html).

## Local verification — 9 September 2026

- Production build passed with the audit environment configuration.
- `npm ci --dry-run --ignore-scripts --offline` passed (lockfile verification).
- `npm run test:lighthouse` passed six CLI integration cases: a healthy report,
  performance score regression, LCP regression, blocking-time regression, layout
  shift regression, and an HTTP error page. Failure cases returned exit code 1.
- The full Chrome run completed all nine audits and saved HTML/JSON reports even
  though assertions returned exit code 1.

| Page | Median performance | Median LCP | Budget outcome |
| --- | --- | --- | --- |
| Home | 63/100 | 7,860 ms | Failed score and LCP |
| About | 81/100 | 3,339 ms | Passed |
| Privacy Policy | 78/100 | 4,236 ms | Failed LCP |

These measurements are from local macOS/Node 24 with mobile simulation, not the
GitHub Ubuntu/Node 22 runner. The integration is functional locally, but the site
currently exceeds the initial budgets. The first hosted PR run and administrator
configuration of the required status check remain necessary before claiming merge
protection is active. No performance budgets were relaxed to make this test pass.
