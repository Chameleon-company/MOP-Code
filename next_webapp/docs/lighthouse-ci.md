# Sprint 3: Lighthouse CI

The repository-root [Lighthouse workflow](../../.github/workflows/lighthouse.yml)
is configured to run when a PR targeting `staging` is opened, receives new commits,
or is reopened. It also defines a manual trigger. It checks
out GitHub's PR merge commit, installs locked dependencies, builds Next.js in
production mode, and audits that build on port 3001. There is no path filter, so
PRs always receive the check. New commits cancel older runs for the same PR.

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

## GitHub setup and reports

The workflow must be included in the pushed commits, not just present locally.
From `next_webapp`, stage it explicitly with
`git add ../.github/workflows/lighthouse.yml` and confirm it appears in
`git diff --cached --name-only` before committing. A `git add .` from
`next_webapp` does not include this parent directory.

The first hosted run has not yet been verified. Open a PR targeting
`staging`, then inspect **Actions → Lighthouse CI → Lighthouse performance budgets**.
Download the `lighthouse-reports-…` artifact to view HTML/JSON results. Reports are
retained for 14 days and uploaded even when an audit assertion fails; a build
failure can occur before any report is available. Reports are not sent to public
Lighthouse storage.

A repository administrator must add **Lighthouse performance budgets** as a
required status check in the `staging` branch protection rule or ruleset after its
first run. The workflow produces a failing status; branch protection makes that
status prevent merging. This repository setting cannot be enforced by a YAML file.

Implementation follows the [official Lighthouse CI configuration documentation](https://googlechrome.github.io/lighthouse-ci/docs/configuration.html).

## Local verification — 9 September 2026

- Production build passed with the audit environment configuration.
- `npm ci --dry-run --ignore-scripts --offline` passed (lockfile verification).
- Workflow YAML parsed successfully; staging trigger, working directory, and
  report upload on failure were checked.
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
