<img src="next_webapp/public/img/new-logo-green.png" alt="Melbourne Open Data Playground" width="140"/>

# Melbourne Open Data Playground (MOP)

**An industry capstone project by [Chameleon](https://github.com/Chameleon-company), sponsored by Deakin University for the [City of Melbourne](https://data.melbourne.vic.gov.au/).**

🌐 **Live site:** [melbourne-open-playground.vercel.app](https://melbourne-open-playground.vercel.app)

---

## About

Melbourne Open Data Playground (MOP) helps the business, research, and developer community discover, explore, and build on the City of Melbourne's Open Data. Since 2014 the City of Melbourne has been an Australian leader in open data, but two gaps remain in the community's ability to make use of it:

- understanding **how to access** Open Data, and
- knowing **how to gain insights** from it to support application development and solve real city problems.

MOP addresses both — publishing runnable Jupyter notebook use cases against City of Melbourne datasets, and a web platform for browsing, showcasing, and interacting with them.

This repository is developed and maintained by **Chameleon**, a Deakin University Industry Capstone team, with each trimester's student cohort contributing new use cases, data science work, and improvements to the web platform.

---

## Repository structure

| Folder | Contents |
|---|---|
| [`next_webapp/`](next_webapp/) | The MOP website — a Next.js application. See its own [README](next_webapp/README.md) for setup instructions. |
| [`usecases/`](usecases/) | Jupyter notebook use cases built on City of Melbourne Open Data, organised by publishing status (`READY TO PUBLISH`, `FINALISED`, `RETIRED`, `UPDATE NEEDED`, `DEPENDENCIES`). See [`Use_Case_Index.md`](usecases/Use_Case_Index.md) for the full index. |
| [`Playground/`](Playground/) | Work-in-progress and experimental notebooks from current and past student contributors. |
| [`documentation/`](documentation/) | Onboarding guides, tooling how-tos, and team process documentation for new contributors. |

---

## Getting started

1. Clone the repository:
   ```bash
   git clone https://github.com/Chameleon-company/MOP-Code.git
   ```
2. To run the **website**, see [`next_webapp/README.md`](next_webapp/README.md) for full setup instructions.
3. To explore a **use case**, open any notebook under [`usecases/`](usecases/) in Jupyter — start with [`usecases/usecase_TEMPLATE.ipynb`](usecases/usecase_TEMPLATE.ipynb) if you're building a new one.

---

## How can I contribute?

1. Set up a [GitHub account](https://github.com/signup) and get familiar with [Git and GitHub](https://lab.github.com/) if you haven't already.
2. Star and watch this repository for updates.
3. Read through [`documentation/`](documentation/) for onboarding guides and team working practices.
4. Explore the existing [use cases](usecases/) and the [website](https://melbourne-open-playground.vercel.app).
5. [Open an issue](https://github.com/Chameleon-company/MOP-Code/issues) if you spot a defect or have an idea for a new feature.
6. Build your own use case using [City of Melbourne Open Data](https://data.melbourne.vic.gov.au/) and the [notebook template](usecases/usecase_TEMPLATE.ipynb), then open a pull request.

Each use-case team/project works in its own branch, merging into `master` once complete — see [`documentation/`](documentation/) for the detailed contribution and pull request workflow.

---

## About Chameleon

[Chameleon](https://github.com/Chameleon-company) is a Deakin University student capstone organisation building applied data and software projects for real-world partners. Beyond MOP, Chameleon also maintains projects such as [EVAT](https://github.com/Chameleon-company/EVAT) and [TreeO2](https://github.com/Chameleon-company/TreeO2).

📧 Contact: chameleon@deakin.edu.au
