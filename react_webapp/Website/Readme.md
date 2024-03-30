# Melbourne Open Playground

This repository houses the code for MOP website.

## Getting Started

1. Clone the reposioty.
2. Install Packages: `npm install`
3. Start the local server: `npm run dev`
___
## Tech Stack

Frontend:
1. **Library**: React | [React Documentation](https://legacy.reactjs.org/docs/getting-started.html)
2. **Stylesheet**: Tailwind CSS & Custom CSS | [Tailwind Documentation](https://v2.tailwindcss.com/docs)
3. **Local Development Server**: Vite | [Vite Documentation](https://vitejs.dev/guide/)


> **Note**: You can either choose to use in-line tailwind css or make your own custom css stylesheets

Backend:
(TBA)

___
## Folder Structure

This is a simplified view of the project structure for the website.

```
└── 📁src
    └── 📁assets -> Contains all the assets e.g., images, banner, icons etc.
    └── 📁components -> Contains all the reusable componenets e.g., header, footer etc.
    └── 📁pages -> Contains all the static pages e.g., Homepage, conatct us etc.
    └── 📁styles -> Conatins the CSS files
    └── App.css
    └── App.jsx
    └── index.css
    └── main.jsx
```
___
## Code Guidelines

The project has to maintain a few set rules in order to maintain structure and uniformity. Such as:

* Files have to stored in the respective folders mentioned [Here](##Folder Structure)
* Ensure that the functions, methods, variables etc. are properly commented
* Reusable UI componenets such as buttons, headers and footers are to be stored as components in the componenets folder to maintain style unitformity and maintain the DRY priciples.
* API keys are to be saved as a ".env" file.

___
## Git Contributions

### Branching

Branches are used for isolated feature developments and should always branch from the source they intend to merge into.
Branches that are created from `main` will return to `main`.

Branch names should have an indicator of which team are you in. For example: `Username_WD` or `Username-webdev`

⚠️ Make sure to "Fetch" to update the repository before you push into the branch you are working on to avoid merge conflicts especially when collaborating with team members.

### Commit Messages

While there are no set rule for formatting commit messages we still have to be make sure the heading and the body are properly written.
You can use tags, as listed below ,which would help us identify what type of of commit is being being made.

| Tags        | Description           |
| ------------- |:-------------:|
|   Build  | Related to build process
|  Docks| Documentation work
|  Feature | Adding new features
| Patch | Bug fixes or hot fixes
| Design | Style changes
| Refactor | Code refactoring
| Test | New or update test cases


### Pull Requests

Some rules to look out for:
1. Make sure your pull request message is not vauge nor too self explanitory.
2. If you are pushing from a branch that has other collaborator you have mentioned their name in the pull request message.
