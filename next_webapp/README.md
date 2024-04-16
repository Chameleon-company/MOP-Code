# Melbourne Open Playground

This repository houses the code for MOP website.

## Getting Started

1. Clone the reposioty.
2. Install Packages: `npm install`
3. Start the local server: `npm run dev`
4. View the website: Open [http://localhost:3000] (http://localhost:3000) with your browser to see the result. You can also just click on the link provided in the terminal when starting the dev server.

## Architecture

![Architecture](/markdownAssets/arch1.png)

Above is an illustration of how the software is designed. We will be using a microservices architechture for out website. Imagine having three different container images. One that serves the website, one that serves the mongoDB database and one that serves the python libraries for the casestudies to be interactive. We will be using kubernetes in the future to manage all these container images. 

The deployment is done in three stages:

1. Build: Build the image using google cloud build.
2. Store: Store the image in artifact registry.
3. Deploy: Deploy image using cloud run.

## Alternative Design:

![Alternate Design](/markdownAssets/archalt.png)

Until we can get the mongodb database up and running we will be replacing it with firebase. We will be using 3 firebase features:

1. Real-time Database.
2. Cloud storage.
3. Authentication.

> Firebase will be a temporary solution until the mongodb database is being served as a microservice.

## Database schema:

![Schema](/markdownAssets/Schema.png)

We will be using a noSQL database. The schema is just a represenation of how the database works. We will be creating collections based on trimester. Each document will hold the information stated as below with a link to the file stored in firebase storage.

## Render Casestudies

![Render](/markdownAssets/Render.png)

once the user from DS uploads the casestudies through the website, the database would store the `HTML` file. We can then fetch the file from the database and render directly into the website. Since the casestudies are interactable we can create an API that would fetch the libraries from the python libraries microservice. 

## Tech Stack

### Frontend:
1. **Library**: React | [React Documentation](https://legacy.reactjs.org/docs/getting-started.html)
2. **Stylesheet**: Tailwind CSS & Custom CSS | [Tailwind Documentation](https://v2.tailwindcss.com/docs)

> **Note**: You can either choose to use in-line tailwind css or make your own custom css stylesheets

### Backend:

1. **Firebase**: The website would be utilising the firebase storage and firebase database to store casestudies and map the casestudies with user generated information. The website would also be using authentication features provided by firebase.

2. **MongoDB**: In the long run MongoDB would be replacing firebase once all the features are completed and the database is being served as microservice. 

### Deployment (GCP):

1. **Cloud Build**: We encourage members to use google cloud build instead of the docker build command. Googles platform has a set standard for how docker images are built. For example a docker image that was built on an arm64 machine would not run in GCP, as the platform only supports x86 architecture. To remove any issues regarding building images we recommend using google build command that would automatically build and upload the image on googles artifact registry. 

2. **Artifact Registry**: Google Artifact Registry is your cloud storage for the docker images and packages.It makes it easier to work with other Google Cloud tools and makes it to manage them. Makes storing, sharing, and building software faster and easier. The repository also allows for easier CI/CD and git integration.

3. **Cloud Run**: Google Cloud Run lets us run our code on gcp serverless style. This makes deployment easier as it allows us to deploy the images uploaded in the artifact registry. Cloud Run scales automatically to handle traffic and you only pay for what you use.

## Folder Structure

This is a simplified view of the project structure for the website. We would be focusing on two main folders. `src` and `public`.

```
└── 📁src -> Holds the source code for the website and database
    └── 📁app
        └── 📁about
        └── 📁casestudies
        └── 📁contact
        └── 📁datasets
        └── 📁licensing
        └── 📁privacypolicy
    └── 📁assets
    └── 📁components -> Contains all the reusable componenets e.g., header, footer etc.
    └── 📁firebase -> Contains the firebase config and functions.
```

```
└── 📁public -> Holds the assests such as images, stylesheets, icons and banner.
    └── 📁img
    └── 📁styles
```

> Routing in nextJS is simplified where each of the folders inside the app folder represents the page it would be directing to. 
> For example: 'http://localhost:3000/about' link would display the `page.tsx` in the about folder. This is just a simple explanantion of how routing works in next. To learn more please visit the [site](https://nextjs.org/docs/app/building-your-application/routing).

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

## Resources