## Deployment & Architecture
Below is outlined a pathway for deploying current webapp application (that is deployed in AWS) into Google Cloud.

### Google Cloud Run
Great tool for deploying docker applications as serverless web applications.
[Learn why here.](https://cloud.google.com/blog/topics/developers-practitioners/cloud-run-story-serverless-containers)

#### Steps to Develop
1. Install the google cloud SDK
1. Run the following login inside of the Google Console  
    1. ```gcloud auth login``` - login with deakin
    1. ```gcloud auth application-default login``` - this will setup your environment with user account principal so you can run and work with gcloud resources on your local machine (like cloud storage)  
    
    [View Auth Documentation here](https://googleapis.dev/python/google-api-core/latest/auth.html)
1. At this point you should have everything setup on your machine to work with gcloud cli and gcloud resources.

#### Steps to deploy
1. Install Google Cloud SDK Cli
1. Login using Deakin Credentials
1. Copy Dockerfile to `../webapp` directory
1. Run `gcloud run deploy test-cloud-run-again --source  . --allow-unauthenticated` from the webapp directory.
1. Select 10 as region
1. Application should be deployed
    * In the Deakin Project, you will likely not have permission to perform the `--allow-unauthenticated` action.  
    You can still test the instance as authenticated user using the steps outlined [here](https://cloud.google.com/run/docs/authenticating/developers).
    * To deploy to production / public, check in with `Nghia Dang` to request `Cloud Run Invoker` role be added to anonymous users for your cloud run application ([Documentation here](https://cloud.google.com/run/docs/authenticating/public)).
1. You can now test the application by calling 
    ``` 
        curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" [url to deployed instance]
    ```