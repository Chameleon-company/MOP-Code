## Deployment & Architecture
Below is outlined a pathway for deploying current webapp application (that is deployed in AWS) into Google Cloud.

### Google Cloud Run
Great tool for deploying docker applications as serverless web applications.

#### Steps to deploy
1. Install Google Cloud SDK Cli
1. Login using Deakin Credentials
1. Copy Dockerfile to `../webapp` directory
1. Run `gcloud run deploy test-cloud-run-again --source  . --allow-unauthenticated` from the webapp directory.
1. Select 10 as region
1. Application should be deployed
    * In the Deakin Project, you will likely not have permission to perform the `--allow-unauthenticated` action.  
    In this case you can use developer access as described here: `https://cloud.google.com/run/docs/authenticating/developers`
    * To deploy to production / public, check in with `Nghia Dang` to request `Cloud Run Invoker` role be added to anonymous users.
1. You can now test the application by calling 
    ``` 
        curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" https://test-cloud-run-again-hewios74ha-ts.a.run.app
    ```