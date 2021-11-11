# Melbourne Data Playground

## Development
In order to setup your development environment and run the application locally please follow these steps:
1. Open a Anaconda Prompt
1. Navigate to this webapp folder
1. Type the following command ```conda activate base```
1. Type the following command ```conda env create --file dev_environment.yml```
1. Type the following command ```conda activate melbourne_playground_webapp```
1. To fix an error I was receiving - Type the following command ```conda update rtee```
1. To run the app from command line type ```flask run```
  

> <b>NB!</b> When making changes to python dependencies make sure that you also update the requirements.txt file so that the web application has the dependencies that it needs to run in the docker container.

## Deployment
This application takes a code first approach to defining the infrastructure to run the flask application.

The template structure we use is that of [AWS SAM](https://docs.aws.amazon.com/serverless-application-model/index.html).
I followed the design patterns laid out [here](https://pritul95.github.io/blogs/aws/2020/12/25/flask-aws-containerized-lambda/)

Look in the `template.yaml` file to understand how the architecture is working.

> <b>NB!</b> Make sure you have copied your aws credentials file to aws/credential in this folder.

Run the `update.sh` from a bash prompt with access to docker to build a new docker image, upload changes to ECR, and deploy API gateway with Lambda function.

