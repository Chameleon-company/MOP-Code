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

## Using Jupyter with new environment
Jupyter Notebook and Jupyter Lab (an improved version of the notebook) are included in the environment. To run them:
1. Switch to the environment using ```conda activate melbourne_playground_webapp```
1. Run Jupyter Lab: ```jupyter lab``` **OR** Run Jupyter Notebook: ```jupyter notebook```

**(OPTIONAL)** Alternatively, you can register the melbourne_playground_webapp environment with a version of Jupyter Lab/Notebook already installed in your base environment. This allows you to run a Jupyter session from the base environment and switch between any environments that you have registered. To do this, please follow these steps:
1. First ensure Jupyter Lab/Notebook is installed in the base environment.
1. Switch to the dev environment: ```conda activate melbourne_playground_webapp```
1. Install the Jupyter kernel (should be already installed though): ```conda install ipykernel```
1. Register kernel with Jupyter in base environment:
- ```install --user --name=melbourne_playground_webapp``` (for user-wide conda installations) **OR**
- ```install --name=melbourne_playground_webapp``` (for system-wide conda installations)
1. Switch back to the base environment: ```conda deactivate```
1. Run Jupyter Lab: ```jupyter lab``` **OR** Run Jupyter Notebook: ```jupyter notebook```
1. From within Jupyter Lab/Notebook, you can got to the "Kernel" menu at the top, and then "Change Kernel".

> <b>NB!</b> When making changes to python dependencies make sure that you also update the requirements.txt file so that the web application has the dependencies that it needs to run in the docker container.

## Deployment
This application takes a code first approach to defining the infrastructure to run the flask application.

The template structure we use is that of [AWS SAM](https://docs.aws.amazon.com/serverless-application-model/index.html).
I followed the design patterns laid out [here](https://pritul95.github.io/blogs/aws/2020/12/25/flask-aws-containerized-lambda/)

Look in the `template.yaml` file to understand how the architecture is working.

> <b>NB!</b> Make sure you have copied your aws credentials file to aws/credential in this folder.

Run the `update.sh` from a bash prompt with access to docker to build a new docker image, upload changes to ECR, and deploy API gateway with Lambda function.

