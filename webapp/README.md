# Melbourne Data Playground

## Development
In order to setup your development environment and run the application locally please follow these steps:
1. Open a Anaconda Prompt
2. Navigate to this webapp folder
3. Type the following command: ```conda activate base```
4. Type the following command: ```conda env create --file dev_environment.yml```
5. Type the following command: ```conda activate melbourne_playground_webapp```
5b. To fix an error I was receiving - Type the following command: ```conda update rtee```
6. To run the app from command line type: ```flask run```
7. Go to localhost:5000

> **NB!** If you get an error like '*Found conflicts! Looking for incompatible packages.*' and the environment fails to create, you may need to configure conda with the following command: ```conda config --set channel_priority flexible```

## Using Jupyter with new environment
Jupyter Notebook and Jupyter Lab (an improved version of the notebook) are included in the environment. To run them:
1. Switch to the environment using ```conda activate melbourne_playground_webapp```
2. Run Jupyter Lab: ```jupyter lab``` **OR** Run Jupyter Notebook: ```jupyter notebook```

**(OPTIONAL)** Alternatively, you can register the melbourne_playground_webapp environment with a version of Jupyter Lab/Notebook already installed in your base environment. This allows you to run a Jupyter session from base and switch between any environments that you have registered. To do this, please follow these steps:
1. Make sure Jupyter Lab/Notebook is installed in the base environment
2. Switch to the dev environment: ```conda activate melbourne_playground_webapp```
3. Install the Jupyter kernel (should be already installed though): ```conda install ipykernel```
4. Register kernel with Jupyter in base environment:
- ```ipython kernel install --user --name=melbourne_playground_webapp``` (for user-wide conda installations) **OR**
- ```ipython kernel install --name=melbourne_playground_webapp``` (for system-wide conda installations)
5. Switch back to the base environment: ```conda deactivate```
6. Run Jupyter Lab/Notebook, go to the 'Kernel' menu at the top, then 'Change Kernel' and select '*melbourne_playground_webapp*'

> **NB!** When making changes to python dependencies make sure that you also update the requirements.txt file so that the web application has the dependencies that it needs to run in the docker container.

## Deployment
This application takes a code first approach to defining the infrastructure to run the flask application.

The template structure we use is that of [AWS SAM](https://docs.aws.amazon.com/serverless-application-model/index.html).
I followed the design patterns laid out [here](https://pritul95.github.io/blogs/aws/2020/12/25/flask-aws-containerized-lambda/)

Look in the `template.yaml` file to understand how the architecture is working.

> **NB!** Make sure you have copied your aws credentials file to aws/credential in this folder.

Run the `update.sh` from a bash prompt with access to docker to build a new docker image, upload changes to ECR, and deploy API gateway with Lambda function.

