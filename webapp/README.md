# Melbourne Data Playground

## Web Development

In order to setup your development environment and run the application locally please follow these steps:

1. Open a Anaconda/miniconda Prompt
2. Navigate to this webapp folder
3. Type the following command: ```conda activate base```
4. Type the following command: ```conda env create --file dev_environment.yml``` or ```conda env create -f dev_environment.yml```
5. Type the following command: ```conda activate melbourne_playground_webapp```
6. To run the app from command line type: ```flask run``` or go the run tab in visual studio code and select " Run without debugging ". 
7. You should get a link in the terminal, follow the link to view the website on your local machine.

> Known Issues
> 1. If you get an error like '*Found conflicts! Looking for incompatible packages.*' and the environment fails to create, you may need to configure conda with the following command: ```conda config --set channel_priority flexible```
> 2. To fix an error I was receiving - Type the following command: ```conda update rtee```
> 3. QuickFix for conda environemnts not activating in powershell : 
>- Step 1 : Run powershell as admin
>- Step 2 : Change the execution policy by typing ```Set-ExecutionPolicy RemoteSigned```
>- Step 3 : Restart powershell
>- Step 4 : Initialise the conda environment by typing ```conda init```\
> Read more about the issue [here](https://github.com/conda/conda/issues/8428)

## Running and Building a Conda Dev Container in VS Code

### Prerequisites

1. **Docker Installation:**
   Ensure that Docker is installed on your system, and the Docker daemon is running.

2. **Install VS Code Extension:**
   Install the "Dev Containers" extension for Visual Studio Code. You can find it in the Extensions view (Ctrl+Shift+X).


### Getting the Dev Container Running

1. **Open Project in VS Code:**
   Open your project folder in Visual Studio Code.

2. **Navigate to the Devcontainer File:**
   Navigate to the `.devcontainer` folder within your project. If the folder does not exist, you may need to create it.

3. **Trigger Rebuild and Run:**
   - For macOS: Press `Cmd + Shift + P`.
   - For Windows: Press `Ctrl + Shift + P`.
   
   In the command palette, type and select "Dev Containers: Rebuild and Run Container."

4. **Monitor Container Status:**
   Once the container is successfully built and running, you should see the name of the dev container in the bottom left corner of VS Code.

## Additional Information

- The "Dev Containers" extension allows you to develop inside a Docker container, providing a consistent and reproducible development environment.

- The `.devcontainer` folder contains configuration files, such as `devcontainer.json` and `Dockerfile`, specifying the setup of your development container.

- Customization of the dev container, including Conda environment setup, can be done in the `Dockerfile` and `environment.yml` files.

- To stop or restart the dev container, you can use the options available in the bottom-left status bar of VS Code.

This guide outlines the essential steps to set up and run a Conda dev container in Visual Studio Code, ensuring a seamless and isolated development environment for your project.

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

> Currently the website is not being hosted in AWS, in the near future the website would be hosted in GCP

This application takes a code first approach to defining the infrastructure to run the flask application.

The template structure we use is that of [AWS SAM](https://docs.aws.amazon.com/serverless-application-model/index.html).
I followed the design patterns laid out [here](https://pritul95.github.io/blogs/aws/2020/12/25/flask-aws-containerized-lambda/)

Look in the `template.yaml` file to understand how the architecture is working.

> **NB!** Make sure you have copied your aws credentials file to aws/credential in this folder.

Run the `update.sh` from a bash prompt with access to docker to build a new docker image, upload changes to ECR, and deploy API gateway with Lambda function.

