#  Chameleon Company - Melbourne Open Data Playground Code repository
This is a public repository designed for the Chameleon-City of Melbourne Open Data project team to manage their codebase and artifacts for development and deployment of the Melbourne Open Data Playground and associated Public GitHub Repository(MOP).

**Note: This repository is Private and contains the code base for the web application and any associated collatoral for our project.**

## Sub-folders

**Only publish content to this repository to the folders managed by your team!**

### Data Science team folders 
For the Data Science team you should only publish to the following folders:
- *Playground*  - create your named folder in here for private WIP content
- *datascience* - content you wish to share or collaborate on with the webapp team or others in the datascience team
- *public_github/MOP* - Staging and testing content before being published to the MOP repository.

### Web Development Team Folders 
Similarly, the Web Development team should only publish to the following folders:
- *faq*  - contains all the queries created for the FAQs folder.
- *google_cloud_test* - Under this folder, contains the instance of the web application in the form of an image.
- *webapp* - This is the web application itself, an executable instance for the team project.

### Previous Folders
Please note that the folders can be overlooked as these all the previous contributions:
- *ETL* - This is the server that consists of our parking sensors’ recorded data. They are recorded daily.
- *prior_work* - These consist of all of the progress made by the previous trimesters (by all the Teams).
  
## Setting up the Repository:

**Please follow these instructions to set up your environment. This is especially useful for all Junior Students!**

### Pre-Requisites:
Please also make sure you have these tools downloaded on your desktop:
- [Microsoft VS Code (Word Editor)](https://code.visualstudio.com/Download)
- [GitHub Desktop (An alternative for handling GitHub CLI) ](https://desktop.github.com/)
- [Miniconda3 (Environment Path for the WebApp)](https://docs.conda.io/en/latest/miniconda.html)

**NOTE:** Please install only the setup files for the first two applications only. Then make sure to install MiniConda3 while proceeding to Step 4. 

### Step 1: Clone the Repository:
Once we have downloaded the pre-requisites for the web application, proceed to copy the link via the GitHub Desktop option, as shown in the picture link below: 

<a href="https://drive.google.com/file/d/1LinRBm13b1x7AGfAIMEjejavz6SH0SSJ/view?usp=sharing">Step 1a</a>

You should get your GitHub Desktop opened and would click on ‘Clone’ after you proceed to direct your repository to the place where you wish to save it. It is better to save all the repositories under the “GitHub” folder, which will be under your “Documents” folder.

<a href="https://drive.google.com/file/d/1XR_bUQHsqNIFjQULohcI9N8spTGbxvr_/view?usp=sharing">Step 1b</a>

### Step 2: Link the repository with your VS Code
After cloning the repository, type in “VSCode” on your Desktop’s search bar.

<a href="https://drive.google.com/file/d/19fyVrszUfjkBryORR8WW9cBHQoK-SepO/view?usp=sharing">Step 2a</a>

Open your VSCode and on the top left, you should be able to see the explorer folder, as shown by the yellow arrow in the following picture. This is how you will be able to open up a new folder.

<a href="https://drive.google.com/file/d/1TNEcRiL627QD_Ha8euEop0t0E6zvsaaH/view?usp=sharing">Step 2b</a>

Now proceed to Documents > GitHub > MOP-Code. Select the “webapp” folder and click on “Select Folder”.

<a href="https://drive.google.com/file/d/17Dd-iGftFM0Jaf_kg6nZgTDUOVCv2ETW/view?usp=sharing">Step 2c</a>

### Step 3: Add the Extensions
Now add the extensions from VSCode as well. Note that you have a “Extensions” Tab on the left side as shown in the picture below:

<a href="https://drive.google.com/file/d/17Yd1Pr0X69iH4FMpV8eOsco6oHDXSZnI/view?usp=sharing">Step 3</a>

Now add the following extensions by simply typing in:
-	GitHub
-	Docker
-	Python
Finally, install these extensions!

### Step 4: Installing Miniconda3
Install Miniconda3 using this [link](https://docs.conda.io/en/latest/miniconda.html)

**NOTE:** While installing your environment on your machine, please always make sure to check the first box when you reach this stage of pre-installation. Otherwise, you will have to set the environment as shown here; see Troubleshooting – Scenario 1.

**ONCE AGAIN, MAKE SURE TO TICK “ADD MINICONDA3 TO MY ENVIRONMENT VARIABLE” (DEFAULT IS UNTICKED):**

<a href="https://drive.google.com/file/d/1LqfHVBcMHsMzfq1sjuSacybwtCk8OG3p/view?usp=sharing)">Step 4</a>

### Step 5: Create the Environment
Now that we have our web application linked to VSCode, note that there is an “environment.yml” file. This is essentially a kind of file used for the web application’s configuration. This directly implies that it will set and install all the dependencies needed for it. To create our environment, please copy and execute this terminal command:

```bash
conda env create --file dev_environment.yml
```

### Step 6: Select Miniconda3 Environment from the List of Environment
From Step 3, you have only added a new environment for your webapp. However, this path hasn’t been linked yet. Now to ensure to check your miniconda3 environment, first execute this command:

```bash
conda env list
```

Under this command, you should be able to see your list of environments. We need to set our environment, which will exist under “melbourne_open_playground”. Please copy and execute this command in the terminal:

```bash
conda activate melbourne_playground_webapp
```

### Step 7: Run the WebApp
Finally, you will run the web application by clicking on the “Run without Debugging” Option, which is under the “Run” Tab. You will get see your application running and should give a message like:

<a href="https://drive.google.com/file/d/1Xw74LJ4ZyywxBmFxeJ6ssFdqVDt-yyiD/view?usp=sharing">Step 7a</a>

Now just click on the website link mentioned on the terminal. This will lead you to the web application, running as shown here. Now you are all ready to develop this web application!

<a href="https://drive.google.com/file/d/16my1X_XI_Y9URfFhWgaoC5lLAq2-pOBG/view?usp=sharing">Step 7b</a>

## Troubleshooting Corner

**The following sub-headings might address some of the issues our web development has experienced during the upskilling phase of our team project. So here are some of the possible fixes you need to know:**

### Scenario 1: Setup environment variables for windows 
Please follow these steps, in case you are having trouble with your environment path: 
- Right-Click on My Computer
- Click on Advanced System Settings
- Click on Environment Variables or you can simply search for environment variables as follows

<a href="https://drive.google.com/file/d/1ZGOP6QIJEbP7gxQxCTqBP6FrL4L1BlGy/view?usp=sharing">Scenario 1a</a>

- Then, under System Variables, look for the path variable and click edit

<a href="https://drive.google.com/file/d/1jja4Fv45e-Tde8u3iAeQicwpQPWf2Bdk/view?usp=sharing">Scenario 1b</a>

- Add the paths as shown in the below picture. This has four paths added as shown:

<a href="https://drive.google.com/file/d/1r5oXS_DHMLr0L-Ltx2JOm_5BQJWDuw6c/view?usp=sharing">Scenario 1c</a>

### Scenario 2: Use-Case Templates not appearing in the Development environment – (Registry Fix)
In case your templates are not showing as shown in the picture from the last step of the set up procedures, please head on to the “Registry Editor”.

<a href="https://drive.google.com/file/d/15R-UgQwCS54bC-K_ooBt-f_rpIab9CBk/view?usp=sharing">Scenario 2a</a>

On the left hand of the Registry Editor, look for the “HKEY_CLASSES_ROOT”. From there, proceed to the ‘.js’ folder and then select the ‘Content Type’ file. The content type would initially be as “text/plain” but please set this as “text/javascript”. 

<a href="https://drive.google.com/file/d/1K6KM0-23PUIL4AQgcib8vcbbpng5x4oW/view?usp=sharing">Scenario 2b</a>

<a href="https://drive.google.com/file/d/1jauw9tgivgZhHPeXBL-Vb4wJbU8xH50j/view?usp=sharing">Scenario 2c</a>

<a href="https://drive.google.com/file/d/1CNXXFsfSxenF_CqxnKCOn3ZZpac1gO6M/view?usp=sharing">Scenario 2d</a>


Now, similar will also be done under the “HKEY_LOCAL_MACHINE” folder. Please repeat the above steps:

<a href="https://drive.google.com/file/d/1dnBnmaSVoc9lZtNCqCuS4b9rkr8zoyyh/view?usp=sharing">Scenario 2e</a>

### Scenario 3: Conda Environment Activation not working in PowerShell
Please proceed to this link in case your environment is not working on [Conda Environment Activation.](https://github.com/conda/conda/issues/8428)




## License

This is the license we have for our Repository

[MIT](https://choosealicense.com/licenses/mit/)


## Authors

[Muhammad Sohaib bin Kashif (T1 2022)](https://github.com/M-S-Kashif)

