# How to navigate and use the Chameleon MOP Repository for the Datascience team
Written Trimester 3, Dec 2023

If you are new to the Capstone project or new to the datascience team, please read the entire document.

### At the beginning of the trimester you will be required to clone the Chameleon MOP-Code repository and create your own branch. 
First you will need to install git onto your computer. You can also download Github Desktop if you do not want to use git commands in a terminal.
You must then clone this repository onto your local machine which means that this repository will be copied onto your computer and will appear as a folder. 
You must also create your own branch using your name. You can do this in GitHub Desktop or the Github Website or using git commands. 

### Navigate to the repo on your computer and go to MOP-Code > Playground
This is the folder where you will work out of during the trimester. Create a new folder using your name. Inside this folder you can place the files that you are working on and create new files. 
This is your workspace for the trimester. 

Remember your workspace is located in: GitHub > MOP-Code > Playground > folder with your name

### Committing work to your branch

At first the work will only be stored on your local computer. You must commit the work to your branch. Again, there are a variety of ways to do this. For those who are not familiar with github, the easiest 
way is to use Github Desktop. Make sure you are committing to your branch, not to anyone else's and not to the master repo. 

### Committing to the master repository

Once you have committed work to your branch you can push it to the master repo. You can do this in Github Desktop, it will be labelled "push to origin". You will then go to the repo on the github website, 
click on the master branch and click "compare and pull request". This will create a pull request which is attemping to merge the content of your branch to the master repo. Someone will be required to review
and then merge your request. 

#### Note!
Other people are committing to the repo too. If your branch is behind commits to master, you must rebase your branch first to avoid conflicts. ALWAYS do this before you push your commits to master. 
To do this, click on the link where it says "you are x commits behind" and then create a pull request to merge those commits from master into your own branch. After you do this, you can then push to master. You can also use GitHub Desktop, it has a "update from master" under Branch menu.

## At the end of the trimester

Eventually you will get to the end of the trimester and be wrapping everything up. You might be wondering where to place your work. Your team leaders will give the green light to start pushing your final work to the master repo

Navigate to MOP-Code > datascience > usecases

If you have completed a new use case, you will need to place it in the folder named READY TO PUBLISH. The team leaders should create a folder named eg: T3 2023 and your can place your work in there. 

If you have completed repointing an old use case you must place it in the folder named REWORK COMPLETED. 

For the completed cases, we need three files: .html, .json, .jpynb. HTML is output from the jpynb file. Please refer to usecases/Output_html_file. The JSON file can be created by vscode. Please refer to the completed cases in READY TO PUBLISH. Please name these three files with the same name and put them in a folder with the same name.

If you have any unfinished work or are a junior planning to continue with your work next trimester, place it in the folder named PLAYGROUND. This is the folder that holds all unfinished work. At the start
of the trimester if you would like to pick up someone's work you can find unfinished work in this folder. Please make sure that the work doesn't belong to anyone else in the team first, such as a junior coming
back from the previous trimester because it might be their work. Otherwise any other work is fair game and you can call dibs and start working on it. In this folder you might find old incomplete use cases or use cases that need to be repointed but
weren't fully completed in the past. 

### Templates

In the datascience > usecase folder you will also find some templates. The "Import Dataset API template" is a data importing template. You can use it to import any dataset from the Open Data City of Melbourne website as a pandas dataframe. 
It is designed to take a lot of trouble out of using the APIs to import the datasets but you must still aim to understand how it works in case future APIs are released and you need to alter the code. 

The second template named "usecase_TEMPLATE" is a template which you can use to start your new use case. All use cases follow the general format of this template.

### If you have any questions about any of this, please contact your current Datascience Leadership team. 




