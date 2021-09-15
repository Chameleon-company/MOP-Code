# Melbourne Data Playground


## Deployment
This application takes a code first approach to defining the infrastructure to run the flask application.

The template structure we use is that of [AWS SAM](https://docs.aws.amazon.com/serverless-application-model/index.html).
I followed the design patterns laid out [here](https://pritul95.github.io/blogs/aws/2020/12/25/flask-aws-containerized-lambda/)

Look in the `template.yaml` file to understand how the architecture is working.

Run the `update.sh` script to build a new docker image, upload changes to ECR, and deploy API gateway with Lambda function.

