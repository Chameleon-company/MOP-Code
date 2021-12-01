FROM public.ecr.aws/lambda/python:3.7

RUN yum update -y

COPY requirements.txt .

RUN pip3 install -r requirements.txt --target ${LAMBDA_TASK_ROOT}

# COPY all python files into the lambda task root
COPY *.py ${LAMBDA_TASK_ROOT}/

CMD ["function.run"]