FROM public.ecr.aws/lambda/python:3.7

COPY logic ./logic

COPY function.py requirements.txt ./

RUN python3.7 -m pip install -r requirements.txt -t .

# Command can be overwritten by providing a different command in the template directly.
CMD ["function.run"]