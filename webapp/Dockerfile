FROM public.ecr.aws/lambda/python:3.7

COPY devops/serverless_wsgi.py ./
COPY devops/wsgi_handler.py ./
COPY devops/wsgi.py ./ 
COPY devops/.serverless-wsgi ./
COPY devops/entry.sh ./

COPY app.py requirements.txt  ./

RUN python3.7 -m pip install -r requirements.txt -t .
RUN pip install awslambdaric

RUN chmod +rx entry.sh

ADD flaskr ./flaskr/

ENTRYPOINT [ "/bin/bash", "./entry.sh" ]
CMD [ "wsgi_handler.handler" ]