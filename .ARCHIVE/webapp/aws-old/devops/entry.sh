#!/bin/sh

if [ -z "${AWS_LAMBDA_RUNTIME_API}" ]; then
    exec aws-lambda-rie python -m awslambdaric $1
else
    exec python -m awslambdaric $1
fi