aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 119291007423.dkr.ecr.ap-southeast-2.amazonaws.com

sam build && yes "y" | sam deploy --capabilities CAPABILITY_NAMED_IAM