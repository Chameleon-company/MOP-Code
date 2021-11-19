aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 119291007423.dkr.ecr.ap-southeast-2.amazonaws.com

docker build -t deakin_melbournecity . -f deployment/Dockerfile

docker tag deakin_melbournecity:latest 119291007423.dkr.ecr.ap-southeast-2.amazonaws.com/deakin_melbournecity:latest

docker push 119291007423.dkr.ecr.ap-southeast-2.amazonaws.com/deakin_melbournecity:latest

aws lambda update-function-code --region ap-southeast-2 --function-name ParkingAvailability --image-uri 119291007423.dkr.ecr.ap-southeast-2.amazonaws.com/deakin_melbournecity:latest