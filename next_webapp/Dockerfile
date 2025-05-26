FROM node:18-alpine

# Set working directory
WORKDIR /app

# Install openssl for secure connections to MongoDB Atlas
RUN apk add --no-cache openssl

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of your code
COPY . .

# Expose dev port
EXPOSE 3000

# Start the app in dev mode
CMD ["npm", "run", "dev"]
