# /frontend/Dockerfile
FROM node:20

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

RUN npm run build

# Use a server to serve the build, or use Nginx in production
CMD ["npm", "start"]
