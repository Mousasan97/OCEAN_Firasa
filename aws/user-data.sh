#!/bin/bash
# EC2 User Data Script - Auto-setup Docker and deploy OCEAN API
# This runs automatically when the EC2 instance starts

set -e

# Update system
yum update -y

# Install Docker
yum install -y docker git
systemctl start docker
systemctl enable docker

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Add ec2-user to docker group
usermod -aG docker ec2-user

# Create app directory
mkdir -p /opt/ocean
cd /opt/ocean

# Clone the repository
git clone https://github.com/Mousasan97/OCEAN_Firasa.git .

# Create .env file (will be populated via SSM Parameter Store or manually)
cat > /opt/ocean/.env << 'ENVFILE'
# Production environment
ENVIRONMENT=production
DEBUG=false
MODEL_DEVICE=cpu
MODEL_WARMUP=true

# AI Provider - OpenAI
AI_REPORT_ENABLED=true
AI_PROVIDER=openai
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
OPENAI_MODEL=gpt-4.1-mini

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# CORS (update with your domain)
CORS_ORIGINS=["*"]
ENVFILE

# Create docker-compose.yml
cat > /opt/ocean/docker-compose.prod.yml << 'COMPOSEFILE'
version: '3.8'

services:
  ocean-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ocean-api
    restart: always
    ports:
      - "80:8000"
    env_file:
      - .env
    volumes:
      - ./uploads:/app/uploads
      - ./processed:/app/processed
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
COMPOSEFILE

# Build and start the container
cd /opt/ocean
docker-compose -f docker-compose.prod.yml up -d --build

# Create a simple status check script
cat > /opt/ocean/status.sh << 'STATUSFILE'
#!/bin/bash
echo "=== OCEAN API Status ==="
docker ps
echo ""
echo "=== Recent Logs ==="
docker logs ocean-api --tail 50
STATUSFILE
chmod +x /opt/ocean/status.sh

# Log completion
echo "OCEAN API deployment complete!" >> /var/log/user-data.log
