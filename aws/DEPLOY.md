# AWS EC2 Deployment Guide for OCEAN Personality API

## Quick Start (5 minutes)

### Step 1: Launch EC2 Instance

1. Go to **EC2 Dashboard** → **Launch Instance**

2. **Name**: `ocean-personality-api`

3. **AMI**: Amazon Linux 2023 AMI (free tier eligible)

4. **Instance Type**:
   - For testing: `t3.medium` (2 vCPU, 4GB RAM) - ~$30/mo on-demand
   - For production: `t3.large` (2 vCPU, 8GB RAM) - ~$60/mo on-demand
   - **Cost saving**: Use Spot Instance for ~70% discount

5. **Key Pair**: Create new or select existing (for SSH access)

6. **Network Settings**:
   - Allow SSH (port 22) from your IP
   - Allow HTTP (port 80) from anywhere
   - Allow HTTPS (port 443) from anywhere (if using SSL)

7. **Storage**: 20 GB gp3 (default is fine)

8. **Advanced Details** → **User Data**:
   - Copy contents of `aws/user-data.sh` into the text box
   - **IMPORTANT**: Edit the `.env` section to add your `GOOGLE_API_KEY`

9. Click **Launch Instance**

### Step 2: Configure API Key

After instance launches (~3-5 minutes for setup):

1. Get the **Public IP** from EC2 console

2. SSH into the instance:
   ```bash
   ssh -i your-key.pem ec2-user@YOUR_PUBLIC_IP
   ```

3. Edit the environment file:
   ```bash
   sudo nano /opt/ocean/.env
   ```

4. Replace `YOUR_GOOGLE_API_KEY_HERE` with your actual API key

5. Restart the container:
   ```bash
   cd /opt/ocean
   sudo docker-compose -f docker-compose.prod.yml restart
   ```

### Step 3: Access Your API

Your API is now available at:
```
http://YOUR_PUBLIC_IP/
```

Test it:
```
http://YOUR_PUBLIC_IP/health
http://YOUR_PUBLIC_IP/docs
```

---

## Cost Optimization: Spot Instances

Spot instances can save ~70% but may be interrupted (rare).

1. When launching, click **Advanced Details**
2. **Purchasing option**: Select "Spot Instance"
3. **Spot instance type**: "Persistent"
4. **Interruption behavior**: "Stop"

Estimated costs (us-east-1):
- t3.medium spot: ~$7-10/mo
- t3.large spot: ~$15-20/mo

---

## Manual Deployment (Alternative)

If you prefer to set up manually instead of using user-data:

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ec2-user@YOUR_PUBLIC_IP

# Install Docker
sudo yum update -y
sudo yum install -y docker git
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone and deploy
sudo mkdir -p /opt/ocean
cd /opt/ocean
sudo git clone https://github.com/Mousasan97/OCEAN_Firasa.git .

# Create .env file
sudo nano .env
# Add:
# ENVIRONMENT=production
# MODEL_DEVICE=cpu
# AI_PROVIDER=gemini
# GOOGLE_API_KEY=your-key-here

# Build and run
sudo docker-compose -f docker-compose.prod.yml up -d --build

# Check logs
sudo docker logs ocean-api -f
```

---

## Useful Commands

```bash
# Check status
sudo docker ps

# View logs
sudo docker logs ocean-api -f

# Restart
cd /opt/ocean
sudo docker-compose -f docker-compose.prod.yml restart

# Update to latest code
cd /opt/ocean
sudo git pull
sudo docker-compose -f docker-compose.prod.yml up -d --build

# Stop
sudo docker-compose -f docker-compose.prod.yml down
```

---

## Optional: Add HTTPS with Let's Encrypt

1. Get a domain and point it to your EC2 IP

2. Install Certbot:
   ```bash
   sudo yum install -y certbot
   ```

3. Get certificate:
   ```bash
   sudo certbot certonly --standalone -d yourdomain.com
   ```

4. Update docker-compose to use nginx with SSL (advanced)

---

## Security Checklist

- [ ] Restrict SSH access to your IP only
- [ ] Don't commit API keys to git
- [ ] Use AWS Secrets Manager for production
- [ ] Enable CloudWatch monitoring
- [ ] Set up billing alerts

---

## Troubleshooting

**Container not starting?**
```bash
sudo docker logs ocean-api
```

**Out of memory?**
- Upgrade to t3.large (8GB RAM)
- Or reduce MODEL_WARMUP=false

**Slow predictions?**
- Expected: ~1-2 sec/frame on t3.medium
- Upgrade to t3.xlarge for faster
