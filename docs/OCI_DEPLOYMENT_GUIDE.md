# Oracle Cloud Infrastructure (OCI) Deployment Guide

This guide covers deploying the FWAI WebRTC Gemini application on Oracle Cloud's Always Free Tier.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Cost Overview](#cost-overview)
- [Step 1: Create Oracle Cloud Account](#step-1-create-oracle-cloud-account)
- [Step 2: Create Virtual Cloud Network (VCN)](#step-2-create-virtual-cloud-network-vcn)
- [Step 3: Configure Security Rules](#step-3-configure-security-rules)
- [Step 4: Create Compute Instance](#step-4-create-compute-instance)
- [Step 5: Connect via SSH](#step-5-connect-via-ssh)
- [Step 6: Server Setup](#step-6-server-setup)
- [Step 7: Deploy Application](#step-7-deploy-application)
- [Step 8: Configure Systemd Service](#step-8-configure-systemd-service)
- [Step 9: Update Webhook URLs](#step-9-update-webhook-urls)
- [Maintenance Commands](#maintenance-commands)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Oracle Cloud account (free tier available)
- SSH key pair (public/private)
- Git repository with your application code
- API keys (Google Gemini, Plivo/WhatsApp)

---

## Cost Overview

Oracle Cloud Always Free Tier includes:

| Resource | Free Allocation | Monthly Cost |
|----------|-----------------|--------------|
| AMD Compute (VM.Standard.E2.1.Micro) | 2 instances, 1GB RAM each | $0 |
| ARM Compute (VM.Standard.A1.Flex) | 4 OCPUs, 24GB RAM total | $0 |
| Block Storage | 200 GB total | $0 |
| Outbound Data Transfer | 10 TB/month | $0 |
| Public IP | Included | $0 |

**Total: $0/month** (Always Free)

---

## Step 1: Create Oracle Cloud Account

1. Go to [Oracle Cloud Free Tier](https://www.oracle.com/cloud/free/)
2. Click "Start for free"
3. Fill in your details and verify email
4. Add payment method (required but won't be charged for free tier)
5. Select home region (choose closest to your users):
   - India: `ap-mumbai-1`, `ap-hyderabad-1`, `ap-chennai-1`
   - US: `us-ashburn-1`, `us-phoenix-1`
   - Europe: `eu-frankfurt-1`, `uk-london-1`

---

## Step 2: Create Virtual Cloud Network (VCN)

### Using VCN Wizard (Recommended)

1. Navigate to: `Menu (☰) → Networking → Virtual Cloud Networks`
2. Click **"Start VCN Wizard"**
3. Select **"Create VCN with Internet Connectivity"**
4. Click **"Start VCN Wizard"**
5. Fill in:
   ```
   VCN Name: fwai-vcn
   Compartment: (your compartment)
   VCN CIDR Block: 10.0.0.0/16 (default)
   Public Subnet CIDR: 10.0.0.0/24 (default)
   Private Subnet CIDR: 10.0.1.0/24 (default)
   ```
6. Click **"Next"** → **"Create"**

This automatically creates:
- VCN with Internet Gateway
- Public Subnet (for your app)
- Private Subnet (for future databases)
- Route Tables (correctly configured)
- Security Lists

---

## Step 3: Configure Security Rules

### Add Ingress Rules

1. Navigate to: `VCN → Security Lists → Default Security List`
2. Click **"Add Ingress Rules"**
3. Add the following rules:

| Stateless | Source CIDR | Protocol | Port Range | Description |
|-----------|-------------|----------|------------|-------------|
| No | 0.0.0.0/0 | TCP | 22 | SSH |
| No | 0.0.0.0/0 | TCP | 80 | HTTP |
| No | 0.0.0.0/0 | TCP | 443 | HTTPS |
| No | 0.0.0.0/0 | TCP | 3000 | FastAPI App |
| No | 0.0.0.0/0 | TCP | 8003 | Gemini Live Service |
| No | 0.0.0.0/0 | UDP | 10000-20000 | WebRTC Media |

**Note:** Keep "Stateless" unchecked (OFF) for all rules.

---

## Step 4: Create Compute Instance

1. Navigate to: `Menu (☰) → Compute → Instances`
2. Click **"Create Instance"**
3. Configure:

### Basic Details
```
Name: fwai-webrtc
Compartment: (your compartment)
```

### Image and Shape
```
Image: Ubuntu 22.04 (Canonical)
Shape: VM.Standard.E2.1.Micro (AMD, Always Free)
   - 1 OCPU
   - 1 GB Memory

Alternative (if available):
Shape: VM.Standard.A1.Flex (ARM, Always Free)
   - 1-4 OCPUs
   - 6-24 GB Memory
```

### Networking
```
VCN: fwai-vcn
Subnet: Public Subnet-fwai-vcn
Public IPv4 address: Assign a public IPv4 address ✓ (REQUIRED)
```

### SSH Keys
- Upload your public key OR
- Generate a new key pair and download the private key

### Boot Volume
```
Size: 50 GB (default, free)
Performance: Balanced
```

4. Click **"Create"**
5. Wait for instance status: **RUNNING**
6. Note the **Public IP Address**

---

## Step 5: Connect via SSH

### Set Key Permissions
```bash
chmod 400 /path/to/your-private-key.key
```

### Connect
```bash
ssh -i /path/to/your-private-key.key ubuntu@<PUBLIC_IP>
```

### Troubleshooting SSH
If connection hangs:
1. Verify Security List has port 22 rule
2. Check instance is RUNNING
3. Use verbose mode: `ssh -v -i key.key ubuntu@IP`

---

## Step 6: Server Setup

### Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### Install Dependencies
```bash
sudo apt install -y python3.10 python3.10-venv python3-pip git
sudo apt install -y libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config libsrtp2-dev ffmpeg
sudo apt install -y libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libswresample-dev
```

### Create Swap File (Important for 1GB RAM instances)
```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Verify swap:
```bash
free -h
```

### Configure Ubuntu Firewall
```bash
sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 3000 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 8003 -j ACCEPT
sudo iptables -I INPUT -p udp --dport 10000:20000 -j ACCEPT
sudo apt install -y iptables-persistent
sudo netfilter-persistent save
```

---

## Step 7: Deploy Application

### Create App Directory
```bash
sudo mkdir -p /opt/fwai
sudo chown ubuntu:ubuntu /opt/fwai
cd /opt/fwai
```

### Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/FWAI-GeminiLive.git
cd FWAI-GeminiLive
```

### Setup Python Environment
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### Install Dependencies
```bash
pip install aiortc
pip install fastapi uvicorn httpx websockets numpy python-dotenv loguru python-multipart google-generativeai "google-genai>=1.0.0" pydantic scipy
```

### Create Environment File
```bash
nano .env
```

Add your configuration:
```env
# Call Provider
CALL_PROVIDER=plivo

# Server Configuration
HOST=0.0.0.0
PORT=3000
DEBUG=false

# Google Gemini API
GOOGLE_API_KEY=your_gemini_api_key

# Plivo Configuration
PLIVO_AUTH_ID=your_plivo_auth_id
PLIVO_AUTH_TOKEN=your_plivo_auth_token
PLIVO_PHONE_NUMBER=+91xxxxxxxxxx
PLIVO_CALLBACK_URL=http://<YOUR_PUBLIC_IP>:3000

# WhatsApp Configuration (if using)
PHONE_NUMBER_ID=your_phone_number_id
META_ACCESS_TOKEN=your_meta_access_token
META_APP_ID=your_app_id
META_APP_SECRET=your_app_secret
META_VERIFY_TOKEN=your_verify_token

# Gemini Live Service
GEMINI_LIVE_HOST=localhost
GEMINI_LIVE_PORT=8003
TTS_VOICE=Puck

# Logging
LOG_LEVEL=INFO
ENABLE_TRANSCRIPTS=true
ENABLE_DETAILED_LOGGING=true
```

### Test Run
```bash
python run.py
```

Test in browser: `http://<PUBLIC_IP>:3000`

Expected response:
```json
{"status":"ok","service":"WhatsApp Voice Calling with Gemini Live","version":"1.0.0"}
```

---

## Step 8: Configure Systemd Service

### Create Service File
```bash
sudo nano /etc/systemd/system/fwai-app.service
```

Paste:
```ini
[Unit]
Description=FWAI WebRTC Gemini App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/fwai/FWAI-GeminiLive
Environment=PATH=/opt/fwai/FWAI-GeminiLive/venv/bin
ExecStart=/opt/fwai/FWAI-GeminiLive/venv/bin/python run.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Enable and Start
```bash
sudo systemctl daemon-reload
sudo systemctl enable fwai-app
sudo systemctl start fwai-app
```

### Verify
```bash
sudo systemctl status fwai-app
```

---

## Step 9: Update Webhook URLs

### Plivo Console
1. Go to Plivo Console → Applications
2. Update Answer URL: `http://<PUBLIC_IP>:3000/plivo/answer`
3. Update Hangup URL: `http://<PUBLIC_IP>:3000/plivo/hangup`

### Meta/WhatsApp (if using)
1. Go to Meta Developer Console
2. Update Webhook URL: `http://<PUBLIC_IP>:3000/webhook`
3. Verify webhook with your META_VERIFY_TOKEN

---

## Maintenance Commands

| Task | Command |
|------|---------|
| Check app status | `sudo systemctl status fwai-app` |
| Restart app | `sudo systemctl restart fwai-app` |
| Stop app | `sudo systemctl stop fwai-app` |
| View live logs | `sudo journalctl -u fwai-app -f` |
| View recent logs | `sudo journalctl -u fwai-app -n 100` |
| Check memory | `free -h` |
| Monitor resources | `htop` |
| Check disk space | `df -h` |

### Update Application
```bash
cd /opt/fwai/FWAI-GeminiLive
git pull origin main
sudo systemctl restart fwai-app
```

---

## Troubleshooting

### SSH Connection Timeout
1. Check Security List has port 22 ingress rule
2. Verify instance is RUNNING
3. Check correct username (`ubuntu` for Ubuntu images)
4. Verify key permissions: `chmod 400 key.key`

### App Not Accessible from Browser
1. Check app is running: `sudo systemctl status fwai-app`
2. Check Ubuntu firewall: `sudo iptables -L INPUT -n | grep 3000`
3. Check Security List has port 3000 rule
4. Test locally: `curl http://localhost:3000`

### Out of Memory
1. Check swap is enabled: `free -h`
2. If no swap, create it (see Step 6)
3. Consider upgrading to ARM instance with more RAM

### Package Installation Errors
1. For `av` package errors: `pip install av --only-binary=:all:`
2. For dependency conflicts: `pip install <package> --no-deps`

### Instance Out of Capacity (ARM)
ARM instances are popular. Try:
1. Different Availability Domain (if available)
2. Different time (early morning, weekends)
3. Use AMD E2.1.Micro instead (always available)
4. Different region

---

## Security Recommendations

1. **Use HTTPS**: Set up Nginx reverse proxy with Let's Encrypt SSL
2. **Restrict SSH**: Change Security List SSH rule to your IP only
3. **Environment Variables**: Never commit `.env` to git
4. **Regular Updates**: `sudo apt update && sudo apt upgrade`
5. **Monitor Logs**: Check for suspicious activity regularly

---

## Next Steps

- [Production Architecture](./PRODUCTION_ARCHITECTURE.md)
- Set up HTTPS with Nginx and Let's Encrypt
- Configure monitoring and alerts
- Set up automated backups
