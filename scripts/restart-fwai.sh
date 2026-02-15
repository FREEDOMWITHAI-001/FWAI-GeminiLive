#!/bin/bash
# Graceful restart: stop accepting new calls, drain active ones, then restart
# Crontab: 0 */8 * * * /opt/fwai/FWAI-GeminiLive/scripts/restart-fwai.sh >> /var/log/fwai-restart.log 2>&1

DRAIN_WAIT=60
APP_DIR="/opt/fwai/FWAI-GeminiLive"

echo "$(date) - Starting graceful restart..."

# Set maintenance flag - app returns 503 to new calls
touch "$APP_DIR/.maintenance"
echo "$(date) - Maintenance flag set, rejecting new calls"

# Wait for active calls to finish
echo "$(date) - Waiting ${DRAIN_WAIT}s for active calls to drain..."
sleep $DRAIN_WAIT

# Restart (use sudo for manual runs; root crontab won't need it)
sudo systemctl restart fwai-app
sleep 5

# Remove flag
rm -f "$APP_DIR/.maintenance"
echo "$(date) - Restart complete, accepting calls"
