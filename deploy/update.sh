#!/usr/bin/env bash
# deploy/update.sh — pull latest code and restart ppb-mcp on the server.
#
# Run this on the server after pushing a new commit to origin/main:
#   ssh user@mcp.poorpaul.dev
#   sudo /opt/ppb-mcp/deploy/update.sh
#
# What it does:
#   1. Pulls latest code from git
#   2. Rebuilds the Docker image
#   3. Restarts the container (zero-downtime via docker compose up -d)
#   4. Reloads nginx if the config changed

set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/opt/ppb-mcp}"

log() { printf '\033[1;34m[update]\033[0m %s\n' "$*"; }

log "Pulling latest code..."
git -C "$INSTALL_DIR" pull --ff-only

log "Rebuilding Docker image..."
docker compose -f "$INSTALL_DIR/docker-compose.yml" build --pull

log "Restarting service (zero-downtime)..."
docker compose -f "$INSTALL_DIR/docker-compose.yml" up -d

log "Checking if nginx config changed..."
if ! diff -q "$INSTALL_DIR/deploy/nginx-ppb-mcp.conf" /etc/nginx/sites-available/ppb-mcp.conf &>/dev/null; then
    log "Updating nginx config..."
    cp "$INSTALL_DIR/deploy/nginx-ppb-mcp.conf" /etc/nginx/sites-available/ppb-mcp.conf
    nginx -t && systemctl reload nginx
    log "nginx reloaded."
else
    log "nginx config unchanged."
fi

log "Done. Waiting for health check..."
for i in $(seq 1 12); do
    if curl -fsS "http://0.0.0.0:9933/health" >/dev/null 2>&1; then
        log "Server is healthy."
        exit 0
    fi
    sleep 5
done
log "WARNING: health check did not pass within 60 s — check 'docker compose logs'."
exit 1
