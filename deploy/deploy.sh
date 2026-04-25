#!/usr/bin/env bash
# deploy/deploy.sh — install ppb-mcp on a fresh Ubuntu 22.04 / 24.04 host.
# Run as a sudoer; this script uses sudo where needed.
#
# What it does:
#   1. Installs Docker, docker-compose-plugin, nginx, certbot
#   2. Clones the repo to /opt/ppb-mcp (or pulls if present)
#   3. Builds the Docker image, runs via docker compose
#   4. Installs systemd unit and starts it on boot
#   5. Installs nginx site and (optionally) provisions TLS via certbot

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/paulplee/ppb-mcp.git}"
INSTALL_DIR="${INSTALL_DIR:-/opt/ppb-mcp}"
DOMAIN="${DOMAIN:-mcp.poorpaul.dev}"
EMAIL="${EMAIL:-paul@poorpaul.dev}"

log() { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }

log "Installing system packages..."
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release nginx certbot python3-certbot-nginx git

if ! command -v docker >/dev/null 2>&1; then
    log "Installing Docker Engine..."
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

log "Cloning / updating repo at $INSTALL_DIR"
if [ -d "$INSTALL_DIR/.git" ]; then
    sudo git -C "$INSTALL_DIR" pull --ff-only
else
    sudo git clone "$REPO_URL" "$INSTALL_DIR"
fi

log "Building image..."
cd "$INSTALL_DIR"
sudo docker compose build

log "Installing systemd unit..."
sudo cp "$INSTALL_DIR/deploy/ppb-mcp.service" /etc/systemd/system/ppb-mcp.service
sudo systemctl daemon-reload
sudo systemctl enable --now ppb-mcp.service

log "Installing nginx site..."
sudo cp "$INSTALL_DIR/deploy/nginx-ppb-mcp.conf" /etc/nginx/sites-available/ppb-mcp.conf
sudo ln -sf /etc/nginx/sites-available/ppb-mcp.conf /etc/nginx/sites-enabled/ppb-mcp.conf
sudo nginx -t
sudo systemctl reload nginx

if [ "${SKIP_CERTBOT:-0}" != "1" ]; then
    log "Provisioning TLS for $DOMAIN via certbot..."
    sudo certbot --nginx --non-interactive --agree-tos -m "$EMAIL" -d "$DOMAIN" --redirect || \
        log "certbot failed; you can re-run manually: sudo certbot --nginx -d $DOMAIN"
fi

log "Done. Health: curl -s https://$DOMAIN/health"
