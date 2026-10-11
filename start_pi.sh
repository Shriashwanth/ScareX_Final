#!/usr/bin/env bash
# ==========================================================
# ScareX - Raspberry Pi 5 Standalone Autonomous Startup
# Starts local AI inference, camera pipeline, hardware control,
# SQLite database, and Web Dashboard on port 5000.
# No laptop required!
# ==========================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "          Starting ScareX Raspberry Pi 5 Standalone       "
echo "=========================================================="

# Release port 5000 if occupied
fuser -k 5000/tcp > /dev/null 2>&1

# Export display if running headless
export DISPLAY=:0

# Get local IP address for easy phone access
IP_ADDR=$(hostname -I | cut -d' ' -f1)

# Activate virtual environment if present (Debian Bookworm PEP 668 compliance)
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "Virtual environment activated: $SCRIPT_DIR/venv"
elif [ -f "$SCRIPT_DIR/../venv/bin/activate" ]; then
    source "$SCRIPT_DIR/../venv/bin/activate"
    echo "Virtual environment activated: $SCRIPT_DIR/../venv"
elif [ -f "$HOME/scarex_env/bin/activate" ]; then
    source "$HOME/scarex_env/bin/activate"
    echo "Virtual environment activated: $HOME/scarex_env"
fi

echo "Local IP Address: http://${IP_ADDR}:5000"
echo "mDNS URL:         http://raspberrypi.local:5000"
echo "Starting master orchestrator..."
echo "=========================================================="

python3 -m raspberry_pi.main
