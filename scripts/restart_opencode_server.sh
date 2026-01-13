#!/bin/bash
# Restart the OpenCode server to pick up the latest code changes

echo "Finding OpenCode server processes on port 4096..."
SERVER_PID=$(lsof -ti :4096)

if [ -n "$SERVER_PID" ]; then
    echo "Found server process: $SERVER_PID"
    echo "Stopping server..."
    kill $SERVER_PID
    sleep 2

    # Force kill if still running
    if ps -p $SERVER_PID > /dev/null 2>&1; then
        echo "Server didn't stop gracefully, force killing..."
        kill -9 $SERVER_PID
    fi
    echo "Server stopped."
else
    echo "No server found on port 4096"
fi

echo ""
echo "To start the server again with updated code, run:"
echo "  cd /home/phil65/dev/oss/agentpool"
echo "  agentpool ui desktop"
echo ""
echo "Or if you want to run just the server (without desktop app):"
echo "  agentpool serve-opencode --port 4096"
