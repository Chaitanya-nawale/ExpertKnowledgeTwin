#!/bin/bash

PROJECT_DIR="$HOME/ExpertDigitalTwin/embeddingModel"
VENV_DIR="$HOME/ExpertDigitalTwin/.venv"
APP_PORT=${APP_PORT:-16022}
PID_FILE="$PROJECT_DIR/server.pid"
HF_TOKEN=${HF_TOKEN:-""}

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

start() {
    echo -e "${GREEN}Setting up...${NC}"
    mkdir -p "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    
    # Create virtual environment if needed
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${GREEN}Creating venv...${NC}"
        python3 -m venv "$VENV_DIR"
        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip > /dev/null 2>&1
        pip install -r "$PROJECT_DIR/requirements.txt"
    fi
    
    source "$VENV_DIR/bin/activate"
    
    # Stop existing server
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        kill $OLD_PID 2>/dev/null || true
        sleep 1
    fi
    
    # Start server
    echo -e "${GREEN}Starting server on port $APP_PORT...${NC}"
    nohup uvicorn app:app --host 0.0.0.0 --port $APP_PORT > /dev/null 2>&1 &
    
    SERVER_PID=$!
    echo $SERVER_PID > "$PID_FILE"
    sleep 2
    
    echo -e "${GREEN}Server started (PID: $SERVER_PID)${NC}"
    echo -e "${GREEN}Test: curl -X POST http://localhost:$APP_PORT/embed -H 'Content-Type: application/json' -d '{\"texts\": [\"hello\", \"world\"]}'${NC}"
}

stop() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        kill $PID 2>/dev/null && echo -e "${GREEN}Server stopped${NC}" || echo -e "${RED}Server not running${NC}"
        rm "$PID_FILE"
    fi
}

status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 $PID 2>/dev/null; then
            echo -e "${GREEN}Server running (PID: $PID)${NC}"
        else
            echo -e "${RED}Server not running${NC}"
        fi
    else
        echo -e "${RED}Server not running${NC}"
    fi
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        sleep 1
        start
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        ;;
esac
