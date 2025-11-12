#!/bin/bash
# Start server script with port conflict checking

PORT=8000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "  Office Map Server - Startup Check"
echo "=========================================="
echo ""

# Check if port is in use
echo "Checking port $PORT..."
CONFLICTS=$(lsof -i :$PORT 2>/dev/null | grep LISTEN)

if [ -n "$CONFLICTS" ]; then
    echo "WARNING: Port $PORT is already in use!"
    echo ""
    echo "Processes using port $PORT:"
    echo "$CONFLICTS" | while read line; do
        echo "  $line"
    done
    echo ""
    echo "Options:"
    echo "  1. Kill conflicting processes (will prompt for each)"
    echo "  2. Use a different port (set PORT environment variable)"
    echo "  3. Exit and handle manually"
    echo ""
    read -p "Choose option (1/2/3): " choice
    
    case $choice in
        1)
            echo "$CONFLICTS" | awk '{print $2}' | sort -u | while read pid; do
                if [ "$pid" != "PID" ]; then
                    process_info=$(ps -p $pid -o comm= 2>/dev/null)
                    if [ -n "$process_info" ]; then
                        read -p "Kill process $pid ($process_info)? (y/N): " kill_choice
                        if [[ $kill_choice =~ ^[Yy]$ ]]; then
                            kill $pid 2>/dev/null
                            echo "  Killed process $pid"
                        fi
                    fi
                fi
            done
            # Check again
            sleep 1
            REMAINING=$(lsof -i :$PORT 2>/dev/null | grep LISTEN)
            if [ -n "$REMAINING" ]; then
                echo ""
                echo "ERROR: Port $PORT is still in use. Please handle manually."
                exit 1
            fi
            ;;
        2)
            read -p "Enter new port number: " new_port
            PORT=$new_port
            export PORT
            ;;
        3)
            echo "Exiting. Please handle port conflicts manually."
            exit 1
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
else
    echo "Port $PORT is available"
fi

echo ""
echo "=========================================="
echo "  Starting server on port $PORT"
echo "=========================================="
echo ""

# Change to server directory (parent of scripts/)
cd "$SCRIPT_DIR/../server"

# Start the server
if [ -n "$PORT" ] && [ "$PORT" != "8000" ]; then
    PORT=$PORT python3 main.py
else
    python3 main.py
fi

