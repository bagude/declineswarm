#!/bin/bash
# Kill all Claude Code (node) instances except the current session

echo "=== Claude Code Instance Killer ==="
echo ""

# Find claude-related processes
PIDS=$(ps aux | grep -i '[c]laude' | grep -v "grep" | grep -v "$$" | awk '{print $1}')

if [ -z "$PIDS" ]; then
    echo "No Claude Code processes found."
    exit 0
fi

echo "Found Claude Code processes:"
ps aux | grep -i '[c]laude' | grep -v "grep"
echo ""

read -p "Kill all listed processes? (y/N): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

for PID in $PIDS; do
    kill "$PID" 2>/dev/null && echo "Killed PID $PID" || echo "Failed to kill PID $PID"
done

sleep 1

# Verify
REMAINING=$(ps aux | grep -i '[c]laude' | grep -v "grep")
if [ -z "$REMAINING" ]; then
    echo ""
    echo "All Claude Code processes terminated successfully."
else
    echo ""
    echo "Some processes still running:"
    echo "$REMAINING"
    read -p "Force kill (SIGKILL)? (y/N): " FORCE
    if [[ "$FORCE" == "y" || "$FORCE" == "Y" ]]; then
        ps aux | grep -i '[c]laude' | grep -v "grep" | awk '{print $1}' | xargs kill -9 2>/dev/null
        echo "Force killed remaining processes."
    fi
fi
