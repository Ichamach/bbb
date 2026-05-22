#!/bin/bash
# NeuroViral Lab — Complete Backend Startup Script
# Starts all four modules in separate terminal sessions (requires tmux or screen)
# Usage: bash start_all_backends.sh

set -e

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORK_DIR"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  NeuroViral Lab — Backend Startup                         ║"
echo "║  Starting all 4 computation engines + static server       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"
echo ""

# Check required modules
echo "Checking dependencies..."
python3 -c "import fastapi; print('✓ FastAPI installed')" 2>/dev/null || { echo "❌ FastAPI not found. Run: pip install fastapi uvicorn[standard]"; exit 1; }
python3 -c "import numpy; print('✓ NumPy installed')" 2>/dev/null || { echo "❌ NumPy not found. Run: pip install numpy"; exit 1; }
echo ""

# Create a file to track running processes
PIDS_FILE="/tmp/neuroviral_pids.txt"
> "$PIDS_FILE"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  Shutting down all backends...                            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    if [ -f "$PIDS_FILE" ]; then
        while read pid; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
                echo "✓ Killed process $pid"
            fi
        done < "$PIDS_FILE"
        rm "$PIDS_FILE"
    fi
}

trap cleanup EXIT

# ─── BACKEND 1: BBB Permeability (Port 8000) ──────────────────
echo "Starting Module 1: BBB Permeability Simulator (port 8000)..."
python3 -m uvicorn main:app --reload --port 8000 > /tmp/bbb_8000.log 2>&1 &
BBB_PID=$!
echo "$BBB_PID" >> "$PIDS_FILE"
echo "✓ BBB engine started (PID: $BBB_PID)"
sleep 2

# Health check
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Port 8000: BBB engine responding"
else
    echo "⚠️  Port 8000: Not responding yet (startup may be slow)"
fi
echo ""

# ─── BACKEND 2: RABV Neuroinvasion (Port 8001) ────────────────
echo "Starting Module 2: RABV Neuroinvasion Simulator (port 8001)..."
python3 -m uvicorn rabv_main:app --reload --port 8001 > /tmp/rabv_8001.log 2>&1 &
RABV_PID=$!
echo "$RABV_PID" >> "$PIDS_FILE"
echo "✓ RABV engine started (PID: $RABV_PID)"
sleep 2

if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ Port 8001: RABV engine responding"
else
    echo "⚠️  Port 8001: Not responding yet (startup may be slow)"
fi
echo ""

# ─── BACKEND 3: Split-Protein Designer (Port 8002) ────────────
echo "Starting Module 3: Split-Protein Designer (port 8002)..."
python3 -m uvicorn split_main:app --reload --port 8002 > /tmp/split_8002.log 2>&1 &
SPLIT_PID=$!
echo "$SPLIT_PID" >> "$PIDS_FILE"
echo "✓ Split-Protein Designer engine started (PID: $SPLIT_PID)"
sleep 2

if curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo "✅ Port 8002: Split-Protein Designer engine responding"
else
    echo "⚠️  Port 8002: Not responding yet (startup may be slow)"
fi
echo ""

# ─── BACKEND 4: Protein Splitter (Port 8003) ──────────────────
echo "Starting Module 4: Protein Splitter (port 8003)..."
python3 -m uvicorn splitter_main:app --reload --port 8003 > /tmp/splitter_8003.log 2>&1 &
SPLITTER_PID=$!
echo "$SPLITTER_PID" >> "$PIDS_FILE"
echo "✓ Protein Splitter engine started (PID: $SPLITTER_PID)"
sleep 2

if curl -s http://localhost:8003/health > /dev/null 2>&1; then
    echo "✅ Port 8003: Protein Splitter engine responding"
else
    echo "⚠️  Port 8003: Not responding yet (startup may be slow)"
fi
echo ""

# ─── STATIC SERVER (Port 3000) ─────────────────────────────────
echo "Starting static file server (port 3000)..."
python3 -m http.server 3000 > /tmp/http_3000.log 2>&1 &
HTTP_PID=$!
echo "$HTTP_PID" >> "$PIDS_FILE"
echo "✓ HTTP server started (PID: $HTTP_PID)"
sleep 1

if curl -s http://localhost:3000/split_protein_simulator.html > /dev/null 2>&1; then
    echo "✅ Port 3000: Static server responding"
else
    echo "⚠️  Port 3000: Not responding yet"
fi
echo ""

# ─── WAIT AND DISPLAY STATUS ──────────────────────────────────
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  All backends started. Health check in 3 seconds...        ║"
echo "╚════════════════════════════════════════════════════════════╝"
sleep 3

echo ""
echo "HEALTH CHECK RESULTS:"
echo "──────────────────────────────────────────────────────────────"

# BBB health
if curl -s http://localhost:8000/health | grep -q '"status":"ok"'; then
    echo "✅ Port 8000 (BBB Permeability)        — HEALTHY"
else
    echo "❌ Port 8000 (BBB Permeability)        — NOT RESPONDING"
    echo "   → Check log: tail -f /tmp/bbb_8000.log"
fi

# RABV health
if curl -s http://localhost:8001/health | grep -q '"status":"ok"'; then
    echo "✅ Port 8001 (RABV Neuroinvasion)      — HEALTHY"
else
    echo "❌ Port 8001 (RABV Neuroinvasion)      — NOT RESPONDING"
    echo "   → Check log: tail -f /tmp/rabv_8001.log"
fi

# Split-Protein Designer health
if curl -s http://localhost:8002/health | grep -q '"status":"ok"'; then
    echo "✅ Port 8002 (Split-Protein Designer)  — HEALTHY"
else
    echo "❌ Port 8002 (Split-Protein Designer)  — NOT RESPONDING"
    echo "   → Check log: tail -f /tmp/split_8002.log"
fi

# Protein Splitter health
if curl -s http://localhost:8003/health | grep -q '"status":"ok"'; then
    echo "✅ Port 8003 (Protein Splitter)        — HEALTHY"
else
    echo "❌ Port 8003 (Protein Splitter)        — NOT RESPONDING"
    echo "   → Check log: tail -f /tmp/splitter_8003.log"
fi

# HTTP server health
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Port 3000 (Static HTTP Server)      — HEALTHY"
else
    echo "❌ Port 3000 (Static HTTP Server)      — NOT RESPONDING"
    echo "   → Check log: tail -f /tmp/http_3000.log"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ READY TO USE                                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Access simulators:"
echo "  • BBB Permeability:       http://localhost:3000/bbb_simulator.html"
echo "  • RABV Neuroinvasion:     http://localhost:3000/rabies_simulator.html"
echo "  • Split-Protein Designer: http://localhost:3000/split_protein_simulator.html"
echo "  • Protein Splitter:       http://localhost:3000/protein_splitter.html"
echo ""
echo "Monitor logs:"
echo "  • tail -f /tmp/bbb_8000.log     (BBB engine)"
echo "  • tail -f /tmp/rabv_8001.log    (RABV engine)"
echo "  • tail -f /tmp/split_8002.log   (Split-Protein Designer engine)"
echo "  • tail -f /tmp/splitter_8003.log (Protein Splitter engine)"
echo "  • tail -f /tmp/http_3000.log    (Static server)"
echo ""
echo "To stop, press Ctrl+C"
echo ""

# Keep script running
wait
