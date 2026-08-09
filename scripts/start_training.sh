#!/bin/bash
# Modified by NextoCR contributors; see NOTICE for attribution.
# Start the Java bridge server, run PPO training, and clean up on exit.
#
# Usage:
#   ./scripts/start_training.sh
#   ./scripts/start_training.sh --timesteps 100000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

# Multi-process ZMQ training launches its own servers, while JPype needs none.
NEEDS_SINGLE_SERVER=1
EXPECT_NUM_ENVS=0
for ARG in "$@"; do
    if [ "$EXPECT_NUM_ENVS" -eq 1 ]; then
        if [ "$ARG" -gt 1 ] 2>/dev/null; then
            NEEDS_SINGLE_SERVER=0
        fi
        EXPECT_NUM_ENVS=0
        continue
    fi
    case "$ARG" in
        --jpype)
            NEEDS_SINGLE_SERVER=0
            ;;
        --num-envs)
            EXPECT_NUM_ENVS=1
            ;;
        --num-envs=*)
            NUM_ENVS="${ARG#*=}"
            if [ "$NUM_ENVS" -gt 1 ] 2>/dev/null; then
                NEEDS_SINGLE_SERVER=0
            fi
            ;;
    esac
done

if [ "$NEEDS_SINGLE_SERVER" -eq 0 ]; then
    echo "Training mode manages its own bridge process(es)."
    exec python3 python/examples/train_ppo.py "$@"
fi

# Build the Java server
echo "Building Java bridge server..."
./gradlew :gym-bridge:installDist --quiet

# Start the Java server in the background
echo "Starting bridge server on port 9876..."
./gym-bridge/build/install/gym-bridge/bin/gym-bridge &
SERVER_PID=$!

# Clean up server on exit
cleanup() {
    echo ""
    echo "Stopping bridge server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# Wait for server to start accepting connections
echo "Waiting for server to start..."
for i in $(seq 1 30); do
    if python3 -c "
import zmq, json, sys
ctx = zmq.Context()
s = ctx.socket(zmq.PAIR)
s.setsockopt(zmq.RCVTIMEO, 1000)
s.setsockopt(zmq.SNDTIMEO, 1000)
s.connect('tcp://localhost:9876')
try:
    s.send_string(json.dumps({'type':'init','data':{'blueDeck':['knight','archer','fireball','arrows','giant','musketeer','minions','valkyrie'],'redDeck':['knight','archer','fireball','arrows','giant','musketeer','minions','valkyrie'],'level':11,'ticksPerStep':6}}))
    r = s.recv_string()
    s.send_string(json.dumps({'type':'close'}))
    s.recv_string()
except: pass
s.close()
ctx.term()
sys.exit(0 if 'init_ok' in r else 1)
" 2>/dev/null; then
        echo "Server is ready."
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Error: Server failed to start after 30 seconds"
        exit 1
    fi
    sleep 1
done

# Run training
echo ""
echo "Starting PPO training..."
python3 python/examples/train_ppo.py "$@"
