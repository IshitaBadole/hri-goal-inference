#!/bin/bash

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: ./run_clustering.sh <participant_id> <world> <scenario_no> [options]"
    echo ""
    echo "Example: ./run_clustering.sh p1 apartment.wbt 1"
    echo ""
    echo "Optional parameters:"
    echo "  --eps <value>         DBSCAN epsilon parameter (default: 0.15)"
    echo "  --min-samples <value> DBSCAN min_samples parameter (default: 10)"
    echo "  --start-row <value>   Offset to start trajectory analysis (default: 3500)"
    echo "  --step <value>        Sampling frequency of trajectory (default: 10)"
    echo "  --image-size <w> <h>  Image resizing dimensions (default: 200 200)"
    echo ""
    echo "Available worlds:"
    echo "  - apartment.wbt"
    echo "  - break_room.wbt"
    echo "  - factory.wbt"
    echo "  - hall.wbt"
    echo "  - my_world.wbt"
    exit 1
fi

PARTICIPANT_ID=$1
WORLD=$2
SCENARIO_NO=$3
shift 3  # Remove first 3 arguments, keep the rest for optional parameters

# Extract world name without extension
WORLD_NAME="${WORLD%.wbt}"

# Construct log file path
LOG_FILE="$HOME/webots_server/pedestrian_logs/${PARTICIPANT_ID}_${WORLD_NAME}_s${SCENARIO_NO}.csv"

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "ERROR: Log file not found: $LOG_FILE"
    echo ""
    echo "Make sure you have run the experiment first with:"
    echo "  ./run_experiment.sh $PARTICIPANT_ID $WORLD $SCENARIO_NO"
    exit 1
fi

# Map world to corresponding image file
case "$WORLD_NAME" in
    apartment)
        IMAGE_FILE="media/.apartment_cropped.png"
        ;;
    break_room)
        IMAGE_FILE="media/.break_room_cropped.png"
        ;;
    factory)
        IMAGE_FILE="media/.factory_cropped.png"
        ;;
    hall)
        IMAGE_FILE="media/.hall_cropped.png"
        ;;
    my_world)
        IMAGE_FILE="media/.my_world_cropped.png"
        ;;
    *)
        echo "WARNING: Unknown world '$WORLD_NAME', using apartment image as default"
        IMAGE_FILE="media/.apartment_cropped.png"
        ;;
esac

# Check if image file exists
if [ ! -f "$IMAGE_FILE" ]; then
    echo "ERROR: World image file not found: $IMAGE_FILE"
    echo ""
    echo "Please ensure the corresponding world image exists in the media directory."
    exit 1
fi

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "ERROR: Python virtual environment not found."
    echo "Please run ./setup.sh first to create the environment."
    exit 1
fi

echo "=========================================="
echo "Running Clustering Analysis"
echo "=========================================="
echo "Participant: $PARTICIPANT_ID"
echo "World: $WORLD_NAME"
echo "Scenario: $SCENARIO_NO"
echo "Log file: $LOG_FILE"
echo "World image: $IMAGE_FILE"
echo ""

source venv/bin/activate

# Run clustering with optional parameters
python3 notebooks/find_goals.py "$LOG_FILE" "$IMAGE_FILE" "$@"

CLUSTERING_EXIT_CODE=$?

deactivate

if [ $CLUSTERING_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Clustering analysis complete!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "ERROR: Clustering analysis failed"
    echo "=========================================="
    exit 1
fi
