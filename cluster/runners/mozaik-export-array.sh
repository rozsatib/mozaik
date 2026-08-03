#!/bin/bash

cd /project

TRIAL=${TRIAL:-1}
N_CHUNKS=${N_CHUNKS:-12}
CHUNK_START=${CHUNK_START:-}
CHUNK_END=${CHUNK_END:-}
EXPORT_MODE=${EXPORT_MODE:-}       # "", "--screen-only", or "--spikes-only"
MODALITY_FILTER=${MODALITY_FILTER:-}
BATCH_SIZE=${BATCH_SIZE:-4}

echo "--- Starting Export ---"
echo "Trial: $TRIAL"
echo "N_CHUNKS: $N_CHUNKS"
echo "CHUNK_START: ${CHUNK_START:-0}"
echo "CHUNK_END: ${CHUNK_END:-$N_CHUNKS}"
echo "EXPORT_MODE: ${EXPORT_MODE:-both}"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

CMD="python -u export.py $TRIAL --n-chunks $N_CHUNKS --batch-size $BATCH_SIZE"

[ -n "$CHUNK_START" ]    && CMD="$CMD --chunk-start $CHUNK_START"
[ -n "$CHUNK_END" ]      && CMD="$CMD --chunk-end $CHUNK_END"
[ -n "$EXPORT_MODE" ]    && CMD="$CMD $EXPORT_MODE"
[ -n "$MODALITY_FILTER" ] && CMD="$CMD --modality-filter $MODALITY_FILTER"

echo "Running: $CMD"
$CMD
