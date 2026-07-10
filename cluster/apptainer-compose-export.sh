#!/bin/bash

PROJECT_ROOT="$PWD/../mozaik-models/experanto"
SIF_IMAGE="${SIF_IMAGE:-$PWD/../mozaik-sif/mozaik-opt.sif}"
MOZAIK_ROOT="$PWD"
EXPERANTO_ROOT="$PWD/../../experanto"
DATA_ROOT="/mnt/vast-react/projects/neural_foundation_model"

echo PROJECT_ROOT: $PROJECT_ROOT
echo SIF_IMAGE: $SIF_IMAGE
echo MOZAIK_ROOT: $MOZAIK_ROOT
echo EXPERANTO_ROOT: $EXPERANTO_ROOT
echo DATA_ROOT: $DATA_ROOT

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${OMP_NUM_THREADS:-4}
export VECLIB_MAXIMUM_THREADS=${OMP_NUM_THREADS:-4}

echo "Starting export container for trial ${TRIAL}..."
apptainer exec \
 --cleanenv \
 --env PYTHONPATH="/mozaik:$PYTHONPATH" \
 --env TRIAL="$TRIAL" \
 --env N_CHUNKS="${N_CHUNKS:-12}" \
 --env CHUNK_START="${CHUNK_START:-}" \
 --env CHUNK_END="${CHUNK_END:-}" \
 --env EXPORT_MODE="${EXPORT_MODE:-}" \
 --env CHUNK_DIR="${CHUNK_DIR:-/data/mozaik_chunk}" \
 --env OUTPUT_PREFIX="${OUTPUT_PREFIX:-}" \
 --env MODALITY_FILTER="${MODALITY_FILTER:-}" \
 --env BATCH_SIZE="${BATCH_SIZE:-4}" \
 --env DATASTORE_PREFIX="${DATASTORE_PREFIX:-}" \
 --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
 --env MKL_NUM_THREADS=$MKL_NUM_THREADS \
 --env OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS \
 --env NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS \
 --bind "$PROJECT_ROOT:/project" \
 --bind "$MOZAIK_ROOT:/mozaik" \
 --bind "$EXPERANTO_ROOT:/experanto" \
 --bind "$DATA_ROOT:/data" \
 "$SIF_IMAGE" \
 bash cluster/runners/mozaik-export-array.sh
