#!/usr/bin/env bash
# Source from Dream launchers so artifact and result names identify the decoder.
export DREAM_ALG="${DREAM_ALG:-entropy}"
export DREAM_TEMPERATURE="${DREAM_TEMPERATURE:-0.2}"
export DREAM_TOP_P="${DREAM_TOP_P:-0.95}"
export DREAM_STEPS="${DREAM_STEPS:-256}"
export DREAM_SEED="${DREAM_SEED:-0}"
export DREAM_DECODER_TAG="sparse_v1_${DREAM_ALG}_t${DREAM_TEMPERATURE}_p${DREAM_TOP_P}_s${DREAM_STEPS}_seed${DREAM_SEED}"
DREAM_ARGS=(--dream-alg "$DREAM_ALG" --dream-temperature "$DREAM_TEMPERATURE"
            --dream-top-p "$DREAM_TOP_P" --dream-steps "$DREAM_STEPS")
