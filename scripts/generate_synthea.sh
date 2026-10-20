#!/usr/bin/env bash
# Generate a synthetic adult population with Synthea (CSV export only).
#
# Usage: bash scripts/generate_synthea.sh [population] [seed] [output_dir]
# Example: bash scripts/generate_synthea.sh 1000 42 data/raw/synthea_s42
set -euo pipefail

POPULATION="${1:-1000}"
SEED="${2:-42}"
OUT_DIR="${3:-data/raw/synthea_s${SEED}}"
SYNTHEA_VERSION="v4.0.0"
JAR="data/raw/synthea-with-dependencies.jar"
URL="https://github.com/synthetichealth/synthea/releases/download/${SYNTHEA_VERSION}/synthea-with-dependencies.jar"

mkdir -p data/raw
if ! command -v java >/dev/null 2>&1; then
  echo "Java 17+ is required (Ubuntu/Colab: apt-get install -y openjdk-17-jdk-headless)." >&2
  exit 1
fi
if [ ! -f "$JAR" ]; then
  echo "Downloading Synthea ${SYNTHEA_VERSION}..."
  wget -q -O "$JAR" "$URL"
fi

echo "Generating ${POPULATION} patients aged 40-85 (seed ${SEED}) into ${OUT_DIR} ..."
java -jar "$JAR" \
  -s "$SEED" -cs "$SEED" -p "$POPULATION" -a 40-85 \
  --exporter.fhir.export=false \
  --exporter.csv.export=true \
  --exporter.baseDirectory="$OUT_DIR" \
  Massachusetts

echo "Done. CSV files are in ${OUT_DIR}/csv"
