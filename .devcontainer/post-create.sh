#!/bin/bash
set -e

echo "==> Restoring NuGet packages..."
dotnet restore

echo "==> Building solution..."
dotnet build --no-restore

MODEL_DIR="samples/BasicUsage/models"
MODEL_PATH="$MODEL_DIR/model.onnx"

if [ ! -f "$MODEL_PATH" ]; then
    echo "==> Downloading all-MiniLM-L6-v2 ONNX model (~86MB)..."
    mkdir -p "$MODEL_DIR"
    curl -L -o "$MODEL_PATH" \
        "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
    echo "==> Model downloaded to $MODEL_PATH"
else
    echo "==> Model already exists at $MODEL_PATH, skipping download."
fi

# Copy the model to samples that use the same all-MiniLM-L6-v2 model
for SAMPLE_DIR in samples/ComposablePoolingComparison samples/IntermediateInspection samples/MeaiProviderAgnostic; do
    if [ ! -f "$SAMPLE_DIR/models/model.onnx" ]; then
        echo "==> Copying model to $SAMPLE_DIR..."
        mkdir -p "$SAMPLE_DIR/models"
        cp "$MODEL_PATH" "$SAMPLE_DIR/models/model.onnx"
        # Copy vocab if it exists in BasicUsage
        [ -f "$MODEL_DIR/vocab.txt" ] && cp "$MODEL_DIR/vocab.txt" "$SAMPLE_DIR/models/vocab.txt"
    fi
done

echo ""
echo "Ready! Run the basic sample with:"
echo "  cd samples/BasicUsage && dotnet run"
echo ""
echo "Modular pipeline samples (use same model, already set up):"
echo "  cd samples/ComposablePoolingComparison && dotnet run"
echo "  cd samples/IntermediateInspection && dotnet run"
echo "  cd samples/MeaiProviderAgnostic && dotnet run"
echo ""
echo "Additional samples (BGE, E5, GTE) require downloading their models first:"
echo "  See each sample's README for download instructions:"
echo "    samples/BgeSmallEmbedding/README.md"
echo "    samples/E5SmallEmbedding/README.md"
echo "    samples/GteSmallEmbedding/README.md"
