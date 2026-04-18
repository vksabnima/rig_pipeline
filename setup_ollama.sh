#!/bin/bash
# ============================================================================
# Ollama Setup Script for RIG Pipeline
# Checks for Ollama installation, pulls Llama 3.2, and tests inference.
# ============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "============================================"
echo "  RIG Pipeline — Ollama Setup"
echo "============================================"
echo ""

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
    echo -e "${GREEN}[OK]${NC} Ollama is installed (${OLLAMA_VERSION})"
    echo ""

    # Check if Ollama server is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}[OK]${NC} Ollama server is running"
    else
        echo -e "${YELLOW}[INFO]${NC} Starting Ollama server..."
        ollama serve &
        sleep 3
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo -e "${GREEN}[OK]${NC} Ollama server started"
        else
            echo -e "${RED}[FAIL]${NC} Could not start Ollama server"
            echo "  Try running 'ollama serve' manually in another terminal"
            exit 1
        fi
    fi

    echo ""

    # Pull Llama 3.2 model (matches the model tag used at inference time)
    echo "Pulling llama3.2 model (this may take a while on first run)..."
    if ollama pull llama3.2; then
        echo -e "${GREEN}[OK]${NC} llama3.2 model pulled successfully"
    else
        echo -e "${RED}[FAIL]${NC} Failed to pull llama3.2"
        exit 1
    fi

    echo ""

    # Test inference
    echo "Testing llama3.2 inference..."
    RESPONSE=$(curl -s http://localhost:11434/api/generate \
        -d '{"model": "llama3.2", "prompt": "Respond with only: OK", "stream": false}' \
        2>/dev/null)

    if echo "$RESPONSE" | grep -q "response"; then
        echo -e "${GREEN}[OK]${NC} llama3.2 inference test passed"
        echo ""
        echo "============================================"
        echo -e "  ${GREEN}Ollama is ready!${NC}"
        echo "  Set OLLAMA_AVAILABLE=true in your .env file"
        echo "============================================"
    else
        echo -e "${RED}[FAIL]${NC} llama3.2 inference test failed"
        echo "  Response: $RESPONSE"
        exit 1
    fi

else
    echo -e "${YELLOW}[NOT FOUND]${NC} Ollama is not installed"
    echo ""
    echo "To install Ollama:"
    echo ""
    echo "  Linux/WSL:  curl -fsSL https://ollama.com/install.sh | sh"
    echo "  macOS:      brew install ollama"
    echo "  Windows:    Download from https://ollama.com/download"
    echo ""
    echo "After installation, run this script again."
    echo ""
    echo "NOTE: Ollama/Llama 3.2 is OPTIONAL. The RIG pipeline works"
    echo "      fine with just OpenAI + Anthropic API keys."
    echo "      Set OLLAMA_AVAILABLE=false in your .env file."
    echo ""
    exit 0
fi
