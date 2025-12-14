#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

echo "========================================"
echo "   Poptart Gal Friday V2 - VM Setup"
echo "========================================"

# 1. System Updates & Dependencies
echo ">>> [1/6] Updating System & Installing Dependencies..."
sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv screen build-essential

# 2. Python Virtual Environment
echo ">>> [2/6] Setting up Virtual Environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "    Created 'venv'."
else
    echo "    'venv' already exists."
fi

# Activate venv
source venv/bin/activate

# 3. Python Packages
echo ">>> [3/6] Installing Python Packages..."
pip install --upgrade pip wheel

# Install pandas-ta from development branch (Fixes Pandas 2.0+ compatibility)
echo "    Installing pandas-ta (Dev)..."
pip install git+https://github.com/twopirllc/pandas-ta.git@development

# Install other requirements
echo "    Installing requirements.txt..."
pip install -r requirements.txt

# 4. Configuration
echo ">>> [4/6] Checking Configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "    Created .env from .env.example."
        echo "    ⚠️  IMPORTANT: You must edit .env and add your API keys!"
    else
        echo "    ⚠️  Warning: .env.example not found. Please create .env manually."
    fi
else
    echo "    .env already exists."
fi

# 5. Data Download
echo ">>> [5/6] Downloading Market Data (Binance Vision)..."
# Ensure data directory exists
mkdir -p data
python -m src.tools.download_binance_data

# 6. Model Training
echo ">>> [6/6] Training Models..."

# XRP (Uses both TP/SL and Regime)
echo "    [XRP] Training TP/SL Predictors..."
python -m src.tools.train_models --pair XRPUSDT

echo "    [XRP] Training Regime Classifier..."
python -m src.tools.train_regime_classifier --pair XRPUSDT

# SOL (Uses Regime only)
echo "    [SOL] Training Regime Classifier..."
python -m src.tools.train_regime_classifier --pair SOLUSDT

# BTC (DCA Mode - No ML needed)
echo "    [BTC] Skipping ML training (DCA Mode)."

echo "========================================"
echo "   ✅ Setup Complete!"
echo "========================================"
echo "Next Steps:"
echo "1. Edit your .env file: nano .env"
echo "2. Run WFO for SOL: screen -S wfo_sol python -m src.tools.wfo_optimizer --pair SOLUSDT"
echo "3. Run WFO for XRP: screen -S wfo_xrp python -m src.tools.wfo_optimizer --pair XRPUSDT"
echo ""
