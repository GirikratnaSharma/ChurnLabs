#!/bin/bash

# ChurnLabs Setup Script

echo "🚀 Setting up ChurnLabs..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Initialize database
echo "🗄️  Initializing database..."
python scripts/init_db.py

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created. Please update it with your configuration."
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p ml/models ml/data mlruns

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Train a model: python ml/training/train.py"
echo "3. Start API: uvicorn app.main:app --reload"
echo "4. Start Dashboard: streamlit run dashboard/app.py"
