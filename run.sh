#!/bin/bash

echo "============================================"
echo "   Fake News Detection System Launcher"
echo "============================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "[INFO] Python found!"
python3 --version
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[INFO] Virtual environment not found. Creating..."
    python3 -m venv venv
    echo "[SUCCESS] Virtual environment created!"
    echo
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

echo "[SUCCESS] Virtual environment activated!"
echo

# Check if requirements are installed
if ! python -c "import streamlit" &> /dev/null; then
    echo "[INFO] Installing requirements..."
    pip install -r requirements.txt
    echo "[SUCCESS] Requirements installed!"
    echo
fi

# Show menu
while true; do
    echo "============================================"
    echo "   What would you like to do?"
    echo "============================================"
    echo
    echo "   1. Train Models (First time only)"
    echo "   2. Run Web Application"
    echo "   3. Install/Update Requirements"
    echo "   4. Exit"
    echo
    read -p "Enter your choice (1-4): " choice

    case $choice in
        1)
            echo
            echo "[INFO] Training models..."
            echo "This may take 2-5 minutes..."
            echo
            python train.py
            if [ $? -ne 0 ]; then
                echo
                echo "[ERROR] Training failed!"
            else
                echo
                echo "[SUCCESS] Training completed!"
            fi
            echo
            read -p "Press Enter to continue..."
            ;;
        2)
            echo
            echo "[INFO] Starting web application..."
            echo "The app will open in your browser at http://localhost:8501"
            echo
            echo "Press Ctrl+C to stop the application"
            echo
            streamlit run app.py
            ;;
        3)
            echo
            echo "[INFO] Installing/Updating requirements..."
            pip install --upgrade -r requirements.txt
            echo "[SUCCESS] Requirements updated!"
            echo
            read -p "Press Enter to continue..."
            ;;
        4)
            echo
            echo "[INFO] Deactivating virtual environment..."
            deactivate
            echo "[SUCCESS] Goodbye!"
            echo
            exit 0
            ;;
        *)
            echo "[ERROR] Invalid choice. Please try again."
            echo
            ;;
    esac
done
