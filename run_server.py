#!/usr/bin/env python3
"""
Simple server launcher
"""
import uvicorn
from pathlib import Path
from src.utils.config import settings

if __name__ == "__main__":
    # Check if VAT model exists
    model_path = Path(settings.MODEL_CHECKPOINT_PATH)
    if not model_path.exists():
        print("="*70)
        print("WARNING: VAT model checkpoint not found!")
        print(f"   Expected: {model_path}")
        print("="*70)
        print("\nPlease make sure your model checkpoint is in the correct location.")
        print("Update .env file with MODEL_CHECKPOINT_PATH if it's elsewhere.\n")
        exit(1)

    print("="*70)
    print("Starting OCEAN Personality API Server (VAT Model)")
    print("="*70)
    print(f"\n* Model found: {model_path}")
    print(f"* Video-only mode: 32 frames @ 224x224")
    print(f"* Port: {settings.PORT}")
    print(f"* Configuration: .env")
    print("\nStarting server...\n")

    # Run server
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info"
    )
