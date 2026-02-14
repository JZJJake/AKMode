import argparse
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.fetch_history import fetch_history
from src.train_hybrid import HybridTrainer
from src.inference import InferenceEngine

def main():
    parser = argparse.ArgumentParser(description="AKMode Full System Orchestrator")
    parser.add_argument("--mode", type=str, required=True, choices=["fetch", "train", "inference"], help="Operation Mode")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y%m%d"), help="Target Date (YYYYMMDD)")
    parser.add_argument("--start", type=str, help="Start Date for Fetch (YYYYMMDD)")
    parser.add_argument("--end", type=str, help="End Date for Fetch (YYYYMMDD)")

    args = parser.parse_args()

    if args.mode == "fetch":
        if not args.start or not args.end:
            print("Error: --start and --end are required for fetch mode.")
            sys.exit(1)
        print(f"Starting History Fetch: {args.start} to {args.end}")
        fetch_history(args.start, args.end)

    elif args.mode == "train":
        print("Starting Hybrid Model Retraining...")
        # Assume mock data or real data exists in data/processed
        # We might need to regenerate tensors if new data arrived.
        # Ideally, call FeatureEngine first.
        from src.feature_engine import FeatureEngine
        print("Running Feature Engineering...")
        fe = FeatureEngine()
        fe.process_all()

        print("Training Models...")
        trainer = HybridTrainer()
        trainer.load_data()
        trainer.train_nn(epochs=5)
        trainer.extract_embeddings()
        trainer.train_lgbm()
        print("Training Complete.")

    elif args.mode == "inference":
        print(f"Running Daily Inference for {args.date}...")
        engine = InferenceEngine()
        engine.run_daily_inference(args.date)

if __name__ == "__main__":
    main()
