from src.train_hybrid import HybridTrainer
import sys

def run_phase3():
    print("Starting Phase 3: Hybrid Model Training...")
    try:
        trainer = HybridTrainer()
        trainer.load_data()

        # Train Stage 1: NN
        trainer.train_nn(epochs=5)

        # Extract Embeddings
        trainer.extract_embeddings()

        # Train Stage 2: LightGBM
        trainer.train_lgbm()

        print("\nPhase 3 Completed Successfully!")
    except Exception as e:
        print(f"\nPhase 3 Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_phase3()
