import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
import os
import joblib

from src.model_defs import TimeSeriesNet
from app_config.settings import DATA_DIR

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

class HybridTrainer:
    """
    Manages the Hybrid Stacking Pipeline.
    Stage 1: Train Neural Network (TimeSeriesNet)
    Stage 2: Train LightGBM Classifier using NN Embeddings + Tabular Features
    """

    def __init__(self, data_path=os.path.join(DATA_DIR, "processed", "train_data.npz")):
        self.data_path = data_path
        self.model = None
        self.lgb_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load_data(self):
        """
        Loads training data and prepares tensors/features.
        """
        print(f"Loading data from {self.data_path}")
        data = np.load(self.data_path, allow_pickle=True)
        self.X_raw = data['X'] # (N, 6, 5)
        self.y = data['y']

        # Generate Dummy Tabular Features (N, 2)
        # Feature 1: Market Cap (Random log-normal)
        # Feature 2: Turnover (Random uniform)
        n_samples = self.X_raw.shape[0]
        self.tabular_features = np.random.randn(n_samples, 2).astype(np.float32)

        # Split Train/Val
        self.X_train, self.X_val, self.y_train, self.y_val, self.tab_train, self.tab_val = train_test_split(
            self.X_raw, self.y, self.tabular_features, test_size=0.2, random_state=42
        )

    def train_nn(self, epochs=5):
        """
        Stage 1: Train Neural Network.
        """
        print("\n--- Stage 1: Training Neural Network ---")
        self.model = TimeSeriesNet(input_size=5).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.tensor(self.X_train),
            torch.tensor(self.y_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(self.X_val),
            torch.tensor(self.y_val, dtype=torch.float32)
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {val_loss/len(val_loader):.4f}")

        # Save Model
        torch.save(self.model.state_dict(), os.path.join(MODELS_DIR, "stage1_nn.pth"))
        print("NN Model Saved.")

    def extract_embeddings(self):
        """
        Extracts embeddings for Train and Val sets using the trained NN.
        """
        print("\n--- Extracting Embeddings ---")
        self.model.eval()

        X_train_tensor = torch.tensor(self.X_train).to(self.device)
        X_val_tensor = torch.tensor(self.X_val).to(self.device)

        with torch.no_grad():
            self.train_embeddings = self.model.get_embedding(X_train_tensor).cpu().numpy()
            self.val_embeddings = self.model.get_embedding(X_val_tensor).cpu().numpy()

        print(f"Train Embeddings Shape: {self.train_embeddings.shape}")
        print(f"Val Embeddings Shape: {self.val_embeddings.shape}")

    def train_lgbm(self):
        """
        Stage 2: Train LightGBM Classifier.
        Inputs: Concatenated [Embeddings (32) + Tabular (2)]
        """
        print("\n--- Stage 2: Training LightGBM ---")

        # Concatenate Features
        X_train_final = np.hstack([self.train_embeddings, self.tab_train])
        X_val_final = np.hstack([self.val_embeddings, self.tab_val])

        print(f"Final Feature Shape (Train): {X_train_final.shape}")

        # Initialize LGBM
        # Use scale_pos_weight for class imbalance
        pos_count = sum(self.y_train)
        if pos_count > 0:
            pos_weight = (len(self.y_train) - pos_count) / pos_count
        else:
            pos_weight = 1.0

        print(f"Using scale_pos_weight: {pos_weight:.2f}")

        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            scale_pos_weight=pos_weight,
            random_state=42,
            verbosity=-1
        )

        self.lgb_model.fit(
            X_train_final,
            self.y_train,
            eval_set=[(X_val_final, self.y_val)],
            eval_metric='auc'
        )

        # Evaluate
        y_pred_prob = self.lgb_model.predict_proba(X_val_final)[:, 1]
        y_pred = self.lgb_model.predict(X_val_final)

        try:
            auc = roc_auc_score(self.y_val, y_pred_prob)
        except:
            auc = 0.0 # Handle case with one class only

        acc = accuracy_score(self.y_val, y_pred)

        print(f"\nFinal Validation AUC: {auc:.4f}")
        print(f"Final Validation Accuracy: {acc:.4f}")

        # Save Model
        joblib.dump(self.lgb_model, os.path.join(MODELS_DIR, "stage2_lgb.pkl"))
        print("LGBM Model Saved.")

if __name__ == "__main__":
    trainer = HybridTrainer()
    trainer.load_data()
    trainer.train_nn()
    trainer.extract_embeddings()
    trainer.train_lgbm()
