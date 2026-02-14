import torch
import torch.nn as nn

class TimeSeriesNet(nn.Module):
    def __init__(self, input_size=5):
        super(TimeSeriesNet, self).__init__()

        # 1. Permute + 1D-CNN
        # Input: (Batch, Time=6, Channels=5)
        # Conv1d expects (Batch, Channels, Time) -> (B, 5, 6)
        self.conv1 = nn.Conv1d(
            in_channels=input_size,  # 5
            out_channels=32,         # 32 filters
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)

        # 2. GRU
        # Input: (Batch, Time, Channels) if batch_first=True
        # Conv output (B, 32, 6) -> Permute -> (B, 6, 32)
        # GRU input_size = 32 (from Conv output channels)
        # hidden_size = 64
        self.gru = nn.GRU(
            input_size=32,
            hidden_size=64,
            batch_first=True
        )

        # 3. Embedding Head
        # Projects 64-dim GRU state to 32-dim Embedding
        self.embedding_head = nn.Linear(64, 32)

        # 4. Auxiliary Classifier Head
        # Projects 32-dim Embedding to 1-dim Logit -> Sigmoid
        self.classifier_head = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Standard forward pass for training. Returns probability.
        x: (Batch, 6, 5)
        """
        emb = self.get_embedding(x)
        logits = self.classifier_head(emb)
        prob = self.sigmoid(logits)
        return prob

    def get_embedding(self, x):
        """
        Returns the 32-dimensional embedding vector.
        x: (Batch, 6, 5)
        """
        # Permute for Conv1d: (B, 6, 5) -> (B, 5, 6)
        x = x.permute(0, 2, 1)

        # CNN Block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        # Permute for GRU: (B, 32, 6) -> (B, 6, 32)
        x = x.permute(0, 2, 1)

        # GRU Block
        # output, h_n = gru(input)
        # h_n shape: (num_layers, batch, hidden_size) -> (1, B, 64)
        _, h_n = self.gru(x)

        # Extract last hidden state
        last_hidden = h_n[-1] # (B, 64)

        # Project to Embedding Space
        embedding = self.embedding_head(last_hidden) # (B, 32)

        return embedding
