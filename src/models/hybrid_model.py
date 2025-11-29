import torch
import torch.nn as nn
from transformers import AutoModel


class HybridModel(nn.Module):
    def __init__(
            self,
            phobert_model_name='vinai/phobert-base',
            cnn_out_channels=128,
            lstm_hidden_size=128,
            lstm_layers=1,
            num_classes=2,
            dropout=0.3
    ):
        """
        Hybrid model for Vietnamese counterfeit review classification.

        Args:
            phobert_model_name: PhoBERT model identifier
            cnn_out_channels: Number of output channels for each CNN layer
            lstm_hidden_size: Hidden size for BiLSTM
            lstm_layers: Number of BiLSTM layers
            num_classes: Number of output classes (2: Spam/Normal)
            dropout: Dropout rate
        """
        super(HybridModel, self).__init__()

        # PhoBERT for embeddings
        self.phobert = AutoModel.from_pretrained(phobert_model_name)
        self.embedding_dim = self.phobert.config.hidden_size  # 768 for base model

        # 1D Convolutional layers with different kernel sizes
        # Use 'same' padding by calculating explicit padding values
        self.conv3 = nn.Conv1d(
            in_channels=self.embedding_dim,
            out_channels=cnn_out_channels,
            kernel_size=3,
            padding=1  # (kernel_size - 1) // 2 = (3-1)//2 = 1
        )
        self.conv4 = nn.Conv1d(
            in_channels=self.embedding_dim,
            out_channels=cnn_out_channels,
            kernel_size=4,
            padding=1  # For kernel_size=4, padding=1 gives output_size = input_size - 2
        )
        self.conv5 = nn.Conv1d(
            in_channels=self.embedding_dim,
            out_channels=cnn_out_channels,
            kernel_size=5,
            padding=2  # (kernel_size - 1) // 2 = (5-1)//2 = 2
        )

        # Activation and normalization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Batch normalization for CNN outputs
        self.bn_cnn = nn.BatchNorm1d(cnn_out_channels * 3)

        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=cnn_out_channels * 3,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Batch normalization for LSTM output
        self.bn_lstm = nn.BatchNorm1d(lstm_hidden_size * 2)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids: Token IDs from PhoBERT tokenizer [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Get PhoBERT embeddings
        # outputs.last_hidden_state: [batch_size, seq_len, embedding_dim]
        phobert_output = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        embeddings = phobert_output.last_hidden_state

        # Transpose for Conv1d: [batch_size, embedding_dim, seq_len]
        embeddings = embeddings.transpose(1, 2)

        # Apply 3 parallel CNN layers with different kernel sizes
        conv3_out = self.relu(self.conv3(embeddings))  # [batch_size, cnn_out, seq_len]
        conv4_out = self.relu(self.conv4(embeddings))  # [batch_size, cnn_out, seq_len-2]
        conv5_out = self.relu(self.conv5(embeddings))  # [batch_size, cnn_out, seq_len]

        # Pad conv4_out to match the sequence length of conv3_out and conv5_out
        if conv4_out.size(2) != conv3_out.size(2):
            pad_amount = conv3_out.size(2) - conv4_out.size(2)
            conv4_out = torch.nn.functional.pad(conv4_out, (0, pad_amount), mode='constant', value=0)

        # Concatenate CNN outputs along channel dimension
        # [batch_size, cnn_out*3, seq_len]
        cnn_concat = torch.cat([conv3_out, conv4_out, conv5_out], dim=1)

        # Apply batch normalization and dropout
        cnn_concat = self.bn_cnn(cnn_concat)
        cnn_concat = self.dropout(cnn_concat)

        # Transpose back for LSTM: [batch_size, seq_len, cnn_out*3]
        cnn_concat = cnn_concat.transpose(1, 2)

        # BiLSTM layer
        # lstm_out: [batch_size, seq_len, lstm_hidden_size*2]
        lstm_out, (hidden, cell) = self.bilstm(cnn_concat)

        # Use the last timestep output
        # We take the last valid output based on attention mask
        # For simplicity, we'll use the output at the last position
        last_lstm_out = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_size*2]

        # Apply batch normalization
        last_lstm_out = self.bn_lstm(last_lstm_out)
        last_lstm_out = self.dropout(last_lstm_out)

        # Classification
        logits = self.classifier(last_lstm_out)  # [batch_size, num_classes]

        return logits

