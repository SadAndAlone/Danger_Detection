"""
Model CNN + LSTM do klasyfikacji sekwencji klatek wideo (bójka, palenie, pożar, brak zagrożenia).
"""
import torch
import torch.nn as nn
from .config import IMG_SIZE, NUM_CLASSES, SEQ_LEN

# Po 3 warstwach Conv2d + MaxPool(2): 64 -> 32 -> 16 -> 8
FEATURE_SIZE = 8
CNN_OUTPUT_CHANNELS = 128
CNN_FLAT = CNN_OUTPUT_CHANNELS * FEATURE_SIZE * FEATURE_SIZE  # 8192
LSTM_INPUT_SIZE = 256
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 1


class CNNFeatureExtractor(nn.Module):
    """CNN wyciągający wektor cech z pojedynczej klatki (bez warstwy klasyfikacyjnej)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, CNN_OUTPUT_CHANNELS, 3, padding=1),
            nn.BatchNorm2d(CNN_OUTPUT_CHANNELS),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CNN_FLAT, LSTM_INPUT_SIZE),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, LSTM_INPUT_SIZE)
        x = self.features(x)
        x = self.fc(x)
        return x


class CNNLSTM(nn.Module):
    """
    Sekwencja klatek -> CNN (cechy na klatkę) -> LSTM -> klasyfikator.
    Wejście: (batch, seq_len, C, H, W)
    Wyjście: (batch, num_classes) logity
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        cnn_input_size: tuple = (IMG_SIZE, IMG_SIZE),
        lstm_hidden: int = LSTM_HIDDEN_SIZE,
        lstm_layers: int = LSTM_NUM_LAYERS,
    ):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.lstm = nn.LSTM(
            input_size=LSTM_INPUT_SIZE,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0 if lstm_layers == 1 else 0.2,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.cnn(x)  # (B*T, LSTM_INPUT_SIZE)
        features = features.view(B, T, -1)  # (B, T, LSTM_INPUT_SIZE)
        lstm_out, (h_n, _) = self.lstm(features)
        # Użyj ostatniego wyjścia LSTM
        out = lstm_out[:, -1, :]  # (B, lstm_hidden)
        logits = self.classifier(out)
        return logits
