import torch
from torch import nn, optim
from torchmetrics import CharErrorRate

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

MAX_EPOCHS = 20
LOSS = nn.CrossEntropyLoss()

cnn = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),  
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),  
    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2), 
    nn.Conv2d(256, 512, kernel_size=(2, 3), stride=1, padding=1),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 3),  
    nn.Conv2d(512, 512, kernel_size=(2, 4), stride=1, padding=0),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),
)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        rec_output, _ = self.rnn(input)
        batch_size, seq_len, hidden_size = rec_output.size()
        output = self.embedding(rec_output)
        return output


class CRNN(nn.Module):
    def __init__(self, channels_num, class_num, hidden_size):
        super().__init__()
        self.cnn = cnn
        self.lstm = nn.Sequential(
            BiLSTM(512, hidden_size, hidden_size),
            BiLSTM(hidden_size, hidden_size, class_num)
        )

    def forward(self, input):
        conv = self.cnn(input) 
        conv = conv.squeeze(2) 
        conv = conv.permute(0, 2, 1)  
        output = self.lstm(conv)
        return output


class LCRNN(pl.LightningModule):
    def __init__(self, crnn, criterion, encoder):
        super().__init__()
        self.crnn = crnn
        self.criterion = criterion
        self.encoder = encoder

    def training_step(self, batch, batch_idx):
        images, labels, text_labels = batch
        preds = self.predict(images)
        loss = self.calculate_loss(preds, labels)
        self.log(f"train_loss", loss) 
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, text_labels = batch
        preds = self.predict(images)
        loss = self.calculate_loss(preds, labels)
        cer = self.calculate_cer(preds, text_labels)
        self.log("val_loss", loss)
        self.log("val_CER", cer, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels, text_labels = batch
        preds = self.predict(images)
        loss = self.calculate_loss(preds, labels)
        cer = self.calculate_cer(preds, text_labels)
        self.log("test_loss", loss)
        self.log("test_CER", cer, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, labels, text_labels = batch
        preds = self.predict(images)
        text_preds = self.convert_to_text(preds)
        return text_preds

    def calculate_loss(self, preds, labels):
        loss = self.criterion(preds.permute(0, 2, 1), labels)
        return loss

    def predict(self, images):
        preds = self.crnn(images)
        return preds

    def calculate_cer(self, preds, text_labels):
        text_preds = self.convert_to_text(preds)
        cer = CharErrorRate()(text_preds, text_labels)
        return cer

    def convert_to_text(self, preds):
        ids = torch.argmax(preds, dim=2)
        text_preds = [self.encoder.decode_str(x) for x in ids]
        return text_preds

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-4)
        return optimizer


trainer = pl.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=3, min_delta=0.001)], accelerator='gpu', max_epochs=MAX_EPOCHS)