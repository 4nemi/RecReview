import config
import dataset
import engine
import torch
import pandas as pd
import numpy as np
import torch.nn as nn

from model import BERTModel
from sklearn import model_selection
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import wandb

wandb.init(
    project="RecReview",
    config={
        "participant": config.PARTICIPANT,
        "test_name": config.TEST_NAME,
        "learning_rate": config.ENCODER_LR,
        "epochs": config.EPOCHS,
        "batch_size": config.TRAIN_BATCH_SIZE,
        "dropout": 0.3,
        "target_normalize": False,
    }
)

def train():
    dfx = pd.read_csv(config.TRAIN_FILE)

    df_train, df_valid = dfx[dfx.test == -1], dfx[dfx.test == 1]
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.BERTDataset(
        review=df_train.review.values,
        target=df_train.target.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=2,
        pin_memory=True,
    )

    valid_dataset = dataset.BERTDataset(
        review=df_valid.review.values,
        target=df_valid.target.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2,
        pin_memory=True,
    )

    device = torch.device("cuda")
    model = BERTModel()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer 
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer 
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(
        len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS
    )

    optimizer = AdamW(optimizer_parameters, lr=config.ENCODER_LR)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
    )

    best_rmse = np.inf
    for epoch in range(config.EPOCHS):
        avg_train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        rmse = np.sqrt(nn.MSELoss()(torch.tensor(outputs).float(), torch.tensor(targets).float()))
        print(f"RMSE Score = {rmse}")
        wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "rmse": rmse})
        if rmse < best_rmse:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_rmse = rmse

if __name__ == "__main__":
    train()