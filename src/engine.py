import torch
import torch.nn as nn

def loss_fn(outputs, targets):
    #平均二乗誤差を返す
    return nn.MSELoss()(outputs, targets.view(-1, 1))

def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    scaler = torch.cuda.amp.GradScaler()

    for d in data_loader:
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            loss = loss_fn(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []

    with torch.no_grad():
        for d in data_loader:
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            targets = targets.cpu().detach()
            fin_targets.extend(targets.numpy().tolist())

            outputs = outputs.reshape(-1).cpu().detach()
            fin_outputs.extend(outputs.numpy().tolist())
    return fin_outputs, fin_targets