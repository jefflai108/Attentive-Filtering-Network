import torch
import numpy as np
import logging
import os
import torch.nn.functional as F

## Get the same logger from main"
logger = logging.getLogger("anti-spoofing")

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (_, X1, X2, target) in enumerate(train_loader):
        X1, X2, target = X1.to(device), X2.to(device), target.to(device)
        target = target.view(-1,1).float()
        optimizer.zero_grad()
        #output, hidden = model(data, hidden=None)
        y = model(X1, X2)
        loss = F.binary_cross_entropy(y, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(X1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def snapshot(dir_path, run_name, is_best, state):
    snapshot_file = os.path.join(dir_path,
                    run_name + '-model_best.pth')
    if is_best:
        torch.save(state, snapshot_file)
        logger.info("Snapshot saved to {}\n".format(snapshot_file))
