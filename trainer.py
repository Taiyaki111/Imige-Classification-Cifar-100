import sys, tqdm
from tqdm import tqdm
import math
import utils

def trainer(trainloader, model, device, optimizer, epoch, criterion, args):
    print("---------- Start Training ----------")
    model.train()
    try:
        with tqdm(enumerate(trainloader), total=len(trainloader), ncols=100) as pbar:
            train_loss = 0.0
            for i, batch in pbar:
                images, labels = batch
                utils.adjust_learning_rate(optimizer, i/len(trainloader)+epoch,
                        args.epochs,
                        args.lr,
                        args.min_lr,
                        args.warmup)
                images = images.to(device)
                labels = labels.to(device)

                outputs =  model(images)

                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss
        return train_loss
    except ValueError:
        pass

def validater(validloader, model, device, criterion):
    print("---------- Start Validating ----------")
    model.eval()
    try:
        with tqdm(enumerate(validloader), total=len(validloader), ncols=100) as pbar:
            valid_loss = 0.0
            for i, batch in pbar:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                if not math.isfinite(loss):
                    print("Loss is {}, stopping training".format(loss))
                    print(loss)
                    sys.exit(1)

                valid_loss += loss
        return valid_loss
    except ValueError:
        pass

if __name__ == '__main__':
    pass