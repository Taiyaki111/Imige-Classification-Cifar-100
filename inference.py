import argparse
from model import SimpleCNN
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import datasets
import numpy as np
import torch
from tqdm import tqdm
import timm
import torchvision.models as models

def main(args):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device =  'cpu'
    #model = SimpleCNN(num_class=args.num_class)
    model = models.vit_b_16()
    model.heads = torch.nn.Linear(768, 100)
    model.load_state_dict(torch.load(args.model_name))
    model.to(device)

    transform = T.Compose([
        T.Resize(256),  # (256, 256) で切り抜く。
        T.CenterCrop(224),  # 画像の中心に合わせて、(224, 224) で切り抜く
        T.ToTensor(),  # テンソルにする。
        T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # 標準化する。
        ])
    test_dataset = datasets.CIFAR100(
            './data',
            train = False,
            transform=transform
        )
    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = 1,
        shuffle = False
    )
    try:
        with tqdm(enumerate(test_dataloader), total=len(test_dataloader), ncols=100) as pbar:
            result = np.zeros((args.num_class, args.num_class))
            t_p, f_p, t_n, f_n, = np.zeros(args.num_class), np.zeros(args.num_class), np.zeros(args.num_class), np.zeros(args.num_class)
            model.eval()
            for i, batch in pbar:
                image, label = batch
                image = image.to(device)
                label = label.to(device)
                logit = model(image)
                predict = torch.argmax(torch.softmax(logit, dim=-1)).cpu().numpy()
                label = label.cpu().numpy()
                result[label, predict] += 1
            for posclass in range(args.num_class):
                total = result.sum()
                p = result[:, posclass].sum()
                n = total - p
                t_p[posclass] = result[posclass, posclass]
                f_p[posclass] = p - t_p[posclass]
                tp_and_fn = result[posclass, :].sum()
                f_n[posclass] = tp_and_fn - t_p[posclass]
                t_n[posclass] = n - f_n[posclass]

            accuracy, precision, recall = 0, 0, 0
            for cls in range(args.num_class):
                accuracy += (t_p[cls])/(total)
                precision += (t_p[cls])/(t_p[cls] + f_p[cls])
                recall += (t_p[cls])/(t_p[cls] + f_n[cls])
            accuracy = accuracy
            precision /= args.num_class
            recall /= args.num_class
            f_score = 2/((1/precision) + (1/recall))
            print(f'accuracy:{accuracy}, precision:{precision}, recall:{recall}, F_Score:{f_score}')
    except ValueError:
        pass
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--num_class',  default=100)
    parser.add_argument('--output_dir', default='output')
    args = parser.parse_args()
    main(args)