# 判斷我將 train 資料集切成 train, validation 後，
# 這兩個資料集的分佈是否均勻。
# 做這的目的是因為 train, val 的效果一直差很多，
# 所以才懷疑是否因為在切分資料集的時候，就那麼的剛好，
# 兩個資料集的分佈差很多導致效果差很多，
# 不過想當然 random 是真的滿 random，
# 資料集的分佈很相似，所以不是這個問題。


import argparse
import os
import random

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset

from pysot.core.config import cfg
from pysot.datasets.collate import collate_fn_new
from pysot.datasets.pcbdataset.pcbdataset_search import PCBDataset

plt.style.use('ggplot')


def get_loader(validation_split: float = 0.0, random_seed: int = 42):
    dataset = PCBDataset(args, "train")
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.seed(random_seed)
    random.shuffle(indices)
    split = dataset_size - int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[:split], indices[split:]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    def create_loader(dataset, batch_size, num_workers):
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn_new
        )
        return loader

    train_loader = create_loader(train_dataset, 1, 16)
    val_loader = create_loader(val_dataset, 1, 16)

    return train_loader, val_loader


def loader_to_boxes(loader):
    boxes = list()
    for idx, data in enumerate(loader):
        z_box = data['z_box'][0].squeeze()
        z_w = z_box[2] - z_box[0]
        z_h = z_box[3] - z_box[1]
        boxes.append([z_w, z_h])
    boxes = np.array(boxes)
    return boxes


def draw(train_boxes, val_boxes):
    fig, ax = plt.subplots()
    ax.scatter(x=train_boxes[:, 0], y=train_boxes[:, 1], c="green")
    ax.scatter(x=val_boxes[:, 0], y=val_boxes[:, 1], c="red")
    ax.set_xlim([0, cfg.TRAIN.SEARCH_SIZE])
    ax.set_ylim([0, cfg.TRAIN.SEARCH_SIZE])
    ax.set_xlabel("width")
    ax.set_ylabel("height")
    ax.set_aspect(1)
    save_path = os.path.join("./pysot/datasets/analysis/analysis.jpg")
    plt.savefig(save_path)
    print(f"Save boxes scatter plot to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='draw train & val boxes on plot')
    parser.add_argument('--dataset_name', type=str, default='', help='dataset name')
    parser.add_argument('--dataset', type=str, default='', help='training dataset')
    parser.add_argument('--criteria', type=str, default='', help='criteria of dataset')
    parser.add_argument('--neg', type=float, default=0.0, help='negative pair')
    parser.add_argument('--bg', type=str, help='background of template')
    parser.add_argument('--cfg', type=str,
                                 default='./experiments/siamcar_r50/config.yaml', help='configuration of tracking')
    args = parser.parse_args()

    print("Building datasets...")
    train_loader, val_loader = get_loader(validation_split=0.1)
    assert len(train_loader.dataset) != 0, "ERROR, empty dataset!!"

    print("Turning loaders to boxes...")
    train_boxes = loader_to_boxes(train_loader)
    val_boxes = loader_to_boxes(val_loader)
    print(f"Number of train_boxes: {train_boxes.shape[0]}")
    print(f"Number of val_boxes: {val_boxes.shape[0]}")

    draw(train_boxes, val_boxes)

    print('=' * 20, "Done", '=' * 20)
