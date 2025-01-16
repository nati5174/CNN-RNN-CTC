import os
import glob
import torch
import numpy as np

import albumentations
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
from torch.utils.data import DataLoaders
import config
import dataset


def run_training():
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.png"))
    print(image_files)
    targets_orig = [x.split("/")[-1][:-4] for x in image_files]
    targets = [[c for c in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]

    lbl_enc = preprocessing.LabelEncoder() # gives everything string in ur list a numberal value, dividing them into classes
    lbl_enc.fit(targets_flat)

    targets_enc = [lbl_enc.transform(x) for x in targets_flat] #convert to numerical value
    targets_enc = np.array(targets_enc) + 1

    print(targets)
    print(np.unique(targets_flat))

    (train_imgs, test_imgs, train_target, test_target, train_orig_targets, test_orig_targets) = model_selection.train_test_split(image_files, targets_enc, image_files, test_size=0.1, random_state=42 )
    train_dataset = dataset.ClassificationDataset(train_imgs, train_target, (config.IMAGE_HIGHT, config.widith))
    test_dataset = dataset.ClassificationDataset(test_imgs, test_target, (config.IMAGE_HIGHT, config.widith))

    train_loader = DataLoaders(train_dataset,config.BATCH_SIZE, config.NUM_WORKERS, shuffle=True )
    test_loader = DataLoaders(test_dataset,config.BATCH_SIZE, config.NUM_WORKERS, shuffle=True )

    model = #model to build
if __name__ == "__main__":
    run_training()    
