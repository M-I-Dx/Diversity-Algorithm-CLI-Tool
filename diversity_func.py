import sys
sys.path.insert(0,"./Self-Supervised-Learner")

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from sklearn.preprocessing import LabelEncoder

from da_utils import ModelUtils
from da_algos import DaTechniques

from models import SIMCLR, SIMSIAM


def get_diversity_imgs(DATA_PATH, DIVERSITY_DATA, MODEL_PATH, subset_size, 
                            img_size, embedding_size, technique, sample_size, model_type="SIMCLR"):
    utils = ModelUtils(MODEL_PATH, DATA_PATH, img_size, embedding_size)
    filenames, feature_list = get_matrix(MODEL_PATH, DATA_PATH, img_size, embedding_size, model_type)
    da = DaTechniques(subset_size, filenames, feature_list, sample_size)

    if technique=='DA_STD':
        da_files, da_embeddings, _ = da.min_max_diverse_embeddings(i = da.farthest_point())
    elif technique=='DA_FAST':
        da_files, da_embeddings, _ = da.min_max_diverse_embeddings_fast(i = da.farthest_point())
    utils.filenames = da_files
    utils.embeddings = da_embeddings
    utils.prepare_dataset(DIVERSITY_DATA)


def get_matrix(MODEL_PATH, DATA_PATH, img_size, embedding_size, model_type="SIMCLR"):
    def to_tensor(pil):
        return torch.tensor(np.array(pil)).permute(2,0,1).float()
    t = transforms.Compose([
                            transforms.Resize((img_size, img_size)),
                            transforms.Lambda(to_tensor)
                            ])
    dataset = ImageFolder(DATA_PATH, transform = t)
    if model_type == "SIMCLR":
        model = SIMCLR.SIMCLR.load_from_checkpoint(MODEL_PATH, DATA_PATH)
    else:
        model = SIMSIAM.SIMSIAM.load_from_checkpoint(MODEL_PATH, DATA_PATH)
    model.eval()
    model.cuda()
    with torch.no_grad():
        data_matrix = torch.empty(size = (0, embedding_size)).cuda()
        bs = 32
        if len(dataset) < bs:
          bs = 1
        loader = DataLoader(dataset, batch_size = bs, shuffle = False)
        for batch in tqdm(loader):
            x = batch[0].cuda()
            embeddings = model(x)
            data_matrix = torch.vstack((data_matrix, embeddings))
    paths = [dataset.imgs[i][0] for i in range(len(dataset.imgs))]
    img_embeddings = data_matrix.cpu().detach().numpy()
    return paths, img_embeddings