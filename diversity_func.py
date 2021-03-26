from da_utils import ModelUtils
from da_algos import DaTechniques
import sys
sys.path.insert(0,"./Self-Supervised-Learner")
from models import SIMCLR


def get_diversity_imgs(DATA_PATH, DIVERSITY_DATA, MODEL_PATH, subset_size, 
                            img_size, embedding_size, technique, sample_size):
    utils = ModelUtils(MODEL_PATH, DATA_PATH, img_size, embedding_size)
    filenames, feature_list = get_matrix(MODEL_PATH, DATA_PATH, img_size, embedding_size)
    da = DaTechniques(subset_size, filenames, feature_list, sample_size)

    if technique=='DA_STD':
        da_files, da_embeddings, _ = da.min_max_diverse_embeddings(i = da.farthest_point())
    elif technique=='DA_FAST':
        da_files, da_embeddings, _ = da.min_max_diverse_embeddings_fast(i = da.farthest_point())
    utils.filenames = da_files
    utils.embeddings = da_embeddings
    utils.prepare_dataset(DIVERSITY_DATA)
    utils.plot_umap(DIVERSITY_DATA)



def get_matrix(MODEL_PATH, DATA_PATH, img_size, embedding_size):
    def to_tensor(pil):
        return torch.tensor(np.array(pil)).permute(2,0,1).float()
    t = transforms.Compose([
                            transforms.Resize((img_size, img_size)),
                            transforms.Lambda(to_tensor)
                            ])
    dataset = ImageFolder(DATA_PATH, transform = t)
    model = SIMCLR.SIMCLR.load_from_checkpoint(MODEL_PATH)
    model.eval()
    model.cuda()
    with torch.no_grad():
        data_matrix = torch.empty(embedding_size = (0, embedding_size)).cuda()
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