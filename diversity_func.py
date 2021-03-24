from da_utils import ModelUtils
from da_algos import DaTechniques


def get_diversity_imgs(DATA_PATH, DIVERSITY_DATA, MODEL_PATH, subset_size, 
                            img_size, embedding_size, technique, sample_size):
    utils = ModelUtils(MODEL_PATH, DATA_PATH, img_size, embedding_size)
    filenames, feature_list = utils.get_embeddings()
    da = DaTechniques(subset_size, filenames, feature_list, sample_size)

    if technique=='DA_STD':
        da_files, da_embeddings, _ = da.min_max_diverse_embeddings(i = da.farthest_point())
    elif technique=='DA_FAST':
        da_files, da_embeddings, _ = da.min_max_diverse_embeddings_fast(i = da.farthest_point())
    utils.filenames = da_files
    utils.embeddings = da_embeddings
    utils.prepare_dataset(DIVERSITY_DATA)
    utils.plot_umap(DIVERSITY_DATA)

