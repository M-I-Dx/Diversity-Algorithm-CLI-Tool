from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.datasets import ImageFolder 
from termcolor import colored
from sklearn.preprocessing import LabelEncoder
from matplotlib import colors 
import pickle
import csv 
import os
import numpy as np
import pandas as pd 
import faiss 
import umap
import shutil
import matplotlib.pyplot as plt


def process_faiss(index_path, dataset_path): 

  if dataset_path is not None:
    index = faiss.read_index(index_path)
    feature_list = []
    filenames = []
    dataset = ImageFolder(dataset_path)
    dataset = dataset.imgs
    for i in range(index.ntotal):
      feature_list.append(index.reconstruct(i))
      filenames.append(dataset[i][0])
    return filenames, feature_list
  else:
    print("Invalid DATA_PATH")
    exit()

def converter(instr):
    return np.fromstring(instr[1:-1],sep=' ')

def process_csv(path):
  with open(path,'r') as f:
    reader = csv.reader(f)
    columns = next(reader)
  df = pd.read_csv(path,converters={columns[1]:converter},index_col = 0)
  filenames = [df.values[i][1] for i in range(len(df.values))]
  feature_list = [df.values[i][0] for i in range(len(df.values))]
  return filenames, feature_list

def save_pickle(filename,obj):
  with open(filename, 'wb') as handle:
      pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
  with open(filename, 'rb') as handle:
      embedding_path_tuple = pickle.load(handle)
  filenames = [embedding_path_tuple[i][1] for i in range(len(embedding_path_tuple))]
  feature_list = [embedding_path_tuple[i][0] for i in range(len(embedding_path_tuple))]
  return filenames, feature_list

def plot_umap(feature_list, filenames , path, n_neighbors=20, count = 0):
  
  class_id = []
  for _ in filenames:
    class_id.append(_.split("/")[-2])
  num_points = dict((x,class_id.count(x)) for x in set(class_id))
  txt = ''
  for i in num_points.keys():
    txt += i + ':' + str(num_points[i]) + " "
  le = LabelEncoder()
  class_labels = le.fit_transform(class_id)

  fit = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        metric='euclidean')

  u = fit.fit_transform(feature_list)
  color_map = plt.cm.get_cmap('tab20b_r')
  scatter_plot= plt.scatter(u[:,0], u[:, 1], c=class_labels, cmap = color_map)
  plt.title('UMAP embedding');
  plt.colorbar(scatter_plot)
  
  fname = path + 'UMAP_DA_Embeddings' + '.png'
  print("UMAP Saved at:",fname)
  plt.savefig(fname)
  plt.show();

def min_max_diverse_embeddings(n , filenames, feature_list, i = None) :
  if len(feature_list) != len(filenames) or len(feature_list) == 0 :
      return 'Data Inconsistent'
  n = int(n * len(feature_list))
  print("Len of Filenames and Feature List for sanity check:",len(filenames),len(feature_list))
  filename_copy = filenames.copy()
  set_input = feature_list.copy()
  set_output = []
  filename_output = []
  idx = 0
  if i is None: 
      idx = random.randint(0, len(set_input) -1)
  else:
      idx = i
  set_output.append(set_input[idx])
  filename_output.append(filename_copy[idx])
  min_distances = [1000] * len(set_input)
  # maximizes the minimum distance
  for _ in tqdm(range(n - 1)):
      for i in range(len(set_input)) :
          # distances[i] = minimum of the distances between set_output and one of set_input
          dist = np.linalg.norm(set_input[i] - set_output[-1])
          if min_distances[i] > dist :
              min_distances[i] = dist
      inds = min_distances.index(max(min_distances))
      set_output.append(set_input[inds])
      filename_output.append(filename_copy[inds])

  return filename_output, set_output, min_distances

def farthest_point(embeddings):
  import scipy.spatial.distance as dist
  centroid = sum(embeddings)/len(embeddings)
  distances = [dist.euclidean(i, centroid) for i in embeddings]

  return distances.index(max(distances))

def class_distrib(filenames):
    classes = {}
    for x in filenames:
      if x.split('/')[-2] in classes:
        classes[x.split('/')[-2]] += 1
      else:
        classes[x.split('/')[-2]] = 1

    classes_dict = {}
    for _ in classes.keys():
      print(_ , '= ', classes[_])
      classes_dict[_] = classes[_]
    x = classes_dict
    return x 

def prepare_dataset(dir, paths): 

  try: 
    os.makedirs(dir)
  except:
    shutil.rmtree(dir)
    os.makedirs(dir)
    # os.makedirs(os.path.join(dir, 'train'))

  for x in paths:
    # if x.split("/")[-2] not in os.listdir(os.path.join(dir,"train")):
    if x.split("/")[-2] not in os.listdir(dir):
      # os.mkdir(os.path.join(os.path.join(dir,"train"),x.split("/")[-2]))
      os.mkdir(os.path.join(dir,x.split("/")[-2]))
      shutil.copy(x, os.path.join(os.path.join(dir,x.split("/")[-2])))
    else:
      shutil.copy(x, os.path.join(os.path.join(dir,x.split("/")[-2])))
  print(colored('Subset moved to directory. Dataset abiding formats created successfully','blue'))

def driver():
  parser = ArgumentParser()
  parser.add_argument("--INPUT_FILE_PATH", type=str, help="path to pkl/csv file in format [embedding, filename]")
  parser.add_argument("--OUTPUT_FILE_PATH", default=None, type=str, help="path to store pkl file")
  parser.add_argument("--subset_size", default = 0.1, type=float, help="subset size for DA")
  parser.add_argument("--metric", default = False, type=bool, help="enable count metric")
  parser.add_argument("--DATA_PATH", type=str, default=None, help="Path to dataset in ImageFolder format")
  parser.add_argument("--UMAP", default = False, type=bool, help="enable UMAP")

  args = parser.parse_args()
  INPUT_FILE_PATH = args.INPUT_FILE_PATH
  OUTPUT_FILE_PATH = args.OUTPUT_FILE_PATH
  subset_size = args.subset_size
  metric = args.metric
  UMAP = args.UMAP
  DATA_PATH = args.DATA_PATH

  if '.pkl' in INPUT_FILE_PATH:
    filenames, feature_list = load_pickle(INPUT_FILE_PATH)   
  elif '.csv' in INPUT_FILE_PATH:
    filenames, feature_list = process_csv(INPUT_FILE_PATH) 
  elif '.bin' in INPUT_FILE_PATH:
    filenames, feature_list = process_faiss(INPUT_FILE_PATH, DATA_PATH)

  
  print(colored("Number of Files and Features",'blue'),len(filenames), colored("Embedding Size",'blue'), feature_list[0].shape)

  print(colored("Running DA..",'blue'))
  da_files, da_embeddings, _ = min_max_diverse_embeddings(subset_size, filenames, feature_list, i = farthest_point(feature_list))

  
  if OUTPUT_FILE_PATH != None:
    if '.pkl' in OUTPUT_FILE_PATH:
      embedding_path_tuple = [] 
      for i in range(len(da_embeddings)):
        embedding_path_tuple.append((da_embeddings[i], da_files[i]))
        save_pickle(OUTPUT_FILE_PATH, embedding_path_tuple)
      print("Pkl file saved at:",OUTPUT_FILE_PATH)
    else:
      prepare_dataset(OUTPUT_FILE_PATH,da_files)

  if metric:
      class_distrib(da_files)
  if UMAP:
      plot_umap(da_embeddings, da_files, path=OUTPUT_FILE_PATH.split('/')[-2])

if __name__ == "__main__":
  driver()