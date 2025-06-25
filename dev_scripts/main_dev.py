#--------------------------------
# Author : Serge Zaugg
# Description : 
#--------------------------------

import torch

# dev
from src.fe_saec import FeatureExtractor

# usage
from fe_saec import FeatureExtractor

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize a AEC-extractor instance
path_models = "D:/xc_real_projects/pytorch_hot_models_keep"
model_tag = "20250617_150956"
path_images  = "D:/xc_real_projects/xc_sw_europe/xc_spectrograms"
ae = FeatureExtractor(path_models, model_tag, path_images, device = device)

# extract (will save to disk as npz)
ae.extract(devel = True)

# time_pool_and_dim_reduce
ae.time_pool()

[ae.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]




