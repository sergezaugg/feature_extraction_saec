#--------------------------------
# Author : Serge Zaugg
# Description : 
#--------------------------------

import torch
from fe_saec import FeatureExtractor

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize a AEC-extractor instance
path_models = "D:/xc_real_projects/pytorch_hot_models_keep"
model_tag = "20250617_150956"
path_images  = "D:/xc_real_projects/xc_sw_europe/xc_spectrograms"
ae = FeatureExtractor(path_models, model_tag, path_images, device = device)

# evaluate reconstruction
ae.evaluate_reconstruction_on_examples(n_images = 64, shuffle = False).show()

# extract (will save to disk as npz)
ae.encoder_based_feature_extraction(devel = True)

# time_pool_and_dim_reduce
ae.time_pool_and_dim_reduce(n_neigh = 10, reduced_dim = [2, 4, 8, 16])
