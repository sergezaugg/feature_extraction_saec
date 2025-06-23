#--------------------------------
# Author : Serge Zaugg
# Description : 
#--------------------------------

import torch
from utils import AutoencoderExtract
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize a AEC-extractor instance
ae = AutoencoderExtract(sess = 'extract_02.yaml', device = device)
# evaluate reconstruction
ae.evaluate_reconstruction_on_examples(n_images = 64, shuffle = False).show()
# extract (will save to disk as npz)
ae.encoder_based_feature_extraction(devel = False)
# time_pool_and_dim_reduce
ae.time_pool_and_dim_reduce(n_neigh = 10, reduced_dim = [2, 4, 8, 16])
