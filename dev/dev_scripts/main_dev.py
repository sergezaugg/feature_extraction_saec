#--------------------------------
# Author : Serge Zaugg
# Description : Small script to try methods interactively
#--------------------------------

import torch
# dev
from src.fe_saec import SAEC_extractor
# usage
# from fe_saec import SAEC_extractor

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize a AEC-extractor instance
path_model = "./dev/dev_data/20250614_004030_encoder_script_GenBTP32_CH0256_epo006.pth"
path_images  = "./dev/dev_data/images"
path_save = "./dev/dev_outp"

ae = SAEC_extractor(path_model = path_model, device = device) 
# extract (will save to disk as npz)
ae.extract(image_path = path_images, fe_save_path = path_save, batch_size = 16, shuffle = True , n_batches = 12) 
# time pool
ae.time_pool(ecut=2)
ae.time_pool()
# dim reduce
ae.reduce_dimension(n_neigh = 10, reduced_dim = 10) 

ae.time_stamp_model
ae.X.shape
ae.N.shape
ae.X_pooled.shape
ae.X_2D.shape
ae.X_red.shape
