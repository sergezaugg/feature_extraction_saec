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
path_model = "D:/xc_real_projects/pytorch_hot_models_keep/20250617_150956_encoder_script_GenC_new_TP32_epo007.pth"
path_images  = "D:/xc_real_projects/xc_sw_europe/xc_spectrograms"
path_save = "C:/Users/sezau/Downloads"

ae = SAEC_extractor(path_model = path_model, device = device) 
# extract (will save to disk as npz)
ae.extract(image_path = path_images, fe_save_path = path_save, batch_size = 16, shuffle = True , devel = True) # , ecut = 1)
# time pool
ae.time_pool(ecut=2)
ae.time_pool()
# dim reduce
ae.reduce_dimension(n_neigh = 10, reduced_dim = 8) 

ae.time_stamp_model
ae.X.shape
ae.N.shape
ae.X_pooled.shape
ae.X_2D.shape
ae.X_red.shape
