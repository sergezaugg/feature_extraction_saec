#--------------------------------
# Author : Serge Zaugg
# Description : 
#--------------------------------

import torch
from fe_saec import SAEC_extractor

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define full path to a TorchScript model
path_model = "D:/xc_real_projects/pytorch_hot_models_keep/20250617_150956_encoder_script_GenC_new_TP32_epo007.pth"
# path to dir with images 
path_images  = "D:/xc_real_projects/xc_sw_europe/xc_spectrograms"
# instantiate with a model
ae = SAEC_extractor(path_model = path_model, device = device) 
# extract (will save to disk as npz)
ae.extract(image_path = path_images, batch_size = 16, shuffle = True , devel = True) 
# Check dim of internally stores array 
ae.X.shape
ae.N.shape
# time pool
ae.time_pool(ecut=0)
ae.time_pool(ecut=2)
# dim reduce
[ae.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8]]  

