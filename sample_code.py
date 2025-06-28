#--------------------------------
# Author : Serge Zaugg
# Description : 
#--------------------------------

import torch
from fe_saec import SAEC_extractor

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# path to a TorchScript model
path_model = "./dev/dev_data/20250614_004030_encoder_script_GenBTP32_CH0256_epo006.pth"
# path to dir with images 
path_images  = "./dev/dev_data/images"
# path where features will be saved
path_save = "./dev/dev_outp"

# instantiate with a model
ae = SAEC_extractor(path_model = path_model, device = device) 
# extract (will save to disk as npz)
ae.extract(image_path = path_images, fe_save_path = path_save, batch_size = 16, shuffle = True , n_batches = 2) 
# time pool
ae.time_pool(ecut=0)
ae.time_pool(ecut=2)
# dim reduce
[ae.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8]]  

# Check dim of internally stored arrays 
ae.X.shape
ae.N.shape
ae.X_pooled.shape
ae.X_2D.shape
ae.X_red.shape


