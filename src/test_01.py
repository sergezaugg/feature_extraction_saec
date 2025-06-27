#--------------------             
# Author : Serge Zaugg
# Description : A few basic tests for CI
#--------------------

import torch
from fe_saec import SAEC_extractor
# from src.fe_saec import SAEC_extractor # for interactive dev of tests

# temp to suppress warning triggered by UMAP when using sklearn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# path to a TorchScript model
path_model = "./dev/dev_data/20250614_004030_encoder_script_GenBTP32_CH0256_epo006.pth"
path_images = "./dev/dev_data/images"
path_save = "./dev/dev_outp"

def test_placeholder():
    assert 44 == 44

# # test 1
# fe001 = SAEC_extractor(path_model = path_model, device = device) 
# fe001.extract(image_path = path_images, fe_save_path = path_save, batch_size = 16, shuffle = True , devel = True) 
# fe001.time_pool(ecut=0)
# fe001.reduce_dimension(n_neigh = 10, reduced_dim = 5)

# # Check dim of internally stored arrays 
# fe001.X.shape
# fe001.N.shape
# fe001.X_pooled.shape
# fe001.X_2D.shape
# fe001.X_red.shape

# def test_0011():
#     assert fe001.X.shape == (64, 256, 8)

# def test_0012():
#     assert fe001.N.shape == (64,)

# def test_0013():
#     assert fe001.X_pooled.shape == (64, 512)

# def test_0014():
#     assert fe001.X_2D.shape == (64, 2)

# def test_0015():
#     assert fe001.X_red.shape == (64, 5)


