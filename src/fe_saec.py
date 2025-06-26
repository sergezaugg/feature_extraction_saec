#--------------------------------
# Author : Serge Zaugg
# Description : ML processes are wrapped into classes here
#--------------------------------

import os 
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms.v2 as transforms

class SpectroImageDataset(Dataset):
    """
    PyTorch Dataset for spectrogram images with optional denoising and augmentation.
    Loads PNG images from a directory and returns two versions (x_1, x_2) per sample, 
    each optionally denoised and/or augmented, along with the image filename.
    """

    def __init__(self, imgpath, par = None, augment_1=False, augment_2=False, denoise_1=False, denoise_2=False):
        """
        Initialize the SpectroImageDataset.

        Parameters
        ----------
        imgpath : str
            Directory containing PNG images.
        par : dict, optional
            Parameters for augmentation (par['da']) and denoising (par['den']).
        augment_1, augment_2 : bool, optional
            Apply augmentation to x_1/x_2.
        denoise_1, denoise_2 : bool, optional
            Apply denoising to x_1/x_2.

        Notes
        -----
        If any augmentation or denoising is enabled, constructs a torchvision transforms pipeline
        using parameters from `par['da']`. Only PNG files in `imgpath` are considered as dataset samples.
        """
        self.all_img_files = [a for a in os.listdir(imgpath) if '.png' in a]
        self.imgpath = imgpath
        self.par = par
        self.augment_1 = augment_1
        self.augment_2 = augment_2
        self.denoise_1 = denoise_1
        self.denoise_2 = denoise_2

        if self.augment_1 or self.augment_2 or self.denoise_1 or self.denoise_2:
            self.dataaugm = transforms.Compose([
                transforms.RandomAffine(translate=(self.par['da']['trans_prop'], 0.0), degrees=(-self.par['da']['rot_deg'], self.par['da']['rot_deg'])),
                transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianNoise(mean = 0.0, sigma = self.par['da']['gnoisesigm'], clip=True),]), p=self.par['da']['gnoiseprob']),
                transforms.ColorJitter(brightness = self.par['da']['brightness'] , contrast = self.par['da']['contrast']),
                ])
 
    def __getitem__(self, index):   
        """
        Returns:
            tuple: (x_1, x_2, y)
                x_1, x_2 (Tensor): Processed images.
                y (str): Filename.
        """  
        img = Image.open( os.path.join(self.imgpath,  self.all_img_files[index] ))
        # load pimage and set range to [0.0, 1.0]
        x_1 = pil_to_tensor(img).to(torch.float32) / 255.0
        x_2 = pil_to_tensor(img).to(torch.float32) / 255.0
        # simple de-noising with threshold
        # take random thld between 0.0 and self.par['den']['thld']
        if self.denoise_1: 
            denoize_thld = np.random.uniform(low=self.par['den']['thld_lo'], high=self.par['den']['thld_up'], size=1).item()
            x_1[x_1 < denoize_thld ] = 0.0
        if self.denoise_2:
            denoize_thld = np.random.uniform(low=self.par['den']['thld_lo'], high=self.par['den']['thld_up'], size=1).item() 
            x_2[x_2 < denoize_thld ] = 0.0    
        # data augmentation 
        if self.augment_1: 
            x_1 = self.dataaugm(x_1)  
        if self.augment_2:
            x_2 = self.dataaugm(x_2) 
        # prepare meta-data 
        y = self.all_img_files[index]

        return (x_1, x_2, y)
    
    def __len__(self):
        """Number of images in the dataset."""
        return (len(self.all_img_files))

class SAEC_extractor:
    """
    A class for extracting features and evaluating image reconstructions using auto-encoders.
    This class provides utilities for dimensionality reduction, feature extraction, 
    visualization, and pooling/aggregation of features over time using a trained auto-encoder.
    """
  
    def __init__(self, path_model, device): 
        """
        Initialize the SAEC_extractor instance.
        Loads parameters and sets up paths and device information.
        """
        if not ('encoder_script'in path_model):
            print('Model not loaded! "path_model" must be the path to a TorchScript model that has "encoder_script" in its name')
        else:    
            self.path_enc = path_model
            self.time_stamp_model = os.path.basename(self.path_enc)[0:15]
            self.device = device

    def _dim_reduce(self, X, n_neigh, n_dims_red):
        """
        Perform dimensionality reduction on input features using UMAP, with pre- and post-scaling.
        Args:
            X (np.ndarray): Input feature matrix to reduce.
            n_neigh (int): Number of neighbors for UMAP.
            n_dims_red (int): Number of dimensions for reduction.
        Returns:
            np.ndarray: The dimensionally reduced feature array after scaling.
        """
        scaler = StandardScaler()
        reducer = umap.UMAP(
            n_neighbors = n_neigh, 
            n_components = n_dims_red, 
            metric = 'euclidean',
            n_jobs = -1
            )
        X_scaled = scaler.fit_transform(X)
        X_trans = reducer.fit_transform(X_scaled)
        X_out = scaler.fit_transform(X_trans)
        return(X_out)

    def extract(self, image_path, fe_save_path,  batch_size = 128, shuffle = True, devel = False):
        """
        Extract features from images using a trained encoder and save the latent representation.
        Applies the encoder to all images in the specified directory and saves the resulting
        feature array and filenames as a .npz file.
        Args:
            batch_size (int, optional): Batch size for processing images. Default is 128.
            shuffle (bool, optional): Whether to shuffle the dataset. Default is True.
            devel (bool, optional): If True, only processes a few batches for development/testing. Default is False.
        Returns:
            None
        """
        self.path_images = image_path
        self.fe_save_path = fe_save_path
        # Load TorchScript models 
        model_enc = torch.jit.load(self.path_enc)
        model_enc = model_enc.to(self.device)
        _ = model_enc.eval()
        # prepare dataloader
        test_dataset = SpectroImageDataset(self.path_images, augment_1 = False, denoise_1 = False, augment_2 = False, denoise_2 = False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = shuffle)
        # extract features
        feat_li = []
        imfiles = []
        for i, (data, _, fi) in enumerate(test_loader, 0):    
            print('input data shape', data.shape)
            data = data.to(self.device)
            encoded = model_enc(data).detach().cpu().numpy()
            print('encoded.shape', encoded.shape)
            feat_li.append(encoded)
            imfiles.append(fi)
            print(len(imfiles))
            if devel and i > 2:
                break
        # transform lists to array 
        feat = np.concatenate(feat_li)
        feat = feat.squeeze()
        imfiles = np.concatenate(imfiles)
        # save as npz
        tag = '_'.join(os.path.basename(self.path_enc).split('_')[0:2])     
        out_name = os.path.join(self.fe_save_path, 'full_features_' + 'saec_' + tag + '.npz')
        self.X = feat
        self.N = imfiles
        np.savez(file = out_name, X = feat, N = imfiles)

    def time_pool(self, ecut=0):
        """
        in devel
        """
        print('Feature dim at start:', self.X.shape)
        # cutting time edges
        if ecut != 0:
            X = self.X[:, :, ecut:(-1*ecut)]
        else:
            X = self.X
        print('Feature dim After cutting time edges:', X.shape)
        # full average pool over time 
        X_mea = X.mean(axis=2)
        X_std = X.std(axis=2)
        X = np.concatenate([X_mea, X_std], axis = 1)
        print('Feature dim After average/std pool along time:', X.shape)
        self.X_pooled = X
      
    def reduce_dimension(self, n_neigh = 10, reduced_dim = 8):
        """
        in devel
        """
        if not hasattr(self, 'X_pooled'):
            print("Please first run .time_pool() ")    
        else:
            self.X_2D  = self._dim_reduce(self.X_pooled, n_neigh, 2) # make 2D features needed for plot 
            self.X_red = self._dim_reduce(self.X_pooled, n_neigh, reduced_dim)
            print('Shapes: ', self.X_pooled.shape, self.X_red.shape, self.X_2D.shape, self.N.shape)
            # save as npz
            file_name_out = "dimred_" + str(reduced_dim) + "_neigh_" + str(n_neigh) + "_" + 'saec_' + self.time_stamp_model + '.npz'
            out_name = os.path.join(self.fe_save_path, file_name_out)
            np.savez(file = out_name, X_red = self.X_red, X_2D = self.X_2D, N = self.N)


# devel 
if __name__ == "__main__":
    print(22)


  




