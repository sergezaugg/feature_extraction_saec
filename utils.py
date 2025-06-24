#--------------------------------
# Author : Serge Zaugg
# Description : small helper functions and bigger ML processes are wrapped into functions/classes here
#--------------------------------

import os 
import pickle
import numpy as np
import pandas as pd
import datetime
from PIL import Image
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms.v2 as transforms
import yaml


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

class AutoencoderExtract:
    """
    A class for extracting features and evaluating image reconstructions using autoencoders.
    This class provides utilities for dimensionality reduction, feature extraction, 
    visualization, and pooling/aggregation of features over time using a trained autoencoder.
    """
  
    def __init__(self, path_models, model_tag, path_images, device): 
        """
        Initialize the AutoencoderExtract instance.
        Loads session parameters and configuration files, and sets up paths and device information.
        """
        self.path_models = path_models
        self.time_stamp_model = model_tag
        self.path_images = path_images
        self.device = device

    def dim_reduce(self, X, n_neigh, n_dims_red):
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

    def evaluate_reconstruction_on_examples(self, n_images = 16, shuffle = True):
        """
        Evaluate the quality of autoencoder reconstructions on a sample of images.
        Loads a batch of images, reconstructs them using the trained autoencoder,
        and plots side-by-side comparisons of original and reconstructed images.
        Args:
            n_images (int, optional): Number of images to sample and display. Default is 16.
            shuffle (bool, optional): Whether to shuffle the dataset when sampling. Default is True.
        Returns:
            plotly.graph_objs._figure.Figure: A plotly figure showing original and reconstructed images.
        """
        # ---------------------
        # (1) load a few images 
        test_dataset = SpectroImageDataset(self.path_images, augment_1 = False, denoise_1 = False, augment_2 = False, denoise_2 = False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = n_images, shuffle = shuffle)
        for i_test, (data_1, data_2 , _ ) in enumerate(test_loader, 0):
            if i_test > 0: break
            print(data_1.shape)
            print(data_2.shape)
        # ---------------------
        # (2) load models 
        # NEW with TorchScript models 
        path_enc = [a for a in os.listdir(self.path_models) if self.time_stamp_model in a and 'encoder_script' in a][0]
        path_dec = [a for a in os.listdir(self.path_models) if self.time_stamp_model in a and 'decoder_script' in a][0]
        model_enc = torch.jit.load(os.path.join(self.path_models, path_enc))
        model_dec = torch.jit.load(os.path.join(self.path_models, path_dec))
        model_enc = model_enc.to(self.device)
        model_dec = model_dec.to(self.device)
        _ = model_enc.eval()
        _ = model_dec.eval()
        # ---------------------
        # (3) predict 
        data = data_1.to(self.device)
        encoded = model_enc(data).to(self.device)
        decoded = model_dec(encoded).to(self.device)
        # ---------------------
        # plot 
        fig = make_subplots(rows=n_images, cols=2,)
        for ii in range(n_images) : 
            img_orig = data_2[ii].cpu().numpy()
            # img_orig = img_orig.squeeze() # 1 ch
            img_orig = np.moveaxis(img_orig, 0, 2) # 3 ch
            img_orig = 255.0*img_orig  
            img_reco = decoded[ii].cpu().detach().numpy()
            # img_reco = img_reco.squeeze()  # 1 ch
            img_reco = np.moveaxis(img_reco, 0, 2) # 3 ch
            img_reco = 255.0*img_reco   
            _ = fig.add_trace(px.imshow(img_orig).data[0], row=ii+1, col=1)
            _ = fig.add_trace(px.imshow(img_reco).data[0], row=ii+1, col=2)
        _ = fig.update_layout(autosize=True,height=400*n_images, width = 800)
        _ = fig.update_layout(title="Model ID: " + self.time_stamp_model)
        return(fig)

    def encoder_based_feature_extraction(self, batch_size = 128, shuffle = True, devel = False):
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
        # NEW with TorchScript models 
        path_enc = [a for a in os.listdir(self.path_models) if self.time_stamp_model in a and 'encoder_script' in a][0]
        model_enc = torch.jit.load(os.path.join(self.path_models, path_enc))
    
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
        tag = '_'.join(os.path.basename(path_enc).split('_')[0:2])     
        out_name = os.path.join(os.path.dirname(self.path_images), 'full_features_' + 'saec_' + tag + '.npz')
        np.savez(file = out_name, X = feat, N = imfiles)

    def time_pool_and_dim_reduce(self, n_neigh = 10, reduced_dim = [2,4,8,16,32]):
        """
        Aggregate (pool) features over the time dimension and apply dimensionality reduction.
        Loads previously extracted feature arrays, chops time edges, computes mean and std over time,
        and applies UMAP to obtain reduced representations in various dimensions. Saves the results as .npz files.
        Args:
            n_neigh (int, optional): Number of neighbors for UMAP. Default is 10.
            reduced_dim (list of int, optional): List of target dimensions for reduction. Default is [2,4,8,16,32].
        Returns:
            None
        """
        npzfile_full_path = os.path.join(os.path.dirname(self.path_images), 'full_features_' + 'saec_' + self.time_stamp_model + '.npz')
        file_name_in = os.path.basename(npzfile_full_path)        
        # load full features 
        npzfile = np.load(npzfile_full_path)
        X = npzfile['X']
        N = npzfile['N']
        # combine information over time
        # cutting time edges (currently hard coded to 10% on each side)
        ecut = np.ceil(0.10 * X.shape[2]).astype(int)
        X = X[:, :, ecut:(-1*ecut)] 
        print('Feature dim After cutting time edges:', X.shape)
        # full average pool over time 
        X_mea = X.mean(axis=2)
        X_std = X.std(axis=2)
        X_mea.shape
        X_std.shape
        X = np.concatenate([X_mea, X_std], axis = 1)
        print('Feature dim After average/std pool along time:', X.shape)
        # X.shape
        # N.shape
        # make 2d feats needed for plot 
        X_2D = self.dim_reduce(X, n_neigh, 2)
        for n_dims_red in reduced_dim:
            X_red = self.dim_reduce(X, n_neigh, n_dims_red)
            print(X.shape, X_red.shape, X_2D.shape, N.shape)
            # save as npz
            tag_dim_red = "dimred_" + str(n_dims_red) + "_neigh_" + str(n_neigh) + "_"
            file_name_out = tag_dim_red + '_'.join(file_name_in.split('_')[2:5])
            out_name = os.path.join(os.path.dirname(npzfile_full_path), file_name_out)
            np.savez(file = out_name, X_red = X_red, X_2D = X_2D, N = N)

            
# devel 
if __name__ == "__main__":
    print(22)


  




