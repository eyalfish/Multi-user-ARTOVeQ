# Multi-User ARTOVeQ
import torch
import torch.nn as nn


from models.adaptive_quantizer import AdaptiveVectorQuantizer
from models.user_encoder_manager import UserManager
from models.decoder import *
from models.user_specific_encoder import Autoencoder

import numpy as np


import logging

logging.basicConfig(
    level=logging.DEBUG,  # Set the level to capture all types of logs
    format='%(asctime)s - %(levelname)s - %(message)s',  # Customize log format
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler('debug.log')  # Log to a file
    ]
)


class MultiuserARTOVeQ(nn.Module):
    def __init__(self,encoder_kwargs:dict, quantizer_kwargs:dict, decoder_kwargs:dict, flag_kwargs:dict):

        super(MultiuserARTOVeQ, self).__init__()


        #TODO: add normalization function
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Flag variables (bool)
        self.is_training =flag_kwargs['is_training']
        self.use_quantization = flag_kwargs['use_quantization'] # flag variable for quantizer forward pass


        self.quantizer = AdaptiveVectorQuantizer(**quantizer_kwargs).to(self.device)
        self.encoders = UserManager(Autoencoder, encoder_kwargs).to(self.device)

        # ViT is determined by the latent representation of the image
        self.decoder = VisionTransformer(**decoder_kwargs).to(self.device)

    def add_users(self, num_users):
        for _ in range(num_users):
            self.encoders.create_and_add_user()

    def _encode(self, users_data):

        # logging.info(f" Start of the _encode function for {len(users_data)} users \n")

        users_data= users_data.permute(1,0,2,3,4) #Rearrange tensors from (B,N,C,H,W) to (N,B,C,H,W)

        #unsqueeze is to include the channel dimension
        encoded_features = [user_network(users_data[ii]) for ii, (user_id, user_network) in
                            enumerate(self.encoders.users_networks.items())]

        encoded_tensors = torch.stack(encoded_features, dim = 0)  # Shape: (N,B,C,H,W)

        # logging.info(f'Encoded tensor shape: {encoded_tensors.shape} \n')
        # logging.debug(f' Normalization is missing \n')

        return encoded_tensors

    def _quantize(self, encoded_tensors: torch.Tensor, previous_active_vectors=None, number_active_vectors=None):

        # logging.info(f' Entered _quantize function \n')

        if self.use_quantization:
            # logging.info(f'Quantizing the encoding tensor')


            for quantization_level in range(int(np.log2(number_active_vectors))):  # codebook size = 8
                # logging.debug(f'Quantization level {quantization_level} \n')


                # quantization_kwargs = {previous_active_vectors: None, "out_channels": 6}
                quantized_tensors, vqvae_loss, active_codebook_vectors = self.quantizer(encoded_tensors,
                                                                                    number_active_vectors,
                                                                                    previous_active_vectors)  # TODO: input previous_active vectors is missing or it might not need to be passed
        else:
            # logging.info(f' use_quantization variable is false \n')
            quantized_tensors, vqvae_loss, active_codebook_vectors = [encoded_tensors], [0.0], None

        return quantized_tensors, vqvae_loss, active_codebook_vectors

    def _decode(self, quantized_tensors: list):

        # logging.info(f' Entered _decode function \n')
        #
        # logging.debug(f' type: {type(quantized_tensors)} \t shape: {quantized_tensors[0].shape}')

        prediction_list = []
        for vectors in range(len(quantized_tensors)):
            z_q_actives = quantized_tensors[vectors]
            prediction_list.append(self.decoder(z_q_actives))

        return prediction_list


    def forward(self, users_inputs:list, previous_active_vectors = None, number_active_vectors =None) -> torch.Tensor:
        # Perhaps add previous_active_vectors as an argument to quantization_kwargs
        """
        forward path is executed for all users in parallel via torch.multiprocessing module
        Args:
            inputs (list[torch.Tensor]): List of inputs, one per user
        :return:
            torch.Tensor: Server side classification output

        """
        #TODO: Changed input from list whose of tensors to a tensor of shape B,N,H,W
        # if not isinstance(users_inputs,list):
        #     users_inputs=[users_inputs]
        # # Forward pass through the encoders
        # if len(users_inputs) != self.encoders.user_counter:
        #     raise ValueError("Number of inputs does not match number of users")

        # Forward pass through the encoders
        z_encoded = self._encode(users_inputs)

        # Forward pass through the shared Quantizer
        # number active and previous_cb_vectors needs to be supplied if training
        if self.use_quantization:
            #update previous_active_vectors
            z_quantized, vqvae_loss, active_codebook_vectors = self._quantize(z_encoded, previous_active_vectors, number_active_vectors=number_active_vectors )
        else: # Inference

            z_quantized, vqvae_loss, active_codebook_vectors = self._quantize(z_encoded) # N*B,C,H,W

        # Forward path through decoder and classifier (within decoder)
        z_decoded = self._decode(z_quantized)


        return z_decoded, vqvae_loss, active_codebook_vectors, z_encoded
