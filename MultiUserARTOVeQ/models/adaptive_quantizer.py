import torch
import torch.nn as nn
import torch.nn.functional as F
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

#TODO: Add rotation trick
class AdaptiveVectorQuantizer(nn.Module):

    def __init__(self, codebook_vector_dim: int, codebook_size: int, commitment_loss_weight:float=0.1, proximity_loss_weight:float=0.33):
        """
        Initialize the AdaptiveVectorQuantizer.

        Args:
            codebook_vector_dim (int): Size of the vectors in the codebook.
            codebook_size (int): Number of vectors in the codebook.
            commitment_loss_weight (float, optional): Weight for the commitment loss. Defaults to 0.1.
            proximity_loss_weight (float, optional): Weight for the proximity loss. Defaults to 0.33.
        """
        super(AdaptiveVectorQuantizer, self).__init__()

        #
        self.d = codebook_vector_dim  # Size of the vectors
        self.p = codebook_size  # Number of vectors in the codebook

        # Initialize the codebook as an embedding layer
        self.codebook = nn.Embedding(self.p, self.d)
        self.codebook.weight.data.uniform_(-1 / self.p, 1 / self.p)  # Initialize codebook with Uniform Distribution


        # Loss weights
        self.commitment_loss_weight = commitment_loss_weight
        self.proximity_loss_weight = proximity_loss_weight

        # Select the device (GPU or CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def _compute_quantization(self, num_active_levels:int, flat_input: torch.Tensor, input_shape: tuple):
        """
            Computes quantized vectors for the given level of active vectors.

            Args:
                flat_input (Tensor): Flattened input tensor. (NBHWC//d, d)
                input_shape (tuple): Original input shape (NxBxHxWxC).
                num_active_levels (int): Current level of active vectors.

            Returns:
                tuple: Quantized tensor and active vectors.
        """

        # logging.debug(f"Computing quantization for {num_active_levels+1} with input shape {input_shape}")

        num_vectors = pow(2, num_active_levels + 1)
        active_vectors = self.codebook.weight[:num_vectors]
        # logging.debug(f"Using {num_vectors} active vectors for quantization.")

        # Calculate distances
        distances = (
                torch.sum(flat_input ** 2, dim=1, keepdim=True)
                + torch.sum(active_vectors ** 2, dim=1)
                - 2 * torch.matmul(flat_input, active_vectors.t())
        )
        # logging.debug(f"Calculated distances with shape {distances.shape}")

        # Find closest codebook vectors: j = argmin {||z_t - e_k||_2} for  k in {1,2,...,num_vectors}
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # logging.debug(f"Encoding indices shape: {encoding_indices.shape}")

        encodings = torch.zeros(encoding_indices.shape[0], num_vectors, device=flat_input.device)
        encodings.scatter_(1, encoding_indices, 1) # encoding indices are converted into one-hot encodings
        # logging.debug("Generated one-hot encodings for quantization.")

        # Quantize and reshape
        quantized = torch.matmul(encodings, active_vectors).view(input_shape)
        # logging.info("Quantization completed. \n")

        return quantized, active_vectors

    def _compute_loss(self, quantized: torch.Tensor, input_data: torch.Tensor,
                      num_active_levels: int, previous_active_vectors: torch.Tensor,
                      active_codebook_vectors: torch.Tensor):  #-> Tuple[torch.Tensor, float]
        """
        Compute the loss for the given quantization level.

        Args:
            quantized (Tensor): Quantized tensor of shape (N, B, H, W, C).
            input_data (Tensor): Input tensor of shape (N, B, H, W, C).
            num_active_levels (int): Current quantization level.
            previous_active_vectors (Tensor): Codebook vectors from the previous level.
            active_codebook_vectors (Tensor): Codebook vectors for the current level.

        Returns:
            tuple:
                quantized (Tensor): Modified quantized tensor (N, B, C, H, W).
                codebook_loss (float): Total loss for the current level.
        """
        # logging.info(" Computing Adaptive Quantizer Loss \n")
        quantization_loss = F.mse_loss(quantized, input_data.detach())
        proximity_loss = 0.0
        alignment_loss = 0.0

        if num_active_levels == 0:
            alignment_loss = F.mse_loss(quantized.detach(), input_data)
        else:
            if num_active_levels == 1:
                alignment_loss = F.mse_loss(quantized.detach(), input_data)

            proximity_loss = (num_active_levels * self.proximity_loss_weight) * F.mse_loss(
                previous_active_vectors[:pow(2, num_active_levels + 1) // 2],
                active_codebook_vectors[:pow(2, num_active_levels + 1) // 2]
            )

        current_loss = quantization_loss + self.commitment_loss_weight * alignment_loss + proximity_loss

        # Ensure proper gradient propagation
        quantized = input_data + (quantized - input_data).detach()


        #TODO: Might need to fix dimensions based on the decoder
        # Convert BHWC -> BCHW
        # quantized = quantized.permute(0, 2, 3, 1).contiguous() # changed from (0, 2, 3, 1) # this is for the decoder #TODO: Check input shape

        return quantized, current_loss

    def forward(self, input_data: torch.Tensor, num_active_vectors: int, previous_active_vectors: torch.Tensor) -> tuple:
        """
        Forward pass of the Adaptive Vector Quantizer.

        Args:
            input_data (Tensor): Input tensor of shape (Number of Users, B, C, H, W).
            num_active_vectors (int): Total number of active vectors (powers of 2).
            previous_active_vectors (Tensor): Previously active vectors used in quantization.

        Returns:
            tuple:
                quantized_vectors (List[Tensor]): List of quantized tensors for each level of active vectors.
                    Each tensor is of shape (B, C, H, W).
                losses (List[float]): List of loss values corresponding to each quantization level.
                active_codebook_vectors (Tensor): The final set of active vectors used in the codebook.
        """
        # logging.info("Started Adaptive Quantizer forward pass \n")
        # Rearrange input for processing:Num_UsersxBxHxWxC -> N*BxHXWXC
        #TODO: Might have a problem with the data here due to view

        reshape_data  = input_data.view(-1, input_data.shape[2], input_data.shape[3], input_data.shape[4])
        reshape_data = reshape_data.permute(0, 2, 3, 1).contiguous() # changed from  (0, 2, 3, 1)
        permutated_input_shape = reshape_data.shape

        # Flatten Input
        flattened_input = reshape_data.view(-1, self.d) # flatten input has shape NBHWC//d,d

        # Create empty list for storing quantities per each quantization level
        quantized_vectors = []
        losses = []

        # Process each level of active vectors
        for num_active_levels in range(int(np.log2(num_active_vectors))):

            # Compute quantized vectors per each level
            quantized, active_codebook_vectors = self._compute_quantization(num_active_levels, flattened_input, permutated_input_shape)

            # Compute loss
            if self.training:
                quantized, loss_at_level = self._compute_loss(quantized, reshape_data, num_active_levels, previous_active_vectors, active_codebook_vectors)
            # Perform Inference
            else:
                loss_at_level = 0.0

            #previous_active_vectors = active_codebook_vectors.detach() #Should be outside adaptive quantizer (I THINK)

            # Rearrange quantized vector from (N*B,H,W, C) -> (N*B,C,H,W)
            quantized = quantized.permute(0, 3, 1, 2).contiguous()

            quantized_vectors.append(quantized)
            losses.append(loss_at_level)

        return quantized_vectors, losses, active_codebook_vectors