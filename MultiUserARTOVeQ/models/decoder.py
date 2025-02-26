import torch
import torch.nn as nn
import math

#MULTI INPUT VISION TRANSORMER
# Vision Transformer (ViT)
# ViT divides the image into patches, adds positional encoding, and applies multi-head self-attention with transformer encoder blocks.

class PatchEmbedding(nn.Module):
    """
    This class divides the input image into patches and projects each patch into an embedding of size embed_dim.
    """
    # Transforms a image to a sequence

    def __init__(self, image_size: int, patch_size: int, embed_dim: int, in_channels: int = 1):
        """
        Initializes the PatchEmbedding layer.

        :param image_size: The size of the input image (height and width assumed to be equal).
        :param patch_size: The size of each patch to be extracted.
        :param embed_dim: The dimension of the embedding for each patch.
        :param in_channels: The number of channels in the input image (default is 3 for RGB images).
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


        # Convolution to project patches into an embedding space
        self.projection = nn.Linear(in_features=in_channels*patch_size**2, out_features=self.embed_dim).to(self.device)

        # Check if image size is divisible by patch size
        assert image_size % patch_size == 0, "Image size must be divisible by patch size."

    def image_to_patch_sequence(self, input_image:torch.Tensor, flatten_channels = True) -> torch.Tensor:
        B,C,H,W = input_image.shape
        patch_image = input_image.reshape(B, C , H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size)
        patch_image= patch_image.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
        patch_image = patch_image.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
        if flatten_channels:
            patch_image = patch_image.flatten(2, 4)  # [B, H'*W', C*p_H*p_W], MNIST [B, 16, 49]
        return patch_image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PatchEmbedding. Divides the image into patches and projects them to embeddings.

        :param x: Input image tensor of shape (N, B, C, H, W).
        :return: A tensor of shape (N, B, num_patches, embed_dim), where num_patches = (H * W) / (patch_size ** 2).
        """
        if x.dim() == 4:
            x = self.image_to_patch_sequence(x)
            x = self.projection(x)

        elif x.dim() == 5:
            data = []
            for user_i in range(x.shape[0]):
                user_data = self.image_to_patch_sequence(x[user_i]) # convert to image to sequence
                data.append(self.projection(user_data))

            x = torch.stack(data) # Shape: (N, B, num_patches, embed_dim)
        else:
            raise ValueError ('Dimensions of input x is incorrect')

        return x


class PositionalEncoding(nn.Module):
    """
    Adds positional encodings to the patch embeddings to retain spatial information.
    """

    def __init__(self, image_size: int, patch_size: int, embed_dim: int, num_users=2, encoding_type = 'learned'):
        """
        Initializes the PositionalEncoding layer.
        Require:Positional Encoding is independent of the number of users
            use sinusodial postional encoding
            PE(pos,2i) = sin(pos/10000^(2i/d)), PE(pos,2i+1) = cos(pos/10000^(2i+1/d))
        :param image_size: The size of the input image.
        :param patch_size: The size of the patches.
        :param embed_dim: The embedding dimension for each patch.
        """
        super().__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # Number of patches along one dimension
        num_patches = image_size // patch_size
        sequence_length = num_patches ** 2  # Total number of patches (flattened)
        self.n_groups = num_users
        # Remark: +1 accounts for the class token in vision transformer

        self.encoding_type = encoding_type

        if encoding_type == 'learned':
            # Learned positional encoding: (1, sequence_length, embed_dim)
            self.positional_embedding = nn.Parameter(torch.randn(1,sequence_length, embed_dim)).to(self.device)
            self.cls_positional_embedding = nn.Parameter(torch.randn(1, 1, embed_dim)).to(self.device)
        else:
            # Sinusoidal positional encoding
            self.positional_embedding = self._generate_postional_encoding(sequence_length, embed_dim)


    def _generate_postional_encoding(self, sequence_length: int, embed_dim: int) -> torch.Tensor:
        """
        Generates a sinusodial postional encoding


        :param sequence_length: Number of patches in the flattened image
        :param embed_dim: The dimension of the positional encoding
        :return: A tensor of shape (N,  B, sequence_length, embed_dim)
        """

        # Create a tensor of shape (sequence_length, embeded_dim) for sinusodial encoding
        position = torch.arange(sequence_length).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))

        positional_encoding = torch.zeros(sequence_length, embed_dim)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        return positional_encoding.unsqueeze(0) # Shape ( 1, sequence_length, embeded_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PositionalEncoding. Adds positional information to the input embeddings.
            Step 1: Split users_data N,B,C,H,W to N,B,Num_Patches, D (flatten patch size)
            Step 2: For each user compute positional encoding on Num_Patches, D
            Step 3: Add the positional encoding to the patch embeddings
            Step 4: Return Tensor of shape N,B,Num_Patches, D

        :param x: Input tensor of shape (B, num_patches + 1, embed_dim), where the first token is [CLS].
        :return: The tensor with positional encoding added.
        """
        B , L_total, D = x.shape

        L = (L_total - 1)//self.n_groups # -1 accounts for CLS token

        if self.encoding_type == 'learned':
            # Split into groups
            groups_embeding = [x[:, i * L:(i + 1) * L, :] + self.positional_embedding.expand(B,-1,-1) for i in range(self.n_groups)]
            class_token = x[:, -1:, :] + self.cls_positional_embedding

            return torch.cat(groups_embeding + [class_token], dim=1)

        else:
            pass
            #TODO: fix
            # positional_embedding = self.positional_embedding.expand((N, B, -1, -1))





#TODO vision transformer for mutliple images
class MultiHeadAttention(nn.Module):
    """
    Implements multi-head self-attention.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        """
        Initializes the MultiHeadAttention layer.

        :param embed_dim: The dimension of the embedding space.
        :param num_heads: The number of attention heads.
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads,batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MultiHeadAttention.

        :param x: Input tensor of shape (num_patches, B, embed_dim) for multi-head attention.
        :return: The output tensor after self-attention.
        """
        return self.attention(x, x, x)[0]

class GroupedLayerNorm(nn.Module):

    def __init__(self, embed_dim, num_users, eps = 1e-5, affine = True):
        super(GroupedLayerNorm, self).__init__()
        self.embed_dim = embed_dim
        self.n_groups = num_users
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(embed_dim))  # Scale
            self.beta = nn.Parameter(torch.zeros(embed_dim))  # Shift


    def forward(self, x):

        B, L_total, D = x.shape
        assert D == self.embed_dim, "Embedding dimension mismatch"

        # Calculate L (length of each subsequence)
        L = (L_total - 1) // self.n_groups  # -1 for class token

        # Split into groups
        groups = [x[:, i * L:(i + 1) * L, :] for i in range(self.n_groups)]
        class_token = x[:, -1:, :]

        # Normalize each group
        normed_groups = []
        for group in groups:
            mean = group.mean(dim=-1, keepdim=True)
            std = group.std(dim=-1, keepdim=True)
            normalized = (group - mean) / (std + self.eps)
            if self.affine:
                normalized = normalized * self.gamma + self.beta
            normed_groups.append(normalized)

        # Concatenate with class token
        out = torch.cat(normed_groups + [class_token], dim=1)
        return out


class TransformerEncoderBlock(nn.Module):
    """
    A single transformer encoder block containing multi-head self-attention and an MLP.
    """

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int):
        """
        Initializes the TransformerEncoderBlock.

        :param embed_dim: The embedding dimension for each patch.
        :param num_heads: The number of attention heads in the multi-head attention.
        :param mlp_dim: The dimension of the MLP hidden layer.
        """
        super().__init__()

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.layer_norm1 = GroupedLayerNorm(embed_dim=embed_dim,num_users=2).to(self.device)
        self.attention = MultiHeadAttention(embed_dim, num_heads).to(self.device)
        self.linear = nn.Sequential( nn.Linear(embed_dim, hidden_dim),
                                     nn.GELU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_dim, embed_dim),
                                     nn.GELU(),
                                     nn.Dropout(0.2)
        ).to(self.device)

        self.layer_norm2 =  GroupedLayerNorm(embed_dim=embed_dim,num_users=2).to(self.device)



    def forward(self, x: torch.Tensor, mode: str = 'concat') -> torch.Tensor:
        """
        Forward pass of TransformerEncoderBlock.

        :param x: Input tensor of shape (B, num_patches, embed_dim).
                mode(str): determines the aggregation mechanism
                    Option 1: concat- treats all patches for all users as a single input
                    Option 2: cross-user transformer layer - pass each user sequence separately than combine
        :return: The output tensor after attention and MLP processing.

        """

        # if not isinstance(mode, str):
        #     raise TypeError
        #
        # N, B, sequence_length, embed_dim = x.shape
        # #TODO: Aggregation of all users:
        #Option 1: Concatenate

        if mode == 'concat':
            #TODO: reshape creates multiple cls_tokens that are then shared within the attention mechanism
            # Extends the input sequence to incorporate all users
            # x = x.permute(1, 0, 2, 3).reshape(B, N*sequence_length,embed_dim)
            x = x + self.attention(self.layer_norm1(x))
            x = x + self.linear(self.layer_norm2(x))

        #Option 2: Include an additional transformer

        # Apply multi-head attention and add residual connection

        # Reshape x so attention mechanism can run then reshape it again
        # Question: self attention between users or for separately for each user


        return x


# Check out the attention scores and see if there is a strong correlation between the patches between different users
#TODO: apply vision transformer to multiple images
class VisionTransformer(nn.Module):
    """
    The Vision Transformer model which uses patch embeddings, positional encodings, and transformer encoder blocks.
    """
    #TODO: So far the ViT processes each user's data independently -try averaging or concatenating the representation

    def __init__(self, image_size: int, patch_size: int, num_classes: int, embed_dim: int, depth: int,
                 num_heads: int, hidden_dim: int, dropout: float = 0.1, num_users:int = 2):
        """
        Initializes the VisionTransformer model.

        :param image_size: The size of the input image.
        :param patch_size: The size of the patches to divide the image into.
        :param num_classes: The number of output classes for classification.
        :param embed_dim: The dimension of the patch embeddings.
        :param depth: The number of transformer encoder blocks.
        :param num_heads: The number of attention heads in each block.
        :param hidden_dim: The dimension of the MLP hidden layer in the transformer encoder block.
        :param dropout: The dropout rate.
        """
        super().__init__()
        self.num_users = num_users
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.num_patches = int((image_size/patch_size)**2)

        # Initialize components
        self.patch_embedding = PatchEmbedding(image_size, patch_size, embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_blocks = nn.ModuleList([TransformerEncoderBlock(embed_dim, num_heads, hidden_dim)
                                                 for _ in range(depth)])
        self.classification_head = nn.Linear(embed_dim, num_classes).to(self.device)

        # Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)).to(self.device) # [CLS] token for classification
        # applies 1 token to a single sequence, of size embedding dimension of the model (before broadcasting)
        self.positional_encoding = PositionalEncoding(image_size, patch_size, embed_dim)
        #self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim)).to(self.device)


    # def apply_positional_embedding(self, sequence):
    #     _, L, D = sequence.shape
    #     positional_embedding = torch.zeros(L, D).to(self.device)
    #
    #     # Assign the class token positional embedding
    #     positional_embedding[0,:] = self.positional_embedding[0,0] # class token positional embedding
    #
    #     # Repeat the rest of the positional embedding values
    #     positional_embedding[1:,:] = self.positional_embedding[0,1:].repeat(self.num_users,1)
    #
    #     # Add the positional embedding to the sequence
    #     sequence = sequence + positional_embedding.unsqueeze(0).expand(sequence.shape[0], -1, -1)
    #     return sequence


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of VisionTransformer.

        :param x: Input image tensor of shape (N*B, C, H, W).
        :return: The class prediction for the input image.
        """

        x = x.reshape(self.num_users,-1, x.shape[1], x.shape[2], x.shape[3]) # Quantization output shape (N*B,C,H,W)
        # x = x.permute(1, 0, 2, 3, 4).contiguous()  # B,N,C,H,W -> re-arrange to N,B,C,H,W
        N, B, C, H, W = x.shape

        x_patches = self.patch_embedding(x)  # Shape: (N, B, num_patches, embed_dim)
        #TODO visualizes patch embeddings

        # Try permuting and flatten the approiate dimensions
        # for each batch element make sure the embedded vector is the same
        x_patches = x_patches.permute(1, 0, 2, 3).flatten(1, 2)  # Shape (B, N*L, embed_dim)

        # x_patches = x_patches.view(B,self.num_patches*N,-1) # Shape (B, N*L, embed_dim)

        # Broadcast CLS token for each user and batch
        cls_tokens = self.cls_token.expand(B, -1, -1) # B,1,embed_dim


        x_patches = torch.cat((x_patches,cls_tokens), dim=1)  # (B, N*L+1, embed_dim)  CLS token at the end of the sequence

        # Add positional encoding to the embeddings

        # TODO: Fix problem here. positional embedding is independent of t N and has length L+1 - L is th number of images patches and +1 for CLS token
        #Solution: self.positional embedding [1:] replicate with number of users so the each patch corresponds to the same positional embedding. CLS token is shared amongst all users hence it gets the o element
        x_patches = self.positional_encoding(x_patches)

        x_patches = self.dropout(x_patches) #TODO: CHECK

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            z = block(x_patches)

        # Aggregate information torch.mean(x, dim=0) perhaps use cross attention
        # Classification is based on the [CLS] token's embedding
        return self.classification_head(z[:,-1])  # Shape: (B, num_classes) #TODO: Have issue here


    #Verify that each image patch for each user attains the same positional emedding