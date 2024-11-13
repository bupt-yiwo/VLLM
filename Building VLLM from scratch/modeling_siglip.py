from typing import Optional, Tuple
import torch
import torch.nn as nn


# The visual model provides us with the features of the picture, but its dimensions are not aligned with the language model, so later we will go through a projecter (linear layer) to perform the alignment of the feature dimensions


class SiglipVisionConfig:# NO.0 Configuration file for saved model parameters, to be loaded whenever the model is loaded.

    def __init__(
        self,
        hidden_size=768, # Feature dimensions of the features obtained by the model after processing the image
        intermediate_size=3072, # The MLP layer of each transformer layer of the visual model maps the features to a high dimension and then introduces a non-linearity, this high dimension is typically four times the original feature dimension
        num_hidden_layers=12, # How many layers of transformers are there in the visual model
        num_attention_heads=12, # Number of k, q and v attention heads per transformer layer
        num_channels=3, # Number of channels of the image, RGB images have three channels and greyscale images have one channel
        image_size=224, # The size of the resolution of the image to be processed (224 * 224), if it is not this size then it will be scaled
        patch_size=16, # The size of the patch when the image is processed by conv2d
        layer_norm_eps=1e-6, # A parameter in Layer Normalization, eps denotes epsilon, a very small value usually used to prevent division by zero when calculating standard deviation or variance.
        attention_dropout=0.0, # This parameter is used to control the proportion of Dropout in the attention mechanism, an operation that helps to reduce model overfitting and allows the model to perform more robustly in the face of unseen data.
        num_image_tokens: int = None, # The number of tokens corresponding to the image feature. (224/16) ** 2 = 256
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module): # NO.3
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(  # Convolution to extract features, this part suggests learning the basic convolution.
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # This indicates no padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2 # The number of tokens of the image
        self.num_positions = self.num_patches 
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim) # TOKEN's location code, which is used to add location information to each token afterwards
        self.register_buffer( # Registers a buffer called position_ids to hold position indexes. persistent=False specifies that it does not need to be persisted when the model is saved.
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        ) # Tensors like position_ids are usually computed based on model parameters or input sizes and can be easily reconstructed.It is therefore simpler and more straightforward to recalculate these buffers after the model has been loaded than to save them. This avoids storing unnecessary information.

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [Batch_Size, Channels, Height, Width]
        
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)  
        
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)
        
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings


class SiglipAttention(nn.Module): # NO.6
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # Before looking at this part of the code, it is necessary to understand the formula for multi-headed self-attention
    # formula: softmax(Q * K^T / sqrt(d_k)) * V
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads # feature dimensions processed by each head, we let each head process different feature dimensions instead of one head, which allows for better feature information extraction.
        self.scale = self.head_dim**-0.5 # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        # key, value, query, out weight matrix
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim) 
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, _ = hidden_states.size()
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)
        
        # We need to adjust the dimensions of the matrix to get token-to-token queries
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_dim] -> [Batch_Size, Num_Heads, Num_Patches, Head_dim] 
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate the attention using the formula Q * K^T / sqrt(d_k). attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # Apply the softmax row-wise. attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # Multiply the attention weights by the value states. attn_output: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # [Batch_Size, Num_Patches, Embed_Dim] 
        # The results of each head are independent and parallel, and we get their results to interact through Wo
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module): # NO.7
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Intermediate_Size]
        # Nonlinear transformations are applied to the inputs to improve the representation and performance of the model, and tanh approximations are used to speed up the computation.
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipEncoderLayer(nn.Module): # NO.5
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config) # Multi-head self-attention, each head is responsible for processing different features.
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps) # Layer Normalisation (LN). Layer Normalisation (LN) is performed to enhance the stability of the model and the representation of features.
        self.mlp = SiglipMLP(config) # MLP layer that maps features to higher dimensions, introduces nonlinearities, and then maps back to the original dimensions.
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps) # Layer Normalisation (LN). Layer Normalisation (LN) is performed to enhance the stability of the model and the representation of features.

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states # Residual Connection
        # residual: [Batch_Size, Num_Patches, Embed_Dim] 
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states # Residual Connection
        # The core purpose of residual connectivity is to improve the training stability, information transfer efficiency and expressiveness of the model by introducing jump connections, allowing deeper networks to be trained and reasoned effectively.
        return hidden_states


class SiglipEncoder(nn.Module):# NO.4
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    # Ignore copy
    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module): # NO.2 
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config) # This class extracts features from the image, and the extracted features are subsequently processed through a layer-by-layer transformer.
        self.encoder = SiglipEncoder(config) # A collection of tranformer layers. Transformer gradually learns and integrates different levels of semantic information through different levels of transformations. 
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps) # Layer Normalisation (LN). Layer Normalisation (LN) is performed to enhance the stability of the model and the representation of features.

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        
        hidden_states = self.embeddings(pixel_values) # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]

        last_hidden_state = self.encoder(inputs_embeds=hidden_states) # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim] 

        last_hidden_state = self.post_layernorm(last_hidden_state) # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim] 

        return last_hidden_state # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]


class SiglipVisionModel(nn.Module): # NO.1 In writing the code for the model, the classes are progressively deeper

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config) 

    def forward(self, pixel_values) -> Tuple:
        # The processor will process the image into a tensor, and the next step is to extract features from this tensor.
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim], Num_Patches is the number of tokens of the image
        return self.vision_model(pixel_values=pixel_values) 
