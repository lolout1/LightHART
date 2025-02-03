import torch
import torch.nn as nn
import torch.nn.functional as F

# We will replicate the encoder block as its own PyTorch module
class EncoderBlock(nn.Module):
    """
    Replicates the 'encoder' function from the Keras code:
    
      1) y = LayerNorm(x)
      2) y = MultiHeadAttention(...) (query=x, key=x, value=x)
      3) res = x + y
      4) y = LayerNorm(res)   (But not actually used in feed-forward below!)
      5) y = Dense(mlp_dim)(res)
      6) y = Dropout
      7) y = Dense(embed_dim)
      8) y = Dropout
      9) out = res + y
    """
    def __init__(self, embed_dim, mlp_dim, num_heads, attn_drop_rate, drop_rate):
        super().__init__()
        
        # 1) Equivalent to LayerNormalization(epsilon=1e-6)
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # 2) MultiHeadAttention
        #
        #   In Keras, we do: MultiHeadAttention(num_heads, key_dim=embed_dim, dropout=attn_drop_rate)
        #   In PyTorch, nn.MultiheadAttention expects total embed_dim (not just the key_dim), 
        #   so we give embed_dim here. We'll set 'batch_first=True' so input is (B, L, E).
        #
        #   The "kernel_initializer=TruncatedNormal(stddev=0.02)" we will replicate below
        #   with manual weight initialization.
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=attn_drop_rate,
            batch_first=True
        )
        
        # 3) We'll skip adding here since we do x + attn in forward.
        
        # 4) A second LN, which is not actually used in feed-forward in your Keras code,
        #    but we include it for an exact replication of the lines of code.
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)

        # 5) -> 8) Dense(mlp_dim) + Dropout + Dense(embed_dim) + Dropout
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.dropout1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout2 = nn.Dropout(drop_rate)

    def forward(self, x):
        """
        x: Tensor shape (B, L, E)
        """

        # Step 1) LN - though in Keras code, the LN output isn't used as MHA input.
        # We'll still compute it for correctness in replication.
        y_normed = self.ln1(x)  
        
        # Step 2) MHA with (query=x, key=x, value=x).
        #   Notice the Keras code passes "query=x, value=x, key=x" ignoring the LN result.
        attn_output, _ = self.mha(x, x, x)  # ignoring y_normed as Keras does

        # Step 3) Add skip connection
        res = x + attn_output

        # Step 4) This LN is computed but not actually used for anything in feed-forward
        _ = self.ln2(res)  # Keras code does this but never uses it.

        # Step 5..8) The MLP (Dense -> Dropout -> Dense -> Dropout)
        #            Notice Keras does Dense(mlp_dim) on "res", not on LN(res).
        y = self.fc1(res)
        y = self.dropout1(y)
        y = self.fc2(y)
        y = self.dropout2(y)

        # Step 9) Another skip connection
        out = res + y
        return out


class TransformerModel(nn.Module):
    """
    Replicates the 'transformer' function from Keras.
    
    1) input = (B, length, channels)
    2) First Dense->ReLU->Dropout to project from channels -> embed_dim
    3) num_layers times of the EncoderBlock
    4) GlobalAveragePooling1D (mean across the length dimension)
    5) Dense(32)->ReLU->Dropout
    6) Dense(16)->ReLU->Dropout
    7) Dense(1) with kernel_initializer='zeros', activation='sigmoid'
    """
    def __init__(
        self, 
        length, 
        channels, 
        num_layers, 
        embed_dim, 
        mlp_dim, 
        num_heads, 
        dropout_rate, 
        attention_dropout_rate
    ):
        super().__init__()
        
        # 2) "Dense(embed_dim, kernel_initializer=TruncatedNormal(0.02), activation='relu')"
        self.input_linear = nn.Linear(channels, embed_dim)
        self.input_activation = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout_rate)

        # 3) Stack of Encoders
        self.encoders = nn.ModuleList([
            EncoderBlock(embed_dim, mlp_dim, num_heads, attention_dropout_rate, dropout_rate)
            for _ in range(num_layers)
        ])

        # 4) Instead of GlobalAveragePooling1D(data_format='channels_last'), 
        #    we'll do a simple mean over dimension=1 (the length dimension).
        #    (No trainable parameters, so no special module needed.)
        
        # 5) Dense(32)->ReLU->Dropout
        self.linear1 = nn.Linear(embed_dim, 32)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        # 6) Dense(16)->ReLU->Dropout
        self.linear2 = nn.Linear(32, 16)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        # 7) Dense(1, kernel_initializer="zeros", activation='sigmoid')
        self.linear_out = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

        # ---- Initialization to replicate Keras's TruncatedNormal(stddev=0.02) + zeros ----
        # Note: Keras sets final Dense kernel_initializer="zeros". We'll do that manually.
        self._init_weights_keras_style()

    def _init_weights_keras_style(self):
        """Replicates the truncated normal init with std=0.02 and sets final layer to zeros."""
        for m in self.modules():
            # If it's a Linear layer, do truncated normal (std=0.02)
            if isinstance(m, nn.Linear):
                if m is self.linear_out:
                    # The final output layer must be all zeros:
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                else:
                    # Truncated normal for everything else
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
            # If it's a MultiheadAttention, do truncated normal for in_proj_weight/out_proj.weight
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.trunc_normal_(m.in_proj_weight, std=0.02)
                nn.init.zeros_(m.in_proj_bias)
                nn.init.trunc_normal_(m.out_proj.weight, std=0.02)
                nn.init.zeros_(m.out_proj.bias)

    def forward(self, x):
        """
        x shape: (B, length, channels)
        """
        # 2) First Dense->ReLU->Dropout
        x = self.input_linear(x)
        x = self.input_activation(x)
        x = self.input_dropout(x)

        # 3) Pass through stacked encoders
        for encoder_block in self.encoders:
            x = encoder_block(x)

        # 4) GlobalAveragePooling1D over the "length" dimension => x.mean(dim=1)
        x = x.mean(dim=1)  # (B, embed_dim)

        # 5) Dense(32)->ReLU->Dropout
        x = self.linear1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        # 6) Dense(16)->ReLU->Dropout
        x = self.linear2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        # 7) Dense(1) w/ zero init -> Sigmoid
        x = self.linear_out(x)
        x = self.sigmoid(x)
        return x

