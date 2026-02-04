import torch
import torch.nn as nn
from models.recon_net import ReconNet

### Simplse DNN fusion
import torch.nn.functional as F

class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(198 * 2, 256)  # Combining two input vectors along the last dimension
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # Output single value for each vector at dim 1
        self.dropout = nn.Dropout(0.3)  # Optional dropout for regularization
    
    def forward(self, x1, x2):
        # Concatenate along the last dimension
        x = torch.cat((x1, x2), dim=-1)  # Size: batch x 1024 x (198*2)
        
        # Apply fully connected layers
        x = self.fc1(x)  # Size: batch x 1024 x 256
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # Size: batch x 1024 x 128
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)  # Size: batch x 1024 x 1
        x = x.squeeze(-1)  # Remove the last dimension to make size batch x 1024
        
        return x


### AFF ###

class AFF(nn.Module):

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


###########

class MultiScaleChannelAttention(nn.Module):
    def __init__(self, channels, reduction=1):
        super(MultiScaleChannelAttention, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.local_conv2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global context
        global_context = self.global_pool(x)
        global_context = self.local_conv1(global_context)
        global_context = F.relu(global_context, inplace=True)
        global_context = self.local_conv2(global_context)
        
        # Local context (point-wise convolution)
        local_context = F.relu(self.local_conv1(x), inplace=True)
        local_context = self.local_conv2(local_context)

        # Combine global and local contexts
        attention = self.sigmoid(global_context + local_context)
        return attention

class AttentionalFeatureFusion(nn.Module):
    def __init__(self, channels, reduction=16):
        super(AttentionalFeatureFusion, self).__init__()
        self.attention_module = MultiScaleChannelAttention(channels, reduction)

    def forward(self, x, y):
        # Initial feature integration (element-wise summation)
        initial_integration = x + y

        # Generate attention weights
        attention_weights = self.attention_module(initial_integration)

        # Weighted combination of features
        fused_features = attention_weights * x + (1 - attention_weights) * y
        return fused_features


class MultiHeadCrossAttention2D(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.1):
        super(MultiHeadCrossAttention2D, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        
        # Linear layers for Q, K, V projections
        self.query_proj = nn.Linear(embed_dim, hidden_dim)
        self.key_proj = nn.Linear(embed_dim, hidden_dim)
        self.value_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Linear layer for output projection
        self.out_proj = nn.Linear(hidden_dim, embed_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = self.hidden_dim ** 0.5
        
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Trainable scaling factor
        self.beta = nn.Parameter(torch.tensor(0.5))   # Trainable scaling factor

        self._initialize_proj()

    def _initialize_proj(self):
        nn.init.eye_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.eye_(self.key_proj.weight)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.eye_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.eye_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, head_dim) and transpose for attention.
        Input: [batch, seq_len, hidden_dim]
        Output: [batch, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, hidden_dim = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]

    def combine_heads(self, x):
        """
        Combine the num_heads and head_dim into the last dimension.
        Input: [batch, num_heads, seq_len, head_dim]
        Output: [batch, seq_len, hidden_dim]
        """
        batch_size, num_heads, seq_len, head_dim = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch, seq_len, num_heads, head_dim]
        return x.view(batch_size, seq_len, num_heads * head_dim)  # [batch, seq_len, hidden_dim]

    def forward(self, vector1, vector2, mask=None):
        """
        vector1: [batch, seq_len, embed_dim] - Query
        vector2: [batch, seq_len, embed_dim] - Key/Value
        """
        # Project inputs to Q, K, V
        Q = self.query_proj(vector1)  # [batch, seq_len, hidden_dim]
        K = self.key_proj(vector2)   # [batch, seq_len, hidden_dim]
        V = self.value_proj(vector2) # [batch, seq_len, hidden_dim]
        
        # Split into heads
        Q = self.split_heads(Q)  # [batch, num_heads, seq_len, head_dim]
        K = self.split_heads(K)  # [batch, num_heads, seq_len, head_dim]
        V = self.split_heads(V)  # [batch, num_heads, seq_len, head_dim]

        # Compute attention scores
        Q = Q / self.scale 
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, num_heads, seq_len, seq_len]
        
        # Apply optional mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax and dropout
        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True)[0]  # Stabilize softmax
        attn_weights = self.softmax(attn_scores)  # [batch, num_heads, seq_len, seq_len]
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_len, head_dim]
        
        # Combine heads
        attn_output = self.combine_heads(attn_output)  # [batch, seq_len, hidden_dim]
        
        # Adjust vector1 and combine
        adjusted_vector1 = self.out_proj(vector1)  # [batch, seq_len, embed_dim]
        output = self.alpha * attn_output + self.beta * adjusted_vector1
        
        return output
    
class CrossAttention2D(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(CrossAttention2D, self).__init__()
        # Linear layers for Q, K, V projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        # Linear layer for output projection
        self.out_proj = nn.Linear(embed_dim, hidden_dim)
         # Custom initialization

        self._initialize_proj()

    def _initialize_proj(self):
        # Initialize query_proj to be identity-like
        nn.init.eye_(self.query_proj.weight)  # Identity mapping
        nn.init.zeros_(self.query_proj.bias)  # No initial bias

        # Initialize key_proj and value_proj to learn features gradually
        nn.init.normal_(self.key_proj.weight, mean=0, std=0.01)  # Small random values
        nn.init.zeros_(self.key_proj.bias)

        nn.init.normal_(self.value_proj.weight, mean=0, std=0.01)  # Small random values
        nn.init.zeros_(self.value_proj.bias)

        nn.init.eye_(self.out_proj.weight) # Identity mapping
        nn.init.zeros_(self.out_proj.bias)


    def forward(self, vector1, vector2):
        """
        vector1: [batch, 198, 1024] - Query
        vector2: [batch, 198, 1024] - Key/Value
        """
        # Project vector1 and vector2 into Q, K, V
        #print(f'vector1 shape: {vector1.shape}')
        Q = self.query_proj(vector2)  # [batch, 198, hidden_dim]
        #print(f'Q shape: {Q.shape}')
        K = self.key_proj(vector1)    # [batch, 198, hidden_dim]
        #print(f'K shape: {K.shape}')
        V = self.value_proj(vector2)  # [batch, 198, hidden_dim]
        #print(f'V shape: {V.shape}')
        # Compute attention scores
        # QK^T -> [batch, 198, 198]
        #Q = Q / (self.hidden_dim ** 0.5)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        #print(f'attn_scores shape: {attn_scores.shape}')
        # Apply softmax to get attention weights
        attn_weights = self.softmax(attn_scores) # [batch, 198, 198]
        #print(f'attn_weights shape: {attn_weights.shape}')
        # Use attention weights to compute weighted sum of V
        # [batch, 198, hidden_dim]
        # Compute (I - attn_weights)
        
        batch_size, seq_len, _ = attn_weights.shape
        identity_matrix = torch.eye(seq_len, device=attn_weights.device).unsqueeze(0).expand(batch_size, -1, -1)
        I_minus_attn = identity_matrix - attn_weights  # [batch, 198, 198]

        #adjusted_vector1 = torch.matmul(I_minus_attn, self.out_proj(vector1))  # [batch, 198, 1024]
        adjusted_vector1 = torch.matmul(I_minus_attn, vector1)  # [batch, 198, 1024]

        
        output = torch.matmul(attn_weights, V) +  adjusted_vector1
        output = self.out_proj(output)

        return output



class FeatureAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeatureAttention, self).__init__()
        # Linear layers for Q, K, V projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.epsilon = 1e-6
        self.alpha = nn.Parameter(torch.normal(0, 0.01, size=(198,)))

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        # Linear layer for output projection
        self.out_proj = nn.Linear(embed_dim, hidden_dim)
        self.scale = 1024
        # Custom initialization
        self._initialize_proj()

    def _initialize_proj(self):
        # Initialize query_proj to be identity-like
        nn.init.eye_(self.query_proj.weight)  # Identity mapping
        nn.init.zeros_(self.query_proj.bias)  # No initial bias

        nn.init.eye_(self.key_proj.weight)  # Identity mapping
        nn.init.zeros_(self.key_proj.bias)  # No initial bias

        nn.init.eye_(self.out_proj.weight) # Identity mapping
        nn.init.zeros_(self.out_proj.bias)


    def forward(self, vector1, vector2):
        """
        vector1: [batch, 198, 1024] - Query
        vector2: [batch, 198, 1024] - Key/Value
        """
        # Project vector1 and vector2 into Q, K, V
        batch, _, _ = vector1.shape
        Q = self.query_proj(vector2)  # [batch, 198, hidden_dim]
        ## Attention
        vector_sub = torch.abs(Q - vector1) 
        sum_vector = torch.sum(vector_sub, -2, keepdim=False) 
        div_vector = sum_vector/ ( torch.sum(torch.abs(vector1), -2, keepdim=False)+ self.epsilon )
        attn_weights = ( div_vector / (self.scale ) )
        attn_weights = attn_weights * self.alpha
        attn_weights_expand = attn_weights.unsqueeze(1)
        # Expand to [24, 198, 1024]
        attn_weights_expand = attn_weights_expand.repeat(1, 1024, 1)
        ## multiply with weights
        Addition1 = torch.mul(attn_weights_expand , Q)
        #print(f'Addition1 shape: {Addition1.shape}')
        Addition2 = torch.mul(torch.ones_like(attn_weights_expand)-attn_weights_expand, vector1)
        #print(f'Addition2 shape: {Addition2.shape}')

        return Addition1 + Addition2

class ViTfuser(nn.Module):
    def __init__(self, net, epsilon=1e-6):
        super().__init__()
        self.device = 'cuda:3'

        # ViT layer
        self.recon_net = ReconNet(net).to(self.device)
        self.recon_net_ref = ReconNet(net).to(self.device)

        # Load weights
        cp = torch.load('../checkpoints_trained_start/model_100.pt', map_location=self.device) 
        self.recon_net.load_state_dict(cp['model'])
        self.recon_net_ref.load_state_dict(cp['model'])
        
        # Fusion layers
        self.epsilon = epsilon
        self.param1 = nn.Parameter(torch.normal(0.85, 0.00, size=(198,)))
        self.param2 = nn.Parameter(torch.normal(0.15, 0.05, size=(198,)))


        #self.cross_attn = MultiHeadCrossAttention2D(198, 198, 6)

    def printer(self, x):
        print("Current value of param1 during forward:", self.param1)
        return

    def forward(self, img,ref): 

        in_pad, wpad, hpad = self.recon_net.pad(img)
        ref_pad, wpad, hpad = self.recon_net.pad(ref)
        input_norm,mean,std = self.recon_net.norm(in_pad.float())
        ref_norm,mean_ref,std_ref = self.recon_net.norm(ref_pad.float())
       
        # Feature extract
        features = self.recon_net.net.forward_features(input_norm)
        
        
        features_ref = self.recon_net_ref.net.forward_features(ref_norm)#.permute(0,2,1)

        ########## Feaute fusion start ##########
        batch_size, num_channels, height = features.shape
        features_flat = features.reshape(batch_size, num_channels, -1)
        features_ref_flat = features_ref.reshape(batch_size, num_channels, -1)       
        
        # Reshape params to match the dimensions
        param1_expanded = self.param1.reshape(1, -1, 1)  # Shape: [1, 416, 1]
        param2_expanded = self.param2.reshape(1, -1, 1)  # Shape: [1, 416, 1]
        # Expand params to match the flattened tensor dimensions
        param1_expanded = param1_expanded.expand(batch_size, -1, height)  # Shape: [batch_size, 416, height*width]
        param2_expanded = param2_expanded.expand(batch_size, -1, height)  # Shape: [batch_size, 416, height*width]
        # Calculate weighted sum
        
        weighted_sum = (param1_expanded * features_flat + param2_expanded * features_ref_flat)
        
        # Calculate normalization factor
        normalization_factor = param1_expanded + param2_expanded + self.epsilon
        
        # Normalize
        features_comb = weighted_sum / normalization_factor
        
        # Low Resolution
        features_comb = features_comb.reshape(features_flat.shape[0], 198, 1024)
        
        ########## Feaute fusion end ##########
        
        # Recon Head
        head_out = self.recon_net.net.head(features_comb)
        
        # Low Resolution 
        head_out_img = self.recon_net.net.seq2img(head_out, (180, 110))


        # un-norm
        merged = self.recon_net.unnorm(head_out_img, mean, std) 

        # un-pad 
        im_out = self.recon_net.unpad(merged,wpad,hpad)
        
        return im_out