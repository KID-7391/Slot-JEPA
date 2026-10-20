import torch
import math
import torch.nn as nn

class ContinuousPosEmb(nn.Module):
    def __init__(self, d_model, num_freqs=8):
        super().__init__()
        self.d_model = d_model
        self.num_freqs = num_freqs
        self.proj = nn.Linear(3 * 2 * num_freqs, d_model)
        
    def forward(self, coords):
        B, N, _ = coords.shape # coords: [B, N, 3]
        device = coords.device
        freqs = (2.0 ** torch.arange(self.num_freqs, dtype=torch.float32, device=device)) * math.pi
        scaled = coords.unsqueeze(-1) * freqs.view(1, 1, 1, self.num_freqs)
        emb = torch.stack([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        emb = emb.view(B, N, -1)
        return self.proj(emb)

class StateEncoder(nn.Module):
    def __init__(self, d_model=768, n_heads=8, n_layers=4, state_dim=773):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        
        self.state_proj = nn.Linear(state_dim, d_model)  
        self.pos_emb = nn.Linear(3, d_model)  # x, y, t
        
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        
        self.vjepa_proj = nn.Linear(768, d_model)
        self.out_proj = nn.Linear(d_model, state_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        
    def forward(self, states, vjepa_features, pos_emb_inputs):
        states_emb = self.state_proj(states)
        pos = self.pos_emb(pos_emb_inputs)
        
        x = states_emb + pos
        vjepa_emb = self.vjepa_proj(vjepa_features)
        
        for sa, ca in zip(self.self_attn_layers, self.cross_attn_layers):
            x = sa(x)
            x_ca, _ = ca(query=x, key=vjepa_emb, value=vjepa_emb)
            x = x + x_ca

        return self.out_proj(x) + states

class StatePredictor(nn.Module):
    def __init__(self, d_model=768, n_heads=8, n_layers=4, state_dim=773):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        
        self.pos_emb = nn.Linear(3, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(d_model, state_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        
    def forward(self, masked_pos_emb, unmasked_states):
        B, N_unmasked, _ = unmasked_states.shape
        
        unmasked_emb = self.state_proj(unmasked_states)
        masked_emb = self.pos_emb(masked_pos_emb)
        
        x = torch.cat([unmasked_emb, masked_emb], dim=1)
        
        for layer in self.layers:
            x = layer(x)
            
        predicted_masked_states = x[:, N_unmasked:]
        return self.out_proj(predicted_masked_states)
