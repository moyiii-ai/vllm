import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from deepspeed.moe.layer import MoE
import json

# --------------------------
# Expert network (MLP)
# --------------------------
class Expert(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# --------------------------
# Simple Transformer block
# --------------------------
class TransformerBlock(nn.Module):
    def __init__(self, input_size, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(input_size, input_size * 4),
            nn.ReLU(),
            nn.Linear(input_size * 4, input_size)
        )
        self.ln1 = nn.LayerNorm(input_size)
        self.ln2 = nn.LayerNorm(input_size)

    def forward(self, x):
        # Self-attention
        x2, _ = self.attn(x, x, x)
        x = self.ln1(x + x2)
        # Feed-forward
        x = self.ln2(x + self.ff(x))
        return x

# --------------------------
# Mixed MoE + Transformer model
# --------------------------
class MixedModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts, top_k, ep_size,
                 num_moe_layers=2, num_transformer_layers=1, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        # Add MoE layers
        for _ in range(num_moe_layers):
            self.layers.append(MoE(input_size, Expert(input_size, hidden_size),
                                   num_experts=num_experts,
                                   ep_size=ep_size,
                                   k=top_k))
        # Add Transformer layers
        for _ in range(num_transformer_layers):
            self.layers.append(TransformerBlock(input_size, num_heads))

    def forward(self, x):
        aux_loss_total = 0.0
        for layer in self.layers:
            out = layer(x)
            # MoE layers return (output, aux_loss)
            if isinstance(out, tuple):
                x = out[0]
                aux_loss_layer = out[1]
                aux_loss_total += aux_loss_layer
            else:
                # Transformer layer returns only output
                x = out
        return x, aux_loss_total if aux_loss_total != 0 else None

def main():
    # --------------------------
    # Hyperparameters
    # --------------------------
    input_size = 64
    hidden_size = 128
    num_experts = 8
    top_k = 2
    batch_size = 16
    num_moe_layers = 2
    num_transformer_layers = 0
    num_heads = 4
    num_steps = 5  # simple training steps

    # --------------------------
    # Load DeepSpeed configuration
    # --------------------------
    with open("ds_config.json", "r") as f:
        ds_config = json.load(f)
    ep_size = ds_config["moe"]["expert_model_parallel_size"]

    # --------------------------
    # Initialize model
    # --------------------------
    model = MixedModel(input_size, hidden_size, num_experts, top_k, ep_size,
                       num_moe_layers, num_transformer_layers, num_heads)

    # --------------------------
    # Initialize DeepSpeed
    # --------------------------
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        training_data=None
    )
    model_engine.train()

    # --------------------------
    # Loss function
    # --------------------------
    criterion = nn.MSELoss()

    # --------------------------
    # Training loop
    # --------------------------
    for step in range(num_steps):
        input_data = torch.randn(batch_size, input_size).to(model_engine.device)
        target = torch.randn(batch_size, input_size).to(model_engine.device)

        output, aux_loss = model_engine(input_data)
        loss = criterion(output, target)
        if aux_loss is not None:
            loss += aux_loss

        model_engine.backward(loss)
        model_engine.step()

        if model_engine.global_rank == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | Aux Loss: {aux_loss.item() if aux_loss is not None else 'None'}")

    # --------------------------
    # Print model info
    # --------------------------
    if model_engine.global_rank == 0:
        print(f"Expert parallel size: {model_engine.module.layers[0].ep_size}")
        print(f"Number of local experts: {model_engine.module.layers[0].num_local_experts}")
        print(f"Number of total experts: {model_engine.module.layers[0].num_experts}")
        print(f"Number of layers: {len(model_engine.module.layers)}")

if __name__ == "__main__":
    main()
