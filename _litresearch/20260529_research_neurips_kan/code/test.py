import torch
import torch.nn as nn
import torch.optim as optim
# from torch.nn.functional import interpolate
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Define the target function ---
def f_feynman(theta, sigma=5.0):
    return np.exp(-theta**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)

# --- 2. Generate training data ---
x = np.linspace(-15, 15, 500)
y_true = f_feynman(theta=x)
# Convert to torch tensors
x_train = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_true, dtype=torch.float32).unsqueeze(1)

# --- 3. Define a simple MLP with ReLU ---
class MLPReLU(nn.Module):
    def __init__(self, hidden_size=20):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        return self.model(x)

# --- 4. Define a simple RBF layer ---
class RBFLayer(nn.Module):
    def __init__(self, in_features, out_features, centers=None, gamma=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gamma = gamma
        if centers is None:
            # initialize centers evenly in input space
            self.centers = nn.Parameter(torch.linspace(-15, 15, out_features).unsqueeze(1))
        else:
            self.centers = nn.Parameter(centers)
        self.beta = nn.Parameter(torch.ones(out_features) * gamma)
        
    def forward(self, x):
        # x: [batch, in_features], centers: [out_features, in_features]
        size = (x.size(0), self.out_features, self.in_features)
        x_expanded = x.unsqueeze(1).expand(size)
        c_expanded = self.centers.unsqueeze(0).expand(size)
        # Gaussian RBF
        out = torch.exp(-self.beta.unsqueeze(0) * (x_expanded - c_expanded)**2)
        return out.sum(dim=2)  # sum over input features
# RBF MLP (1 hidden RBF layer)
class MLPRBF(nn.Module):
    def __init__(self, hidden_size=20):
        super().__init__()
        self.rbf = RBFLayer(1, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        return self.linear(self.rbf(x))

# --- 5. Define a simple KAN NN ---

class KAN3Layer(nn.Module):
    def __init__(self, n_splines=20, x_min=-15.0, x_max=15.0, hidden_sizes=[10, 5]):
        """
        3-layer KAN for 1D input → 1D output.
        hidden_sizes: list with two integers [hidden_layer1_size, hidden_layer2_size]
        Each layer has univariate spline neurons.
        Layer3 is a linear combination of Layer2 outputs.
        """
        super().__init__()
        self.n_splines = n_splines
        self.x_min = x_min
        self.x_max = x_max
        
        # --- Layer 1: spline per neuron ---
        self.layer1 = nn.ModuleList([self._make_spline() for _ in range(hidden_sizes[0])])
        
        # --- Layer 2: spline per neuron ---
        self.layer2 = nn.ModuleList([self._make_spline() for _ in range(hidden_sizes[1])])
        
        # --- Layer 3: final linear combination ---
        self.linear_out = nn.Linear(hidden_sizes[1], 1)
        
    def _make_spline(self):
        # Create a learnable spline as a ParameterList of coefficients
        coeffs = nn.Parameter(torch.randn(self.n_splines))
        return nn.ParameterList([coeffs])
    
    def _spline_forward(self, x, coeffs):
        """
        Linear interpolation of spline coefficients.
        x: [batch,1]
        coeffs: [n_splines]
        """
        # Normalize x to [0,1]
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)
        x_norm = torch.clamp(x_norm, 0.0, 1.0)
        
        # Map normalized x to indices
        idx = x_norm * (self.n_splines - 1)
        idx_low = torch.clamp(idx.floor().long(), 0, self.n_splines - 2)
        idx_high = idx_low + 1
        
        w_high = idx - idx_low.float()
        w_low = 1 - w_high
        
        y_out = coeffs[idx_low.squeeze()] * w_low.squeeze() + coeffs[idx_high.squeeze()] * w_high.squeeze()
        return y_out.unsqueeze(1)
    
    def forward(self, x):
        # --- Layer 1 ---
        layer1_out = []
        for spline in self.layer1:
            y = self._spline_forward(x, spline[0])
            layer1_out.append(y)
        layer1_out = torch.cat(layer1_out, dim=1)  # [batch, hidden1]
        
        # --- Layer 2 ---
        layer2_out = []
        # Apply each Layer2 spline to corresponding Layer1 neuron output
        for i, spline in enumerate(self.layer2):
            # pick i-th neuron output from layer1_out
            x_in = layer1_out[:, i:i+1]  # shape [batch,1]
            y = self._spline_forward(x_in, spline[0])
            layer2_out.append(y)
        layer2_out = torch.cat(layer2_out, dim=1)  # [batch, hidden2]
        
        # --- Layer 3 ---
        out = self.linear_out(layer2_out)  # [batch,1]
        return out

# --- 5. Training function ---
def train(model, x_train, y_train, lr=0.01, epochs=5000):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f"\tEpoch {epoch}, Loss={loss.item():.10f}")
    return model

def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
# def get_loss(model, x_train, y_train):
#     # Suppose x_train and y_train are your tensors
#     criterion = nn.MSELoss()  # Mean Squared Error
#     # Forward pass: compute model predictions
#     y_pred = model(x_train)  # shape [batch, 1]
#     # Compute loss
#     loss = criterion(y_pred, y_train)
#     return loss.item()

def get_loss(model, x, y):
    """
    Compute MSE loss of a model on given x, y.
    Automatically converts numpy arrays to torch tensors if needed.
    Returns a float.
    """
    # Convert to torch tensors if not already
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1) if x.ndim == 1 else torch.tensor(x, dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y.ndim == 1 else torch.tensor(y, dtype=torch.float32)

    # Ensure tensors are float32
    x = x.float()
    y = y.float()
    
    # Compute predictions
    y_pred = model(x)
    
    # Compute MSE loss
    criterion = nn.MSELoss()
    loss = criterion(y_pred, y)
    
    # Return as float
    return loss.item()

# --- 6. Initialize and train models ---

print("Training KAN...")
kan_model = KAN3Layer(n_splines=20)
kan_model = train(kan_model, x_train, y_train)
print("")
kan_model_simple = KAN3Layer(n_splines=10)
kan_model_simple = train(kan_model_simple, x_train, y_train)
print("")

print("Training ReLU...")
mlp_relu = MLPReLU(hidden_size=20)
mlp_relu = train(mlp_relu, x_train, y_train)
print("")

print("Training RBF...")
mlp_rbf = MLPRBF(hidden_size=20)
mlp_rbf = train(mlp_rbf, x_train, y_train)
print("")




# --- 7. FROM MLP TO KAN: Initialize and train models  ---

import torch
import numpy as np
from scipy.interpolate import CubicSpline

def mlp_to_kan(mlp_model, n_splines=20, x_min=-15.0, x_max=15.0, hidden_sizes=[20, 20]):
    """
    Convert a trained MLPReLU to a 3-layer KAN3Layer.
    
    hidden_sizes: [hidden1, hidden2] sizes of MLP hidden layers
    """
    # 1. Create KAN model
    kan_model = KAN3Layer(n_splines=n_splines, x_min=x_min, x_max=x_max, hidden_sizes=[hidden_sizes[0], hidden_sizes[1]])
    
    # 2. Define fine input grid
    x_grid = np.linspace(x_min, x_max, 500)
    
    # --- Layer 1: convert MLP layer1 neurons to KAN splines ---
    layer1_weights = mlp_model.model[0].weight.detach().numpy()  # [hidden1,1]
    layer1_bias = mlp_model.model[0].bias.detach().numpy()       # [hidden1]
    
    for i in range(hidden_sizes[0]):
        # Evaluate neuron output on grid: w*x + b, then ReLU
        neuron_out = np.maximum(0, layer1_weights[i] * x_grid + layer1_bias[i])
        # Fit spline
        spline = CubicSpline(x_grid, neuron_out)
        knots = np.linspace(x_min, x_max, n_splines)
        kan_model.layer1[i][0].data = torch.tensor(spline(knots), dtype=torch.float32)
    
    # --- Layer 2: convert MLP layer2 neurons to KAN splines ---
    layer2_weights = mlp_model.model[2].weight.detach().numpy()  # [hidden2, hidden1]
    layer2_bias = mlp_model.model[2].bias.detach().numpy()       # [hidden2]
    
    # Compute Layer1 outputs on grid for all neurons
    # shape [hidden1, grid]
    layer1_outputs = layer1_weights @ x_grid.reshape(1,-1)  # matrix multiplication
    layer1_outputs += layer1_bias[:, None]                  # add bias per neuron
    layer1_outputs = np.maximum(0, layer1_outputs)          # ReLU
    
    for j in range(hidden_sizes[1]):
        # Weighted sum of Layer1 neuron outputs + bias
        neuron_input = np.dot(layer2_weights[j], layer1_outputs) + layer2_bias[j]
        neuron_out = np.maximum(0, neuron_input)  # ReLU
        spline = CubicSpline(x_grid, neuron_out)
        knots = np.linspace(x_min, x_max, n_splines)
        kan_model.layer2[j][0].data = torch.tensor(spline(knots), dtype=torch.float32)
    
    # --- Layer 3: linear output weights ---
    linear_weights = mlp_model.model[4].weight.detach().numpy().flatten()  # shape [1, hidden2]
    linear_bias = mlp_model.model[4].bias.detach().numpy()
    kan_model.linear_out.weight.data = torch.tensor(linear_weights.reshape(1, -1), dtype=torch.float32)
    kan_model.linear_out.bias.data = torch.tensor(linear_bias, dtype=torch.float32)
    
    return kan_model

kan_from_mlp = mlp_to_kan(mlp_relu, n_splines=20, hidden_sizes=[20,20])
# Fine-tune the KAN
kan_from_mlp_fine = train(kan_from_mlp, x_train, y_train, lr=0.01, epochs=2000)



# --- 8. FROM MLP TO KAN: Initialize and train models  ---
def compress_kan(kan_model, x_grid, top_k_layer1=None, top_k_layer2=None, n_splines_new=None):
    """
    Compress a KAN3Layer by:
    1. Pruning neurons with low contribution
    2. Reducing spline resolution
    
    Parameters:
    - kan_model: trained KAN3Layer
    - x_grid: numpy array of input values to evaluate neuron outputs
    - top_k_layer1: int or None, number of Layer1 neurons to keep
    - top_k_layer2: int or None, number of Layer2 neurons to keep
    - n_splines_new: int or None, new number of spline knots (reduces coefficients)
    
    Returns:
    - compressed KAN3Layer
    """
    # Copy old model parameters
    old_layer1 = kan_model.layer1
    old_layer2 = kan_model.layer2
    old_linear = kan_model.linear_out
    
    hidden1 = len(old_layer1)
    hidden2 = len(old_layer2)
    
    # --- 1. Evaluate contributions of Layer1 neurons ---
    layer1_outputs = []
    for i, spline in enumerate(old_layer1):
        y = kan_model._spline_forward(torch.tensor(x_grid, dtype=torch.float32).unsqueeze(1), spline[0])
        layer1_outputs.append(y.detach().numpy().flatten())
    layer1_outputs = np.array(layer1_outputs)  # shape [hidden1, len(x_grid)]
    contributions1 = np.max(np.abs(layer1_outputs), axis=1)
    if top_k_layer1 is None:
        top_k_layer1 = hidden1
    keep_idx1 = np.argsort(contributions1)[-top_k_layer1:]  # keep top-k
    
    # --- 2. Evaluate contributions of Layer2 neurons ---
    layer2_outputs = []
    for j, spline in enumerate(old_layer2):
        # input is Layer1 outputs
        x_in = torch.tensor(layer1_outputs.T[:, :], dtype=torch.float32)  # shape [batch, hidden1]
        # pick corresponding neuron for simplicity
        neuron_input = x_in[:, j % hidden1:j % hidden1 + 1]  # simple mapping
        y = kan_model._spline_forward(neuron_input, spline[0])
        layer2_outputs.append(y.detach().numpy().flatten())
    layer2_outputs = np.array(layer2_outputs)
    contributions2 = np.max(np.abs(layer2_outputs), axis=1)
    if top_k_layer2 is None:
        top_k_layer2 = hidden2
    keep_idx2 = np.argsort(contributions2)[-top_k_layer2:]
    
    # --- 3. Create new compressed KAN ---
    new_hidden_sizes = [top_k_layer1, top_k_layer2]
    n_splines_new = n_splines_new or kan_model.n_splines
    compressed_kan = KAN3Layer(n_splines=n_splines_new, hidden_sizes=new_hidden_sizes,
                               x_min=kan_model.x_min, x_max=kan_model.x_max)
    
    # --- 4. Copy retained spline coefficients (resampled if needed) ---
    # Layer1
    new_knots = np.linspace(kan_model.x_min, kan_model.x_max, n_splines_new)
    for idx_new, idx_old in enumerate(keep_idx1):
        old_knots = np.linspace(kan_model.x_min, kan_model.x_max, kan_model.n_splines)
        old_coeffs = old_layer1[idx_old][0].detach().numpy()
        # Resample spline to new knots
        spline = np.interp(new_knots, old_knots, old_coeffs)
        compressed_kan.layer1[idx_new][0].data = torch.tensor(spline, dtype=torch.float32)
    
    # Layer2
    for idx_new, idx_old in enumerate(keep_idx2):
        old_knots = np.linspace(kan_model.x_min, kan_model.x_max, kan_model.n_splines)
        old_coeffs = old_layer2[idx_old][0].detach().numpy()
        spline = np.interp(new_knots, old_knots, old_coeffs)
        compressed_kan.layer2[idx_new][0].data = torch.tensor(spline, dtype=torch.float32)
    
    # --- 5. Copy linear_out weights for retained Layer2 neurons ---
    compressed_kan.linear_out.weight.data = old_linear.weight.data[:, keep_idx2].clone()
    compressed_kan.linear_out.bias.data = old_linear.bias.data.clone()
    
    return compressed_kan
# Define input grid for evaluating neuron contributions
x_grid = np.linspace(-15, 15, 500)

# Keep top 5 neurons in Layer1, top 3 in Layer2, reduce splines to 10 knots
kan_compressed = compress_kan(kan_from_mlp, x_grid, top_k_layer1=5, top_k_layer2=3, n_splines_new=10)

# Fine-tune compressed KAN
kan_compressed = train(kan_compressed, x_train, y_train, lr=0.01, epochs=1000)



# --- 9. Plot results with losses in legend ---
# --- Get number of parameters ---
params_relu = get_num_params(mlp_relu)
params_rbf = get_num_params(mlp_rbf)
params_kan = get_num_params(kan_model)
params_kan_simple = get_num_params(kan_model_simple)
params_kan_from_mlp = get_num_params(kan_from_mlp)
params_kan_from_mlp_fine = get_num_params(kan_from_mlp_fine)
params_kan_compressed = get_num_params(kan_compressed)
# --- Compute losses ---
loss_relu = get_loss(mlp_relu, x_train, y_train)
loss_rbf = get_loss(mlp_rbf, x_train, y_train)
loss_kan = get_loss(kan_model, x_train, y_train)
loss_kan_simple = get_loss(kan_model_simple, x_train, y_train)
loss_kan_from_mlp = get_loss(kan_from_mlp, x_train, y_train)
loss_kan_from_mlp_fine = get_loss(kan_from_mlp_fine, x_train, y_train)
loss_kan_compressed = get_loss(kan_compressed, x_train, y_train)

# --- Compute predictions ---
y_pred_relu = mlp_relu(x_train).detach().numpy().flatten()
y_pred_rbf = mlp_rbf(x_train).detach().numpy().flatten()
y_pred_kan = kan_model(x_train).detach().numpy().flatten()
y_pred_kan_simple = kan_model_simple(x_train).detach().numpy().flatten()
y_pred_kan_from_mlp = kan_from_mlp(x_train).detach().numpy().flatten()
y_pred_kan_from_mlp_fine = kan_from_mlp_fine(x_train).detach().numpy().flatten()
y_pred_kan_compressed = kan_compressed(x_train).detach().numpy().flatten()

# --- Plot figure ---
plt.figure(figsize=(10,6))
plt.plot(x, y_true, label="f(x) true", color="black", linewidth=2)
plt.plot(x, y_pred_relu, label=f"MLP ReLU (MSE={loss_relu:.2e}, params={params_relu})", color="cyan", linestyle="--")
plt.plot(x, y_pred_rbf, label=f"MLP RBF (MSE={loss_rbf:.2e}, params={params_rbf})", color="red", linestyle=":")
plt.plot(x, y_pred_kan, label=f"KAN (MSE={loss_kan:.2e}, params={params_kan})", color="pink", linestyle=":")
plt.plot(x, y_pred_kan_simple, label=f"KAN simple (MSE={loss_kan_simple:.2e}, params={params_kan_simple})", color="green", linestyle=":")
plt.plot(x, y_pred_kan_from_mlp, label=f"KAN from MLP (MSE={loss_kan_from_mlp:.2e}, params={params_kan_from_mlp})", color="blue", linestyle="-.")
plt.plot(x, y_pred_kan_from_mlp_fine, label=f"KAN from MLP fine-tuned (MSE={loss_kan_from_mlp_fine:.2e}, params={params_kan_from_mlp_fine})", color="orange", linestyle="-.")
plt.plot(x, y_pred_kan_compressed, label=f"KAN compressed (MSE={loss_kan_compressed:.2e}, params={params_kan_compressed})", color="purple", linestyle="--")
plt.xlabel("theta")
plt.ylabel("f(theta)")
plt.title("Function Approximation: Feynman Equation")
plt.legend()
plt.grid(True)
plt.show()