############################################################################### 
#### IMPORTS ##################################################################
###############################################################################
import torch
import torch.nn as nn
import random
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



############################################################################### 
#### FUNCTIONS ################################################################
###############################################################################

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


############################################################################### 
#### CLASSES ##################################################################
###############################################################################

class ModelTools:
    @staticmethod
    def get_nuber_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def generate_network_mlp(device, layer_dims, activation="relu"):
        return MLP(layer_dims, activation).to(device)

    @staticmethod
    def generate_network_kan(device, layer_dims, grid_size, spline_order):
        return KAN(layer_dims, grid_size, spline_order).to(device)

    @staticmethod
    def mlp_to_kan(mlp, device, grid_size=5, spline_order=1):
        """
        Convert an MLP (ReLU or RBF) to a KAN with same layer_dims and approximate parameter transfer.
        Only works for MLPs built via layer_dims.
        """
        layers = mlp.net
        dims = []
        prev_out = None
        # First recover layer_dims
        for module in layers:
            if isinstance(module, nn.Linear):
                if prev_out is None:
                    dims.append(module.in_features)
                dims.append(module.out_features)
                prev_out = module.out_features

        assert len(dims) >= 2, "Cannot recover MLP structure"
        # Start building KAN net
        kan_layers = [nn.Flatten()]
        # build an initial KAN using the recovered dims (previously used [0] which raised unpack error)
        kan = KAN(layer_dims=dims, grid_size=grid_size, spline_order=spline_order).to(device)
        kan.net = nn.Sequential(*kan_layers)
        # Rebuild each layer with KAN, transferring weights
        linear_layers = [m for m in layers if isinstance(m, nn.Linear)]
        assert len(linear_layers) == len(dims) - 1, "Unexpected MLP structure"
        for i in range(len(linear_layers) - 1):
            lin = linear_layers[i]
            in_dim, out_dim = dims[i], dims[i + 1]
            # Transfer Linear → BasisKANLayer
            kan_layer = transfer_linear_to_kan(
                lin.to(device),
                grid_size=grid_size,
                spline_order=spline_order,
            )
            kan_layers.append(kan_layer)

        # Last linear layer (to output_dim)
        final_lin = linear_layers[-1]
        final_kan = transfer_linear_to_kan(
            final_lin.to(device),
            grid_size=grid_size,
            spline_order=spline_order,
        )
        kan_layers.append(final_kan)
        kan.net = nn.Sequential(*kan_layers)
        return kan
    
    @staticmethod
    def kan_to_mlp(kan, device, activation="relu"):
        """
        Convert a KAN into an MLP with same layer_dims, using parameter‑transfer from KAN layers.
        """
        layers = kan.net
        dims = []
        kan_layers = []
        # Extract layer_dims and BasisKANLayer instances
        for module in layers:
            if isinstance(module, BasisKANLayer):
                if not dims:
                    dims.append(module.in_features)
                dims.append(module.out_features)
                kan_layers.append(module)
        assert len(dims) >= 2, "Cannot recover KAN structure"
        # Build MLP net
        mlp_layers = [nn.Flatten()]
        # For each KAN layer, create a Linear layer with transferred weights
        for i, kan_layer in enumerate(kan_layers):
            if i == len(kan_layers) - 1:
                # last layer, no activation after Linear
                linear = transfer_kan_to_linear(kan_layer, device=device)
                mlp_layers.append(linear)
            else:
                # internal layer: Linear + activation
                linear = transfer_kan_to_linear(kan_layer, device=device)
                mlp_layers.append(linear)
                if activation.lower() == "relu":
                    mlp_layers.append(nn.ReLU())
                elif activation.lower() == "rbf":
                    # RBF is not directly reversible; here we just keep ReLU
                    mlp_layers.append(nn.ReLU())
                else:
                    raise ValueError("activation must be 'relu' or 'rbf'")
        mlp = MLP(layer_dims=dims, activation="relu").to(device)
        mlp.net = nn.Sequential(*mlp_layers)
        return mlp

    @staticmethod
    @torch.no_grad()
    def compress_kan(
        kan,
        loader,
        device,
        neuron_threshold=1e-3,    # kill neurons whose mean activation < this
        grid_size_new=3,          # new grid_size after compression
        spline_order=None,        # keep same or overwrite
        verbose=True,
    ):
        """
        Compress KAN in two steps:
        (1) Prune low‑activity neurons.
        (2) Reduce spline grid_size (basis resolution).
        Returns a new, smaller KAN.
        """
        # Step 1: prune neurons (same as before, but return new KAN)
        activations = {}
        for name, module in kan.named_modules():
            if isinstance(module, BasisKANLayer):
                activations[name] = torch.zeros(
                    module.out_features, device=device
                )
        def hook_factory(name):
            def hook(module, inp, out):
                # out: (batch, out_features)
                mean_act = out.abs().mean(dim=0)
                activations[name].add_(mean_act / len(loader))
            return hook
        handles = []
        for name, module in kan.named_modules():
            if isinstance(module, BasisKANLayer):
                handles.append(module.register_forward_hook(hook_factory(name)))
        kan.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                kan(x)
        for h in handles:
            h.remove()
        # Build new layer_dims by pruning neurons
        new_layer_dims = []
        for i, (name, module) in enumerate(kan.named_modules()):
            if isinstance(module, BasisKANLayer):
                if not new_layer_dims:
                    new_layer_dims.append(module.in_features)
                keep_mask = activations[name] >= neuron_threshold
                n_keep = keep_mask.sum().item()
                if verbose:
                    print(
                        f"{name}: {module.out_features} → {n_keep} "
                        f"({n_keep/module.out_features*100:.1f}%)"
                    )
                new_layer_dims.append(n_keep)
        # Step 2: build intermediate KAN with pruned neurons
        if spline_order is None:
            sore_order = kan.net[1].spline_order
        else:
            sore_order = spline_order
        # First KAN: only neuron‑pruned
        pruned_kan = KAN(
            layer_dims=new_layer_dims,
            grid_size=kan.net[1].grid_size,
            spline_order=sore_order,
        ).to(device)
        # Optional: transfer coeffs of kept neurons (skipped here for simplicity)
        # Step 3: compress grid_size inside pruned_kan
        final_kan = KAN(
            layer_dims=new_layer_dims,
            grid_size=grid_size_new,
            spline_order=sore_order,
        ).to(device)
        # Transfer coeffs from pruned_kan to final_kan (coarser grid)
        src_layers = [m for m in pruned_kan.net if isinstance(m, BasisKANLayer)]
        dst_layers = [m for m in final_kan.net if isinstance(m, BasisKANLayer)]
        for src, dst in zip(src_layers, dst_layers):
            old_coeffs = src.coeffs
            old_nbasis = old_coeffs.size(-1)
            new_nbasis = dst.coeffs.size(-1)
            with torch.no_grad():
                if new_nbasis < old_nbasis:
                    ratio = old_nbasis / new_nbasis
                    for i in range(new_nbasis):
                        s = int(round(i * ratio))
                        e = int(round((i + 1) * ratio))
                        e = min(e, old_nbasis)
                        if s < e:
                            dst.coeffs[:, :, i].copy_(
                                old_coeffs[:, :, s:e].mean(dim=-1)
                            )
                        else:
                            dst.coeffs[:, :, i].copy_(
                                old_coeffs[:, :, s]
                            )
                else:
                    # new_nbasis >= old_nbasis
                    dst.coeffs[:, :, :old_nbasis].copy_(old_coeffs)
        return final_kan
    

class EvaluationTools:
    @staticmethod
    def evaluate_classification(model, loader, device):
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        # Accuracy
        acc = (all_preds == all_targets).float().mean().item()
        # Convert to numpy for sklearn
        y_true = all_targets.numpy()
        y_pred = all_preds.numpy()
        f1 = f1_score(y_true, y_pred, average="macro")   # or "micro"/"weighted"
        mcc = matthews_corrcoef(y_true, y_pred)
        return acc, f1, mcc

    @staticmethod
    def evaluate_regression(model, loader, device):
        model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        total_samples = 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                if outputs.shape != y.shape:
                    y = y.view_as(outputs)
                loss = criterion(outputs, y)
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
                all_preds.append(outputs.cpu())
                all_targets.append(y.cpu())
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        # Convert to numpy
        y_true = all_targets.numpy().flatten()
        y_pred = all_preds.numpy().flatten()
        mse  = mean_squared_error(y_true, y_pred)
        rmse = mse**0.5
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        # MAPE (ignore zero targets to avoid division by zero)
        non_zero = y_true != 0
        if non_zero.any():
            mape = 100 * (abs(y_true[non_zero] - y_pred[non_zero]) / abs(y_true[non_zero])).mean()
        else:
            mape = float('nan')
        return mse, rmse, mae, r2, mape

class TrainingEntityTools:
    @staticmethod
    def init(model, criterion, optimizer, epoch, train_loss, seed):
        training_entities = {
            "model": model,
            "criterion": criterion,
            "optimizer": optimizer,
            "epoch": epoch,
            "loss": train_loss,
            "seed": seed,
        }
        return training_entities
    
    @staticmethod
    def save(training_entities, pth_file_path):
        torch.save({
            "model": training_entities["model"].state_dict(),
            "criterion": training_entities["criterion"],
            "optimizer": training_entities["optimizer"].state_dict(),
            "epoch": training_entities["epoch"],
            "loss": training_entities["loss"],
            "seed": training_entities["seed"]
        }, pth_file_path)

    @staticmethod
    def load(training_entities, pth_file_path):
        checkpoint = torch.load(pth_file_path, weights_only=False)
        training_entities["model"].load_state_dict(checkpoint["model"])
        training_entities["criterion"] = checkpoint["criterion"]
        training_entities["optimizer"].load_state_dict(checkpoint["optimizer"])
        training_entities["epoch"] = checkpoint["epoch"] + 1
        training_entities["loss"] = checkpoint["loss"]
        training_entities["seed"] = checkpoint["seed"]
        set_seed(checkpoint["seed"])
        return training_entities
    
    @staticmethod
    def train(training_entities, loader, device, print_freq=-1):
        training_entities["model"].train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            training_entities["optimizer"].zero_grad()
            outputs = training_entities["model"](x)
            loss = training_entities["criterion"](outputs, y)
            loss.backward()
            training_entities["optimizer"].step()
            total_loss += loss.item()
            if print_freq!= -1:
                if batch_idx % print_freq == 0:
                    print(f"Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")
        return total_loss / len(loader)

class RBFLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(out_features, in_features))
        self.gamma = nn.Parameter(torch.ones(out_features))  # width

    def forward(self, x):
        # x: (batch_size, in_features)
        # compute ||x - c||^2
        x = x.unsqueeze(1)  # (batch, 1, in_features)
        c = self.centers.unsqueeze(0)  # (1, out_features, in_features)
        dist = torch.sum((x - c) ** 2, dim=2)  # (batch, out_features)

        return torch.exp(-self.gamma * dist)
    
class MLP(nn.Module):
    def __init__(self, layer_dims, activation="relu"):
        super().__init__()

        self.activation_type = activation.lower()
        input_dim, *hidden_dims, output_dim = layer_dims

        layers = [nn.Flatten()]

        if self.activation_type == "relu":
            # Build hidden layers: each (Linear → ReLU)
            dims = [input_dim] + hidden_dims
            for i in range(len(dims)-1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                layers.append(nn.ReLU())

            # Final output layer (no activation)
            layers.append(nn.Linear(dims[-1], output_dim))

        elif self.activation_type == "rbf":
            # RBF layer expects (in_dim, hidden_dim); rest linear
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            layers.append(nn.ReLU())  # or another activation if you prefer
            # Then RBF and final linear
            layers.append(RBFLayer(hidden_dims[0], hidden_dims[-1]))
            layers.append(nn.Linear(hidden_dims[-1], output_dim))

        else:
            raise ValueError("activation must be 'relu' or 'rbf'")

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    
def get_spline_basis(x, grid, k):
    """
    Compute B‑spline basis of order k (degree = k-1) over grid.
    x: (batch,)
    grid: (n_knots,) in sorted order
    k: spline_order ∈ {1, 2, 3}
    Returns B: (batch, n_basis), where n_basis = n_knots - k
    """
    x = x.unsqueeze(-1)        # (batch, 1)
    grid = grid.unsqueeze(0)   # (1, n_knots)

    # Use xp with shape (batch, 1); broadcasting handles comparisons with knot arrays
    xp = x  # (batch, 1)

    # For order 1: piecewise constant
    if k == 1:
        in_interval = (xp >= grid[:, :-1]) & (xp < grid[:, 1:])
        B = torch.zeros(x.size(0), grid.size(1) - 1, device=x.device)
        B += in_interval.float()
        return B

    # For higher orders: Cox‑de Boor style (simplified)
    # We'll build a matrix of B‑splines recursively
    n_knots = grid.size(1)
    n_basis = n_knots - k

    B = torch.zeros(x.size(0), n_basis, device=x.device)

    # Instead of full recursion, use a simple numerical recipe for k=2,3
    for i in range(n_basis):
        # Use cubic B‑spline-like blending (smooth bell‑shaped kernel)
        center = grid[0, i + k//2]
        width = (grid[0, i + k] - grid[0, i]) / 2 if i + k < n_knots else 1.0

        if k == 2:
            # Piecewise linear B‑spline
            left = grid[0, i]
            right = grid[0, i + 2]
            in_support = (xp >= left) & (xp <= right)
            # Linear blending
            w = torch.clamp((xp - left) / (right - left), 0, 1)
            w = torch.where(in_support, w, 0.0)
            B[:, i] = w.squeeze(-1)
        elif k == 3:
            # Approximate cubic B‑spline (smoother, bell‑shaped)
            # A simple smooth kernel instead of full Cox‑de Boor
            diff = (xp - center) / (width + 1e-8)
            weight = torch.exp(-0.5 * diff**2)  # Gaussian‑like shape
            weight = torch.where(
                (xp >= grid[0, i]) & (xp <= grid[0, i + 3]),
                weight, 0.0
            )
            B[:, i] = weight.squeeze(-1)

    # Normalize so that basis sums to ~1 locally
    norm = B.sum(dim=1, keepdim=True)
    norm = torch.where(norm > 1e-8, norm, torch.ones_like(norm))
    B = B / norm

    return B


class BasisKANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # assert spline_order in 1, "Only linear B‑spline (order 1) implemented here."
        assert spline_order in {1, 2, 3}, "spline_order must be 1, 2, or 3."
        
        # Number of basis functions: grid_size + 1 knots, minus k
        n_basis = grid_size + 1 - spline_order
        assert n_basis > 0, "grid_size too small for spline_order"

        self.coeffs = nn.Parameter(
            torch.randn(out_features, in_features, n_basis) * 0.1
        )

        self.register_buffer(
            "grid", torch.linspace(0, 1, grid_size + 1)
        )
        
    def forward(self, x):
        x = x.clamp(0, 1)
        basis_list = []
        for i in range(self.in_features):
            xp = x[:, i]  # (batch,)
            B = get_spline_basis(xp, self.grid, self.spline_order)  # (batch, n_basis)
            basis_list.append(B.unsqueeze(1))  # (batch, 1, n_basis)
        basis = torch.cat(basis_list, dim=1)  # (batch, in_features, n_basis)
        coeffs = self.coeffs.unsqueeze(0)      # (1, out, in, basis)
        out = (basis.unsqueeze(1) * coeffs).sum(dim=(-1, -2))  # (batch, out)
        return out
    
class KAN(nn.Module):
    def __init__(self, layer_dims, grid_size=5, spline_order=1):
        super().__init__()
        input_dim, *hidden_dims, output_dim = layer_dims
        layers = [nn.Flatten()]

        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(BasisKANLayer(
                dims[i], dims[i+1],
                grid_size=grid_size,
                spline_order=spline_order
            ))

        layers.append(BasisKANLayer(
            dims[-1], output_dim,
            grid_size=grid_size,
            spline_order=spline_order
        ))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    

def transfer_linear_to_kan(
    linear: nn.Linear,
    grid_size=5,
    spline_order=1,
    device=None,
):
    in_features = linear.in_features
    out_features = linear.out_features

    # Number of basis functions
    n_basis = grid_size + 1 - spline_order
    assert n_basis > 0, "grid_size too small for spline_order"

    # Start with a KAN layer
    kan = BasisKANLayer(
        in_features=in_features,
        out_features=out_features,
        grid_size=grid_size,
        spline_order=spline_order
    ).to(linear.weight.device)

    # Use Linear weights as a “mean” transformation
    # coeffs: (out, in, basis) ≈ (out, in, 1)
    # Approximate: coeffs[out, :, mid_basis] ≈ weight[out, :]
    mid_basis = n_basis // 2
    with torch.no_grad():
        # Use mid_basis to hold the linear weight
        kan.coeffs[:, :, mid_basis].copy_(linear.weight)
        # Normalize the rest to small random values
        kan.coeffs[:, :, :mid_basis] *= 0.01
        kan.coeffs[:, :, mid_basis+1:] *= 0.01

    return kan


def transfer_kan_to_linear(kan_layer: BasisKANLayer, device=None):
    in_features = kan_layer.in_features
    out_features = kan_layer.out_features
    n_basis = kan_layer.coeffs.size(-1)
    # Create a Linear layer
    linear = nn.Linear(in_features, out_features).to(kan_layer.coeffs.device)
    # Average coefficients over basis
    # coeffs: (out, in, basis)
    # Take mean over last dim → (out, in)
    with torch.no_grad():
        w_avg = kan_layer.coeffs.mean(dim=-1)  # approximate weight
        # Optionally scale to match Linear initialization
        linear.weight.copy_(w_avg)
        linear.bias.fill_(0.0)

    if device is not None:
        linear = linear.to(device)
    return linear