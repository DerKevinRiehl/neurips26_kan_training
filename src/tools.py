############################################################################### 
#### IMPORTS ##################################################################
###############################################################################
import torch
import torch.nn as nn
import random
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math



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
    def generate_network_kan(
        device,
        layer_dims,
        n_basis=8,
        degree=3,
        grid_min=-3.0,
        grid_max=3.0,
        alpha_init=1.0,
        trainable_alpha=False,
        preserve_relu=True,
    ):
        return HomotopyBSplineKAN(
            layer_dims=layer_dims,
            n_basis=n_basis,
            degree=degree,
            grid_min=grid_min,
            grid_max=grid_max,
            alpha_init=alpha_init,
            trainable_alpha=trainable_alpha,
            preserve_relu=preserve_relu,
        ).to(device)

    @staticmethod
    def mlp_to_kan(
        mlp,
        device,
        n_basis=8,
        degree=3,
        grid_min=-3.0,
        grid_max=3.0,
        alpha_init=1.0,
        trainable_alpha=False,
    ):
        kan_layers = []

        for module in mlp.net:
            if isinstance(module, nn.Flatten):
                kan_layers.append(nn.Flatten())

            elif isinstance(module, nn.Linear):
                kan_layers.append(
                    transfer_linear_to_homotopy_kan(
                        module.to(device),
                        n_basis=n_basis,
                        degree=degree,
                        grid_min=grid_min,
                        grid_max=grid_max,
                        alpha_init=alpha_init,
                        trainable_alpha=trainable_alpha,
                    )
                )

            elif isinstance(module, nn.ReLU):
                kan_layers.append(nn.ReLU())

            else:
                raise ValueError(f"Unsupported module in MLP->KAN conversion: {type(module)}")

        dims = []
        for module in mlp.net:
            if isinstance(module, nn.Linear):
                if not dims:
                    dims.append(module.in_features)
                dims.append(module.out_features)

        kan = HomotopyBSplineKAN(
            layer_dims=dims,
            n_basis=n_basis,
            degree=degree,
            grid_min=grid_min,
            grid_max=grid_max,
            alpha_init=alpha_init,
            trainable_alpha=trainable_alpha,
            preserve_relu=True,
        ).to(device)

        kan.net = nn.Sequential(*kan_layers)
        return kan

    @staticmethod
    def kan_to_mlp(kan, device, activation="relu"):
        mlp_layers = []
        dims = []

        for module in kan.net:
            if isinstance(module, nn.Flatten):
                mlp_layers.append(nn.Flatten())

            elif isinstance(module, HomotopyBSplineKANLayer):
                if not dims:
                    dims.append(module.in_features)
                dims.append(module.out_features)

                linear = nn.Linear(module.in_features, module.out_features).to(device)
                with torch.no_grad():
                    linear.weight.copy_(module.base_weight)
                    linear.bias.copy_(module.base_bias)
                mlp_layers.append(linear)

            elif isinstance(module, nn.ReLU):
                mlp_layers.append(nn.ReLU())

            else:
                raise ValueError(f"Unsupported module in KAN->MLP conversion: {type(module)}")

        mlp = MLP(layer_dims=dims, activation="relu").to(device)
        mlp.net = nn.Sequential(*mlp_layers)
        return mlp

    @staticmethod
    def set_alpha(model, alpha_value):
        for module in model.modules():
            if isinstance(module, HomotopyBSplineKANLayer):
                module.set_alpha(alpha_value)
    

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
    

def make_open_uniform_knots(grid_min, grid_max, n_basis, degree, device=None, dtype=None):
    """
    Create an open-uniform knot vector for B-splines.

    Args:
        grid_min, grid_max: domain endpoints
        n_basis: number of basis functions
        degree: spline degree (0=piecewise constant, 1=linear, 2=quadratic, 3=cubic)

    Returns:
        knots: shape (n_basis + degree + 1,)
    """
    assert n_basis >= degree + 1, "Need n_basis >= degree + 1 for open-uniform B-splines."

    n_knots = n_basis + degree + 1
    n_internal = n_knots - 2 * (degree + 1)

    if n_internal > 0:
        internal = torch.linspace(
            grid_min, grid_max, steps=n_internal + 2, device=device, dtype=dtype
        )[1:-1]
        knots = torch.cat([
            torch.full((degree + 1,), grid_min, device=device, dtype=dtype),
            internal,
            torch.full((degree + 1,), grid_max, device=device, dtype=dtype),
        ])
    else:
        knots = torch.cat([
            torch.full((degree + 1,), grid_min, device=device, dtype=dtype),
            torch.full((degree + 1,), grid_max, device=device, dtype=dtype),
        ])

    return knots


def bspline_basis_1d(x, knots, degree):
    """
    Evaluate all B-spline basis functions of a given degree at points x
    using the Cox-de Boor recursion.

    Args:
        x:     shape (batch,)
        knots: shape (n_knots,)
        degree: spline degree

    Returns:
        basis: shape (batch, n_basis)
               where n_basis = len(knots) - degree - 1
    """
    x = x.unsqueeze(-1)  # (batch, 1)
    device = x.device
    dtype = x.dtype

    n_knots = knots.numel()
    n_basis = n_knots - degree - 1
    assert n_basis > 0, "Invalid knot vector / degree."

    # Degree-0 basis
    # N_{i,0}(x) = 1 if t_i <= x < t_{i+1}, else 0
    # Special-case the right boundary so x == knots[-1] belongs to the last basis.
    B = ((x >= knots[:-1]) & (x < knots[1:])).to(dtype)  # (batch, n_knots-1)
    B_last = (x.squeeze(-1) == knots[-1]).to(dtype)
    if B_last.any():
        B[B_last.bool(), -1] = 1.0

    # Cox-de Boor recursion up to target degree
    for p in range(1, degree + 1):
        new_B = torch.zeros(x.size(0), n_knots - p - 1, device=device, dtype=dtype)

        left_den = knots[p:n_knots - 1] - knots[:n_knots - p - 1]
        right_den = knots[p + 1:n_knots] - knots[1:n_knots - p]

        left_num = x - knots[:n_knots - p - 1].unsqueeze(0)
        right_num = knots[p + 1:n_knots].unsqueeze(0) - x

        left_term = torch.zeros_like(new_B)
        right_term = torch.zeros_like(new_B)

        left_mask = left_den > 0
        right_mask = right_den > 0

        if left_mask.any():
            left_term[:, left_mask] = (
                left_num[:, left_mask] / left_den[left_mask].unsqueeze(0)
            ) * B[:, :n_knots - p - 1][:, left_mask]

        if right_mask.any():
            right_term[:, right_mask] = (
                right_num[:, right_mask] / right_den[right_mask].unsqueeze(0)
            ) * B[:, 1:n_knots - p][:, right_mask]

        B = left_term + right_term

    return B[:, :n_basis]

# kaiming_uniform_ ?
class ProperBSplineKANLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        n_basis=8,
        degree=3,
        grid_min=-3.0,
        grid_max=3.0,
        use_base_linear=True,
    ):
        super().__init__()

        assert degree >= 0
        assert n_basis >= degree + 1

        self.in_features = in_features
        self.out_features = out_features
        self.n_basis = n_basis
        self.degree = degree
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.use_base_linear = use_base_linear

        # Spline residual coefficients: (out_features, in_features, n_basis)
        self.coeffs = nn.Parameter(torch.zeros(out_features, in_features, n_basis))

        if use_base_linear:
            self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
            self.base_bias = nn.Parameter(torch.zeros(out_features))
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

            fan_in = in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.base_bias, -bound, bound)
        else:
            self.register_parameter("base_weight", None)
            self.register_parameter("base_bias", None)

        knots = make_open_uniform_knots(
            grid_min=grid_min,
            grid_max=grid_max,
            n_basis=n_basis,
            degree=degree,
            device=None,
            dtype=torch.float32,
        )
        self.register_buffer("knots", knots)

    def forward(self, x):
        """
        x: shape (batch, in_features)
        returns: shape (batch, out_features)
        """
        batch_size = x.size(0)
        dtype = x.dtype
        device = x.device

        # Evaluate spline basis for each input dimension
        # basis_all: (batch, in_features, n_basis)
        basis_list = []
        for i in range(self.in_features):
            Bi = bspline_basis_1d(x[:, i], self.knots.to(device=device, dtype=dtype), self.degree)
            basis_list.append(Bi.unsqueeze(1))
        basis_all = torch.cat(basis_list, dim=1)

        # Spline contribution:
        # basis_all:          (batch, in, basis)
        # coeffs.unsqueeze:   (1, out, in, basis)
        # result:             (batch, out)
        spline_out = (basis_all.unsqueeze(1) * self.coeffs.unsqueeze(0)).sum(dim=(-1, -2))

        if self.use_base_linear:
            linear_out = x @ self.base_weight.t() + self.base_bias
            return linear_out + spline_out
        else:
            return spline_out

# Proper Cox–de Boor Splines
class ProperBSplineKAN(nn.Module):
    def __init__(
        self,
        layer_dims,
        n_basis=8,
        degree=3,
        grid_min=-3.0,
        grid_max=3.0,
        use_base_linear=True,
    ):
        super().__init__()

        input_dim, *hidden_dims, output_dim = layer_dims
        dims = [input_dim] + hidden_dims + [output_dim]

        layers = [nn.Flatten()]
        for i in range(len(dims) - 1):
            layers.append(
                ProperBSplineKANLayer(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    n_basis=n_basis,
                    degree=degree,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    use_base_linear=use_base_linear,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    
def transfer_linear_to_proper_kan(linear, n_basis=8, degree=3, grid_min=-3.0, grid_max=3.0):
    kan = ProperBSplineKANLayer(
        in_features=linear.in_features,
        out_features=linear.out_features,
        n_basis=n_basis,
        degree=degree,
        grid_min=grid_min,
        grid_max=grid_max,
        use_base_linear=True,
    ).to(linear.weight.device)

    with torch.no_grad():
        kan.base_weight.copy_(linear.weight)
        if linear.bias is not None:
            kan.base_bias.copy_(linear.bias)
        else:
            kan.base_bias.zero_()
        kan.coeffs.zero_()

    return kan


def transfer_proper_kan_to_linear(kan_layer, device=None):
    linear = nn.Linear(kan_layer.in_features, kan_layer.out_features).to(kan_layer.coeffs.device)

    with torch.no_grad():
        if kan_layer.base_weight is not None:
            linear.weight.copy_(kan_layer.base_weight)
            linear.bias.copy_(kan_layer.base_bias)
        else:
            # fallback approximation
            linear.weight.copy_(kan_layer.coeffs.mean(dim=-1))
            linear.bias.zero_()

    if device is not None:
        linear = linear.to(device)
    return linear


class HomotopyBSplineKANLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        n_basis=8,
        degree=3,
        grid_min=-3.0,
        grid_max=3.0,
        alpha_init=1.0,
        trainable_alpha=False,
    ):
        super().__init__()

        assert degree >= 0
        assert n_basis >= degree + 1

        self.in_features = in_features
        self.out_features = out_features
        self.n_basis = n_basis
        self.degree = degree
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.trainable_alpha = trainable_alpha

        # spline residual coefficients
        self.coeffs = nn.Parameter(torch.zeros(out_features, in_features, n_basis))

        # linear path
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.base_bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        fan_in = in_features
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.base_bias, -bound, bound)

        # alpha path coefficient
        if trainable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha_init)))

        knots = make_open_uniform_knots(
            grid_min=grid_min,
            grid_max=grid_max,
            n_basis=n_basis,
            degree=degree,
            device=None,
            dtype=torch.float32,
        )
        self.register_buffer("knots", knots)

    def set_alpha(self, alpha_value: float):
        with torch.no_grad():
            self.alpha.fill_(float(alpha_value))

    def forward(self, x):
        dtype = x.dtype
        device = x.device

        # B-spline basis per input dimension
        basis_list = []
        knots = self.knots.to(device=device, dtype=dtype)
        for i in range(self.in_features):
            Bi = bspline_basis_1d(x[:, i], knots, self.degree)   # (batch, n_basis)
            basis_list.append(Bi.unsqueeze(1))
        basis_all = torch.cat(basis_list, dim=1)  # (batch, in_features, n_basis)

        # spline residual
        spline_out = (basis_all.unsqueeze(1) * self.coeffs.unsqueeze(0)).sum(dim=(-1, -2))

        # linear path
        linear_out = x @ self.base_weight.t() + self.base_bias

        return self.alpha * linear_out + spline_out
    
class HomotopyBSplineKAN(nn.Module):
    def __init__(
        self,
        layer_dims,
        n_basis=8,
        degree=3,
        grid_min=-3.0,
        grid_max=3.0,
        alpha_init=1.0,
        trainable_alpha=False,
        preserve_relu=True,
    ):
        super().__init__()

        input_dim, *hidden_dims, output_dim = layer_dims
        dims = [input_dim] + hidden_dims + [output_dim]

        layers = [nn.Flatten()]
        for i in range(len(dims) - 1):
            layers.append(
                HomotopyBSplineKANLayer(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    n_basis=n_basis,
                    degree=degree,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    alpha_init=alpha_init,
                    trainable_alpha=trainable_alpha,
                )
            )
            if preserve_relu and i < len(dims) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def set_alpha(self, alpha_value: float):
        for module in self.modules():
            if isinstance(module, HomotopyBSplineKANLayer):
                module.set_alpha(alpha_value)
                
def transfer_linear_to_homotopy_kan(
    linear,
    n_basis=8,
    degree=3,
    grid_min=-3.0,
    grid_max=3.0,
    alpha_init=1.0,
    trainable_alpha=False,
):
    kan = HomotopyBSplineKANLayer(
        in_features=linear.in_features,
        out_features=linear.out_features,
        n_basis=n_basis,
        degree=degree,
        grid_min=grid_min,
        grid_max=grid_max,
        alpha_init=alpha_init,
        trainable_alpha=trainable_alpha,
    ).to(linear.weight.device)

    with torch.no_grad():
        kan.base_weight.copy_(linear.weight)
        if linear.bias is not None:
            kan.base_bias.copy_(linear.bias)
        else:
            kan.base_bias.zero_()
        kan.coeffs.zero_()

    return kan




###############################################################################
#### KD / HOMOTOPY / PRUNING UTILITIES ########################################
###############################################################################

class DistillationTools:
    @staticmethod
    def kd_loss(student_logits, teacher_logits, temperature=4.0):
        """
        KL-divergence based knowledge distillation loss.
        """
        log_p_student = torch.log_softmax(student_logits / temperature, dim=1)
        p_teacher = torch.softmax(teacher_logits / temperature, dim=1)
        loss = torch.nn.functional.kl_div(
            log_p_student,
            p_teacher,
            reduction="batchmean"
        ) * (temperature ** 2)
        return loss

    @staticmethod
    def combined_classification_kd_loss(
        student_logits,
        teacher_logits,
        targets,
        alpha_task=0.5,
        alpha_kd=0.5,
        temperature=4.0,
    ):
        ce = torch.nn.functional.cross_entropy(student_logits, targets)
        kd = DistillationTools.kd_loss(student_logits, teacher_logits, temperature=temperature)
        return alpha_task * ce + alpha_kd * kd, ce.detach().item(), kd.detach().item()


class HomotopyTools:
    @staticmethod
    def freeze_base_path(model):
        """
        Freeze base linear parameters, keep spline coeffs trainable.
        """
        for module in model.modules():
            if isinstance(module, HomotopyBSplineKANLayer):
                module.base_weight.requires_grad = False
                module.base_bias.requires_grad = False
                module.coeffs.requires_grad = True
                if isinstance(module.alpha, nn.Parameter):
                    module.alpha.requires_grad = False

    @staticmethod
    def unfreeze_all(model, train_alpha=False):
        """
        Unfreeze everything. Optionally also train alpha if alpha is a Parameter.
        """
        for module in model.modules():
            if isinstance(module, HomotopyBSplineKANLayer):
                module.base_weight.requires_grad = True
                module.base_bias.requires_grad = True
                module.coeffs.requires_grad = True
                if isinstance(module.alpha, nn.Parameter):
                    module.alpha.requires_grad = train_alpha

    @staticmethod
    def set_alpha(model, alpha_value):
        for module in model.modules():
            if isinstance(module, HomotopyBSplineKANLayer):
                module.set_alpha(alpha_value)

    @staticmethod
    def linear_anneal(epoch, total_epochs, start_alpha=1.0, end_alpha=0.0):
        if total_epochs <= 1:
            return end_alpha
        t = epoch / (total_epochs - 1)
        return start_alpha + t * (end_alpha - start_alpha)


class PruningTools:
    @staticmethod
    def prune_spline_coefficients_by_threshold(model, threshold=1e-4):
        """
        Zero out small spline coefficients.
        Returns number pruned and total.
        """
        total = 0
        pruned = 0
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, HomotopyBSplineKANLayer):
                    mask = module.coeffs.abs() < threshold
                    pruned += mask.sum().item()
                    total += mask.numel()
                    module.coeffs[mask] = 0.0
        return pruned, total

    @staticmethod
    def prune_spline_coefficients_by_percentile(model, percentile=20.0):
        """
        Global percentile pruning over all spline coeffs.
        """
        coeffs_all = []
        for module in model.modules():
            if isinstance(module, HomotopyBSplineKANLayer):
                coeffs_all.append(module.coeffs.detach().abs().reshape(-1))
        if len(coeffs_all) == 0:
            return 0, 0, 0.0

        coeffs_all = torch.cat(coeffs_all)
        threshold = torch.quantile(coeffs_all, percentile / 100.0).item()

        pruned, total = PruningTools.prune_spline_coefficients_by_threshold(model, threshold=threshold)
        return pruned, total, threshold


class DiagnosticsTools:
    @staticmethod
    def spline_coefficient_stats(model):
        stats = {}
        total_l1 = 0.0
        total_l2_sq = 0.0
        total_params = 0
        total_nonzero = 0

        layer_idx = 0
        for module in model.modules():
            if isinstance(module, HomotopyBSplineKANLayer):
                coeffs = module.coeffs.detach()
                l1 = coeffs.abs().sum().item()
                l2 = torch.sqrt((coeffs ** 2).sum()).item()
                nnz = (coeffs.abs() > 0).sum().item()
                n = coeffs.numel()

                stats[f"layer_{layer_idx}"] = {
                    "l1": l1,
                    "l2": l2,
                    "nnz": nnz,
                    "total": n,
                    "sparsity": 1.0 - (nnz / n),
                }

                total_l1 += l1
                total_l2_sq += (coeffs ** 2).sum().item()
                total_params += n
                total_nonzero += nnz
                layer_idx += 1

        stats["global"] = {
            "l1": total_l1,
            "l2": total_l2_sq ** 0.5,
            "nnz": total_nonzero,
            "total": total_params,
            "sparsity": 1.0 - (total_nonzero / total_params if total_params > 0 else 0.0),
        }
        return stats

    @staticmethod
    def spline_vs_linear_contribution(model, loader, device, max_batches=10):
        """
        Estimates relative contribution magnitude:
            ratio = ||spline_out|| / (||alpha*linear_out|| + ||spline_out||)
        averaged over layers and batches.
        """
        model.eval()
        layer_ratios = []
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(loader):
                if batch_idx >= max_batches:
                    break

                x = x.to(device)
                h = x

                for module in model.net:
                    if isinstance(module, nn.Flatten):
                        h = module(h)

                    elif isinstance(module, HomotopyBSplineKANLayer):
                        dtype = h.dtype
                        dev = h.device
                        knots = module.knots.to(device=dev, dtype=dtype)

                        basis_list = []
                        for i in range(module.in_features):
                            Bi = bspline_basis_1d(h[:, i], knots, module.degree)
                            basis_list.append(Bi.unsqueeze(1))
                        basis_all = torch.cat(basis_list, dim=1)

                        spline_out = (basis_all.unsqueeze(1) * module.coeffs.unsqueeze(0)).sum(dim=(-1, -2))
                        linear_out = h @ module.base_weight.t() + module.base_bias
                        linear_out = module.alpha * linear_out

                        spline_norm = spline_out.norm(p=2).item()
                        linear_norm = linear_out.norm(p=2).item()
                        denom = spline_norm + linear_norm + 1e-12
                        ratio = spline_norm / denom
                        layer_ratios.append(ratio)

                        h = linear_out + spline_out

                    else:
                        h = module(h)

        if len(layer_ratios) == 0:
            return {"mean_ratio": 0.0, "ratios": []}

        return {
            "mean_ratio": float(np.mean(layer_ratios)),
            "ratios": layer_ratios,
        }


class TrainingEntityToolsExtended:
    @staticmethod
    def train_kd_epoch(
        student_model,
        teacher_model,
        optimizer,
        loader,
        device,
        alpha_task=0.5,
        alpha_kd=0.5,
        temperature=4.0,
        print_freq=-1,
    ):
        student_model.train()
        teacher_model.eval()

        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0

        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher_model(x)

            student_logits = student_model(x)

            loss, ce_value, kd_value = DistillationTools.combined_classification_kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                targets=y,
                alpha_task=alpha_task,
                alpha_kd=alpha_kd,
                temperature=temperature,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_value
            total_kd += kd_value

            if print_freq != -1 and batch_idx % print_freq == 0:
                print(
                    f"Batch {batch_idx}/{len(loader)} | "
                    f"Loss={loss.item():.4f} CE={ce_value:.4f} KD={kd_value:.4f}"
                )

        n_batches = len(loader)
        return {
            "loss": total_loss / n_batches,
            "ce": total_ce / n_batches,
            "kd": total_kd / n_batches,
        }