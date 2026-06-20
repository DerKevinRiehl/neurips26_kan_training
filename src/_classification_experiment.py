from dataset import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import copy

from tools import (
    ModelTools,
    TrainingEntityTools,
    TrainingEntityToolsExtended,
    EvaluationTools,
    DiagnosticsTools,
    HomotopyTools,
    PruningTools,
    set_seed,
)

###############################################################################
#### PARAMETERS ################################################################
###############################################################################

PARAM_SEED = 42
PARAM_BATCH_SIZE = 64
PARAM_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Teacher
PARAM_TEACHER_DIMS = [28 * 28, 16, 8, 10]
PARAM_TEACHER_EPOCHS = 5
PARAM_TEACHER_LR = 1e-3

# Homotopy KAN
PARAM_KAN_N_BASIS = 2
PARAM_KAN_DEGREE = 1
PARAM_GRID_MIN = -3.0
PARAM_GRID_MAX = 3.0

# Stage 1: spline-only transfer
PARAM_STAGE1_EPOCHS = 2
PARAM_STAGE1_LR = 1e-3
PARAM_STAGE1_ALPHA_TASK = 0.3
PARAM_STAGE1_ALPHA_KD = 0.7
PARAM_STAGE1_TEMPERATURE = 4.0

# Stage 2: full fine-tuning with alpha annealing
PARAM_STAGE2_EPOCHS = 5
PARAM_STAGE2_LR = 5e-4
PARAM_STAGE2_ALPHA_TASK = 0.5
PARAM_STAGE2_ALPHA_KD = 0.5
PARAM_STAGE2_TEMPERATURE = 4.0
PARAM_ALPHA_START = 1.0
PARAM_ALPHA_END = 0.0

# Pruning
PARAM_PRUNE_PERCENTILE = 30.0
PARAM_FINETUNE_AFTER_PRUNE_EPOCHS = 2
PARAM_FINETUNE_AFTER_PRUNE_LR = 2e-4

###############################################################################
#### HELPERS ###################################################################
###############################################################################

def print_classification_metrics(tag, model, train_loader, test_loader, device):
    acc_tra, f1_tra, mcc_tra = EvaluationTools.evaluate_classification(model, train_loader, device)
    acc_tst, f1_tst, mcc_tst = EvaluationTools.evaluate_classification(model, test_loader, device)
    print(
        f"{tag} | "
        f"TRAIN(ACC={acc_tra:.4f}, F1={f1_tra:.4f}, MCC={mcc_tra:.4f}) "
        f"TEST(ACC={acc_tst:.4f}, F1={f1_tst:.4f}, MCC={mcc_tst:.4f})"
    )

def print_diagnostics(tag, model, loader, device):
    coeff_stats = DiagnosticsTools.spline_coefficient_stats(model)
    contrib_stats = DiagnosticsTools.spline_vs_linear_contribution(model, loader, device, max_batches=10)

    global_stats = coeff_stats["global"]
    print(
        f"{tag} | "
        f"SPLINE_L1={global_stats['l1']:.6f} "
        f"SPLINE_L2={global_stats['l2']:.6f} "
        f"NNZ={global_stats['nnz']}/{global_stats['total']} "
        f"SPARSITY={global_stats['sparsity']:.4f} "
        f"SPLINE_RATIO={contrib_stats['mean_ratio']:.6f}"
    )

###############################################################################
#### MAIN ######################################################################
###############################################################################

set_seed(PARAM_SEED)

train_dataset, test_dataset = Dataset.mnist()
train_loader = DataLoader(train_dataset, batch_size=PARAM_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=PARAM_BATCH_SIZE, shuffle=False)

device = PARAM_DEVICE

###########################################################################
# 1) TRAIN TEACHER MLP
###########################################################################
teacher = ModelTools.generate_network_mlp(
    device,
    layer_dims=PARAM_TEACHER_DIMS,
    activation="relu"
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(teacher.parameters(), lr=PARAM_TEACHER_LR)

training_entities = TrainingEntityTools.init(
    teacher, criterion, optimizer, epoch=0, train_loss=None, seed=PARAM_SEED
)

print("\n=== STAGE 0: TRAIN TEACHER MLP ===")
for epoch in range(PARAM_TEACHER_EPOCHS):
    train_loss = TrainingEntityTools.train(training_entities, train_loader, device, print_freq=-1)
    print(f"Teacher Epoch {epoch+1}/{PARAM_TEACHER_EPOCHS} | Loss={train_loss:.4f}")
    print_classification_metrics("Teacher", teacher, train_loader, test_loader, device)

teacher = copy.deepcopy(teacher).to(device)
teacher.eval()

print(f"\nTeacher params: {ModelTools.get_nuber_parameters(teacher)}")
print_classification_metrics("Final Teacher", teacher, train_loader, test_loader, device)

###########################################################################
# 2) CONVERT TO HOMOTOPY-KAN
###########################################################################
student = ModelTools.mlp_to_kan(
    teacher,
    device,
    n_basis=PARAM_KAN_N_BASIS,
    degree=PARAM_KAN_DEGREE,
    grid_min=PARAM_GRID_MIN,
    grid_max=PARAM_GRID_MAX,
    alpha_init=1.0,
    trainable_alpha=False,
)

print("\n=== STAGE 1: EXACT MLP -> HOMOTOPY-KAN CONVERSION ===")
print(f"Student params after conversion: {ModelTools.get_nuber_parameters(student)}")
print_classification_metrics("Converted KAN", student, train_loader, test_loader, device)
print_diagnostics("Converted KAN", student, train_loader, device)

###########################################################################
# 3) FREEZE BASE PATH, TRAIN ONLY SPLINES WITH KD + TASK LOSS
###########################################################################
print("\n=== STAGE 2: SPLINE-ONLY TRANSFER (FREEZE BASE PATH) ===")
HomotopyTools.freeze_base_path(student)
HomotopyTools.set_alpha(student, 1.0)

optimizer_stage1 = torch.optim.Adam(
    filter(lambda p: p.requires_grad, student.parameters()),
    lr=PARAM_STAGE1_LR
)

for epoch in range(PARAM_STAGE1_EPOCHS):
    stats = TrainingEntityToolsExtended.train_kd_epoch(
        student_model=student,
        teacher_model=teacher,
        optimizer=optimizer_stage1,
        loader=train_loader,
        device=device,
        alpha_task=PARAM_STAGE1_ALPHA_TASK,
        alpha_kd=PARAM_STAGE1_ALPHA_KD,
        temperature=PARAM_STAGE1_TEMPERATURE,
        print_freq=-1,
    )

    print(
        f"Stage1 Epoch {epoch+1}/{PARAM_STAGE1_EPOCHS} | "
        f"Loss={stats['loss']:.4f} CE={stats['ce']:.4f} KD={stats['kd']:.4f}"
    )
    print_classification_metrics("Spline-only student", student, train_loader, test_loader, device)
    print_diagnostics("Spline-only student", student, train_loader, device)

###########################################################################
# 4) UNFREEZE ALL
# 5) ANNEAL ALPHA FROM 1.0 -> 0.0
###########################################################################
print("\n=== STAGE 3: FULL TRAINING WITH ALPHA ANNEALING ===")
HomotopyTools.unfreeze_all(student, train_alpha=False)

optimizer_stage2 = torch.optim.Adam(student.parameters(), lr=PARAM_STAGE2_LR)

for epoch in range(PARAM_STAGE2_EPOCHS):
    current_alpha = HomotopyTools.linear_anneal(
        epoch=epoch,
        total_epochs=PARAM_STAGE2_EPOCHS,
        start_alpha=PARAM_ALPHA_START,
        end_alpha=PARAM_ALPHA_END,
    )
    HomotopyTools.set_alpha(student, current_alpha)

    stats = TrainingEntityToolsExtended.train_kd_epoch(
        student_model=student,
        teacher_model=teacher,
        optimizer=optimizer_stage2,
        loader=train_loader,
        device=device,
        alpha_task=PARAM_STAGE2_ALPHA_TASK,
        alpha_kd=PARAM_STAGE2_ALPHA_KD,
        temperature=PARAM_STAGE2_TEMPERATURE,
        print_freq=-1,
    )

    print(
        f"Stage2 Epoch {epoch+1}/{PARAM_STAGE2_EPOCHS} | "
        f"alpha={current_alpha:.4f} "
        f"Loss={stats['loss']:.4f} CE={stats['ce']:.4f} KD={stats['kd']:.4f}"
    )
    print_classification_metrics("Annealed student", student, train_loader, test_loader, device)
    print_diagnostics("Annealed student", student, train_loader, device)

###########################################################################
# 6) PRUNE SPLINE COEFFICIENTS
###########################################################################
print("\n=== STAGE 4: PRUNE SPLINE COEFFICIENTS ===")
pruned, total, threshold = PruningTools.prune_spline_coefficients_by_percentile(
    student,
    percentile=PARAM_PRUNE_PERCENTILE
)
print(
    f"Pruned spline coeffs: {pruned}/{total} "
    f"({100.0 * pruned / max(total, 1):.2f}%) "
    f"with threshold={threshold:.8f}"
)
print_classification_metrics("Pruned student", student, train_loader, test_loader, device)
print_diagnostics("Pruned student", student, train_loader, device)

###########################################################################
# 7) BRIEF FINE-TUNE AFTER PRUNING
###########################################################################
print("\n=== STAGE 5: FINE-TUNE AFTER PRUNING ===")
optimizer_stage3 = torch.optim.Adam(student.parameters(), lr=PARAM_FINETUNE_AFTER_PRUNE_LR)

for epoch in range(PARAM_FINETUNE_AFTER_PRUNE_EPOCHS):
    HomotopyTools.set_alpha(student, 0.0)

    stats = TrainingEntityToolsExtended.train_kd_epoch(
        student_model=student,
        teacher_model=teacher,
        optimizer=optimizer_stage3,
        loader=train_loader,
        device=device,
        alpha_task=0.7,
        alpha_kd=0.3,
        temperature=PARAM_STAGE2_TEMPERATURE,
        print_freq=-1,
    )

    print(
        f"Stage3 Epoch {epoch+1}/{PARAM_FINETUNE_AFTER_PRUNE_EPOCHS} | "
        f"Loss={stats['loss']:.4f} CE={stats['ce']:.4f} KD={stats['kd']:.4f}"
    )
    print_classification_metrics("Post-prune fine-tuned student", student, train_loader, test_loader, device)
    print_diagnostics("Post-prune fine-tuned student", student, train_loader, device)
