from dataset import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tools import ModelTools, TrainingEntityTools, EvaluationTools, set_seed
import copy






PARAM_SEED = 42
PARAM_FILE = "test.pth"
PARAM_N_EPOCHS = 1


# SET SEED
set_seed(seed=PARAM_SEED)

# LOAD DATA
train_dataset, test_dataset = Dataset.mnist()
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# PREPARE TRAINING ENTITIES
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
# model = ModelTools.generate_network_mlp(device, layer_dims=[28*28, 256, 128, 10], activation="relu") # activation="rbf"
# Epoch 1: Loss=0.2869, TRAIN(ACC=0.9655, F1=0.9652, MCC=0.9617) TEST(ACC=0.9615, F1=0.9611, MCC=0.9572)
# Epoch 2: Loss=0.1079, TRAIN(ACC=0.9801, F1=0.9801, MCC=0.9779) TEST(ACC=0.9741, F1=0.9740, MCC=0.9712)
# Epoch 3: Loss=0.0701, TRAIN(ACC=0.9869, F1=0.9868, MCC=0.9854) TEST(ACC=0.9769, F1=0.9767, MCC=0.9743)
# Epoch 4: Loss=0.0523, TRAIN(ACC=0.9863, F1=0.9863, MCC=0.9848) TEST(ACC=0.9773, F1=0.9772, MCC=0.9748)
# Epoch 5: Loss=0.0384, TRAIN(ACC=0.9923, F1=0.9923, MCC=0.9915) TEST(ACC=0.9788, F1=0.9786, MCC=0.9764)

model = ModelTools.generate_network_kan(device, layer_dims=[28*28, 32, 10], grid_size=4, spline_order=2)
# Epoch 1: Loss=1.2938, TRAIN(ACC=0.8229, F1=0.8198, MCC=0.8041) TEST(ACC=0.8209, F1=0.8184, MCC=0.8017)
# Epoch 2: Loss=0.5642, TRAIN(ACC=0.8748, F1=0.8725, MCC=0.8611) TEST(ACC=0.8739, F1=0.8715, MCC=0.8601)
# Epoch 3: Loss=0.4430, TRAIN(ACC=0.8634, F1=0.8618, MCC=0.8487) TEST(ACC=0.8601, F1=0.8584, MCC=0.8449)
# Epoch 4: Loss=0.4019, TRAIN(ACC=0.8969, F1=0.8957, MCC=0.8855) TEST(ACC=0.8843, F1=0.8829, MCC=0.8715)
# Epoch 5: Loss=0.3825, TRAIN(ACC=0.8872, F1=0.8862, MCC=0.8750) TEST(ACC=0.8821, F1=0.8811, MCC=0.8694)

# print(ModelTools.get_nuber_parameters(model))


# import sys
# sys.exit(0)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epoch = 0
train_loss = None

# # # PREPARE TRIANING ENTITIES (FROM PREVIOUS RUN)
# training_entities = TrainingEntityTools.init(model, criterion, optimizer, epoch, train_loss, PARAM_SEED)
# training_entities = TrainingEntityTools.load(training_entities, pth_file_path=PARAM_FILE)

# # PREPARE TRAINING ENTITIES (FROM SCRATCH)
training_entities = TrainingEntityTools.init(model, criterion, optimizer, epoch, train_loss, PARAM_SEED)

# # CONDUCT TRAINING
for epoch in range(PARAM_N_EPOCHS):
    train_loss = TrainingEntityTools.train(training_entities, train_loader, device, print_freq=10)
    acc_tra, f1_tra, mcc_tra = EvaluationTools.evaluate_classification(model, train_loader, device)
    acc_tst, f1_tst, mcc_tst = EvaluationTools.evaluate_classification(model, test_loader, device)
    print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, TRAIN(ACC={acc_tra:.4f}, F1={f1_tra:.4f}, MCC={mcc_tra:.4f}) TEST(ACC={acc_tst:.4f}, F1={f1_tst:.4f}, MCC={mcc_tst:.4f})")

# # SAVE RESULT
# TrainingEntityTools.save(training_entities=training_entities, pth_file_path=PARAM_FILE)

# # Converting
model_mlp = copy.deepcopy(model)
acc, f1, mcc = EvaluationTools.evaluate_classification(model_mlp, test_loader, device)
print("MLP", ModelTools.get_nuber_parameters(model_mlp), f"ACC={acc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")
# MLP 235146 ACC=0.9615, F1=0.9611, MCC=0.9572
model_kan = ModelTools.mlp_to_kan(model_mlp, device, grid_size=5, spline_order=1)
acc, f1, mcc = EvaluationTools.evaluate_classification(model_kan, test_loader, device)
print("KAN", ModelTools.get_nuber_parameters(model_kan), f"ACC={acc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")
# KAN 1173760 ACC=0.2596, F1=0.2645, MCC=0.1854
model_mlp2 = ModelTools.kan_to_mlp(model_kan, device)
acc, f1, mcc = EvaluationTools.evaluate_classification(model_mlp2, test_loader, device)

print("MLP2", ModelTools.get_nuber_parameters(model_mlp2), f"ACC={acc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")
# MLP2 235146 ACC=0.9609, F1=0.9604, MCC=0.9566
model_kan2 = ModelTools.mlp_to_kan(model_mlp, device, grid_size=5, spline_order=2)
acc, f1, mcc = EvaluationTools.evaluate_classification(model_kan2, test_loader, device)
print("KAN2", ModelTools.get_nuber_parameters(model_kan2), f"ACC={acc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")
# KAN2 939008 ACC=0.1299, F1=0.0944, MCC=0.0514
model_mlp3 = ModelTools.kan_to_mlp(model_kan2, device)
acc, f1, mcc = EvaluationTools.evaluate_classification(model_mlp3, test_loader, device)
print("MLP3", ModelTools.get_nuber_parameters(model_mlp3), f"ACC={acc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")
# MLP3 235146 ACC=0.9611, F1=0.9606, MCC=0.9568

model_kan3 = ModelTools.compress_kan(model_kan, test_loader, device)
acc, f1, mcc = EvaluationTools.evaluate_classification(model_kan3, test_loader, device)
print("KAN3", ModelTools.get_nuber_parameters(model_kan3), f"ACC={acc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")
# KAN3 704256 ACC=0.1034, F1=0.0549, MCC=0.0023
