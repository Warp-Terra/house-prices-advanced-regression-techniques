import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data_processing import load_raw_data, preprocess_for_linear_baseline, ID_COL

config = {
    "batch_size": 128,
    "learning_rate": 0.001,
    "num_epochs": 1000,
    "k": 5
}

def prepare_data(project_root):
    train_df, test_df = load_raw_data(project_root)
    test_ids = test_df[ID_COL]
    X_train, X_test, y_log = preprocess_for_linear_baseline(train_df, test_df)
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_log.values, dtype=torch.float32)
    return X_train_tensor, X_test_tensor, y_tensor, test_ids

def build_model(input_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

    return model

def build_dataloader(X_tensor, y_tensor, batch_size, shuffle=True):
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_model(model, train_loader, device, config):
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    for epoch in range(config["num_epochs"]):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = torch.sqrt(mse_loss(outputs, labels.unsqueeze(1)))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, RMSE: {epoch_loss / len(train_loader):.4f}")
    return model

def evaluate_model(model, val_loader, device):
    mse_loss = nn.MSELoss()
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = torch.sqrt(mse_loss(outputs, labels.unsqueeze(1)))
            total_loss += loss.item()
    return total_loss / len(val_loader)

def k_fold_data(k, i, X, y):
    fold_number = X.shape[0] // k
    X_train, y_train, X_val, y_val = None, None, None, None

    for j in range(k):
        start = j * fold_number
        end = (j + 1) * fold_number
        X_part, y_part = X[start:end], y[start:end]

        if j == i:
            X_val, y_val = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_val, y_val

def k_fold_train_and_predict(k, X, y, X_test, config, device):
    """K-Fold 交叉验证 + 集成预测"""
    val_rmse_sum = 0
    test_pred_sum = None
    for i in range(k):
        X_train, y_train, X_val, y_val = k_fold_data(k, i, X, y)
        train_loader = build_dataloader(X_train, y_train, config["batch_size"], shuffle=True)
        val_loader = build_dataloader(X_val, y_val, config["batch_size"], shuffle=False)
        model = build_model(X_train.shape[1]).to(device)
        model = train_model(model, train_loader, device, config)
        val_loss = evaluate_model(model, val_loader, device)
        val_rmse_sum += val_loss
        fold_pred = predict(model, X_test, device)
        if test_pred_sum is None:
            test_pred_sum = fold_pred
        else:
            test_pred_sum += fold_pred
        print(f"Fold {i+1}, Validation RMSE: {val_loss:.4f}")
    
    avg_rmse = val_rmse_sum / k
    print(f"\n===== K-Fold 平均验证 RMSE: {avg_rmse:.4f} =====\n")
    avg_pred = test_pred_sum / k
    return avg_rmse, avg_pred

def predict(model, X_tensor, device):
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor.to(device))
        predictions = np.expm1(outputs.cpu().numpy().flatten())
    return predictions

def create_submission(test_ids, predictions, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df = pd.DataFrame({"Id": test_ids, "SalePrice": predictions})
    df.to_csv(output_path, index=False)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    project_root = ".."
    X_train_tensor, X_test_tensor, y_tensor, test_ids = prepare_data(project_root)
    
    # 1. K-Fold 评估 + 集成预测
    print("=" * 50)
    print("K-Fold 交叉验证评估 + 集成预测")
    print("=" * 50)
    avg_rmse, predictions = k_fold_train_and_predict(
        config["k"], X_train_tensor, y_tensor, X_test_tensor, config, device
    )
    
    # 2. 用集成结果生成提交文件
    create_submission(test_ids, predictions, "../output/submission.csv")
    print("提交文件已生成: ../output/submission.csv")

if __name__ == "__main__":
    main()
