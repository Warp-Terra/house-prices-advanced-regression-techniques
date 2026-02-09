import os
import numpy as np

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data_processing import load_raw_data, preprocess_for_linear_baseline, ID_COL


def prepare_data(project_root):
    """Load and preprocess data, return tensors and test IDs."""
    # TODO: load raw data
    train_df, test_df = load_raw_data(project_root)
    # TODO: save test IDs for submission
    test_ids = test_df[ID_COL]
    # TODO: preprocess data
    X_train, X_test, y_log = preprocess_for_linear_baseline(train_df, test_df)
    # TODO: convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_log.values, dtype=torch.float32)
    # TODO: return X_train_tensor, X_test_tensor, y_tensor, test_ids
    return X_train_tensor, X_test_tensor, y_tensor, test_ids


def build_dataloader(X_tensor, y_tensor, batch_size):
    """Create DataLoader from tensors."""
    # TODO: create TensorDataset
    dataset = TensorDataset(X_tensor, y_tensor)
    # TODO: create and return DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def build_model(input_dim):
    """Build a simple MLP model using nn.Sequential."""
    # TODO: create nn.Sequential with:
    #   - Linear(input_dim, 64), ReLU
    #   - Linear(64, 32), ReLU
    #   - Linear(32, 1)
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
    # TODO: return model
    return model


def train_model(model, train_loader, device, config):
    """Train the model."""
    # PyTorch 没有 RMSELoss，用 MSELoss + sqrt 实现
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    for epoch in range(config["num_epochs"]):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # RMSE = sqrt(MSE)
            loss = torch.sqrt(mse_loss(outputs, labels.unsqueeze(1)))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, RMSE: {epoch_loss / len(train_loader):.4f}")
    return model


def predict(model, X_tensor, device):
    """Make predictions and inverse log transform."""
    # TODO: set model to eval mode
    model.eval()
    # TODO: with torch.no_grad(), forward pass
    with torch.no_grad():
        inputs = X_tensor.to(device)
        outputs = model(inputs)
    # TODO: apply np.expm1 to reverse log transform
    predictions = np.expm1(outputs.cpu().numpy()).flatten()
    # TODO: return predictions
    return predictions


def create_submission(test_ids, predictions, output_path):
    """Create Kaggle submission file."""
    # 自动创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # TODO: create DataFrame with Id and SalePrice
    df = pd.DataFrame({"Id": test_ids, "SalePrice": predictions})
    # TODO: save to CSV
    df.to_csv(output_path, index=False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = {
        "batch_size": 64,
        "learning_rate": 0.0005,
        "num_epochs": 2000
    }

    # TODO: set project_root
    project_root = ".."
    # TODO: prepare_data
    X_train_tensor, X_test_tensor, y_tensor, test_ids = prepare_data(project_root)
    # TODO: build_dataloader
    train_loader = build_dataloader(X_train_tensor, y_tensor, config["batch_size"])
    # TODO: build_model and move to device
    model = build_model(X_train_tensor.shape[1]).to(device)
    # TODO: train_model
    model = train_model(model, train_loader, device, config)
    # TODO: predict
    predictions = predict(model, X_test_tensor, device)
    # TODO: create_submission
    create_submission(test_ids, predictions, "../output/submission.csv")
    print("Training complete!")


if __name__ == "__main__":
    main()
