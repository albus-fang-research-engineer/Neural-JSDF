import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import time
from scipy.io import loadmat
from sdf.robot_sdf import RobotSdfCollisionNet


def create_dataset_2d():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_args = {'device': device, 'dtype': torch.float32}

    # -------------------------------------------------
    # LOAD DATASET
    # -------------------------------------------------
    data = np.load("../dataset/turtlebot2d.npy")

    x = torch.tensor(data[:, 0:4], **tensor_args)   # [xr, yr, px, py]
    y = torch.tensor(data[:, 4:5], **tensor_args)   # [distance]

    print("Dataset:", x.shape, y.shape)

    # -------------------------------------------------
    # TRAIN / VAL / TEST SPLIT
    # -------------------------------------------------
    n = x.shape[0]

    train_ratio = 0.98
    val_ratio = 0.01

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    x_train = x[:n_train]
    y_train = y[:n_train]

    x_val = x[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]

    x_test = x[n_train+n_val:]
    y_test = y[n_train+n_val:]

    # -------------------------------------------------
    # MODEL
    # -------------------------------------------------
    in_channels = 4
    out_channels = 1

    hidden = 128
    n_layers = 4

    model = RobotSdfCollisionNet(
        in_channels=in_channels,
        out_channels=out_channels,
        layers=[hidden] * n_layers,
        skips=[]
    ).model

    model.to(**tensor_args)

    print(model)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    # -------------------------------------------------
    # OPTIMIZER
    # -------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5000
    )

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # -------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------
    epochs = 80000
    best_val = 1e9

    for e in range(epochs):

        t0 = time.time()

        # TRAIN
        model.train()

        with torch.cuda.amp.autocast():
            y_pred = model(x_train)
            train_loss = F.mse_loss(y_pred, y_train)

        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # VALIDATION
        model.eval()

        with torch.no_grad(), torch.cuda.amp.autocast():
            y_val_pred = model(x_val)
            val_loss = F.mse_loss(y_val_pred, y_val)

        scheduler.step(val_loss)

        # SAVE BEST
        if val_loss < best_val:
            best_val = val_loss
            print("saving model", val_loss.item())

            torch.save(
                {
                    "model": model.state_dict(),
                },
                "sdf_2d.pt"
            )

        print(
            f"Epoch {e:05d} | "
            f"Train {train_loss.item():.6f} | "
            f"Val {val_loss.item():.6f} | "
            f"Time {time.time()-t0:.3f}s"
        )


if __name__ == "__main__":
    create_dataset_2d()