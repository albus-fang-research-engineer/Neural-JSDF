import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import time
from scipy.io import loadmat
from sdf.stochastic_robot_sdf import RobotSdfCollisionNet


def create_dataset_2d():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_args = {'device': device, 'dtype': torch.float32}

    # -------------------------------------------------
    # LOAD DATASET
    # -------------------------------------------------
    data = np.load("../dataset/turtlebot2d_truelabel.npy")

    x = torch.tensor(data[:, 0:4], **tensor_args)   # [xr, yr, px, py]
    y = torch.tensor(data[:, 4:5], **tensor_args)   # [distance]
    prior_mu = y
    sigma = torch.tensor(data[:, 5:6], **tensor_args)
    prior_var = sigma ** 2
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
    prior_mu_train = prior_mu[:n_train]
    prior_var_train = prior_var[:n_train]

    x_val = x[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    prior_mu_val = prior_mu[n_train:n_train+n_val]
    prior_var_val = prior_var[n_train:n_train+n_val]

    x_test = x[n_train+n_val:]
    y_test = y[n_train+n_val:]
    prior_mu_test = prior_mu[n_train+n_val:]
    prior_var_test = prior_var[n_train+n_val:]
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
    beta_KL = 1e-3
    for e in range(epochs):

        t0 = time.time()

        # TRAIN
        model.train()

        # with torch.cuda.amp.autocast():
        #     y_pred = model(x_train)
        #     train_loss = F.mse_loss(y_pred, y_train)
        with torch.cuda.amp.autocast():

            pred = model(x_train)

            mu, logvar = torch.chunk(pred, 2, dim=-1)
            logvar = torch.clamp(logvar, -10.0, 5.0)
            # --- NLL ---
            inv_var = torch.exp(-logvar)
            nll = 0.5 * (logvar + (y_train - mu) ** 2 * inv_var)

            # --- KL ---
            prior_var_train_clamped = torch.clamp(prior_var_train, min=1e-8)
            var = torch.exp(logvar)

            kl = 0.5 * (
                torch.log(prior_var_train_clamped)
                - logvar
                + (var + (mu - prior_mu_train) ** 2) / prior_var_train_clamped
                - 1.0
            )

            # train_loss = nll.mean() + beta_KL * kl.mean()
            if e < 2000:
                train_loss = nll.mean()
            else:
                train_loss = nll.mean() + beta_KL * kl.mean()
            # Diagnostic MSEs
            mse_mu_train = F.mse_loss(mu, prior_mu_train)
            mse_var_train = F.mse_loss(var, prior_var_train)
        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # VALIDATION
        model.eval()

        with torch.no_grad(), torch.cuda.amp.autocast():
            # y_val_pred = model(x_val)
            # val_loss = F.mse_loss(y_val_pred, y_val)
            pred_val = model(x_val)

            mu_val, logvar_val = torch.chunk(pred_val, 2, dim=-1)
            var_val = torch.exp(logvar_val)

            mse_mu = F.mse_loss(mu_val, prior_mu_val)
            mse_var = F.mse_loss(var_val, prior_var_val)

            val_loss = mse_mu + mse_var
            print(
                "Sanity Check: "
                f"μ_mean {mu_val.mean().item():.4f} | "
                f"σ²_mean {var_val.mean().item():.6f}"
            )

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

        # print(
        #     f"Epoch {e:05d} | "
        #     f"Train {train_loss.item():.6f} | "
        #     f"Val {val_loss.item():.6f} | "
        #     f"Time {time.time()-t0:.3f}s"
        # )

        print(
            f"Epoch {e:05d} | "
            f"TrainLoss {train_loss.item():.6f} | "
            f"Train μMSE {mse_mu_train.item():.6f} | "
            f"Train σ²MSE {mse_var_train.item():.6f} | "
            f"Val μMSE {mse_mu.item():.6f} | "
            f"Val σ²MSE {mse_var.item():.6f} | "
            f"Time {time.time()-t0:.3f}s"
        )


if __name__ == "__main__":
    create_dataset_2d()