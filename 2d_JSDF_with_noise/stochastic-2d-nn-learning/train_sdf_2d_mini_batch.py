import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import time
from sdf.stochastic_robot_sdf import RobotSdfCollisionNet

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_args = {'device': device, 'dtype': torch.float32}

    # -------------------------------------------------
    # LOAD + SHUFFLE DATASET
    # -------------------------------------------------
    data = np.load("../dataset/turtlebot2d_geom.npy")
    data = torch.tensor(data, **tensor_args)

    perm = torch.randperm(data.shape[0], device=device)
    data = data[perm]

    x = data[:, 0:4]
    y = data[:, 4:5]
    prior_mu  = data[:, 5:6]
    prior_var = data[:, 6:7]

    print("Dataset:", x.shape)

    # -------------------------------------------------
    # SPLIT
    # -------------------------------------------------
    n = x.shape[0]
    n_train = int(n * 0.98)
    n_val   = int(n * 0.01)

    train_set = TensorDataset(
        x[:n_train], y[:n_train],
        prior_mu[:n_train], prior_var[:n_train]
    )

    val_set = TensorDataset(
        x[n_train:n_train+n_val],
        y[n_train:n_train+n_val],
        prior_mu[n_train:n_train+n_val],
        prior_var[n_train:n_train+n_val]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=512,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=512,
        shuffle=False,
        pin_memory=True
    )

    # -------------------------------------------------
    # MODEL
    # -------------------------------------------------
    model = RobotSdfCollisionNet(
        in_channels=4,
        out_channels=1,
        layers=[128] * 4,
        skips=[]
    ).model.to(device)

    print("Parameters:", sum(p.numel() for p in model.parameters()))

    # -------------------------------------------------
    # OPTIM
    # -------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=500
    )

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # -------------------------------------------------
    # TRAIN LOOP
    # -------------------------------------------------
    epochs = 80000
    beta_KL = 1e-2
    best_val = 1e9

    for e in range(epochs):

        t0 = time.time()
        model.train()

        train_mu_mse = 0
        train_var_mse = 0
        train_loss_total = 0

        for xb, yb, mu_prior_b, var_prior_b in train_loader:

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():

                pred = model(xb)
                mu, logvar = torch.chunk(pred, 2, dim=-1)

                var = torch.exp(logvar)

                # ---------------- NLL ----------------
                inv_var = torch.exp(-logvar)
                nll = 0.5 * (logvar + (yb - mu) ** 2 * inv_var)

                # ---------------- KL ----------------
                var_prior_b = torch.clamp(var_prior_b, min=1e-8)

                kl = 0.5 * (
                    torch.log(var_prior_b)
                    - logvar
                    + (var + (mu - mu_prior_b) ** 2) / var_prior_b
                    - 1.0
                )

                loss = nll.mean() + beta_KL * kl.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item()
            train_mu_mse += F.mse_loss(mu, mu_prior_b).item()
            train_var_mse += F.mse_loss(var, var_prior_b).item()

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------
        model.eval()

        val_mu_mse = 0
        val_var_mse = 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            for xb, yb, mu_prior_b, var_prior_b in val_loader:

                pred = model(xb)
                mu, logvar = torch.chunk(pred, 2, dim=-1)
                var = torch.exp(logvar)

                val_mu_mse += F.mse_loss(mu, mu_prior_b).item()
                val_var_mse += F.mse_loss(var, var_prior_b).item()

        val_loss = val_mu_mse + val_var_mse
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict()}, "sdf_2d.pt")
            print("Saved best model:", val_loss)

        print(
            f"Epoch {e:05d} | "
            f"TrainLoss {train_loss_total:.6f} | "
            f"Train μMSE {train_mu_mse:.6f} | "
            f"Train σ²MSE {train_var_mse:.6f} | "
            f"Val μMSE {val_mu_mse:.6f} | "
            f"Val σ²MSE {val_var_mse:.6f} | "
            f"Time {time.time()-t0:.2f}s"
        )


if __name__ == "__main__":
    train()