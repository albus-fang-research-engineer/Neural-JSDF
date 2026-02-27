import numpy as np

DATA_PATH = "../dataset/turtlebot2d_geom.npy"

DIST_TOL = 1e-6
VAR_TOL_ABS = 1e-9
NUM_RANDOM = 5
RADIUS = 0.105
SIGMA = 0.02
EXPECTED_VAR = SIGMA ** 2

data = np.load(DATA_PATH)

robot_xy = data[:, 0:2]
points   = data[:, 2:4]

noisy_dist = data[:, 4]
prior_mu   = data[:, 5]   # GT distance
prior_var  = data[:, 6]

# -------------------------------------------------
# helper
# -------------------------------------------------
def inspect_row(i):
    euclid_gt = np.linalg.norm(points[i] - robot_xy[i]) - RADIUS
    noise = noisy_dist[i] - prior_mu[i]

    print(f"INDEX: {i}")
    print("GT dist (stored) :", prior_mu[i])
    print("GT dist (euclid) :", euclid_gt)
    print("noisy_dist       :", noisy_dist[i])
    print("noise            :", noise)
    print("prior_var        :", prior_var[i])
    print("-" * 50)

# -------------------------------------------------
# 1) GT DISTANCE CONSISTENCY
# -------------------------------------------------
euclid_gt = np.linalg.norm(points - robot_xy, axis=1) - RADIUS
gt_error = np.abs(euclid_gt - prior_mu)
gt_fail_mask = gt_error > DIST_TOL

# -------------------------------------------------
# 2) NOISE STATISTICS
# -------------------------------------------------
noise = noisy_dist - prior_mu

noise_mean = np.mean(noise)
noise_var  = np.var(noise)

# -------------------------------------------------
# 3) VARIANCE CONSISTENCY
# -------------------------------------------------
var_error = np.abs(prior_var - EXPECTED_VAR)
var_fail_mask = var_error > VAR_TOL_ABS

# -------------------------------------------------
# STATS
# -------------------------------------------------
print("\n================ DATASET VALIDATION ================")

print("\n--- GT distance check ---")
print("Max abs error :", gt_error.max())
print("Mean abs error:", gt_error.mean())
print("Failures      :", np.sum(gt_fail_mask))

print("\n--- Noise statistics ---")
print("Mean (should be ~0)     :", noise_mean)
print("Var  (should be 0.0004) :", noise_var)

print("\n--- Stored variance check ---")
print("Max abs error :", var_error.max())
print("Mean abs error:", var_error.mean())
print("Failures      :", np.sum(var_fail_mask))

# -------------------------------------------------
# DEBUG ROWS
# -------------------------------------------------
print("\n========== FIRST 5 ROWS ==========")
for i in range(5):
    inspect_row(i)

print("\n========== RANDOM ROWS ==========")
for i in np.random.choice(len(data), NUM_RANDOM, replace=False):
    inspect_row(i)