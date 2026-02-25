import numpy as np

DATA_PATH = "../dataset/turtlebot2d_geom.npy"

VAR_TOL_ABS = 1e-6
DIST_TOL = 1e-6
NUM_RANDOM = 5
RADIUS = 0.105

data = np.load(DATA_PATH)

robot_xy = data[:, 0:2]
points   = data[:, 2:4]

noisy_dist = data[:, 4]
prior_mu   = data[:, 5]   # gt distance
prior_var  = data[:, 6]

# -------------------------------------------------
# helper
# -------------------------------------------------
def inspect_row(i):
    r = np.linalg.norm(points[i] - robot_xy[i]) - RADIUS

    range_true = prior_mu[i] + RADIUS
    sigma = 0.002 + 0.0015 * range_true**2
    expected_var = sigma**2

    print(f"INDEX: {i}")
    print("noisy_dist     :", noisy_dist[i])
    print("euclid_noisy   :", r)
    print("prior_mu (gt)  :", prior_mu[i])
    print("prior_var      :", prior_var[i])
    print("expected_var   :", expected_var)
    print("-" * 50)

# -------------------------------------------------
# 1) DISTANCE CONSISTENCY
# -------------------------------------------------
euclid_noisy = np.linalg.norm(points - robot_xy, axis=1) - RADIUS
dist_error = np.abs(euclid_noisy - noisy_dist)
dist_fail_mask = dist_error > DIST_TOL

# -------------------------------------------------
# 2) VARIANCE MODEL CONSISTENCY
# -------------------------------------------------
ranges = prior_mu + RADIUS
expected_sigma = 0.002 + 0.0015 * ranges**2
expected_var = expected_sigma**2

var_error = np.abs(expected_var - prior_var)
var_fail_mask = var_error > VAR_TOL_ABS

# -------------------------------------------------
# STATS
# -------------------------------------------------
print("\n================ DATASET VALIDATION ================")

print("\n--- Noisy distance consistency ---")
print("Max abs error :", dist_error.max())
print("Mean abs error:", dist_error.mean())
print("Failures      :", np.sum(dist_fail_mask))

print("\n--- Variance model check ---")
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