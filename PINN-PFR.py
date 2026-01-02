# pfr_pinn.py
# Physics-Informed Neural Network for CH4 -> C + 2 H2 in a packed-bed PFR
# - Normalization, positivity (softplus), safe fractional powers
# - Temperature profile included (Arrhenius kinetics vary with z)
#
# Requirements: tensorflow, numpy, matplotlib
# Run: python pfr_pinn.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# -----------------------------
# Utilities & problem setup
# -----------------------------
R_u = 8.314459848  # J/mol/K
# Reactor geometry & kinetics (from your main function)
d = 0.073
eps = 0.4
L_tot = 2.5
Ac = 0.25 * np.pi * d**2

Af, Ab = 8.5708e12, 1.1190e7
Eaf, Eab = 337.12e3, 243.16e3
nf, mb = 1.123, 0.9296

# Molecular weights (kg/mol)
M_CH4 = 16.04e-3
M_N2 = 28.02e-3
M_H2 = 2.0158814e-3

# Case input (pick one of your earlier cases)
case = {
    "Q_CH4_in_dm3min": 5.0,   # dm3/min
    "Q_N2_in_dm3min": 5.0,
    "T_C": [800, 850, 900, 950, 1000, 1020, 1040, 1050],  # Celsius profile at measurement x_m
    "x_m": [0.4, 0.6, 0.9, 1.2, 1.4, 1.7, 2, 2.4],        # positions of those temps
    "p_in": 2e5  # Pa
}

# convert dm3/min to m3/s
def dm3minTom3s(dm3min):
    return dm3min * 1.66666667e-5

# Ideal gas density (kg/m^3)
def rho_ideal(T, p, M):
    return (p * M) / (R_u * T)

# -----------------------------
# Compute inlet molar flows (approx ideal gas, using inlet T at z=0)
# -----------------------------
Q_CH4_in = dm3minTom3s(case["Q_CH4_in_dm3min"])
Q_N2_in = dm3minTom3s(case["Q_N2_in_dm3min"])
p_in = case["p_in"]

# We'll set inlet temperature = first T_C entry (converted to K)
T_profile_C = np.array(case["T_C"])
x_m = np.array(case["x_m"])
T0_K = T_profile_C[0] + 273.15

rho_ch4_in = rho_ideal(T0_K, p_in, M_CH4)
rho_n2_in = rho_ideal(T0_K, p_in, M_N2)

m_CH4_in = Q_CH4_in * rho_ch4_in
m_N2_in = Q_N2_in * rho_n2_in

N_CH4_in = m_CH4_in / M_CH4   # mol/s
N_N2_in = m_N2_in / M_N2
# assume no H2 or C at inlet
N_H2_in = 0.0
N_C_in = 0.0

print("Inlet molar flows (mol/s): N_CH4_in=", N_CH4_in, " N_N2_in=", N_N2_in)

# -----------------------------
# Temperature interpolation along reactor (physical z)
# -----------------------------
# build full T(z) using linear interpolation over measured x_m and endpoints like original
x_m_full = np.concatenate(([0.0], x_m, [L_tot]))
T_C_full = np.concatenate(([20.0], T_profile_C, [T_profile_C[-1] - 30.0]))  # same trick as original
T_K_full = T_C_full + 273.15

# function to get T(z) (physical z)
def T_of_z(z_phys):
    # numpy linear interpolation: z_phys can be array
    return np.interp(z_phys, x_m_full, T_K_full)

# -----------------------------
# Discretize z for collocation & compute local Arrhenius kf/kb at those z
# -----------------------------
N_collocation = 300
z_phys = np.linspace(0.0, L_tot, N_collocation)  # physical z
T_z = T_of_z(z_phys)  # K

kf_z = Af * np.exp(-Eaf / (R_u * T_z))
kb_z = Ab * np.exp(-Eab / (R_u * T_z))

# convert to tensorflow constants (float32)
z_phys_tf = tf.convert_to_tensor(z_phys.reshape(-1, 1).astype(np.float32))
kf_z_tf = tf.convert_to_tensor(kf_z.reshape(-1, 1).astype(np.float32))
kb_z_tf = tf.convert_to_tensor(kb_z.reshape(-1, 1).astype(np.float32))
T_z_tf = tf.convert_to_tensor(T_z.reshape(-1, 1).astype(np.float32))

# -----------------------------
# Normalize inputs and outputs for better training
# - z_scaled in [0,1]
# - outputs: predict N_hat where N = N_hat * N_CH4_in (so inlet target is [1, 0, 0])
# -----------------------------
z_scaled = (z_phys / L_tot).reshape(-1, 1).astype(np.float32)
z_scaled_tf = tf.convert_to_tensor(z_scaled)

NCH4_scale = N_CH4_in  # characteristic scale for CH4
NH2_scale = 2.0 * N_CH4_in  # characteristic scale for H2 (max possible roughly 2*N_CH4_in)
NC_scale = N_CH4_in  # for solid carbon

# -----------------------------
# Build the PINN model
# -----------------------------
tf.keras.backend.set_floatx('float32')

def build_model(n_layers=4, n_units=64):
    inputs = tf.keras.Input(shape=(1,), dtype=tf.float32)
    x = inputs
    for _ in range(n_layers):
        x = tf.keras.layers.Dense(n_units, activation='tanh',
                                  kernel_initializer='glorot_normal')(x)
    x = tf.keras.layers.Dense(3)(x)           # raw outputs
    outputs = tf.keras.layers.Activation('softplus')(x)  # ensure non-negative
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = build_model(n_layers=4, n_units=64)

# -----------------------------
# Physics residual function & losses
# -----------------------------
# We'll treat the model output as normalized molar flows:
#   N_CH4(z) = model(z)[:,0] * NCH4_scale
#   N_H2 (z) = model(z)[:,1] * NH2_scale
#   N_C  (z) = model(z)[:,2] * NC_scale

# Small floor for safety before fractional powers
SMALL = 1e-12

def physics_residuals(model, z_scaled_tensor, z_phys_tensor, kf_tensor, kb_tensor):
    # z_scaled_tensor: shape (N,1) in [0,1], z_phys_tensor: (N,1) actual z in meters (float32)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(z_scaled_tensor)
        y_hat = model(z_scaled_tensor)  # non-negative via softplus
        N_CH4_hat = y_hat[:, 0:1]  # normalized
        N_H2_hat = y_hat[:, 1:2]
        N_C_hat = y_hat[:, 2:3]

        # physical molar flows
        N_CH4 = N_CH4_hat * tf.cast(NCH4_scale, tf.float32)
        N_H2  = N_H2_hat  * tf.cast(NH2_scale, tf.float32)
        N_C   = N_C_hat   * tf.cast(NC_scale, tf.float32)

        # safe positive values for fractional powers
        N_CH4_safe = tf.maximum(N_CH4, SMALL)
        N_H2_safe = tf.maximum(N_H2, SMALL)

        # reaction rate at each collocation point (use local kf/kb)
        # r has units mol/(m^3 s) if N are mol/s and dividing by reactor volumetric flow would be needed,
        # but we are using the same balance form as your original code:
        # dN/dz = +/- r * eps * Ac
        # Here we interpret r as reaction rate in mol/(m^3 s) consistent with original kf*concentration^n
        # Because we are using molar flows directly, this is an approximation similar to your discretized code.
        r = kf_tensor * tf.pow(N_CH4_safe, nf) - kb_tensor * tf.pow(N_H2_safe, mb)

    # compute dN/dz (note: N depends on z through model: z_scaled -> model)
    dN_dz = tape.gradient(N_CH4, z_scaled_tensor)  # dN_CH4/d(z_scaled)
    # chain rule: dN/d(z_phys) = dN/d(z_scaled) * d(z_scaled)/d(z_phys)
    # but z_scaled = z_phys / L_tot -> d(z_scaled)/d(z_phys) = 1/L_tot
    dN_CH4_dz_phys = dN_dz * (1.0 / L_tot)

    dN_H2_dz = tape.gradient(N_H2, z_scaled_tensor)
    dN_H2_dz_phys = dN_H2_dz * (1.0 / L_tot)

    dN_C_dz = tape.gradient(N_C, z_scaled_tensor)
    dN_C_dz_phys = dN_C_dz * (1.0 / L_tot)

    # Residuals based on balances:
    # dN_CH4/dz + r * eps * Ac = 0
    # dN_H2/dz - 2 r * eps * Ac = 0
    # dN_C/dz - r * eps * Ac = 0
    f_ch4 = dN_CH4_dz_phys + r * eps * Ac
    f_h2 = dN_H2_dz_phys - 2.0 * r * eps * Ac
    f_c  = dN_C_dz_phys - r * eps * Ac

    return f_ch4, f_h2, f_c

# -----------------------------
# Composite loss + training loop
# -----------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

# prepare tensors for training collocation points
z_scaled_all = tf.convert_to_tensor(z_scaled, dtype=tf.float32)
z_phys_all = tf.convert_to_tensor(z_phys.reshape(-1,1).astype(np.float32))
kf_all = tf.convert_to_tensor(kf_z.reshape(-1,1).astype(np.float32))
kb_all = tf.convert_to_tensor(kb_z.reshape(-1,1).astype(np.float32))

# boundary condition target in normalized units
N_target_norm = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # N_CH4 normalized = 1 at inlet

# training
N_epochs = 6000
print_every = 500

for epoch in range(1, N_epochs + 1):
    with tf.GradientTape() as tape:
        # physics residuals at collocation points
        f_ch4, f_h2, f_c = physics_residuals(model, z_scaled_all, z_phys_all, kf_all, kb_all)
        physics_loss = tf.reduce_mean(tf.square(f_ch4)) + tf.reduce_mean(tf.square(f_h2)) + tf.reduce_mean(tf.square(f_c))

        # boundary loss at z=0 (in normalized coordinates z_scaled=0)
        z0 = tf.convert_to_tensor(np.array([[0.0]], dtype=np.float32))
        y0 = model(z0)
        bc_loss = tf.reduce_mean(tf.square(y0 - N_target_norm))

        # total loss
        loss = physics_loss + 1000.0 * bc_loss  # weight boundary loss more strongly initially

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # optionally reduce BC weight after some epochs (simple schedule)
    if epoch == 2000:
        # reduce bc weight to let physics refine interior later
        pass

    if epoch % print_every == 0 or epoch == 1:
        print(f"Epoch {epoch:5d} | Loss {loss.numpy():.3e} | Physics {physics_loss.numpy():.3e} | BC {bc_loss.numpy():.3e}")

# -----------------------------
# Postprocess: predict and plot
# -----------------------------
z_plot = np.linspace(0.0, L_tot, 300).reshape(-1,1).astype(np.float32)
z_plot_scaled = (z_plot / L_tot).astype(np.float32)
Nhat_pred = model.predict(z_plot_scaled, verbose=0)

N_CH4_pred = Nhat_pred[:,0:1].flatten() * NCH4_scale
N_H2_pred  = Nhat_pred[:,1:2].flatten() * NH2_scale
N_C_pred   = Nhat_pred[:,2:3].flatten() * NC_scale

# conversion of CH4
CH4_conv = (N_CH4_in - N_CH4_pred) / N_CH4_in

plt.figure(figsize=(7,5))
plt.plot(z_plot, CH4_conv, lw=2, label='CH4 conversion (PINN)')
plt.xlabel('Reactor length (m)')
plt.ylabel('CH4 conversion')
plt.grid(True, ls='--', alpha=0.6)
plt.legend()
plt.title('CH4 conversion along reactor (PINN)')
plt.show()

plt.figure(figsize=(7,5))
plt.plot(z_plot, N_CH4_pred, label='N_CH4 (mol/s)')
plt.plot(z_plot, N_H2_pred, label='N_H2 (mol/s)')
plt.plot(z_plot, N_C_pred,  label='N_C (mol/s)')
plt.xlabel('Reactor length (m)')
plt.ylabel('Molar flow rate (mol/s)')
plt.grid(True, ls='--', alpha=0.6)
plt.legend()
plt.title('Molar flows (PINN)')
plt.show()
