# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 16:38:25 2025

@author: Mostafa 

Object-oriented, modular rewrite of the PINN for CH4 -> C + 2 H2 in a packed-bed PFR.

Usage: python pfr_pinn_oop.py

Contains:
 - ReactorPINN class: encapsulates model build, physics residuals, training, prediction and plotting
 - Default example case copied from the original script

Requirements: tensorflow, numpy, matplotlib
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, Optional

# -----------------------------
# Constants / helpers
# -----------------------------
R_u = 8.314459848  # J/mol/K
SMALL = 1e-12


def dm3min_to_m3s(dm3min: float) -> float:
    return dm3min * 1.66666667e-5


def rho_ideal(T: float, p: float, M: float) -> float:
    return (p * M) / (R_u * T)


# -----------------------------
# ReactorPINN class
# -----------------------------
class ReactorPINN:
    """Physics-Informed Neural Network for the packed-bed PFR reaction
       CH4 -> C + 2 H2

    The class is designed to be modular and easy to extend.
    """

    def __init__(self, case: Dict, L_tot: float = 2.5, d: float = 0.073, eps: float = 0.4,
                 Af: float = 8.5708e12, Ab: float = 1.1190e7,
                 Eaf: float = 337.12e3, Eab: float = 243.16e3,
                 nf: float = 1.123, mb: float = 0.9296,
                 M_CH4: float = 16.04e-3, M_N2: float = 28.02e-3, M_H2: float = 2.0158814e-3,
                 n_collocation: int = 300,
                 n_layers: int = 4, n_units: int = 64,
                 bc_weight_init: float = 1000.0,
                 learning_rate: float = 5e-4):
        self.case = case
        self.L_tot = L_tot
        self.d = d
        self.eps = eps
        self.Ac = 0.25 * np.pi * d ** 2

        # Kinetic parameters
        self.Af = Af
        self.Ab = Ab
        self.Eaf = Eaf
        self.Eab = Eab
        self.nf = nf
        self.mb = mb

        # molecular weights
        self.M_CH4 = M_CH4
        self.M_N2 = M_N2
        self.M_H2 = M_H2

        # collocation / discretization
        self.n_collocation = n_collocation
        self.z_phys = np.linspace(0.0, self.L_tot, self.n_collocation)

        # build temperature profile from case
        self._prepare_inlet_and_temperature()

        # compute local Arrhenius constants along z
        self.kf_z = self.Af * np.exp(-self.Eaf / (R_u * self.T_z))
        self.kb_z = self.Ab * np.exp(-self.Eab / (R_u * self.T_z))

        # tensorflow constants
        self.z_phys_tf = tf.convert_to_tensor(self.z_phys.reshape(-1, 1).astype(np.float32))
        self.kf_z_tf = tf.convert_to_tensor(self.kf_z.reshape(-1, 1).astype(np.float32))
        self.kb_z_tf = tf.convert_to_tensor(self.kb_z.reshape(-1, 1).astype(np.float32))
        self.T_z_tf = tf.convert_to_tensor(self.T_z.reshape(-1, 1).astype(np.float32))

        # scaling for normalized model outputs
        self.NCH4_scale = self.N_CH4_in
        self.NH2_scale = 2.0 * self.N_CH4_in
        self.NC_scale = self.N_CH4_in

        # normalized z for NN input
        self.z_scaled = (self.z_phys / self.L_tot).reshape(-1, 1).astype(np.float32)
        self.z_scaled_tf = tf.convert_to_tensor(self.z_scaled)

        # build model
        tf.keras.backend.set_floatx('float32')
        self.model = self.build_model(n_layers=n_layers, n_units=n_units)

        # training pieces
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.bc_weight = bc_weight_init

    def _prepare_inlet_and_temperature(self):
        # convert inlet flows and compute inlet molar flows (mol/s)
        Q_CH4_in = dm3min_to_m3s(self.case["Q_CH4_in_dm3min"])
        Q_N2_in = dm3min_to_m3s(self.case["Q_N2_in_dm3min"])
        p_in = self.case["p_in"]

        T_profile_C = np.array(self.case["T_C"])
        x_m = np.array(self.case["x_m"])
        T0_K = T_profile_C[0] + 273.15

        rho_ch4_in = rho_ideal(T0_K, p_in, self.M_CH4)
        rho_n2_in = rho_ideal(T0_K, p_in, self.M_N2)

        m_CH4_in = Q_CH4_in * rho_ch4_in
        m_N2_in = Q_N2_in * rho_n2_in

        self.N_CH4_in = m_CH4_in / self.M_CH4
        self.N_N2_in = m_N2_in / self.M_N2
        self.N_H2_in = 0.0
        self.N_C_in = 0.0

        # temperature interpolation (physical z)
        x_m_full = np.concatenate(([0.0], x_m, [self.L_tot]))
        T_C_full = np.concatenate(([20.0], T_profile_C, [T_profile_C[-1] - 30.0]))
        self.T_z = np.interp(self.z_phys, x_m_full, T_C_full) + 273.15

    def build_model(self, n_layers: int = 4, n_units: int = 64) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(1,), dtype=tf.float32)
        x = inputs
        for _ in range(n_layers):
            x = tf.keras.layers.Dense(n_units, activation='tanh',
                                      kernel_initializer='glorot_normal')(x)
        x = tf.keras.layers.Dense(3)(x)
        outputs = tf.keras.layers.Activation('softplus')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def physics_residuals(self, z_scaled_tensor: tf.Tensor, z_phys_tensor: tf.Tensor,
                          kf_tensor: tf.Tensor, kb_tensor: tf.Tensor):
        # compute residuals using inner GradientTape (w.r.t z_scaled)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(z_scaled_tensor)
            y_hat = self.model(z_scaled_tensor)

            N_CH4_hat = y_hat[:, 0:1]
            N_H2_hat = y_hat[:, 1:2]
            N_C_hat  = y_hat[:, 2:3]

            N_CH4 = N_CH4_hat * tf.cast(self.NCH4_scale, tf.float32)
            N_H2  = N_H2_hat  * tf.cast(self.NH2_scale, tf.float32)
            N_C   = N_C_hat   * tf.cast(self.NC_scale, tf.float32)

            N_CH4_safe = tf.maximum(N_CH4, SMALL)
            N_H2_safe = tf.maximum(N_H2, SMALL)

            r = kf_tensor * tf.pow(N_CH4_safe, self.nf) - kb_tensor * tf.pow(N_H2_safe, self.mb)

        # gradients w.r.t scaled coordinate
        dN_CH4_dzscaled = tape.gradient(N_CH4, z_scaled_tensor)
        dN_H2_dzscaled = tape.gradient(N_H2, z_scaled_tensor)
        dN_C_dzscaled  = tape.gradient(N_C,  z_scaled_tensor)
        del tape

        # chain rule: d/dz_phys = d/dz_scaled * (1/L_tot)
        dN_CH4_dz_phys = dN_CH4_dzscaled * (1.0 / self.L_tot)
        dN_H2_dz_phys  = dN_H2_dzscaled  * (1.0 / self.L_tot)
        dN_C_dz_phys   = dN_C_dzscaled   * (1.0 / self.L_tot)

        f_ch4 = dN_CH4_dz_phys + r * self.eps * self.Ac
        f_h2  = dN_H2_dz_phys - 2.0 * r * self.eps * self.Ac
        f_c   = dN_C_dz_phys - r * self.eps * self.Ac

        return f_ch4, f_h2, f_c

    def loss(self):
        # compute physics residuals and BC loss, return combined loss and components
        f_ch4, f_h2, f_c = self.physics_residuals(self.z_scaled_tf, self.z_phys_tf, self.kf_z_tf, self.kb_z_tf)
        physics_loss = tf.reduce_mean(tf.square(f_ch4)) + tf.reduce_mean(tf.square(f_h2)) + tf.reduce_mean(tf.square(f_c))

        z0 = tf.convert_to_tensor(np.array([[0.0]], dtype=np.float32))
        y0 = self.model(z0)
        N_target_norm = tf.convert_to_tensor(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        bc_loss = tf.reduce_mean(tf.square(y0 - N_target_norm))

        total_loss = physics_loss + self.bc_weight * bc_loss
        return total_loss, physics_loss, bc_loss

    def train(self, epochs: int = 6000, print_every: int = 500):
        for epoch in range(1, epochs + 1):
            with tf.GradientTape() as tape:
                total_loss, physics_loss, bc_loss = self.loss()

            grads = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # simple BC weight schedule example
            if epoch == int(0.4 * epochs):
                self.bc_weight *= 0.1

            if epoch % print_every == 0 or epoch == 1:
                print(f"Epoch {epoch:5d} | Loss {total_loss.numpy():.3e} | Physics {physics_loss.numpy():.3e} | BC {bc_loss.numpy():.3e}")

    def predict(self, z_plot: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        if z_plot is None:
            z_plot = np.linspace(0.0, self.L_tot, 300).reshape(-1, 1).astype(np.float32)
        z_plot_scaled = (z_plot / self.L_tot).astype(np.float32)
        Nhat = self.model.predict(z_plot_scaled, verbose=0)

        N_CH4 = Nhat[:, 0:1].flatten() * self.NCH4_scale
        N_H2  = Nhat[:, 1:2].flatten() * self.NH2_scale
        N_C   = Nhat[:, 2:3].flatten() * self.NC_scale

        return {
            'z': z_plot.flatten(),
            'N_CH4': N_CH4,
            'N_H2': N_H2,
            'N_C': N_C,
        }

    def plot_results(self, results: Dict[str, np.ndarray]):
        z_plot = results['z']
        N_CH4 = results['N_CH4']
        N_H2  = results['N_H2']
        N_C   = results['N_C']

        CH4_conv = (self.N_CH4_in - N_CH4) / self.N_CH4_in

        plt.figure(figsize=(7, 5))
        plt.plot(z_plot, CH4_conv, lw=2, label='CH4 conversion (PINN)')
        plt.xlabel('Reactor length (m)')
        plt.ylabel('CH4 conversion')
        plt.grid(True, ls='--', alpha=0.6)
        plt.legend()
        plt.title('CH4 conversion along reactor (PINN)')
        plt.show()

        plt.figure(figsize=(7, 5))
        plt.plot(z_plot, N_CH4, label='N_CH4 (mol/s)')
        plt.plot(z_plot, N_H2, label='N_H2 (mol/s)')
        plt.plot(z_plot, N_C,  label='N_C (mol/s)')
        plt.xlabel('Reactor length (m)')
        plt.ylabel('Molar flow rate (mol/s)')
        plt.grid(True, ls='--', alpha=0.6)
        plt.legend()
        plt.title('Molar flows (PINN)')
        plt.show()


# -----------------------------
# Example / main
# -----------------------------
if __name__ == '__main__':
    # example case (copied from original)
    case = {
        "Q_CH4_in_dm3min": 5.0,
        "Q_N2_in_dm3min": 5.0,
        "T_C": [800, 850, 900, 950, 1000, 1020, 1040, 1050],
        "x_m": [0.4, 0.6, 0.9, 1.2, 1.4, 1.7, 2, 2.4],
        "p_in": 2e5
    }

    pinn = ReactorPINN(case=case, n_collocation=300, n_layers=4, n_units=64)
    print("Inlet molar flows (mol/s): N_CH4_in=", pinn.N_CH4_in, " N_N2_in=", pinn.N_N2_in)

    pinn.train(epochs=2000, print_every=200)

    results = pinn.predict()
    pinn.plot_results(results)


