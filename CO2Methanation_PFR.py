# -*- coding: utf-8 -*-
# rev finalize with good results
"""
Professional Python Developer
CO2 Methanation Reactor Model

This module implements a plug flow reactor (PFR) model for CO2 methanation reaction.
The model includes reaction kinetics, mass balances, and methods for solving the system.

Author: Mostafa Haghighi
Date: April 16, 2025
"""

import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp
from typing import Tuple, List, Union
import matplotlib.pyplot as plt


class CO2MethanationReactor:
    """
    A class representing a CO2 methanation plug flow reactor.

    This model simulates the conversion of CO2 and H2 to CH4 and H2O
    in a fixed-bed catalytic reactor under specified operating conditions.

    Reaction:
    CO2 + 4H2 ⇌ CH4 + 2H2O
    """

    def __init__(self,
                 temperature: float = 723.15,  # K
                 pressure: float = 5.0,        # bar
                 length: float = 3.0,          # m
                 diameter: float = 0.01,       # m
                 catalyst_density: float = 23000,  # kg/m³
                 bed_porosity: float = 0.4):
        """
        Initialize the reactor with physical parameters and operating conditions.

        Args:
            temperature: Reaction temperature (K)
            pressure: Total pressure (bar)
            length: Reactor length (m)
            diameter: Reactor diameter (m)
            catalyst_density: Catalyst bulk density (kg/m³)
            bed_porosity: Bed porosity (dimensionless)
        """
        # Universal constants
        self.R = 8.314  # Universal gas constant (J/mol·K)

        # Operating conditions
        self.T0 = temperature
        self.Pt = pressure
        self.L = length
        self.d = diameter
        self.rho_cat = catalyst_density
        self.epsilon = bed_porosity

        # Derived parameters
        self.At = np.pi * (self.d**2) / 4  # Reactor cross-sectional area (m²)

        # Initialize kinetic parameters
        self._init_kinetic_parameters()

        # Default inlet conditions
        self._set_default_inlet_conditions()

    def _init_kinetic_parameters(self) -> None:
        """
        Calculate kinetic parameters based on temperature.

        Uses Arrhenius equations to determine rate and equilibrium constants.
        """
        # Reference temperature for kinetic parameters = 555 K
        # Reaction rate constant (mol/kg·s)
        self.k = 6.41e-5 * np.exp((93.6e3 / self.R) * (1 / 555 - 1 / self.T0))

        # Adsorption constant (bar^-0.5)
        self.k_ads = 0.62e-5 * np.exp((64.3e3 / self.R) * (1 / 555 - 1 / self.T0))

        # Equilibrium constant (dimensionless)
        self.k_eq = 137 * self.T0**(-3.998) * np.exp(158.5e3 / (self.R * self.T0))

    def _set_default_inlet_conditions(self) -> None:
        """Set default inlet flow conditions."""
        # Initial conditions (inlet molar flow rates in mol/s)
        self.FA0 = 94.74    # CO2
        self.FB0 = 378.9    # H2
        self.FC0 = 0.0      # CH4
        self.FD0 = 0.0      # H2O

        self.inlet_flows = [self.FA0, self.FB0, self.FC0, self.FD0]

    def set_inlet_flows(self,
                      co2_flow: float,
                      h2_flow: float,
                      ch4_flow: float = 0.0,
                      h2o_flow: float = 0.0) -> None:
        """
        Set custom inlet flow rates for the reactor.

        Args:
            co2_flow: CO2 inlet flow rate (mol/s)
            h2_flow: H2 inlet flow rate (mol/s)
            ch4_flow: CH4 inlet flow rate (mol/s), default is 0
            h2o_flow: H2O inlet flow rate (mol/s), default is 0
        """
        self.FA0 = co2_flow
        self.FB0 = h2_flow
        self.FC0 = ch4_flow
        self.FD0 = h2o_flow

        self.inlet_flows = [self.FA0, self.FB0, self.FC0, self.FD0]

    def calculate_reaction_rate(self,
                             FA: Union[float, tf.Tensor],
                             FB: Union[float, tf.Tensor],
                             FC: Union[float, tf.Tensor],
                             FD: Union[float, tf.Tensor]) -> Union[float, tf.Tensor]:
        """
        Calculate reaction rate based on current flow rates.

        Args:
            FA: CO2 flow rate (mol/s)
            FB: H2 flow rate (mol/s)
            FC: CH4 flow rate (mol/s)
            FD: H2O flow rate (mol/s)

        Returns:
            Reaction rate (mol/kg_cat·s)
        """
        # Total flow rate
        Ft = FA + FB + FC + FD

        # Partial pressures (bar)
        P_CO2 = (FA/Ft) * self.Pt
        P_H2 = (FB/Ft) * self.Pt
        P_CH4 = (FC/Ft) * self.Pt
        P_H2O = (FD/Ft) * self.Pt

        # Rate equation components
        numerator = P_H2**0.31 * P_CO2**0.16
        denominator = 1 + self.k_ads * (P_H2O / tf.sqrt(P_H2))

        # Approach to equilibrium term
        driving_force = (1 - (P_CH4 * P_H2O**2) / (P_H2**4 * P_CO2 * self.k_eq))

        # Combined rate equation
        coefficient = self.rho_cat * 0.6 * self.k
        reaction_rate = coefficient * (numerator / denominator) * driving_force

        return reaction_rate

    def _reactor_odes(self, V: float, flows: List[float]) -> List[float]:
        """
        Define the system of ODEs for the plug flow reactor.

        Args:
            V: Reactor volume (independent variable)
            flows: List of flow rates [FA, FB, FC, FD]

        Returns:
            List of derivatives [dFA/dV, dFB/dV, dFC/dV, dFD/dV]
        """
        FA, FB, FC, FD = flows

        # Calculate reaction rate
        rA = self.calculate_reaction_rate(FA, FB, FC, FD)

        # Mass balance equations (mol/m³)
        dFA_dV = -rA            # CO2
        dFB_dV = -4 * rA        # H2
        dFC_dV = rA             # CH4
        dFD_dV = 2 * rA         # H2O

        return [dFA_dV, dFB_dV, dFC_dV, dFD_dV]

    def solve_reactor(self, method: str = 'RK45') -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the reactor model using numerical integration.

        Args:
            method: Integration method for solve_ivp

        Returns:
            tuple: (reactor_positions, solution_array)
                reactor_positions: Array of positions along reactor length (m)
                solution_array: Array of flow rates at each position
                                Format: [[FA1, FB1, FC1, FD1], [FA2, FB2, FC2, FD2], ...]
        """
        # Solve ODE system from 0 to reactor length
        solution = solve_ivp(
            self._reactor_odes,
            [0, self.L],
            self.inlet_flows,
            method=method,
            rtol=1e-6,
            atol=1e-8,
            dense_output=True
        )

        # Extract solution at more regularly spaced points
        positions = np.linspace(0, self.L, 100)
        flows = solution.sol(positions)

        # Convert to more readable format: rows are positions, columns are flows
        flow_array = np.vstack([flows[0], flows[1], flows[2], flows[3]]).T

        return positions, flow_array

    def calculate_conversion(self, flow_array: np.ndarray) -> np.ndarray:
        """
        Calculate CO2 conversion along the reactor.

        Args:
            flow_array: Solution array from solve_reactor

        Returns:
            Array of CO2 conversion values (0-1)
        """
        fa_values = flow_array[:, 0]  # CO2 flow rates
        conversion = (self.FA0 - fa_values) / self.FA0
        return conversion

    def calculate_selectivity(self, flow_array: np.ndarray) -> np.ndarray:
        """
        Calculate CH4 selectivity along the reactor.

        Args:
            flow_array: Solution array from solve_reactor

        Returns:
            Array of CH4 selectivity values (0-1)
        """
        fa_initial = flow_array[0, 0]  # Initial CO2
        fa_values = flow_array[:, 0]   # CO2 flows
        fc_values = flow_array[:, 2]   # CH4 flows

        co2_consumed = fa_initial - fa_values
        selectivity = fc_values / co2_consumed

        # Handle division by zero at the inlet
        selectivity[0] = selectivity[1] if len(selectivity) > 1 else 0

        return selectivity

    def plot_results(self, positions: np.ndarray, flow_array: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the results of the reactor simulation.

        Args:
            positions: Reactor positions from solve_reactor
            flow_array: Flow rates from solve_reactor

        Returns:
            fig, axs: Figure and axes objects
        """
        # Calculate conversion
        conversion = self.calculate_conversion(flow_array)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot flow rates
        ax1.plot(positions, flow_array[:, 0], 'b-', label='CO₂')
        ax1.plot(positions, flow_array[:, 1], 'r-', label='H₂')
        ax1.plot(positions, flow_array[:, 2], 'g-', label='CH₄')
        ax1.plot(positions, flow_array[:, 3], 'c-', label='H₂O')

        ax1.set_xlabel('Reactor Length (m)')
        ax1.set_ylabel('Molar Flow Rate (mol/s)')
        ax1.set_title('Species Flow Rates')
        ax1.legend()
        ax1.grid(True)

        # Plot conversion
        ax2.plot(positions, conversion * 100, 'k-', linewidth=2)
        ax2.set_xlabel('Reactor Length (m)')
        ax2.set_ylabel('CO₂ Conversion (%)')
        ax2.set_title('CO₂ Conversion Along Reactor')
        ax2.grid(True)

        plt.tight_layout()

        return fig, (ax1, ax2)


if __name__ == "__main__":
    # usage
    reactor = CO2MethanationReactor()

    # Solve the reactor model
    positions, flows = reactor.solve_reactor()

    # Plot results
    fig, axs = reactor.plot_results(positions, flows)

    # Calculate final conversion
    final_conversion = reactor.calculate_conversion(flows)[-1] * 100
    print(f"Final CO2 conversion: {final_conversion:.2f}%")

    plt.show()