# run_pinn_with_module.py
"""
Example script that imports ReactorPINN from pfr_pinn_oop.py and runs it for arbitrary reactor
conditions. This file shows:
 - how to import the ReactorPINN class
 - how to define different `case` dictionaries (any reactor conditions)
 - a helper `run_case` to train, predict and plot
 - optional CLI to run a JSON case file

Usage examples:
  python run_pinn_with_module.py            # runs built-in examples
  python run_pinn_with_module.py --case case_example.json  # use a JSON case file

Note: place this file in the same folder as pfr_pinn_oop.py or install that module.
"""

import argparse
import json
from pfr_pinn_oop import ReactorPINN


def run_case(case: dict, train_epochs: int = 2000, print_every: int = 200, **pinn_kwargs):
    """Create ReactorPINN from `case`, train it and return results.

    Arguments:
        case: dictionary containing keys used by ReactorPINN (Q_CH4_in_dm3min, Q_N2_in_dm3min, T_C, x_m, p_in)
        train_epochs: number of training epochs
        print_every: print frequency
        pinn_kwargs: passed to ReactorPINN constructor (e.g., n_collocation, n_layers)
    Returns:
        pinn, results dict
    """
    pinn = ReactorPINN(case=case, **pinn_kwargs)
    print("Starting run for case. Inlet molar flows (mol/s): N_CH4_in=", pinn.N_CH4_in, "N_N2_in=", pinn.N_N2_in)

    pinn.train(epochs=train_epochs, print_every=print_every)

    results = pinn.predict()
    pinn.plot_results(results)
    return pinn, results


# -----------------------------
# Example cases
# -----------------------------
case_default = {
    "Q_CH4_in_dm3min": 5.0,
    "Q_N2_in_dm3min": 5.0,
    "T_C": [800, 850, 900, 950, 1000, 1020, 1040, 1050],
    "x_m": [0.4, 0.6, 0.9, 1.2, 1.4, 1.7, 2, 2.4],
    "p_in": 2e5
}

# Example: higher methane flow and different temperature ramp
case_high_CH4 = {
    "Q_CH4_in_dm3min": 20.0,
    "Q_N2_in_dm3min": 2.0,
    "T_C": [750, 820, 880, 920, 960, 990, 1010, 1020],
    "x_m": [0.2, 0.5, 0.9, 1.3, 1.6, 1.9, 2.2, 2.4],
    "p_in": 1.5e5
}

# Example: low pressure slow feed
case_lowp = {
    "Q_CH4_in_dm3min": 1.0,
    "Q_N2_in_dm3min": 1.0,
    "T_C": [700, 760, 800, 840, 880, 900, 920, 930],
    "x_m": [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4],
    "p_in": 1.0e5
}


def main():
    parser = argparse.ArgumentParser(description="Run ReactorPINN for arbitrary reactor conditions")
    parser.add_argument('--case', type=str, default=None, help='Path to JSON file with case dict')
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
    parser.add_argument('--print_every', type=int, default=200, help='Print frequency during training')
    args = parser.parse_args()

    if args.case:
        with open(args.case, 'r') as f:
            case = json.load(f)
        run_case(case, train_epochs=args.epochs, print_every=args.print_every)
    else:
        print('Running default example case...')
        run_case(case_default, train_epochs=args.epochs, print_every=args.print_every, n_collocation=300)

        print('\nRunning high-CH4 example case...')
        run_case(case_high_CH4, train_epochs=1500, print_every=150, n_collocation=300)

        print('\nRunning low-pressure example case...')
        run_case(case_lowp, train_epochs=1500, print_every=150, n_collocation=300)


if __name__ == '__main__':
    main()
