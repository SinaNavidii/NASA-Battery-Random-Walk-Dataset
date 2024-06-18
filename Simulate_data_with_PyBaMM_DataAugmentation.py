import random
import pybamm
import matplotlib.pyplot as plt

# Set plot parameters
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '20'

# Define model options
options = {
    "SEI": "reaction limited",
    "SEI porosity change": "true",
    "lithium plating": "irreversible",
    "lithium plating porosity change": "true",
    "particle mechanics": "swelling and cracking",
    "SEI on cracks": "true",
    "loss of active material": "stress-driven",
    "calculate discharge energy": "true",
}

def create_model_and_parameters():
    model = pybamm.lithium_ion.DFN(options)
    parameter_values = pybamm.ParameterValues("OKane2022")

    parameter_values.update({
        'Lower voltage cut-off [V]': "3.2",
        'Initial concentration in positive electrode [mol.m-3]': "20500.0",
        'Initial concentration in negative electrode [mol.m-3]': "1250.0",
        'Maximum concentration in positive electrode [mol.m-3]': "53000.0",
        'Nominal cell capacity [A.h]': "2.1",
        'Positive electrode thickness [m]': '9.5e-05',
        'Negative electrode thickness [m]': '9e-05',
        'Electrode width [m]': "2.4",
        'Current function [A]': '2.1',
        'Ambient temperature [K]': '313.15'
    })
    
    return model, parameter_values

def create_random_experiment(num_iterations):
    RW_experiment = []
    for _ in range(num_iterations):
        discharge_rates = tuple(f"Discharge at {random.uniform(0.5, 5)}A for 3 minutes" for _ in range(1))
        discharge_rates += (
            f"Discharge at {random.uniform(0.5, 5)}A until 3.2 V",
            "Charge at 2 A until 4.2 V",
            "Hold at 4.2 V until 20 mA"
        )
        RW_experiment.append(discharge_rates)
    return pybamm.Experiment(RW_experiment)

def create_RPT_experiment():
    return pybamm.Experiment([("Discharge at 0.2A until 3.2V",)])

def solve_cycles(model, parameter_values, num_iterations, num_repeats):
    RW_sols = []
    RPT_sols = []

    for i in range(num_repeats):
        RW_experiment = create_random_experiment(num_iterations)
        RPT_experiment = create_RPT_experiment()
        
        if i == 0:
            sim = pybamm.Simulation(model, experiment=RW_experiment, parameter_values=parameter_values)
            RW_sol = sim.solve(calc_esoh=False)
            sim = pybamm.Simulation(model, experiment=RPT_experiment, parameter_values=parameter_values)
            RPT_sol = sim.solve(starting_solution=RW_sol, calc_esoh=False)
        else:
            sim = pybamm.Simulation(model, experiment=RW_experiment, parameter_values=parameter_values)
            RW_sol = sim.solve(starting_solution=RPT_sol, calc_esoh=False)
            sim = pybamm.Simulation(model, experiment=RPT_experiment, parameter_values=parameter_values)
            RPT_sol = sim.solve(starting_solution=RW_sol, calc_esoh=False)
        
        RW_sols.append(RW_sol)
        RPT_sols.append(RPT_sol)

    return RW_sols, RPT_sols, RPT_sol

def plot_results(RPT_sol, num_iterations, num_repeats):
    RW_cycles = []
    RW_capacities = []
    RPT_cycles = []
    RPT_capacities = []

    for i in range(num_repeats):
        for j in range(num_iterations):
            RW_cycles.append(i * (num_iterations + 1) + j + 1)
            start_capacity = RPT_sol.cycles[i * (num_iterations + 1) + j].steps[2]["Discharge capacity [A.h]"].entries[0]
            end_capacity = RPT_sol.cycles[i * (num_iterations + 1) + j].steps[2]["Discharge capacity [A.h]"].entries[-1]
            RW_capacities.append(end_capacity - start_capacity)
        
        RPT_cycles.append(((i + 1) * (num_iterations + 1)) / 50)
        start_capacity = RPT_sol.cycles[(i + 1) * (num_iterations + 1) - 1]["Discharge capacity [A.h]"].entries[0]
        end_capacity = RPT_sol.cycles[(i + 1) * (num_iterations + 1) - 1]["Discharge capacity [A.h]"].entries[-1]
        RPT_capacities.append(end_capacity - start_capacity)
    
    plt.figure(figsize=(8, 6), dpi=600)
    plt.scatter(RPT_cycles, RPT_capacities, label="RPT cycles", marker='^', s=250, color='crimson')
    plt.xlabel("Load number")
    plt.ylabel("Discharge capacity [A.h]")
    plt.yticks([1.2, 1.4, 1.6, 1.8, 2])
    plt.legend()
    plt.show()

# Main execution
if __name__ == "__main__":
    num_iterations = 50
    num_repeats = 20

    model, parameter_values = create_model_and_parameters()
    RW_sols, RPT_sols, final_RPT_sol = solve_cycles(model, parameter_values, num_iterations, num_repeats)
    final_RPT_sol.save("RPT_sol_20RPT_50RWcycles_v0_calcesohFALSE_(0.5-4).pkl")
    plot_results(final_RPT_sol, num_iterations, num_repeats)
