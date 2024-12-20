# VQE
Evaluating the ground state of molecular hamiltonian by using the VQE (Variational Quantum EigenSolver) ¶
A potential energy curve provides the ground-state energy of a molecule as a function of the distances between its constituent atoms. The global minima of the curve indicates the binding energy and internuclear distance for the stable molecule. Therefore, such a curve can be a powerful tool in computational chemistry, material science, Condensed Matter Physics, Optimization Problems, and many other fields.
The global minimum of this curve corresponds to the binding energy and the equilibrium bond length, making it a critical tool for predicting molecular structure and spectra.
This project demonstrates how to use Classiq's Variational Quantum Eigensolver (VQE) package to construct a potential energy curve for a diatomic molecule.
The resulting VQE energies are compared with those obtained from the Hartree-Fock approximation and the exact solution derived from Hamiltonian diagonalization.
Through this computational approach, users can explore molecular interactions and gain insights into quantum chemistry with high precision and efficiency.

Slide Type
Slide
0. Requirments:
The model is using several Classiq's libraries

Slide Type
Sub-Slide
# Import necessary libraries
import time
import matplotlib.pyplot as plt
import numpy as np
​
# We use the Classiq platform to build our model, execute it, and get the outcome.
​
from classiq import *                                    #this imports all modules and functions from the Classiq framework.
from classiq.applications.chemistry import (
    ChemistryExecutionParameters,                        #this package configures execution settings.
    HEAParameters,                                       #this package defines the Hardware Efficient Ansatz (HEA) parameters.
    Molecule,                                            #this package represents a molecule and its properties.
    MoleculeProblem,                                     #this package defines the problem to solve, here finding molecular energy levels.
    UCCParameters,                                       #this package configures the Unitary Coupled Cluster (UCC) method for quantum chemistry simulations.
)
from classiq.execution import OptimizerType              #this package specifies the optimization algorithms for parameter tuning in VQE.
Slide Type
Slide
1. Initialization:
In this section, we define the range of internuclear distances for the model to simulate and choose the number of sampling points in this range.

Slide Type
Sub-Slide
# define the sampling parameters
num1 = 5                       # number of sampling points
start1 = 0.20                  # the sampling start distance
stop1 = 1                      # the sampling end distance
num2 = 7                       # how many sampling points
start2 = 1.4                   # the sampling start distance
stop2 = 3.5                    # the sampling end distance
​
distance = np.append(np.linspace(start1, stop1, num1), np.linspace(start2, stop2, num2))
print(distance)
[0.2  0.4  0.6  0.8  1.   1.4  1.75 2.1  2.45 2.8  3.15 3.5 ]
Slide Type
Slide
2. Define the model:
2.1. In the next several code cells we have developed the model by creating the molecule.
Slide Type
Sub-Slide
# Define the function for creating the molecule
def create_molecule(x):
    """Create a molecule with a distance 'x' between hydrogen atoms."""
    return Molecule(atoms=[("H", (0.0, 0.0, 0.0)),              # The first atom is at the origin
                           ("H", (0.0, 0.0, float(x)))          # The second atom's position depends on the distance x (along the z-axis)
                          ])
Slide Type
Slide
2.2. Then we proposed the problem where our molecule is a hydrogen atom, we used Jordan–Wigner transformation to map the Hamiltonian to the qubits. We used Z2-Symmetries to reduce the problem using known symmetries and Frozen-Core Approximation to not include the core electorns. This speed ups the calculation power.
Slide Type
Sub-Slide
# Define the problem to solve: finding the ground state energy of the hydrogen molecule
def problem(molecule):
    """Define the quantum chemistry problem."""
    return MoleculeProblem(
        molecule=molecule,                    # Associates the problem with the molecule defined above
        mapping="jordan_wigner",              # Specifies how the problem (Hamiltonian) is mapped to qubits
        z2_symmetries=True,                   # Indicates whether to reduce the problem size using known symmetries, speeding up computations
        freeze_core=True,                     # Uses the frozen-core approximation, where core electrons are not included in calculations
    )
Slide Type
Slide
2.3. After defining the problem, we developed the propsed model using Hartree-Fock approximation. We use Unitary Coupled Cluster(UCC) Ansatz to use the potential of single and double excitation, COBYLA optimization method for minimizing energy in VQE. We limit our iteration to optimize the calculation.
Slide Type
Sub-Slide
# Create a quantum model for solving the problem
def model(chemistry_problem):
    """Create the quantum model based on the chemistry problem."""
    return construct_chemistry_model(
        chemistry_problem=chemistry_problem,                  #creates a quantum model for solving the chemistry problem
        use_hartree_fock=True,                                # Specifies the Hartree-Fock initial state for the calculation
        ansatz_parameters=UCCParameters(excitations=[1, 2]),  # Specifies that the ansatz should include both single and double excitations
        execution_parameters=ChemistryExecutionParameters(
            optimizer=OptimizerType.COBYLA,                   # COBYLA optimization method is used for minimizing energy in VQE
            max_iteration=30,                                 # Limits the optimization to 30 iterations
            initial_point=None,                               # No specific initial point for the optimization
        ),
    )
Slide Type
Slide
2.4. Then we synthesize our model and execute the model to get the desired results.
Slide Type
Sub-Slide
def synthesize_and_execute(qmod, is_last=False):
    """Execute the quantum program and return results.
    Args:
        qmod: The quantum model.
        is_last (bool): If True, show the quantum program for the last execution.
    """
    quantum_program = synthesize(qmod)  # Synthesize the quantum program from the model
    
    # Show the quantum program only for the last execution
    if is_last:
        show(quantum_program)  # Display the quantum circuit
    
    job = execute(quantum_program)  # Execute the quantum program
    results_dict = job.result()[1].value  # Retrieve the results from the execution
    return results_dict
​
Slide Type
Slide
2.5. We define a function to find the exact energy of the model/system. To do so we create the operator and the convert this operator to matrix. Then we diagonalize the hamiltonian to find the exact energy eigenvalues.
Slide Type
Sub-Slide
# Compute exact energy (diagonalizing the Hamiltonian)
def exact_energy(chemistry_problem, results_dict):
    """
    Computes the exact energy of the molecule by diagonalizing the Hamiltonian
    and adding the nuclear repulsion energy from the results dictionary.
    
    Args:
        chemistry_problem: The problem that includes the molecule and its Hamiltonian.
        results_dict: The results dictionary containing nuclear repulsion energy.
​
    Returns:
        float: The computed exact energy.
    """
    # Compute exact energy (diagonalizing the Hamiltonian)
    operator = chemistry_problem.generate_hamiltonian()
    mat = operator.to_matrix()                               # Convert operator to matrix form
    w, v = np.linalg.eig(mat)                                # Diagonalize the Hamiltonian (find eigenvalues)
    
    # Calculate exact energy (lowest eigenvalue + nuclear repulsion energy)
    return np.real(min(w)) + results_dict["nuclear_repulsion_energy"]
​
Slide Type
Slide
2.6. We again define a function to store our outcome energies i.e. hartee - fock energy, VQE energy and exact energy.
Slide Type
Sub-Slide
# Store energies and intermediate results
def store_energies(results_dict, exact_energy, VQE_energy, HF_energy, exact_energies):
    """Store computed energies in respective lists."""
    VQE_energy.append(results_dict["total_energy"])
    HF_energy.append(results_dict["hartree_fock_energy"])
    
    # Ensure exact_energy is always a list
    if isinstance(exact_energy, list):
        exact_energies.extend(exact_energy)           # If it's a list, append the values
    else:
        exact_energies.append(exact_energy)           # If it's not, append the single value
​
Slide Type
Slide
3. Run the model:
Slide Type
Sub-Slide
# Main loop to process different distances
VQE_energy = []                            # Initialize as an empty list
HF_energy = []                             # Initialize as an empty list
exact_energies = []                        # Initialize as an empty list
​
# Loop through each distance value
for x in distance:                         # Loop through each distance value
    time1 = time.time()
​
    # Create the molecule for this distance
    molecule = create_molecule(x)
    
    # Solve the problem
    chemistry_problem = problem(molecule)
    
    # Create the quantum model
    qmod = model(chemistry_problem)
    
    # Check if it's the last distance and pass the flag
    # is_last = (x == distance[-1])
    
    # Synthesize and execute the quantum program
    results_dict = synthesize_and_execute(qmod, is_last=False)
    
    # Compute exact energy using the exact_energy function
    result_exact = exact_energy(chemistry_problem, results_dict)
​
    # Store energies (pass all lists here)
    store_energies(results_dict, result_exact, VQE_energy, HF_energy, exact_energies)
​
    time2 = time.time()
    print(f"Time taken for distance {x:.2f}: {(time2 - time1):.2f} seconds")
Time taken for distance 0.20: 8.89 seconds
Time taken for distance 0.40: 6.50 seconds
Time taken for distance 0.60: 7.50 seconds
Time taken for distance 0.80: 7.28 seconds
Time taken for distance 1.00: 13.49 seconds
Time taken for distance 1.40: 6.14 seconds
Time taken for distance 1.75: 7.00 seconds
Time taken for distance 2.10: 7.35 seconds
Time taken for distance 2.45: 6.67 seconds
Time taken for distance 2.80: 6.03 seconds
Time taken for distance 3.15: 7.01 seconds
Time taken for distance 3.50: 7.18 seconds
Slide Type
Slide
4. Plot the data i.e. distance versus energy graph:
Slide Type
Sub-Slide
plt.plot(distance, VQE_energy, "r--", 
         distance, HF_energy, "bs", 
         distance, exact_energies, "go")
​
​
plt.xlabel(r"distance between atoms [ $\AA$ ]")
plt.ylabel("energy [ $H_a$ ]")
plt.legend(["Classiq's VQE", "Hartree-Fock", "Exact solution"])
plt.title("Binding Curve for $H_{2}$ molecule")
plt.grid()
​
plt.show()

Slide Type
Slide
5. Results:
We have extracted total energy, the repulsion energy, the hartee-fock energy, the VQE data, the hamiltonian of the system.

Slide Type
Sub-Slide
# Extract the intermediate results and other relevant data from the results
nuclear_repulsion_energy = results_dict.get('nuclear_repulsion_energy')
total_energy = results_dict.get('total_energy')
hartree_fock_energy = results_dict.get('hartree_fock_energy')
vqe_result = results_dict.get('vqe_result')
​
intermediate_results = results_dict.get('vqe_result', {}).get('intermediate_results', [])
​
# Print the energy data first
print(f"Nuclear Repulsion Energy: {nuclear_repulsion_energy}")
print(f"Total Energy: {total_energy}")
print(f"Hartree Fock Energy: {hartree_fock_energy}")
print(f"VQE Energy: {vqe_result}")
​
​
operator = chemistry_problem.generate_hamiltonian()
gs_problem = chemistry_problem.update_problem(operator.num_qubits)
print("The hamiltonian is :", operator.show(), sep = "\n")
Nuclear Repulsion Energy: 0.15119348883428574
Total Energy: -0.9331720448208493
Hartree Fock Energy: -0.629820110121346
VQE Energy: {'energy': -1.084365533655135, 'time': 0.46463489532470703, 'solution': None, 'eigenstate': {'1': '(0.6651186454310238+0j)', '0': '(0.7467376965842826+0j)'}, 'reduced_probabilities': {'1': 0.4423828125, '0': 0.5576171875}, 'optimized_circuit_sample_results': {'vendor_format_result': {}, 'counts': {'1': 906, '0': 1142}, 'counts_lsb_right': True, 'probabilities': {}, 'parsed_states': {'1': {'qbv': [1]}, '0': {'qbv': [0]}}, 'histogram': None, 'output_qubits_map': {'qbv': [0]}, 'state_vector': None, 'parsed_state_vector_states': None, 'physical_qubits_map': {'qbv': [0]}, 'num_shots': 2048}, 'intermediate_results': [{'utc_time': '2024-12-11T23:09:53.456489Z', 'iteration_number': 1, 'parameters': [1.6049725266901218], 'mean_all_solutions': -0.7839700315356857, 'solutions': [], 'standard_deviation': 0.0}, {'utc_time': '2024-12-11T23:09:53.482540Z', 'iteration_number': 2, 'parameters': [2.604972526690122], 'mean_all_solutions': -1.052122028994532, 'solutions': [], 'standard_deviation': 0.0}, {'utc_time': '2024-12-11T23:09:53.511640Z', 'iteration_number': 3, 'parameters': [3.604972526690122], 'mean_all_solutions': -0.5289363303126603, 'solutions': [], 'standard_deviation': 0.0}, {'utc_time': '2024-12-11T23:09:53.540514Z', 'iteration_number': 4, 'parameters': [3.104972526690122], 'mean_all_solutions': -0.802297762237776, 'solutions': [], 'standard_deviation': 0.0}, {'utc_time': '2024-12-11T23:09:53.564319Z', 'iteration_number': 5, 'parameters': [2.354972526690122], 'mean_all_solutions': -1.084192940768657, 'solutions': [], 'standard_deviation': 0.0}, {'utc_time': '2024-12-11T23:09:53.589763Z', 'iteration_number': 6, 'parameters': [2.104972526690122], 'mean_all_solutions': -1.0439454080489687, 'solutions': [], 'standard_deviation': 0.0}, {'utc_time': '2024-12-11T23:09:53.619712Z', 'iteration_number': 7, 'parameters': [2.229972526690122], 'mean_all_solutions': -1.0747482983832566, 'solutions': [], 'standard_deviation': 0.0}, {'utc_time': '2024-12-11T23:09:53.647147Z', 'iteration_number': 8, 'parameters': [2.417472526690122], 'mean_all_solutions': -1.084365533655135, 'solutions': [], 'standard_deviation': 0.0}, {'utc_time': '2024-12-11T23:09:53.675409Z', 'iteration_number': 9, 'parameters': [2.448722526690122], 'mean_all_solutions': -1.0812555039164897, 'solutions': [], 'standard_deviation': 0.0}, {'utc_time': '2024-12-11T23:09:53.704191Z', 'iteration_number': 10, 'parameters': [2.401847526690122], 'mean_all_solutions': -1.0840693984901046, 'solutions': [], 'standard_deviation': 0.0}, {'utc_time': '2024-12-11T23:09:53.728125Z', 'iteration_number': 11, 'parameters': [2.425285026690122], 'mean_all_solutions': -1.0829915411510387, 'solutions': [], 'standard_deviation': 0.0}, {'utc_time': '2024-12-11T23:09:53.755470Z', 'iteration_number': 12, 'parameters': [2.413566276690122], 'mean_all_solutions': -1.0842914998638775, 'solutions': [], 'standard_deviation': 0.0}, {'utc_time': '2024-12-11T23:09:53.792016Z', 'iteration_number': 13, 'parameters': [2.419425651690122], 'mean_all_solutions': -1.0838142914581352, 'solutions': [], 'standard_deviation': 0.0}, {'utc_time': '2024-12-11T23:09:53.823269Z', 'iteration_number': 14, 'parameters': [2.416495964190122], 'mean_all_solutions': -1.0843490817015222, 'solutions': [], 'standard_deviation': 0.0}, {'utc_time': '2024-12-11T23:09:53.852318Z', 'iteration_number': 15, 'parameters': [2.417960807940122], 'mean_all_solutions': -1.084365533655135, 'solutions': [], 'standard_deviation': 0.0}, {'utc_time': '2024-12-11T23:09:53.883879Z', 'iteration_number': 16, 'parameters': [2.4175225266901217], 'mean_all_solutions': -1.084365533655135, 'solutions': [], 'standard_deviation': 0.0}], 'optimal_parameters': {'param_0': 2.417472526690122}, 'convergence_graph_str': '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', 'num_solutions': None, 'num_shots': 2048}
The hamiltonian is :
-0.773 * I
-0.008 * Z
+0.312 * X
Slide Type
Slide
# show(quantum_program)
is_last = (x == distance[-1])
​
# Synthesize and execute the quantum program: this will give the circuit diagram.
results_dict = synthesize_and_execute(qmod, is_last=is_last)
Opening: https://platform.classiq.io/circuit/f93339d2-276c-43df-a811-0c6a48c78e8a?version=0.61.0
Slide Type
Slide
After going to this above link(which is linked via API), We get the quantum circuit for this method:

Slide Type
Slide
5.1. We have presented our data in terms of iteration number, error and standard deviation.
Slide Type
Sub-Slide
# Extracting the intermediate results
intermediate_results = results_dict["vqe_result"]["intermediate_results"]
​
# Printing the table header
print("Iteration\tParameters\t\tMean All Solutions\tError (\u0394)\tStandard Deviation")
print("="*100)
​
# Loop through the intermediate results and calculate the errors
previous_mean = None
for result in intermediate_results:
    iteration = result["iteration_number"]
    parameters = ", ".join(f"{p:.6f}" for p in result["parameters"])  # Format parameters to 6 decimal places
    mean_solution = result["mean_all_solutions"]  # Current mean_all_solutions
    error = abs(mean_solution - previous_mean) if previous_mean is not None else 0.0  # Calculate error
    SD = result["standard_deviation"]
    print(f"{iteration}\t\t{parameters}\t\t{mean_solution:.6f}\t\t{error:.6f}\t\t{SD}")
    previous_mean = mean_solution  # Update the previous mean for the next iteration
​
Iteration	Parameters		Mean All Solutions	Error (Δ)	Standard Deviation
====================================================================================================
1		1.604973		-0.783970		0.000000		0.0
2		2.604973		-1.052122		0.268152		0.0
3		3.604973		-0.528936		0.523186		0.0
4		3.104973		-0.802298		0.273361		0.0
5		2.354973		-1.084193		0.281895		0.0
6		2.104973		-1.043945		0.040248		0.0
7		2.229973		-1.074748		0.030803		0.0
8		2.417473		-1.084366		0.009617		0.0
9		2.448723		-1.081256		0.003110		0.0
10		2.401848		-1.084069		0.002814		0.0
11		2.425285		-1.082992		0.001078		0.0
12		2.413566		-1.084291		0.001300		0.0
13		2.419426		-1.083814		0.000477		0.0
14		2.416496		-1.084349		0.000535		0.0
15		2.417961		-1.084366		0.000016		0.0
16		2.417523		-1.084366		0.000000		0.0
Slide Type
Slide
6. Comments:
Through this project we have seen that how the Variational Quantum Eigensolver (VQE) can accurately approximate molecular ground-state energies and potential energy curves. It gives us more accurate ground state energy compared to classical hartee - fock method. The results demonstrate that VQE provides better accuracy compared to the Hartree-Fock method by effectively capturing electron correlations. This project also give an intuition about the importance of potential energy curves in predicting molecular properties such as bond lengths, binding energies, and spectra, which are crucial for understanding molecular stability and behavior. The use of the Classiq framework simplifies the implementation of quantum algorithms, making it easier to solve complex problems in computational chemistry.

Slide Type
Slide
7. References:
Peruzzo, A., McClean, J., Shadbolt, P., Yung, M.-H., Zhou, X.-Q., Love, P. J., Aspuru-Guzik, A., & O'Brien, J. L. (2014). A variational eigenvalue solver on a photonic quantum processor. Nature Communications, 5, 4213.

McClean, J. R., Romero, J., Babbush, R., & Aspuru-Guzik, A. (2016). The theory of variational hybrid quantum-classical algorithms. New Journal of Physics, 18, 023023.

Kandala, A., Mezzacapo, A., Temme, K., Takita, M., Brink, M., Chow, J. M., & Gambetta, J. M. (2017). Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets. Nature, 549, 242-246.

Cerezo, M., Arrasmith, A., Babbush, R., Benjamin, S. C., Endo, S., Fujii, K., McClean, J. R., Mitarai, K., Yuan, X., Cincio, L., & Coles, P. J. (2021). Variational quantum algorithms. Nature Reviews Physics, 3, 625-644.

Naeij, H. R., Mahmoudi, E., Yeganeh, H. D., & Akbari, M. (2024). Molecular Electronic Structure Calculation via a Quantum Computer. arXiv preprint.

Seeley, J. T., Richard, M. J., & Love, P. J. (2012). The Bravyi-Kitaev transformation for quantum computation of electronic structure. Journal of Chemical Physics, 137, 224109.

Grimsley, H. R., Economou, S. E., Barnes, E., & Mayhall, N. J. (2019). An adaptive variational algorithm for exact molecular simulations on a quantum computer. Nature Communications, 10, 1.

Classiq Platform for running the code.

Maomin Qing and Wei Xie, Use VQE to Calculate the Ground Energy of Hydrogen Molecules on IBM Quantum, arXiv:2305.06538v1 (2023). https://arxiv.org/abs/2305.06538

Naeij, H. R., Mahmoudi, E., Davoodi Yeganeh, H., & Akbari, M. (2023). Molecular electronic structure calculation via a quantum computer. arXiv. https://arxiv.org/abs/2303.09911

