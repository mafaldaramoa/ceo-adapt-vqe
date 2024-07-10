# ADAPT-VQE Simulation Code

This repository contains code to simulate a wide array of variants of the Adaptive Derivative-Assembled Problem-Tailored (ADAPT) - Variational Quantum Eigensolver (VQE). In particular, this code was used in the following papers:

* [Reducing the Resources Required by ADAPT-VQE Using Coupled Exchange Operators and Improved Subroutines]()
* [Reducing measurement costs by recycling the Hessian in adaptive variational quantum algorithms](https://arxiv.org/abs/2401.05172)

## Installation Requirements

You can install the required packages as follows:

```
pip install qiskit[visualization]==0.43.3
pip install pyscf==2.2.0
pip install openfermion==1.5.0
pip install openfermionpyscf
pip install openfermion==1.5.1
pip install scipy==1.10.1
```

Creating a clean virtual environment using Anaconda is recommended. Note that PySCF does not support Windows. You can use Windows Subsystem for Linux (WSL) to install a Linux distribution (e.g. Ubunto), then install Anaconda.

## Running Simulations

Before running the scripts, you must append the repository path to the Python path: 

```
import sys
sys.path.append("/path/to/repo")
```

If done this way, the change is temporary. A permanent change is achieved via modification of `PYTHONPATH`.

* Test Systems

All example scripts use the $H_2$ molecule as the simplest example. All other molecules featured in the papers can be created similarly (see submodule `molecules`). 

## Simulation Time

For larger molecules, such as $H_6$, simulations might take several hours to complete. To speed up simulations, you may create an eigendecomposition of the pool (see method `create_eig_decomp` in submodule `pools` and method `load` in `algorithms.adapt_vqe`). While the eigendecomposition itself takes hours to compute, once it is created it can be used for the simulation of any system with the same number of qubits (with the same pool).

## Supported Variants

For all options regarding the ADAPT-VQE implementation, see `AdaptVQE` class constructor in `algorithms.adaptvqe`. The current implemention supports Hessian recycling [2], TETRIS [4] and orbital optimization [7], as well as a variety of selection and convergence criteria.

A variety of pool options are also supported, namely all CEO variants (OVP, MVP, DVG, DVE) [1], the qubit pool [5], the QE pool [6], and fermionic pools - GSD, SD, Spin-Adapted GSD, etc [3]. For details, see submodule `pools`. 

## References

[1] [Reducing the Resources Required by ADAPT-VQE Using Coupled Exchange Operators and Improved Subroutines]()

[2] [Reducing measurement costs by recycling the Hessian in adaptive variational quantum algorithms](https://arxiv.org/abs/2401.05172)

[3] [An adaptive variational algorithm for exact molecular simulations on a quantum compute](https://www.nature.com/articles/s41467-019-10988-2)

[4] [TETRIS-ADAPT-VQE: An adaptive algorithm that yields shallower, denser circuit ansätze](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.013254)

[5] [Qubit-ADAPT-VQE: An Adaptive Algorithm for Constructing Hardware-Efficient Ansätze on a Quantum Processor](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.020310)

[6] [Qubit-excitation-based adaptive variational quantum eigensolver](https://www.nature.com/articles/s42005-021-00730-0)

[7] [Self-Consistent Field Approach for the Variational Quantum Eigensolver: Orbital Optimization Goes Adaptive](https://pubs.acs.org/doi/10.1021/acs.jpca.3c05882)
