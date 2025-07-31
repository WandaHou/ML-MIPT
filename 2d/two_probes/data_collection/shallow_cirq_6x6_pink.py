"""Quantum circuit generation for 6x6 grid systems.

This module provides functions for creating and manipulating quantum circuits
on a 6x6 grid of qubits, with support for different distance parameters and
ancilla qubits.
"""

from typing import List, Tuple, Sequence
import cirq  # type: ignore
import cirq_google as cg  # type: ignore
import numpy as np  # type: ignore

# Constants
GRID_SIZE = 6
VALID_DISTANCES = (3, 4, 5, 6)

def get_two_qubits_6x6(d: int = 6) -> Tuple[List[List[cirq.GridQubit]], 
                                           List[cirq.GridQubit], 
                                           List[List[List[cirq.GridQubit]]], 
                                           np.ndarray]:
    """Returns qubit configurations for a 6x6 grid.

    Args:
        d: Distance parameter, must be 3, 4, 5, or 6.

    Returns:
        A tuple containing:
        - qubits_matrix: 2D list of GridQubits
        - probe_qubits: List of probe qubits
        - anc_pairs: List of ancilla qubit pairs
        - all_qubits: Array of all qubits

    Raises:
        ValueError: If d is not 3, 4, 5, or 6.
    """
    if d not in VALID_DISTANCES:
        raise ValueError(f"d must be one of {VALID_DISTANCES}")

    # Define probe qubits based on distance
    if d == 6:
        probe_qubits = [cirq.GridQubit(3, 4), cirq.GridQubit(3, 9)] # this is not used
    elif d == 5:
        probe_qubits = [cirq.GridQubit(3, 4), cirq.GridQubit(3, 8)]
    elif d == 4:
        probe_qubits = [cirq.GridQubit(3, 5), cirq.GridQubit(3, 8)]
    elif d == 3:
        probe_qubits = [cirq.GridQubit(3, 5), cirq.GridQubit(3, 7)]

    # Create qubit matrix
    qubits_matrix = []
    for x in range(3, 9):
        qbs = []
        for y in range(4, 10):
            qbs.append(cirq.GridQubit(x, y))
        qubits_matrix.append(qbs)

    # Add ancilla pairs, [phy, anc]
    anc_pairs = [
        # Top row
        [[cirq.GridQubit(3, 4), cirq.GridQubit(2, 4)],
         [cirq.GridQubit(3, 5), cirq.GridQubit(2, 5)],
         [cirq.GridQubit(3, 6), cirq.GridQubit(2, 6)],
         [cirq.GridQubit(3, 7), cirq.GridQubit(2, 7)],
         [cirq.GridQubit(3, 8), cirq.GridQubit(2, 8)],
         [cirq.GridQubit(3, 9), cirq.GridQubit(2, 9)],
         # Bottom row
         [cirq.GridQubit(8, 4), cirq.GridQubit(9, 4)],
         [cirq.GridQubit(8, 5), cirq.GridQubit(9, 5)],
         [cirq.GridQubit(8, 6), cirq.GridQubit(9, 6)],
         [cirq.GridQubit(8, 7), cirq.GridQubit(9, 7)],
         [cirq.GridQubit(8, 8), cirq.GridQubit(9, 8)],
         [cirq.GridQubit(8, 9), cirq.GridQubit(9, 9)],
         # Left column
         [cirq.GridQubit(3, 4), cirq.GridQubit(3, 3)],
         [cirq.GridQubit(4, 4), cirq.GridQubit(4, 3)],
         [cirq.GridQubit(5, 4), cirq.GridQubit(5, 3)],
         [cirq.GridQubit(6, 4), cirq.GridQubit(6, 3)],
         [cirq.GridQubit(7, 4), cirq.GridQubit(7, 3)],
         [cirq.GridQubit(8, 4), cirq.GridQubit(8, 3)],
         # Right column
         [cirq.GridQubit(3, 9), cirq.GridQubit(3, 10)],
         [cirq.GridQubit(4, 9), cirq.GridQubit(4, 10)],
         [cirq.GridQubit(5, 9), cirq.GridQubit(5, 10)],
         [cirq.GridQubit(6, 9), cirq.GridQubit(6, 10)],
         [cirq.GridQubit(7, 9), cirq.GridQubit(7, 10)],
         [cirq.GridQubit(8, 9), cirq.GridQubit(8, 10)]]
    ]

    # Get all qubits
    all_qubits = np.unique(np.concatenate(
        [np.array(qubits_matrix).flatten()] + 
        [np.array(anc_pair_set).flatten() for anc_pair_set in anc_pairs]
    ))
    
    return qubits_matrix, probe_qubits, anc_pairs, all_qubits

def get_circuit(
    qubits_matrix: List[List[cirq.GridQubit]],
    theta: float,
    phi: float,
    probe_qubits: List[cirq.GridQubit],
    basis: List[int] = [0, 0],
    anc_pairs: List[List[List[cirq.GridQubit]]] = []
) -> cirq.Circuit:
    """Creates a quantum circuit with the specified parameters.

    Args:
        qubits_matrix: 2D list of GridQubits
        theta: Rotation angle for Y gate
        phi: Rotation angle for Z gate
        probe_qubits: List of probe qubits
        basis: List of basis states for probe qubits
        anc_pairs: List of ancilla qubit pairs

    Returns:
        A Cirq circuit with the specified operations

    Raises:
        ValueError: If number of basis states doesn't match probe qubits
    """
    if len(basis) != len(probe_qubits):
        raise ValueError("The number of basis states must match the number of probe qubits.")

    circ = cirq.Circuit()
    qubits_list = np.array(qubits_matrix).flatten()
    num_qubits = len(qubits_list)
    num_col = len(qubits_matrix[0])
    num_row = len(qubits_matrix)

    # Hadamard on all qubits
    circ.append([cirq.H(q) for q in qubits_list])

    # Horizontal bonds
    for i in range(num_row):
        for j in range(0, num_col-1, 2):
            circ.append(cirq.CZ(qubits_matrix[i][j], qubits_matrix[i][j+1]))
            circ.append(cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=-0.5).on(qubits_matrix[i][j]))
            circ.append(cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=-0.5).on(qubits_matrix[i][j+1]))
        for j in range(1, num_col-1, 2):
            circ.append(cirq.CZ(qubits_matrix[i][j], qubits_matrix[i][j+1]))
            circ.append(cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=-0.5).on(qubits_matrix[i][j]))
            circ.append(cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=-0.5).on(qubits_matrix[i][j+1]))

    # Vertical bonds
    for j in range(num_col):
        for i in range(0, num_row-1, 2):
            circ.append(cirq.CZ(qubits_matrix[i][j], qubits_matrix[i+1][j]))
            circ.append(cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=-0.5).on(qubits_matrix[i][j]))
            circ.append(cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=-0.5).on(qubits_matrix[i+1][j]))
        for i in range(1, num_row-1, 2):
            circ.append(cirq.CZ(qubits_matrix[i][j], qubits_matrix[i+1][j]))
            circ.append(cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=-0.5).on(qubits_matrix[i][j]))
            circ.append(cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=-0.5).on(qubits_matrix[i+1][j]))

    # Single qubit rotations
    for i in range(num_qubits):
        circ.append(cirq.rz(-phi).on(qubits_list[i]))
        circ.append(cirq.ry(-theta).on(qubits_list[i]))

    # Rotate probe qubits: x y z -> 0 1 2
    for q, b in zip(probe_qubits, basis):
        if b == 0:
            circ.append(cirq.H.on(q), strategy=cirq.circuits.InsertStrategy.INLINE)
        elif b == 1:
            circ.append(cirq.rx(np.pi/2).on(q), strategy=cirq.circuits.InsertStrategy.INLINE)

    # Ancilla pairs
    for anc_pair_set in anc_pairs:
        for anc_pair in anc_pair_set:
            circ.append(cirq.H.on(anc_pair[1]), strategy=cirq.circuits.InsertStrategy.INLINE)
        for anc_pair in anc_pair_set:
            circ.append(cirq.CZ(*anc_pair))
        for anc_pair in anc_pair_set:
            circ.append(cirq.H.on(anc_pair[1]), strategy=cirq.circuits.InsertStrategy.INLINE)
    
    return circ