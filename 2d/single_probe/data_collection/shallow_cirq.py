import cirq # type: ignore
import cirq_google as cg # type: ignore
import numpy as np # type: ignore

def get_two_qubits(size, anc=False):
    if size == '3x3':
        probe_qubits = [cirq.GridQubit(4,5), cirq.GridQubit(4,7)] # at opposite faces corners
        qubits_matrix = []
        for x in [4,5,6]:
            qbs = []
            for y in [5,6,7]:
                qbs.append(cirq.GridQubit(x,y))
            qubits_matrix.append(qbs)

        # add ancilla pairs, [phy, anc]
        anc_pairs = [
            # top row
            [[cirq.GridQubit(4,5), cirq.GridQubit(3,5)],
            [cirq.GridQubit(4,6), cirq.GridQubit(3,6)],
            [cirq.GridQubit(4,7), cirq.GridQubit(3,7)],
            # bottom row
            [cirq.GridQubit(6,5), cirq.GridQubit(7,5)],
            [cirq.GridQubit(6,6), cirq.GridQubit(7,6)],
            [cirq.GridQubit(6,7), cirq.GridQubit(7,7)],
            # left column
            [cirq.GridQubit(4,5), cirq.GridQubit(4,4)],
            [cirq.GridQubit(5,5), cirq.GridQubit(5,4)],
            [cirq.GridQubit(6,5), cirq.GridQubit(6,4)],
            # right column
            [cirq.GridQubit(4,7), cirq.GridQubit(4,8)],
            [cirq.GridQubit(5,7), cirq.GridQubit(5,8)],
            [cirq.GridQubit(6,7), cirq.GridQubit(6,8)]]
        ] if anc else []

    elif size == '5x5':
        # 5x5 grid
        probe_qubits = [cirq.GridQubit(3,5), cirq.GridQubit(7,5)] # at opposite faces center
        qubits_matrix = []
        for x in [3,4,5,6,7]:
            qbs = []
            for y in [3,4,5,6,7]:
                qbs.append(cirq.GridQubit(x,y))
            qubits_matrix.append(qbs)

        # add ancilla pairs, [phy, anc]
        anc_pairs = [
            # top row
            [[cirq.GridQubit(3,3), cirq.GridQubit(2,3)],
            [cirq.GridQubit(3,4), cirq.GridQubit(2,4)],
            [cirq.GridQubit(3,5), cirq.GridQubit(2,5)],
            [cirq.GridQubit(3,6), cirq.GridQubit(2,6)],
            [cirq.GridQubit(3,7), cirq.GridQubit(2,7)],
            # bottom row
            [cirq.GridQubit(7,3), cirq.GridQubit(8,3)],
            [cirq.GridQubit(7,4), cirq.GridQubit(8,4)],
            [cirq.GridQubit(7,5), cirq.GridQubit(8,5)],
            [cirq.GridQubit(7,6), cirq.GridQubit(8,6)],
            [cirq.GridQubit(7,7), cirq.GridQubit(8,7)],
            # left column
            [cirq.GridQubit(3,3), cirq.GridQubit(3,2)],
            [cirq.GridQubit(4,3), cirq.GridQubit(4,2)],
            [cirq.GridQubit(5,3), cirq.GridQubit(5,2)],
            [cirq.GridQubit(6,3), cirq.GridQubit(6,2)],
            [cirq.GridQubit(7,3), cirq.GridQubit(7,2)],
            # right column
            [cirq.GridQubit(3,7), cirq.GridQubit(3,8)],
            [cirq.GridQubit(4,7), cirq.GridQubit(4,8)],
            [cirq.GridQubit(5,7), cirq.GridQubit(5,8)],
            [cirq.GridQubit(6,7), cirq.GridQubit(6,8)],
            [cirq.GridQubit(7,7), cirq.GridQubit(7,8)]],
            # two extra pairs
            [[cirq.GridQubit(2,5), cirq.GridQubit(1,5)],
            [cirq.GridQubit(8,5), cirq.GridQubit(9,5)]]
        ] if anc else []

    elif size == '6x6':
        # 6x6 grid
        #probe_qubits = [cirq.GridQubit(3,4), cirq.GridQubit(8,5)] # at opposite faces center
        probe_qubits = [cirq.GridQubit(3,7), cirq.GridQubit(8,7)] # d=5
        qubits_matrix = []
        for x in [3,4,5,6,7,8]:
            qbs = []
            for y in [2,3,4,5,6,7]:
                qbs.append(cirq.GridQubit(x,y))
            qubits_matrix.append(qbs)

        # add ancilla pairs, [phy, anc]
        anc_pairs = [
            # top row
            [[cirq.GridQubit(3,3), cirq.GridQubit(2,3)],
            [cirq.GridQubit(3,4), cirq.GridQubit(2,4)],
            [cirq.GridQubit(3,5), cirq.GridQubit(2,5)],
            [cirq.GridQubit(3,6), cirq.GridQubit(2,6)],
            [cirq.GridQubit(3,7), cirq.GridQubit(2,7)],
            # bottom row
            [cirq.GridQubit(8,4), cirq.GridQubit(9,4)],
            [cirq.GridQubit(8,5), cirq.GridQubit(9,5)],
            [cirq.GridQubit(8,6), cirq.GridQubit(9,6)],
            [cirq.GridQubit(8,7), cirq.GridQubit(9,7)],
            # left column
            [cirq.GridQubit(4,2), cirq.GridQubit(4,1)],
            [cirq.GridQubit(5,2), cirq.GridQubit(5,1)],
            [cirq.GridQubit(6,2), cirq.GridQubit(6,1)],
            [cirq.GridQubit(7,2), cirq.GridQubit(7,1)],
            # right column
            [cirq.GridQubit(3,7), cirq.GridQubit(3,8)],
            [cirq.GridQubit(4,7), cirq.GridQubit(4,8)],
            [cirq.GridQubit(5,7), cirq.GridQubit(5,8)],
            [cirq.GridQubit(6,7), cirq.GridQubit(6,8)],
            [cirq.GridQubit(7,7), cirq.GridQubit(7,8)],
            [cirq.GridQubit(8,7), cirq.GridQubit(8,8)]],
            # two extra pairs
            # [[cirq.GridQubit(2,4), cirq.GridQubit(1,4)],
            # [cirq.GridQubit(9,5), cirq.GridQubit(10,5)]]
        ] if anc else []

    else:
        raise ValueError(f"Invalid grid size: {size}")
    
    # get all qubits
    all_qubits = np.unique(np.concatenate(
        [np.array(qubits_matrix).flatten()]+[np.array(anc_pair_set).flatten() for anc_pair_set in anc_pairs]))
    return qubits_matrix, probe_qubits, anc_pairs, all_qubits

def get_single_qubit(size, anc=False):
    if size == '3x3':
        probe_qubits = [cirq.GridQubit(4,7)] # at top right corner
        qubits_matrix = []
        for x in [4,5,6]:
            qbs = []
            for y in [5,6,7]:
                qbs.append(cirq.GridQubit(x,y))
            qubits_matrix.append(qbs)

        # add ancilla pairs, [phy, anc]
        anc_pairs = [
            # top row
            [[cirq.GridQubit(4,5), cirq.GridQubit(3,5)],
            [cirq.GridQubit(4,6), cirq.GridQubit(3,6)],
            [cirq.GridQubit(4,7), cirq.GridQubit(3,7)],
            # bottom row
            [cirq.GridQubit(6,5), cirq.GridQubit(7,5)],
            [cirq.GridQubit(6,6), cirq.GridQubit(7,6)],
            [cirq.GridQubit(6,7), cirq.GridQubit(7,7)],
            # left column
            [cirq.GridQubit(4,5), cirq.GridQubit(4,4)],
            [cirq.GridQubit(5,5), cirq.GridQubit(5,4)],
            [cirq.GridQubit(6,5), cirq.GridQubit(6,4)],
            # right column
            [cirq.GridQubit(4,7), cirq.GridQubit(4,8)],
            [cirq.GridQubit(5,7), cirq.GridQubit(5,8)],
            [cirq.GridQubit(6,7), cirq.GridQubit(6,8)]]
        ] if anc else []

    elif size == '5x5':
        # 5x5 grid
        probe_qubits = [cirq.GridQubit(3,7)] # at top right corner
        qubits_matrix = []
        for x in [3,4,5,6,7]:
            qbs = []
            for y in [3,4,5,6,7]:
                qbs.append(cirq.GridQubit(x,y))
            qubits_matrix.append(qbs)

        # add ancilla pairs, [phy, anc]
        anc_pairs = [
            # top row
            [[cirq.GridQubit(3,3), cirq.GridQubit(2,3)],
            [cirq.GridQubit(3,4), cirq.GridQubit(2,4)],
            [cirq.GridQubit(3,5), cirq.GridQubit(2,5)],
            [cirq.GridQubit(3,6), cirq.GridQubit(2,6)],
            [cirq.GridQubit(3,7), cirq.GridQubit(2,7)],
            # bottom row
            [cirq.GridQubit(7,3), cirq.GridQubit(8,3)],
            [cirq.GridQubit(7,4), cirq.GridQubit(8,4)],
            [cirq.GridQubit(7,5), cirq.GridQubit(8,5)],
            [cirq.GridQubit(7,6), cirq.GridQubit(8,6)],
            [cirq.GridQubit(7,7), cirq.GridQubit(8,7)],
            # left column
            [cirq.GridQubit(3,3), cirq.GridQubit(3,2)],
            [cirq.GridQubit(4,3), cirq.GridQubit(4,2)],
            [cirq.GridQubit(5,3), cirq.GridQubit(5,2)],
            [cirq.GridQubit(6,3), cirq.GridQubit(6,2)],
            [cirq.GridQubit(7,3), cirq.GridQubit(7,2)],
            # right column
            [cirq.GridQubit(3,7), cirq.GridQubit(3,8)],
            [cirq.GridQubit(4,7), cirq.GridQubit(4,8)],
            [cirq.GridQubit(5,7), cirq.GridQubit(5,8)],
            [cirq.GridQubit(6,7), cirq.GridQubit(6,8)],
            [cirq.GridQubit(7,7), cirq.GridQubit(7,8)]]
        ] if anc else []

    elif size == '6x6':
        # 6x6 grid
        probe_qubits = [cirq.GridQubit(8,7)] # at top right corner
        qubits_matrix = []
        for x in [3,4,5,6,7,8]:
            qbs = []
            for y in [2,3,4,5,6,7]:
                qbs.append(cirq.GridQubit(x,y))
            qubits_matrix.append(qbs)

        # add ancilla pairs, [phy, anc]
        anc_pairs = [
            # top row
            [[cirq.GridQubit(3,3), cirq.GridQubit(2,3)],
            [cirq.GridQubit(3,4), cirq.GridQubit(2,4)],
            [cirq.GridQubit(3,5), cirq.GridQubit(2,5)],
            [cirq.GridQubit(3,6), cirq.GridQubit(2,6)],
            [cirq.GridQubit(3,7), cirq.GridQubit(2,7)],
            # bottom row
            [cirq.GridQubit(8,4), cirq.GridQubit(9,4)],
            [cirq.GridQubit(8,5), cirq.GridQubit(9,5)],
            [cirq.GridQubit(8,6), cirq.GridQubit(9,6)],
            [cirq.GridQubit(8,7), cirq.GridQubit(9,7)],
            # left column
            [cirq.GridQubit(4,2), cirq.GridQubit(4,1)],
            [cirq.GridQubit(5,2), cirq.GridQubit(5,1)],
            [cirq.GridQubit(6,2), cirq.GridQubit(6,1)],
            [cirq.GridQubit(7,2), cirq.GridQubit(7,1)],
            # right column
            [cirq.GridQubit(3,7), cirq.GridQubit(3,8)],
            [cirq.GridQubit(4,7), cirq.GridQubit(4,8)],
            [cirq.GridQubit(5,7), cirq.GridQubit(5,8)],
            [cirq.GridQubit(6,7), cirq.GridQubit(6,8)],
            [cirq.GridQubit(7,7), cirq.GridQubit(7,8)],
            [cirq.GridQubit(8,7), cirq.GridQubit(8,8)]]
        ] if anc else []

    else:
        raise ValueError(f"Invalid grid size: {size}")
    
    # get all qubits
    all_qubits = np.unique(np.concatenate(
        [np.array(qubits_matrix).flatten()]+[np.array(anc_pair_set).flatten() for anc_pair_set in anc_pairs]))
    
    return qubits_matrix, probe_qubits, anc_pairs, all_qubits

def get_circuit(qubits_matrix, theta, phi, probe_qubits, basis=[0,0], anc_pairs=[]):
    circ = cirq.Circuit()
    qubits_list = np.array(qubits_matrix).flatten()
    num_qubits = len(qubits_list)
    num_col = len(qubits_matrix[0])
    num_row = len(qubits_matrix)
    if len(basis) != len(probe_qubits):
        raise ValueError("The number of basis states must match the number of probe qubits.")

    # Hadamard on all qubits
    circ.append([cirq.H(q) for q in qubits_list])

    # Horizontal bonds
    for i in range(num_row):
        for j in range(0, num_col-1, 2):
            circ.append(cirq.CZ(qubits_matrix[i][j],qubits_matrix[i][j+1]))
            circ.append(cirq.PhasedXZGate(axis_phase_exponent=0,x_exponent=0,z_exponent=-0.5).on(qubits_matrix[i][j]))
            circ.append(cirq.PhasedXZGate(axis_phase_exponent=0,x_exponent=0,z_exponent=-0.5).on(qubits_matrix[i][j+1]))
        for j in range(1, num_col-1, 2):
            circ.append(cirq.CZ(qubits_matrix[i][j],qubits_matrix[i][j+1]))
            circ.append(cirq.PhasedXZGate(axis_phase_exponent=0,x_exponent=0,z_exponent=-0.5).on(qubits_matrix[i][j]))
            circ.append(cirq.PhasedXZGate(axis_phase_exponent=0,x_exponent=0,z_exponent=-0.5).on(qubits_matrix[i][j+1]))

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