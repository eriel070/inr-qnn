import torch
import torch.nn as nn
import numpy as np
import qutip as qt

class DQNN:
    """
    Dissipative Quantum Neural Network for audio representation.
    """
    def __init__(self, qnn_arch):
        """
        Initialize the DQNN.
        
        Args:
            qnn_arch: List specifying the number of qubits in each layer
        """
        self.qnn_arch = qnn_arch
        
        # Initialize the network unitaries
        self.unitaries = self._initialize_unitaries()
    
    def _initialize_unitaries(self):
        """Initialize the unitaries for each layer"""
        unitaries = [[]]  # Empty list for input layer
        
        for l in range(1, len(self.qnn_arch)):
            num_input_qubits = self.qnn_arch[l-1]
            num_output_qubits = self.qnn_arch[l]
            
            layer_unitaries = []
            for j in range(num_output_qubits):
                # Create a unitary for each output qubit
                if num_output_qubits - 1 != 0:
                    # If more than one output qubit, extend with identity
                    unitary = qt.tensor(self._random_qubit_unitary(num_input_qubits + 1), 
                                        self._tensoredId(num_output_qubits - 1))
                    unitary = self._swappedOp(unitary, num_input_qubits, num_input_qubits + j)
                else:
                    unitary = self._random_qubit_unitary(num_input_qubits + 1)
                
                layer_unitaries.append(unitary)
                
            unitaries.append(layer_unitaries)
            
        return unitaries
    
    def _random_qubit_unitary(self, numQubits):
        """Generate a random unitary for the specified number of qubits"""
        dim = 2 ** numQubits
        
        # Create random complex matrix
        matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
        # Orthogonalize to get a unitary
        matrix = np.linalg.qr(matrix)[0]
        
        # Convert to QuTip Qobj
        unitary = qt.Qobj(matrix)
        
        # Set dimensions
        dims = [2] * numQubits
        unitary.dims = [dims.copy(), dims.copy()]
        
        return unitary
    
    def _tensoredId(self, N):
        """Create a tensored identity operator for N qubits"""
        if N <= 0:
            return qt.qeye(1)
        
        # Create identity matrix
        res = qt.qeye(2 ** N)
        
        # Set dimensions
        dims = [2] * N
        res.dims = [dims.copy(), dims.copy()]
        
        return res
    
    def _tensoredQubit0(self, N):
        """Create a tensored |0⟩⟨0| for N qubits"""
        if N <= 0:
            return qt.qeye(1)
            
        # Create projection onto |0⟩
        res = qt.fock(2 ** N).proj()
        
        # Set dimensions
        dims = [2] * N
        res.dims = [dims.copy(), dims.copy()]
        
        return res
    
    def _swappedOp(self, obj, i, j):
        """Swap qubits i and j in the operator"""
        if i == j:
            return obj
            
        num_qubits = len(obj.dims[0])
        permute = list(range(num_qubits))
        permute[i], permute[j] = permute[j], permute[i]
        
        return obj.permute(permute)
    
    def _make_layer_channel(self, l, input_state):
        """Apply the layer channel for layer l on the input state"""
        num_input_qubits = self.qnn_arch[l-1]
        num_output_qubits = self.qnn_arch[l]
        
        # Tensor input state with |0⟩⟨0| for output qubits
        state = qt.tensor(input_state, self._tensoredQubit0(num_output_qubits))
        
        # Calculate layer unitary (product of all perceptron unitaries in the layer)
        layer_uni = self.unitaries[l][0].copy()
        for i in range(1, num_output_qubits):
            layer_uni = self.unitaries[l][i] * layer_uni
        
        # Apply unitary and trace out input qubits
        transformed_state = layer_uni * state * layer_uni.dag()
        return self._partial_trace_rem(transformed_state, list(range(num_input_qubits)))
    
    def _make_adjoint_layer_channel(self, l, output_state):
        """Apply the adjoint layer channel for layer l on the output state"""
        num_input_qubits = self.qnn_arch[l-1]
        num_output_qubits = self.qnn_arch[l]
        
        # Prepare needed states
        input_id = self._tensoredId(num_input_qubits)
        state1 = qt.tensor(input_id, self._tensoredQubit0(num_output_qubits))
        state2 = qt.tensor(input_id, output_state)
        
        # Calculate layer unitary
        layer_uni = self.unitaries[l][0].copy()
        for i in range(1, num_output_qubits):
            layer_uni = self.unitaries[l][i] * layer_uni
        
        # Apply adjoint unitary and trace out output qubits
        transformed_state = state1 * layer_uni.dag() * state2 * layer_uni
        return self._partial_trace_keep(transformed_state, list(range(num_input_qubits)))
    
    def _partial_trace_rem(self, obj, rem):
        """Partial trace removing specified qubits"""
        # Prepare keep list
        rem = sorted(rem, reverse=True)
        keep = list(range(len(obj.dims[0])))
        for x in rem:
            keep.pop(x)
        
        # Return partial trace
        if len(keep) != len(obj.dims[0]):
            return obj.ptrace(keep)
        return obj
    
    def _partial_trace_keep(self, obj, keep):
        """Partial trace keeping only specified qubits"""
        # Return partial trace
        if len(keep) != len(obj.dims[0]):
            return obj.ptrace(keep)
        return obj
    
    def quantum_forward(self, input_state):
        """
        Forward pass through the DQNN for a quantum input state.
        
        Args:
            input_state: Input quantum state
            
        Returns:
            Output quantum state
        """
        current_state = input_state
        
        # Feed forward through each layer
        for l in range(1, len(self.qnn_arch)):
            current_state = self._make_layer_channel(l, current_state)
            
        return current_state
    
    def feedforward(self, training_data):
        """
        Feed forward all training data through the network.
        
        Args:
            training_data: List of [input_state, output_state] pairs
            
        Returns:
            List of stored states for each layer
        """
        stored_states = []
        
        for x in range(len(training_data)):
            # Get input state and ensure it's a density matrix
            input_state = training_data[x][0]
            if input_state.type == 'ket':
                current_state = input_state * input_state.dag()
            else:
                current_state = input_state
                
            layerwise_list = [current_state]
            
            # Feed forward through each layer
            for l in range(1, len(self.qnn_arch)):
                current_state = self._make_layer_channel(l, current_state)
                layerwise_list.append(current_state)
                
            stored_states.append(layerwise_list)
            
        return stored_states
    
    def _calculate_loss(self, training_data, output_states):
        """Calculate the fidelity-based loss function"""
        if len(training_data) == 0:
            return 1.0
            
        loss_sum = 0
        for i in range(len(training_data)):
            target = training_data[i][1]
            output = output_states[i]
            
            # Calculate fidelity based on state types
            if target.type == 'ket' and output.type == 'oper':
                # For pure target and mixed output
                product = target.dag() * output * target
                if isinstance(product, complex):
                    fidelity = float(abs(product))
                else:
                    fidelity = float(abs(product[0,0]))
            else:
                # Convert to density matrices if needed
                if target.type != 'oper':
                    target = target * target.dag()
                if output.type != 'oper':
                    output = output * output.dag()
                fidelity = float(qt.fidelity(target, output))
            
            loss_sum += fidelity
            
        return loss_sum / len(training_data)
    
    def _update_matrix_first_part(self, l, j, x, stored_states):
        """Calculate the first part of the update matrix"""
        num_input_qubits = self.qnn_arch[l-1]
        num_output_qubits = self.qnn_arch[l]
        
        # Tensor input state with zero state for output qubits
        state = qt.tensor(stored_states[x][l-1], self._tensoredQubit0(num_output_qubits))
        
        # Calculate product unitary up to perceptron j
        product_uni = self.unitaries[l][0]
        for i in range(1, j+1):
            product_uni = self.unitaries[l][i] * product_uni
        
        # Apply unitary transformation
        return product_uni * state * product_uni.dag()
    
    def _update_matrix_second_part(self, l, j, x, training_data):
        """Calculate the second part of the update matrix"""
        num_input_qubits = self.qnn_arch[l-1]
        num_output_qubits = self.qnn_arch[l]
        
        # Calculate sigma state by back-propagating from output
        target = training_data[x][1]
        if target.type == 'ket':
            state = target * target.dag()
        else:
            state = target
            
        for i in range(len(self.qnn_arch)-1, l, -1):
            state = self._make_adjoint_layer_channel(i, state)
            
        # Tensor sigma state
        state = qt.tensor(self._tensoredId(num_input_qubits), state)
        
        # Calculate product unitary from perceptron j+1 to end
        product_uni = self._tensoredId(num_input_qubits + num_output_qubits)
        for i in range(j+1, num_output_qubits):
            product_uni = self.unitaries[l][i] * product_uni
        
        # Apply unitary transformation
        return product_uni.dag() * state * product_uni
    
    def _make_update_matrix(self, training_data, stored_states, l, j, lda, ep):
        """Calculate the update matrix for perceptron j in layer l"""
        num_input_qubits = self.qnn_arch[l-1]
        S = len(training_data)
        
        # Calculate the sum in the update formula
        sum_matrix = 0
        for x in range(S):
            # First part from feedforward
            first_part = self._update_matrix_first_part(l, j, x, stored_states)
            # Second part from backpropagation
            second_part = self._update_matrix_second_part(l, j, x, training_data)
            # Calculate commutator
            mat = qt.commutator(first_part, second_part)
            
            # Trace out the other qubits
            keep = list(range(num_input_qubits))
            keep.append(num_input_qubits + j)
            mat = self._partial_trace_keep(mat, keep)
            
            # Add to sum
            sum_matrix = sum_matrix + mat
        
        # Calculate the update matrix from the sum
        factor = -ep * (2 ** num_input_qubits) / (lda * S)
        return (factor * sum_matrix).expm()
    
    def _make_update_matrix_tensored(self, l, j, training_data, stored_states, lda, ep):
        """Create the final tensored update matrix"""
        num_input_qubits = self.qnn_arch[l-1]
        num_output_qubits = self.qnn_arch[l]
        
        # Get the basic update matrix
        res = self._make_update_matrix(training_data, stored_states, l, j, lda, ep)
        
        # Tensor with identity if needed
        if num_output_qubits - 1 != 0:
            res = qt.tensor(res, self._tensoredId(num_output_qubits - 1))
            
        # Swap qubits to get correct arrangement
        return self._swappedOp(res, num_input_qubits, num_input_qubits + j)
    
    def train(self, training_data, lda, ep, training_rounds, verbose=False):
        """
        Train the DQNN on the given training data.
        
        Args:
            training_data: List of [input_state, output_state] pairs
            lda: Lambda parameter (regularization)
            ep: Epsilon parameter (learning rate)
            training_rounds: Number of training rounds
            verbose: Whether to print training progress
            
        Returns:
            List of training loss values
        """
        s = 0
        loss_values = []
        
        # Initial feedforward
        stored_states = self.feedforward(training_data)
        
        # Calculate initial loss
        output_states = [stored_states[k][-1] for k in range(len(stored_states))]
        loss = self._calculate_loss(training_data, output_states)
        loss_values.append([s, loss])
        
        if verbose:
            print(f"Initial fidelity: {loss:.6f}")
        
        # Training rounds
        for k in range(training_rounds):
            if verbose and (k+1) % 100 == 0:
                print(f"Training round {k}, fidelity: {loss:.6f}")
                
            # Update unitaries
            for l in range(1, len(self.qnn_arch)):
                num_output_qubits = self.qnn_arch[l]
                
                # Update each perceptron
                for j in range(num_output_qubits):
                    update_matrix = self._make_update_matrix_tensored(
                        l, j, training_data, stored_states, lda, ep)
                    self.unitaries[l][j] = update_matrix * self.unitaries[l][j]
            
            # Update step counter and stored states
            s += ep
            stored_states = self.feedforward(training_data)
            
            # Calculate new loss
            output_states = [stored_states[k][-1] for k in range(len(stored_states))]
            loss = self._calculate_loss(training_data, output_states)
            loss_values.append([s, loss])
            
        if verbose:
            print(f"Final fidelity: {loss:.6f}")
            
        return loss_values
    
    def predict(self, input_state):
        """
        Predict the output for a given input state.
        
        Args:
            input_state: Input quantum state
            
        Returns:
            Output quantum state
        """
        # Convert to density matrix if needed
        if input_state.type == 'ket':
            input_state = input_state * input_state.dag()
            
        # Forward pass
        return self.quantum_forward(input_state)
    
    def count_parameters(self):
        """
        Count the number of parameters in the DQNN model.
        
        Returns:
            Total number of parameters
        """
        total_params = 0
        
        for l in range(1, len(self.qnn_arch)):
            num_input_qubits = self.qnn_arch[l-1]
            num_output_qubits = self.qnn_arch[l]
            
            # Each perceptron has 2^(m_in+1) x 2^(m_in+1) complex parameters
            # Due to unitarity, the number of free parameters is 4^(m_in+1) - 1
            params_per_perceptron = 4 ** (num_input_qubits + 1) - 1
            
            # Total parameters for this layer
            layer_params = num_output_qubits * params_per_perceptron
            
            total_params += layer_params
        
        return total_params


def get_dqnn_model(qnn_arch=None, hidden_size=2, hidden_layers=2):
    """
    Create a DQNN model for audio representation.
    
    Args:
        qnn_arch: Architecture of the DQNN, list of qubit counts per layer
        hidden_size: Number of qubits in hidden layers (if qnn_arch not provided)
        hidden_layers: Number of hidden layers (if qnn_arch not provided)
        
    Returns:
        Initialized DQNN model
    """
    if qnn_arch is None:
        # Default architecture: 1 qubit in, 1 qubit out, with hidden layers
        qnn_arch = [1] + [hidden_size] * hidden_layers + [1]
        
    model = DQNN(qnn_arch=qnn_arch)
    
    return model