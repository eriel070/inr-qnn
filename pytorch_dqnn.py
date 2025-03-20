import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Union
from tqdm import tqdm

class UnitaryModule(nn.Module):
    """
    Wrapper module for a unitary matrix parameter.
    This allows us to register unitaries as proper PyTorch modules.
    """
    def __init__(self, unitary_tensor: torch.Tensor):
        super().__init__()
        self.unitary = nn.Parameter(unitary_tensor)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the unitary transform to input"""
        return self.unitary @ x @ self.unitary.conj().T
    
    def matrix(self) -> torch.Tensor:
        """Get the unitary matrix"""
        return self.unitary

class PyTorchDQNN(nn.Module):
    """
    PyTorch-native implementation of a Dissipative Quantum Neural Network (DQNN)
    for Implicit Neural Representation of audio signals.
    
    This implementation doesn't rely on QuTiP and implements all quantum
    operations directly in PyTorch.
    """
    
    def __init__(
        self, 
        qnn_arch: List[int], 
        device: Optional[str] = None, 
        dtype: torch.dtype = torch.complex128
    ):
        """
        Initialize a DQNN with the specified architecture.
        
        Args:
            qnn_arch: List of integers specifying the number of qubits in each layer
                     [input_qubits, hidden1_qubits, ..., output_qubits]
            device: Device to run the PyTorch tensors on ('cpu' or 'cuda')
                   If None, selects cuda if available, otherwise cpu
            dtype: Data type for complex number calculations
        """
        super(PyTorchDQNN, self).__init__()
        
        # Auto-select device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.dtype = dtype
        self.qnn_arch = qnn_arch
        self.num_layers = len(qnn_arch) - 1  # Number of perceptron layers
        
        # Check architecture validity
        if len(qnn_arch) < 2:
            raise ValueError("QNN architecture must have at least input and output layers")
        
        if qnn_arch[0] != 1:
            raise ValueError("Input layer must have 1 qubit for time encoding")
            
        if qnn_arch[-1] != 1:
            raise ValueError("Output layer must have 1 qubit for amplitude output")
        
        # Initialize network unitaries as PyTorch modules
        self.layer_unitaries = nn.ModuleList()
        
        for l in range(1, len(qnn_arch)):
            num_input_qubits = qnn_arch[l-1]
            num_output_qubits = qnn_arch[l]
            
            layer_unitaries = nn.ModuleList()
            for j in range(num_output_qubits):
                # For each perceptron, create a unitary matrix parameter
                # Dimension: 2^(input_qubits + 1) x 2^(input_qubits + 1)
                dim = 2 ** (num_input_qubits + 1)
                unitary = self._random_unitary(dim)
                # Create a module to wrap the unitary parameter
                unitary_module = UnitaryModule(unitary)
                layer_unitaries.append(unitary_module)
                
            self.layer_unitaries.append(layer_unitaries)
            
        # Initialize matrices for Pauli operators
        self._init_pauli_matrices()
        
        # Store training data for reference
        self.training_data = None
        self.current_loss = 0.0
        
        # Move model to specified device
        self.to(self.device)
    
    def _random_unitary(self, dim: int) -> torch.Tensor:
        """
        Generate a random unitary matrix of given dimension.
        
        Args:
            dim: Matrix dimension
            
        Returns:
            A random unitary matrix
        """
        # Create random complex matrix
        mat = torch.randn(dim, dim, dtype=self.dtype, device=self.device) + \
              1j * torch.randn(dim, dim, dtype=self.dtype, device=self.device)
        
        # Perform QR decomposition for orthogonalization
        q, r = torch.linalg.qr(mat)
        
        # Adjust phases to ensure determinant is 1
        phases = torch.diag(r) / torch.abs(torch.diag(r))
        phases = torch.diag(phases)
        unitary = q @ phases
        
        return unitary
    
    def _init_pauli_matrices(self) -> None:
        """Initialize the Pauli matrices for quantum operations."""
        # Pauli matrices
        self.pauli_i = torch.tensor([[1, 0], [0, 1]], dtype=self.dtype, device=self.device)
        self.pauli_x = torch.tensor([[0, 1], [1, 0]], dtype=self.dtype, device=self.device)
        self.pauli_y = torch.tensor([[0, -1j], [1j, 0]], dtype=self.dtype, device=self.device)
        self.pauli_z = torch.tensor([[1, 0], [0, -1]], dtype=self.dtype, device=self.device)
        
        # Projection operators
        self.proj_0 = torch.tensor([[1, 0], [0, 0]], dtype=self.dtype, device=self.device)
        self.proj_1 = torch.tensor([[0, 0], [0, 1]], dtype=self.dtype, device=self.device)
    
    def _tensor_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute tensor product between two matrices.
        
        Args:
            a, b: Input matrices
            
        Returns:
            Tensor product a ⊗ b
        """
        return torch.kron(a, b)
    
    def _partial_trace(self, rho: torch.Tensor, dim_keep: int, dim_trace: int) -> torch.Tensor:
        """
        Compute partial trace of a density matrix.
        
        Args:
            rho: Density matrix
            dim_keep: Dimension of subsystem to keep
            dim_trace: Dimension of subsystem to trace out
            
        Returns:
            Partially traced density matrix
        """
        # Reshape to tensor product structure
        rho_reshaped = rho.reshape(dim_keep, dim_trace, dim_keep, dim_trace)
        
        # Trace out second subsystem
        result = torch.trace(rho_reshaped.permute(0, 2, 1, 3))
        
        return result
    
    def _time_to_qubit_state(self, time_value: float) -> torch.Tensor:
        """
        Convert a classical time value to a quantum state.
        
        For simplicity, we use amplitude encoding: 
        |ψ(t)⟩ = cos(πt/2)|0⟩ + sin(πt/2)|1⟩
        
        Args:
            time_value: Normalized time value in [0, 1]
            
        Returns:
            Quantum state vector representing the time value
        """
        # Create a superposition state based on time
        theta = np.pi * time_value / 2
        state_vector = torch.tensor(
            [np.cos(theta), np.sin(theta)], 
            dtype=self.dtype, 
            device=self.device
        )
        
        return state_vector
    
    def _amplitude_to_qubit_state(self, amplitude: float) -> torch.Tensor:
        """
        Convert a classical amplitude value to a quantum state.
        
        Args:
            amplitude: Normalized amplitude in [-1, 1]
            
        Returns:
            Quantum state representing the amplitude
        """
        # Map amplitude to angle in [0, π]
        theta = (amplitude + 1) * np.pi / 4
        
        # Create quantum state
        state_vector = torch.tensor(
            [np.cos(theta), np.sin(theta)], 
            dtype=self.dtype, 
            device=self.device
        )
        
        return state_vector
    
    def _qubit_state_to_amplitude(self, state: torch.Tensor) -> float:
        """
        Convert a quantum state to a classical amplitude value.
        
        We use the expectation value of the Z operator (scaled and shifted)
        to map from quantum states back to classical amplitudes in [-1, 1].
        
        Args:
            state: Single-qubit quantum state density matrix
            
        Returns:
            Amplitude value in [-1, 1]
        """
        # Calculate expectation value of Pauli-Z
        expectation = torch.trace(state @ self.pauli_z).real
        
        return expectation.item()
    
    def _make_layer_channel(self, l: int, input_state: torch.Tensor) -> torch.Tensor:
        """
        Apply the layer unitaries to process the input state through layer l.
        
        Args:
            l: Layer index
            input_state: Quantum state from previous layer (density matrix)
            
        Returns:
            Output quantum state after processing through layer l
        """
        num_input_qubits = self.qnn_arch[l-1]
        num_output_qubits = self.qnn_arch[l]
        
        # Tensor input state with |0⟩⟨0| states for the output layer
        output_state = self.proj_0
        for _ in range(num_output_qubits-1):
            output_state = self._tensor_product(output_state, self.proj_0)
        
        state = self._tensor_product(input_state, output_state)
        
        # Apply unitaries for this layer
        for j in range(num_output_qubits):
            # Get unitary matrix for this perceptron
            unitary = self.layer_unitaries[l-1][j].matrix()
            # Apply unitary transformation
            state = unitary @ state @ unitary.conj().T
            
        # Trace out input qubits to get output state
        dim_input = 2 ** num_input_qubits
        dim_output = 2 ** num_output_qubits
        
        # Trace out input qubits
        output = self._partial_trace(state, dim_output, dim_input)
        
        return output
    
    def _make_adjoint_layer_channel(self, l: int, output_state: torch.Tensor) -> torch.Tensor:
        """
        Apply the adjoint channel for back-propagation through layer l.
        
        Args:
            l: Layer index
            output_state: Quantum state from the next layer
            
        Returns:
            Input quantum state obtained through the adjoint channel
        """
        num_input_qubits = self.qnn_arch[l-1]
        num_output_qubits = self.qnn_arch[l]
        
        # Prepare input identity
        input_id = torch.eye(2**num_input_qubits, dtype=self.dtype, device=self.device)
        
        # Prepare |0⟩⟨0| tensor product for output qubits
        output_zero = self.proj_0
        for _ in range(num_output_qubits-1):
            output_zero = self._tensor_product(output_zero, self.proj_0)
        
        # State1 = input_id ⊗ |0...0⟩⟨0...0|
        state1 = self._tensor_product(input_id, output_zero)
        
        # State2 = input_id ⊗ output_state
        state2 = self._tensor_product(input_id, output_state)
        
        # Get unitaries for this layer
        # We need to apply them in reverse for the adjoint channel
        unitaries = [self.layer_unitaries[l-1][j].matrix() for j in range(num_output_qubits)]
        
        # Apply adjoint unitaries in reverse order
        result = state1
        for j in range(num_output_qubits-1, -1, -1):
            unitary = unitaries[j]
            result = unitary.conj().T @ state2 @ unitary
            state2 = result
        
        # Trace out output qubits, keeping only input qubits
        dim_input = 2 ** num_input_qubits
        dim_output = 2 ** num_output_qubits
        
        # Trace out the second subsystem (output qubits)
        input_state = self._partial_trace(result, dim_input, dim_output)
        
        return input_state
    
    def forward(self, time_values: torch.Tensor) -> torch.Tensor:
        """
        Process time values through the DQNN to produce amplitude predictions.
        
        Args:
            time_values: Tensor of normalized time values in [0, 1]
            
        Returns:
            Tensor of predicted amplitude values
        """
        batch_size = time_values.shape[0]
        amplitudes = torch.zeros(batch_size, device=self.device)
        
        # Process each time value individually
        for i in range(batch_size):
            t = time_values[i].item()
            
            # Convert time to quantum state
            input_state_vec = self._time_to_qubit_state(t)
            
            # Convert to density matrix
            input_state = input_state_vec.unsqueeze(1) @ input_state_vec.conj().unsqueeze(0)
            
            # Process through all layers
            current_state = input_state
            for l in range(1, len(self.qnn_arch)):
                current_state = self._make_layer_channel(l, current_state)
                
            # Convert output to amplitude
            amplitudes[i] = self._qubit_state_to_amplitude(current_state)
            
        return amplitudes
    
    def _calculate_fidelity(self, state1: torch.Tensor, state2: torch.Tensor) -> float:
        """
        Calculate quantum fidelity between two density matrices.
        
        For pure states: F(|ψ⟩, |φ⟩) = |⟨ψ|φ⟩|²
        For mixed states: F(ρ, σ) = Tr(√(√ρ σ √ρ))²
        
        Args:
            state1, state2: Quantum states as density matrices
            
        Returns:
            Fidelity value in [0, 1]
        """
        # Check if states are pure (rank-1)
        try:
            rank1 = torch.matrix_rank(state1) <= 1
            rank2 = torch.matrix_rank(state2) <= 1
            
            if rank1 and rank2:
                # For pure states, use simpler formula
                # Get dominant eigenvectors
                evals1, evecs1 = torch.linalg.eigh(state1)
                evals2, evecs2 = torch.linalg.eigh(state2)
                idx1 = torch.argmax(evals1.real)
                idx2 = torch.argmax(evals2.real)
                vec1 = evecs1[:, idx1]
                vec2 = evecs2[:, idx2]
                
                # Calculate |⟨ψ|φ⟩|²
                overlap = torch.abs(vec1.conj().T @ vec2) ** 2
                return overlap.item()
        except:
            # Fall back to general formula for any issues
            pass
            
        # For mixed states or if pure state check failed
        # Use formula F(ρ,σ) = Tr(√ρσ√ρ)²
        # We compute √ρσ√ρ using eigendecomposition
        try:
            sqrt_state1 = self._matrix_sqrt(state1)
            product = sqrt_state1 @ state2 @ sqrt_state1
            
            # Compute eigenvalues
            evals = torch.linalg.eigvalsh(product)
            # Sum of positive eigenvalues (numerical stability)
            fidelity = torch.sum(torch.clamp(evals.real, min=0, max=None))
            return fidelity.item()
        except:
            # If all else fails, use the trace distance
            diff = state1 - state2
            trace_dist = 0.5 * torch.norm(diff, p='nuc')
            fidelity = 1 - trace_dist.item()
            return max(0, min(fidelity, 1))  # Clamp to [0,1]
    
    def _matrix_sqrt(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute the matrix square root.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Square root of the matrix
        """
        # Use eigendecomposition for matrix sqrt
        evals, evecs = torch.linalg.eigh(matrix)
        # Ensure eigenvalues are positive (numerical stability)
        sqrt_evals = torch.sqrt(torch.clamp(evals.real, min=1e-10))
        # Reconstruct matrix
        sqrt_matrix = evecs @ torch.diag(sqrt_evals) @ evecs.conj().T
        return sqrt_matrix
    
    def calculate_loss(self, 
                      time_values: torch.Tensor, 
                      target_amplitudes: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss based on quantum fidelity between output and target states.
        
        Args:
            time_values: Tensor of normalized time values
            target_amplitudes: Tensor of target amplitudes
            
        Returns:
            Loss value as a PyTorch tensor
        """
        batch_size = time_values.shape[0]
        loss = torch.tensor(0.0, device=self.device)
        
        for i in range(batch_size):
            t = time_values[i].item()
            target_amp = target_amplitudes[i].item()
            
            # Convert time to input quantum state
            input_state_vec = self._time_to_qubit_state(t)
            input_state = input_state_vec.unsqueeze(1) @ input_state_vec.conj().unsqueeze(0)
            
            # Create target quantum state based on amplitude
            target_state_vec = self._amplitude_to_qubit_state(target_amp)
            target_state = target_state_vec.unsqueeze(1) @ target_state_vec.conj().unsqueeze(0)
            
            # Process through network
            current_state = input_state
            for l in range(1, len(self.qnn_arch)):
                current_state = self._make_layer_channel(l, current_state)
            
            # Calculate fidelity between output and target
            fidelity = self._calculate_fidelity(current_state, target_state)
            
            # Add to batch loss (we want to maximize fidelity)
            loss += torch.tensor(fidelity / batch_size, device=self.device)
            
        # Return negative loss for minimization
        return -loss
    
    def train_step(self, 
                  time_values: torch.Tensor, 
                  target_amplitudes: torch.Tensor,
                  learning_rate: float = 0.01) -> float:
        """
        Perform one training step using gradient descent.
        
        Args:
            time_values: Tensor of normalized time values
            target_amplitudes: Tensor of target amplitudes
            learning_rate: Learning rate for parameter updates
            
        Returns:
            Current loss value
        """
        # Create optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Calculate loss
        loss = self.calculate_loss(time_values, target_amplitudes)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Ensure unitaries remain unitary after gradient update
        with torch.no_grad():
            for layer in self.layer_unitaries:
                for unitary_module in layer:
                    u = unitary_module.unitary
                    # Polar decomposition to find nearest unitary matrix
                    u_polar, _ = torch.linalg.polar(u)
                    # Update the parameter directly
                    unitary_module.unitary.copy_(u_polar)
        
        # Return loss value
        return loss.item()
    
    def train(self, 
             dataloader: torch.utils.data.DataLoader, 
             num_epochs: int, 
             learning_rate: float = 0.01,
             progress_bar: bool = True) -> List[float]:
        """
        Train the DQNN on audio data.
        
        Args:
            dataloader: DataLoader providing (time, amplitude) pairs
            num_epochs: Number of training epochs
            learning_rate: Learning rate for updates
            progress_bar: Whether to show progress bar
            
        Returns:
            List of loss values per epoch
        """
        losses = []
        
        # Training loop
        range_func = tqdm(range(num_epochs)) if progress_bar else range(num_epochs)
        for epoch in range_func:
            epoch_loss = 0.0
            
            # Process each batch
            for batch_idx, (times, amplitudes) in enumerate(dataloader):
                times = times.to(self.device)
                amplitudes = amplitudes.to(self.device)
                
                # Perform training step
                loss = self.train_step(times, amplitudes, learning_rate)
                epoch_loss += loss
                
                if progress_bar:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    desc = f"Epoch {epoch}, Loss: {avg_loss:.6f}"
                    range_func.set_description(desc)
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / len(dataloader)
            losses.append(avg_epoch_loss)
            
            # Print progress if no progress bar
            if not progress_bar and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")
                
        return losses
    
    def evaluate(self, time_values: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the DQNN on a set of time values.
        
        Args:
            time_values: Tensor of normalized time values
            
        Returns:
            Tensor of predicted amplitude values
        """
        with torch.no_grad():
            return self.forward(time_values)
    
    def get_num_parameters(self) -> int:
        """
        Calculate the total number of parameters in the DQNN.
        
        Returns:
            Number of parameters
        """
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()
                
        return total_params