import torch
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import os
import time
import IPython.display as ipd
import soundfile as sf
from tqdm.notebook import tqdm

from dqnn_core import get_dqnn_model


class AudioQuantumDataset:
    """
    Dataset for training quantum neural networks on audio signals.
    Converts classical audio data to quantum states for DQNN training.
    """
    def __init__(self, audio_path, max_seconds=None, normalize_time=True, normalize_amplitude=True):
        """
        Initialize dataset from audio file.
        
        Args:
            audio_path: Path to audio file
            max_seconds: Maximum number of seconds to use (None = use all)
            normalize_time: If True, normalize time indices to [0, 1]
            normalize_amplitude: If True, normalize amplitudes to [-1, 1]
        """
        # Load audio file - explicitly set sample rate to None to use file's rate
        data, self.sample_rate = sf.read(audio_path, dtype='float32')
        
        # Convert to mono if stereo (averaging channels)
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = np.mean(data, axis=1)
        
        # Limit length if specified
        if max_seconds is not None and max_seconds > 0:
            max_samples = int(max_seconds * self.sample_rate)
            if max_samples < len(data):
                data = data[:max_samples]
                
        # Print data information
        print(f"Audio loaded: {len(data)} samples, {len(data)/self.sample_rate:.2f} seconds at {self.sample_rate}Hz")
        print(f"Original amplitude range: ({np.min(data):.4f}, {np.max(data):.4f})")
        
        # Store original data
        self.original_data = data.copy()
        self.data_length = len(data)
        
        # Set up amplitude normalization
        self.normalize_amplitude = normalize_amplitude
        if normalize_amplitude:
            self.amplitude_min = np.min(data)
            self.amplitude_max = np.max(data)
            self.amplitude_range = self.amplitude_max - self.amplitude_min
            
            # If range is too small, use absolute max instead
            abs_max = np.max(np.abs(data))
            if self.amplitude_range < 1e-6:
                data_normalized = data / (abs_max if abs_max > 0 else 1.0)
            else:
                # Normalize to [-1, 1]
                data_normalized = 2.0 * (data - self.amplitude_min) / self.amplitude_range - 1.0
                
            print(f"Normalized amplitude range: ({np.min(data_normalized):.4f}, {np.max(data_normalized):.4f})")
        else:
            data_normalized = data
            
        # Set up time normalization
        self.normalize_time = normalize_time
        self.time_indices = np.arange(len(data))
        
        if normalize_time:
            # Normalize time to [0, 1]
            self.time_indices_normalized = self.time_indices / (len(data) - 1)
            print(f"Normalized time range: [0, 1] ({len(data)} samples)")
        else:
            self.time_indices_normalized = self.time_indices
            print(f"Time range: [0, {len(data)-1}] ({len(data)} samples)")
            
        # Store normalized data
        self.data_normalized = data_normalized
        
        # Convert classical data to quantum format
        self._prepare_quantum_data()
        
    def _prepare_quantum_data(self):
        """Prepare quantum data from classical data"""
        # Amplitude encoding is used to encode normalized amplitude into a qubit state
        # |ψ(t)⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        # where θ = π(amplitude + 1)/2 which maps [-1,1] to [0,π]
        
        self.quantum_data = []
        
        for i in range(self.data_length):
            time = self.time_indices_normalized[i]
            amplitude = self.data_normalized[i]
            
            # Encode amplitude into single-qubit state
            theta = np.pi * (amplitude + 1) / 2
            qubit_state = np.cos(theta/2) * qt.basis(2, 0) + np.sin(theta/2) * qt.basis(2, 1)
            
            # Store time and quantum state
            self.quantum_data.append((time, qubit_state))
    
    def amplitude_to_quantum(self, amplitude):
        """Convert a classical amplitude to a quantum state"""
        # Normalize amplitude if needed
        if self.normalize_amplitude:
            if self.amplitude_range < 1e-6:
                abs_max = np.max(np.abs(self.original_data))
                amplitude_normalized = amplitude / (abs_max if abs_max > 0 else 1.0)
            else:
                amplitude_normalized = 2.0 * (amplitude - self.amplitude_min) / self.amplitude_range - 1.0
        else:
            amplitude_normalized = amplitude
            
        # Convert to quantum state
        theta = np.pi * (amplitude_normalized + 1) / 2
        qubit_state = np.cos(theta/2) * qt.basis(2, 0) + np.sin(theta/2) * qt.basis(2, 1)
        
        return qubit_state
    
    def quantum_to_amplitude(self, quantum_state):
        """Convert a quantum state to a classical amplitude"""
        # Extract amplitude from quantum state
        # For a state |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        # Extract θ and map [0,π] back to [-1,1]
        
        # Calculate probability of |1⟩, which is sin^2(θ/2)
        if quantum_state.type == 'ket':
            prob_1 = abs(quantum_state[1][0][0]) ** 2
        else:  # Density matrix
            prob_1 = quantum_state[1,1].real
            
        # Extract θ
        theta = 2 * np.arcsin(np.sqrt(prob_1))
        
        # Map [0,π] to [-1,1]
        amplitude_normalized = 2 * theta / np.pi - 1
        
        # Denormalize if needed
        if self.normalize_amplitude:
            if self.amplitude_range < 1e-6:
                abs_max = np.max(np.abs(self.original_data))
                amplitude = amplitude_normalized * (abs_max if abs_max > 0 else 1.0)
            else:
                amplitude = (amplitude_normalized + 1) * self.amplitude_range / 2 + self.amplitude_min
        else:
            amplitude = amplitude_normalized
            
        return amplitude
        
    def __len__(self):
        return self.data_length
    
    def __getitem__(self, idx):
        return self.quantum_data[idx]
    
    def get_batch(self, batch_size, evenly_spaced=False):
        """
        Get a batch of training data.
        
        Args:
            batch_size: Size of the batch
            evenly_spaced: If True, select evenly spaced samples, otherwise random
            
        Returns:
            List of [input_state, target_state] pairs
        """
        if evenly_spaced:
            # Select evenly spaced indices
            indices = np.linspace(0, len(self) - 1, batch_size, dtype=int)
        else:
            # Select random indices
            indices = np.random.choice(len(self), batch_size)
            
        # Prepare training pairs
        training_pairs = []
        
        for idx in indices:
            time, amplitude_state = self.quantum_data[idx]
            
            # Create input state from time (classical parameter)
            # Amplitude encoding: |ψ⟩ = cos(πt)|0⟩ + sin(πt)|1⟩
            input_state = np.cos(np.pi * time) * qt.basis(2, 0) + np.sin(np.pi * time) * qt.basis(2, 1)
            
            # The target is the amplitude encoded as a quantum state
            target_state = amplitude_state
            
            # Add to training pairs
            training_pairs.append([input_state, target_state])
            
        return training_pairs
    
    def get_sequential_data(self, num_points=None):
        """
        Get sequential data for evaluation/visualization.
        
        Args:
            num_points: Number of points to return (None = all)
            
        Returns:
            List of [input_state, target_state] pairs
        """
        if num_points is None:
            num_points = len(self)
            
        # Create evenly spaced time points
        if self.normalize_time:
            times = np.linspace(0, 1, num_points)
        else:
            times = np.linspace(0, self.data_length-1, num_points)
            
        # Create sequential data
        sequential_data = []
        
        for i, time in enumerate(times):
            # Find nearest time point in our dataset
            idx = int(time * (self.data_length - 1) if self.normalize_time else time)
            idx = min(idx, self.data_length - 1)
            
            # Get amplitude state for this time
            _, amplitude_state = self.quantum_data[idx]
            
            # Create input state
            input_state = np.cos(np.pi * time) * qt.basis(2, 0) + np.sin(np.pi * time) * qt.basis(2, 1)
            
            # Add to sequential data
            sequential_data.append([input_state, amplitude_state])
            
        return sequential_data
    
    def get_original_audio(self):
        """Get the original unprocessed audio data"""
        return self.original_data


def train_dqnn(model, dataset, num_epochs=1000, batch_size=128, training_rounds=1, lda=1.0, ep=0.01, log_interval=5):
    """
    Train a DQNN model on an audio dataset using batched training.
    
    Args:
        model: The DQNN model to train
        dataset: AudioQuantumDataset instance
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        training_rounds: Number of training rounds per batch
        lda: Lambda parameter (regularization)
        ep: Epsilon parameter (learning rate)
        log_interval: Interval for logging progress
        
    Returns:
        Trained model and history of losses
    """
    losses = []
    best_loss = float('-inf')  # Maximizing fidelity (higher is better)
    
    print(f"\nStarting DQNN batched training for {num_epochs} epochs with {training_rounds} training rounds per batch...")
    
    # Training loop
    pbar = tqdm(range(num_epochs), desc="Training")
    start_time = time.time()
    
    for epoch in pbar:
        # Get random batch
        training_data = dataset.get_batch(batch_size)
        
        # Train for specified number of rounds
        loss_values = model.train(training_data, lda, ep, training_rounds, verbose=False)
        current_loss = loss_values[-1][1]
            
        # Store loss
        losses.append(current_loss)
        
        # Track best model (highest fidelity)
        if current_loss > best_loss:
            best_loss = current_loss
        
        # Update progress bar
        if epoch % log_interval == 0 or epoch == num_epochs - 1:
            pbar.set_description(f"DQNN Training: quantum fidelity={current_loss:.6f}, best={best_loss:.6f}")
                
    training_time = time.time() - start_time
    print(f"DQNN Training completed in {training_time:.2f} seconds. Final quantum fidelity: {losses[-1]:.6f}, Best: {best_loss:.6f}")
    
    return model, losses


def train_dqnn_full_dataset(model, dataset, training_rounds=100, max_points=None, lda=1.0, ep=0.01):
    """
    Train a DQNN model on an audio dataset using full dataset training approach.
    Matches the authors' original implementation more closely.
    
    Args:
        model: The DQNN model to train
        dataset: AudioQuantumDataset instance
        training_rounds: Number of training rounds on the full dataset
        max_points: Maximum number of data points to use (None = all)
        lda: Lambda parameter (regularization)
        ep: Epsilon parameter (learning rate)
        
    Returns:
        Trained model and history of losses
    """
    # Get sequential data (full dataset or limited size)
    training_data = dataset.get_sequential_data(num_points=max_points)
    
    print(f"\nStarting DQNN full dataset training with {len(training_data)} data points for {training_rounds} rounds...")
    start_time = time.time()
    
    # Train on the full dataset for multiple rounds
    loss_values = model.train(training_data, lda, ep, training_rounds, verbose=True)
    
    training_time = time.time() - start_time
    final_loss = loss_values[-1][1]
    
    print(f"DQNN full dataset training completed in {training_time:.2f} seconds.")
    print(f"Final quantum fidelity: {final_loss:.6f}")
    
    # Extract just the loss values for plotting
    losses = [loss[1] for loss in loss_values]
    
    return model, losses


def evaluate_dqnn(model, dataset, batch_size=None, num_points=None):
    """
    Evaluate a trained DQNN by generating audio samples in batches.
    
    Args:
        model: Trained DQNN model
        dataset: AudioQuantumDataset instance
        batch_size: Batch size for evaluation (None = use entire dataset at once)
        num_points: Number of points to evaluate (None = all)
    
    Returns:
        Generated audio samples at the original sample rate
    """
    # Get sequential data for evaluation
    if num_points is None:
        num_points = len(dataset)
        
    sequential_data = dataset.get_sequential_data(num_points)
    
    # Use default batch size if none provided
    if batch_size is None or batch_size <= 0:
        batch_size = 128  # Reasonable default
    
    # Generate predictions
    predictions = []
    
    # Process in batches with progress bar for better user experience
    pbar = tqdm(range(0, len(sequential_data), batch_size), desc="Evaluating DQNN")
    
    for i in pbar:
        batch = sequential_data[i:min(i+batch_size, len(sequential_data))]
        batch_predictions = []
        
        for input_state, _ in batch:
            # Get prediction
            output_state = model.predict(input_state)
            
            # Convert quantum output to classical amplitude
            amplitude = dataset.quantum_to_amplitude(output_state)
            batch_predictions.append(amplitude)
            
        predictions.extend(batch_predictions)
    
    return np.array(predictions)


def plot_audio_waveform(predictions, ground_truth, sample_rate=44100, title="Waveform Comparison", seconds=5):
    """
    Plot comparison between predicted and ground truth audio waveforms.
    
    Args:
        predictions: Predicted audio samples
        ground_truth: Ground truth audio samples
        sample_rate: Audio sample rate
        title: Plot title
        seconds: Number of seconds to show
    """
    # Determine number of samples to show
    samples = min(int(seconds * sample_rate), len(predictions), len(ground_truth))
    
    plt.figure(figsize=(12, 5))
    t = np.arange(samples) / sample_rate
    plt.plot(t, ground_truth[:samples], label='Ground Truth', alpha=0.7)
    plt.plot(t, predictions[:samples], label='DQNN Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()


def play_audio(audio_data, sample_rate=44100, title="Audio"):
    """
    Play audio data using IPython display.
    
    Args:
        audio_data: Audio samples
        sample_rate: Audio sample rate
        title: Audio display title
    """
    print(f"{title} ({len(audio_data)/sample_rate:.2f} seconds):")
    return ipd.Audio(audio_data, rate=sample_rate)


def evaluate_and_visualize(model, dataset, model_name="DQNN", seconds_to_show=5, 
                          save_audio=None, save_plot=None, batch_size=128):
    """
    Comprehensive evaluation and visualization of a trained model.
    
    Args:
        model: Trained DQNN model
        dataset: AudioQuantumDataset instance
        model_name: Name of the model for display
        seconds_to_show: Number of seconds to visualize
        save_audio: Optional path to save audio output (None to skip saving)
        save_plot: Optional path to save waveform plot (None to skip saving)
        batch_size: Batch size for evaluation
    
    Returns:
        predictions, ground_truth, mse, psnr
    """
    # Get original ground truth data
    ground_truth = dataset.get_original_audio()
    
    # Generate predictions
    predictions = evaluate_dqnn(model, dataset, batch_size=batch_size)
    
    # Handle NaN or Inf values
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Make sure predictions match the length of ground truth
    if len(predictions) != len(ground_truth):
        print(f"Warning: prediction length ({len(predictions)}) doesn't match ground truth ({len(ground_truth)})")
        # Adjust lengths to match for comparison purposes
        min_length = min(len(predictions), len(ground_truth))
        predictions = predictions[:min_length]
        ground_truth = ground_truth[:min_length]
    
    # Calculate classical domain metrics
    mse = np.mean((predictions - ground_truth)**2)
    max_amp = np.max(np.abs(ground_truth))
    if max_amp > 0:
        psnr = 20 * np.log10(max_amp / np.sqrt(mse))
    else:
        psnr = 0.0
    
    print(f"=== {model_name} Evaluation (Classical Domain) ===")
    print(f"Classical MSE: {mse:.6f}")
    print(f"Classical PSNR: {psnr:.2f} dB")
    print(f"Note: Training used quantum fidelity but evaluation uses classical metrics")
    
    # Visualize waveform (first few seconds)
    plt.figure(figsize=(12, 5))
    samples = min(int(seconds_to_show * dataset.sample_rate), len(predictions), len(ground_truth))
    t = np.arange(samples) / dataset.sample_rate
    plt.plot(t, ground_truth[:samples], label='Ground Truth', alpha=0.7)
    plt.plot(t, predictions[:samples], label=f'{model_name} Prediction', alpha=0.7)
    plt.title(f"{model_name} Waveform Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    # Save plot if path provided
    if save_plot:
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')
        print(f"Saved waveform plot to {save_plot}\n")
        
    plt.show()
    
    # Play audio clips (limited to 5 seconds for convenience)
    seconds_to_play = min(5, len(ground_truth)/dataset.sample_rate)
    samples_to_play = int(seconds_to_play * dataset.sample_rate)
    
    # Play ground truth audio
    display(play_audio(ground_truth[:samples_to_play], dataset.sample_rate, "Ground Truth Audio"))
    
    # Play predicted audio
    display(play_audio(predictions[:samples_to_play], dataset.sample_rate, f"{model_name} Predicted Audio"))
    
    # Save audio if path provided
    if save_audio:
        sf.write(save_audio, predictions, dataset.sample_rate)
        print(f"Saved predicted audio to {save_audio}")
    
    return predictions, ground_truth, mse, psnr


def create_dqnn_trained_model(audio_file, output_path=None, qnn_arch=None, hidden_size=2, num_layers=2, 
                             lda=1.0, ep=0.01, num_epochs=500, batch_size=64, training_rounds=1,
                             use_full_dataset=False, max_data_points=None):
    """
    Create and train a DQNN model on an audio file.
    
    Args:
        audio_file: Path to audio file
        output_path: Path to save outputs (None to skip saving)
        qnn_arch: Architecture of the DQNN as a list of qubit counts per layer
        hidden_size: Number of qubits in hidden layers (used only if qnn_arch is None)
        num_layers: Number of hidden layers (used only if qnn_arch is None)
        lda: Lambda parameter (regularization)
        ep: Epsilon parameter (learning rate)
        num_epochs: Number of training epochs (used only for batched training)
        batch_size: Batch size for training (used only for batched training)
        training_rounds: Number of training rounds per batch or on full dataset
        use_full_dataset: If True, use full dataset training approach instead of batching
        max_data_points: Maximum number of data points to use for full dataset training
        
    Returns:
        Trained model, dataset, and evaluation results
    """ 
    # Create dataset
    dataset = AudioQuantumDataset(
        audio_file,
        normalize_time=True,
        normalize_amplitude=True
    )
    
    # Configure model
    if qnn_arch is None:
        qnn_arch = [1] + [hidden_size] * num_layers + [1]
    
    model = get_dqnn_model(qnn_arch=qnn_arch)
    
    # Print model details
    num_params = model.count_parameters()
    print(f"DQNN architecture: {qnn_arch}")
    print(f"DQNN model parameters: {num_params:,}")
    
    # Train model using the selected approach
    if use_full_dataset:
        # Use full dataset training approach (authors' method)
        model, losses = train_dqnn_full_dataset(
            model,
            dataset,
            training_rounds=training_rounds,
            max_points=max_data_points,
            lda=lda,
            ep=ep
        )
        
        # For plotting - create a list of epochs/rounds
        x_values = list(range(len(losses)))
    else:
        # Use batched training approach
        model, losses = train_dqnn(
            model,
            dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            training_rounds=training_rounds,
            lda=lda,
            ep=ep
        )
        
        # x_values will be epoch numbers
        x_values = list(range(len(losses)))
    
    # Plot training curve
    fig = plt.figure(figsize=(10, 4))
    plt.plot(x_values, losses)
    
    if use_full_dataset:
        plt.title("DQNN Training Fidelity")
        plt.xlabel("Round")
    else:
        plt.title("DQNN Training Fidelity")
        plt.xlabel("Epoch")
        
    plt.ylabel("Quantum Fidelity (Higher is better)")
    plt.grid(True)
    plt.show()
    
    if output_path:
        fig.savefig(f"{output_path}/dqnn_loss.png", dpi=300, bbox_inches='tight')
    
    # Evaluate model
    print("\nEvaluating DQNN model...")
    save_audio = f"{output_path}/dqnn_audio.wav" if output_path else None
    save_plot = f"{output_path}/dqnn_waveform.png" if output_path else None
    
    predictions, ground_truth, mse, psnr = evaluate_and_visualize(
        model,
        dataset,
        model_name="DQNN",
        save_audio=save_audio,
        save_plot=save_plot,
        batch_size=batch_size  # Use same batch size as training
    )
    
    return model, dataset, (predictions, ground_truth, mse, psnr)