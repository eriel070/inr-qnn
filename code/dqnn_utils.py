import torch
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import os
import time
import IPython.display as ipd
import soundfile as sf
import librosa
from tqdm.notebook import tqdm

from dqnn_core import get_dqnn_model


class AudioQuantumDataset:
    """
    Dataset for training quantum neural networks on audio signals.
    Converts classical audio data to quantum states for DQNN training using
    a bijective encoding-decoding framework.
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
        data, self.sample_rate = sf.read(audio_path, dtype='float64')   # audio_path data is float32
        
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
        
        # Store pairs of time and amplitude for easier access
        self.data_pairs = list(zip(self.time_indices_normalized, self.data_normalized))
        
        # Convert classical data to quantum format
        self._prepare_quantum_data()
        
    def _prepare_quantum_data(self):
        """Prepare quantum data from classical data"""
        self.quantum_data = []
        
        for i in range(self.data_length):
            time = self.time_indices_normalized[i]
            amplitude = self.data_normalized[i]
            
            # Encode time and amplitude into quantum states
            time_state = self.encode_time(time)
            amplitude_state = self.encode_amplitude(amplitude)
            
            # Store time and quantum states
            self.quantum_data.append((time, time_state, amplitude_state))

    @staticmethod
    def encode_time(time):
        """Convert normalized time to a quantum state
        
        Maps time ∈ [0, 1] to ω ∈ [0, π) and creates a pure state
        |ψ⟩ = cos(ω)|0⟩ + sin(ω)|1⟩
        """
        # Map time ∈ [0, 1] to ω ∈ [0, π)
        omega = np.pi * time
        
        # Create pure state with φ = 0
        state = np.cos(omega) * qt.basis(2, 0) + np.sin(omega) * qt.basis(2, 1)
        return state

    @staticmethod
    def decode_time(quantum_state):
        """Convert a quantum state to normalized time
        
        Works with both pure states and mixed states (density matrices)
        """
        # Extract Bloch vector components
        if quantum_state.type == 'ket':
            rho = quantum_state * quantum_state.dag()
        else:
            rho = quantum_state
        
        # Safely extract real numerical values from QuTip objects
        if isinstance(rho[0,1], qt.Qobj):
            r_x = float(2 * np.real(rho[0,1].full()[0,0]))
            r_y = float(2 * np.imag(rho[0,1].full()[0,0]))
            r_z = float(np.real((rho[0,0] - rho[1,1]).full()[0,0]))
        else:
            r_x = float(2 * np.real(rho[0,1]))
            r_y = float(2 * np.imag(rho[0,1]))
            r_z = float(np.real(rho[0,0] - rho[1,1]))
        
        # Calculate ω using both x and z components
        # Use arctan2 for correct quadrant handling
        omega = 0.5 * np.arctan2(r_x, r_z)
        
        # Ensure omega is in [0, π)
        if omega < 0:
            omega += np.pi
            
        # Map ω ∈ [0, π) back to time ∈ [0, 1]
        time = omega / np.pi
        
        return time

    @staticmethod
    def encode_amplitude(amplitude):
        """Convert a normalized amplitude to a quantum state
        
        Maps amplitude ∈ [-1, 1] to ω ∈ [0, π) and creates a pure state
        |ψ⟩ = cos(ω)|0⟩ + sin(ω)|1⟩
        """
        # Map amplitude ∈ [-1, 1] to ω ∈ [0, π)
        omega = np.pi * (amplitude + 1) / 2
        
        # Create pure state with φ = 0
        state = np.cos(omega) * qt.basis(2, 0) + np.sin(omega) * qt.basis(2, 1)
        return state

    @staticmethod
    def decode_amplitude(quantum_state):
        """Convert a quantum state to normalized amplitude
        
        Works with both pure states and mixed states (density matrices)
        """
        # Extract Bloch vector components
        if quantum_state.type == 'ket':
            rho = quantum_state * quantum_state.dag()
        else:
            rho = quantum_state
        
        # Safely extract real numerical values from QuTip objects
        # The .full() method converts Qobj to numpy array
        if isinstance(rho[0,1], qt.Qobj):
            r_x = float(2 * np.real(rho[0,1].full()[0,0]))
            r_y = float(2 * np.imag(rho[0,1].full()[0,0]))
            r_z = float(np.real((rho[0,0] - rho[1,1]).full()[0,0]))
        else:
            r_x = float(2 * np.real(rho[0,1]))
            r_y = float(2 * np.imag(rho[0,1]))
            r_z = float(np.real(rho[0,0] - rho[1,1]))
        
        # Calculate ω using both x and z components
        # Use arctan2 for correct quadrant handling
        omega = 0.5 * np.arctan2(r_x, r_z)
        
        # Ensure omega is in [0, π)
        if omega < 0:
            omega += np.pi
            
        # Map ω ∈ [0, π) back to amplitude ∈ [-1, 1]
        amplitude = 2 * (omega / np.pi) - 1
        
        return amplitude
    
    def amplitude_to_original(self, amplitude_normalized):
        """Convert normalized amplitude to original amplitude value"""
        if self.normalize_amplitude:
            if self.amplitude_range < 1e-6:
                abs_max = np.max(np.abs(self.original_data))
                return amplitude_normalized * (abs_max if abs_max > 0 else 1.0)
            else:
                return (amplitude_normalized + 1) * self.amplitude_range / 2 + self.amplitude_min
        return amplitude_normalized
        
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
            _, time_state, amplitude_state = self.quantum_data[idx]
            
            # The input is the time encoded as a quantum state
            input_state = time_state
            
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
        
        for time in times:
            # Create input state (time encoded as quantum state)
            input_state = self.encode_time(time)
            
            # Find nearest time point in dataset for the target
            idx = int(time * (self.data_length - 1) if self.normalize_time else time)
            idx = min(idx, self.data_length - 1)
            
            # Get target amplitude state
            _, _, amplitude_state = self.quantum_data[idx]
            
            # Add to sequential data
            sequential_data.append([input_state, amplitude_state])
            
        return sequential_data
    
    def get_original_audio(self):
        """Get the original unprocessed audio data"""
        return self.original_data


def verify_quantum_encoding(dataset, seconds_to_show=None, save_plot=None, num_samples=None):
    """
    Verify the quantum encoding/decoding process by comparing the original signal
    with a version where both time and amplitude have been encoded to quantum states
    and decoded back.
    
    Args:
        dataset: AudioQuantumDataset instance
        seconds_to_show: Number of seconds of audio to show in the plot
        num_samples: Number of samples to use for verification
        
    Returns:
        Dictionary with metrics and signals
    """
    print("\nVerifying quantum encoding/decoding fidelity...")
    
    # Use entire dataset time if seconds_to_show is None
    max_points = len(dataset) if seconds_to_show is None else min(len(dataset), int(seconds_to_show * dataset.sample_rate))
    
    # Use all available points if num_samples is None
    num_points = max_points if num_samples is None else min(max_points, num_samples)
    
    # Create evenly spaced time points for the selected range
    original_times = np.linspace(0, 1, num_points) if dataset.normalize_time else np.linspace(0, dataset.data_length-1, num_points)
    
    # Get original amplitudes for these times
    original_amplitudes = []
    for t in original_times:
        idx = int(t * (dataset.data_length - 1) if dataset.normalize_time else t)
        idx = min(idx, dataset.data_length - 1)
        original_amplitudes.append(dataset.data_normalized[idx])
    
    # Encode and decode both time and amplitude
    decoded_times = []
    decoded_amplitudes = []
    
    print(f"Processing {num_points} samples for both time and amplitude encoding/decoding...")
    for i, (time, amplitude) in enumerate(zip(original_times, original_amplitudes)):
        # Encode to quantum states
        time_state = dataset.encode_time(time)
        amplitude_state = dataset.encode_amplitude(amplitude)
        
        # Convert to density matrices (simulating what would happen in DQNN)
        time_density_matrix = time_state * time_state.dag()
        amplitude_density_matrix = amplitude_state * amplitude_state.dag()
        
        # Decode back to classical values
        decoded_time = dataset.decode_time(time_density_matrix)
        decoded_amplitude = dataset.decode_amplitude(amplitude_density_matrix)
        
        decoded_times.append(decoded_time)
        decoded_amplitudes.append(decoded_amplitude)
    
    # Calculate metrics for time and amplitude
    original_times_array = np.array(original_times)
    decoded_times_array = np.array(decoded_times)
    original_amplitudes_array = np.array(original_amplitudes)
    decoded_amplitudes_array = np.array(decoded_amplitudes)
    
    # Calculate error metrics
    time_mse = np.mean((original_times_array - decoded_times_array)**2)
    amplitude_mse = np.mean((original_amplitudes_array - decoded_amplitudes_array)**2)
    
    # Convert to seconds for plotting
    original_seconds = original_times_array * (dataset.data_length - 1) / dataset.sample_rate if dataset.normalize_time else original_times_array / dataset.sample_rate
    decoded_seconds = decoded_times_array * (dataset.data_length - 1) / dataset.sample_rate if dataset.normalize_time else decoded_times_array / dataset.sample_rate
    
    # Denormalize amplitudes for visualization
    original_denorm = dataset.amplitude_to_original(original_amplitudes_array)
    decoded_denorm = dataset.amplitude_to_original(decoded_amplitudes_array)
    
    # Plot comparison of original vs. fully decoded waveform
    plt.figure(figsize=(9, 3))
    plt.plot(original_seconds, original_denorm, label='Pre-Encoding', alpha=0.7)
    plt.plot(decoded_seconds, decoded_denorm, label='Post-Decoding', alpha=0.7, color='green')
    plt.suptitle("Quantum-Classical Transfer Integrity", fontsize=16, y=1.06)
    plt.title(f"Time MSE={time_mse:.3e}, Amplitude MSE={amplitude_mse:.3e}", fontsize=13, pad=10)
    plt.xlabel('Time (s)', fontsize=13)
    plt.ylabel('Amplitude', fontsize=13)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True) 
    
    if save_plot:
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')
        print(f"Saved encoding verification plot to {save_plot}\n")
    
    plt.show()
    
    print(f"Encoding/decoding verification metrics:")
    print(f"Time MSE: {time_mse:.4e}")
    print(f"Amplitude MSE: {amplitude_mse:.4e}")
    
    return {
        'time_mse': time_mse,
        'amplitude_mse': amplitude_mse,
        'original_times': original_seconds,
        'decoded_times': decoded_seconds,
        'original_amplitudes': original_denorm,
        'decoded_amplitudes': decoded_denorm
    }


def train_dqnn(model, dataset, num_epochs=1000, batch_size=128, training_rounds=1, lda=1.0, ep=0.01, log_interval=1):
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


def evaluate_dqnn(model, dataset, num_points=None, collect_quantum_states=False):
    """
    Evaluate a trained DQNN by generating audio samples and optionally collecting quantum states.
    
    Args:
        model: Trained DQNN model
        dataset: AudioQuantumDataset instance
        num_points: Number of points to evaluate (None = all)
        collect_quantum_states: If True, collect quantum states for analysis
    
    Returns:
        Generated audio samples and optionally quantum states for analysis
    """
    # Get sequential data for evaluation
    if num_points is None:
        num_points = len(dataset)
        
    sequential_data = dataset.get_sequential_data(num_points)
    
    # Generate predictions
    predictions = []
    
    # For quantum state analysis
    target_states = []
    output_states = []
    
    # Process in a single loop with progress bar
    pbar = tqdm(sequential_data, desc="Evaluating DQNN")
    
    for input_state, target_state in pbar:
        # Get prediction
        output_state = model.predict(input_state)
        
        # Convert quantum output to classical amplitude
        amplitude = dataset.decode_amplitude(output_state)
        predictions.append(amplitude)
        
        # Collect quantum states if requested
        if collect_quantum_states:
            target_states.append(target_state)
            output_states.append(output_state)
    
    # Convert normalized predictions to original scale
    predictions = dataset.amplitude_to_original(np.array(predictions))
    
    if collect_quantum_states:
        return predictions, target_states, output_states
    
    return predictions


def plot_audio_waveform(predictions, ground_truth, sample_rate=44100, title="Waveform Comparison",
                        seconds=None, save_path=None, color_original='tab:blue', color_prediction='tab:orange'):
    """
    Plot comparison between predicted and ground truth audio waveforms.
    
    Args:
        predictions: Predicted audio samples
        ground_truth: Ground truth audio samples
        sample_rate: Audio sample rate
        title: Plot title
        seconds: Number of seconds to show (None = show all)
        save_path: Optional path to save the plot
    """
    # Determine number of samples to show
    if seconds is None:
        samples = min(len(predictions), len(ground_truth))
    else:
        samples = min(int(seconds * sample_rate), len(predictions), len(ground_truth))
    
    fig = plt.figure(figsize=(12, 5))
    t = np.arange(samples) / sample_rate
    plt.plot(t, ground_truth[:samples], label='Target', alpha=0.7, color=color_original)
    plt.plot(t, predictions[:samples], label='Output', alpha=0.7, color=color_prediction)
    plt.title(title, fontsize=22)
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    plt.legend(fontsize=14, loc='upper left')
    plt.grid(True)
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved waveform plot to {save_path}\n")
        
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


def evaluate_and_visualize(model, dataset, model_name="DQNN", seconds_to_show=None, 
                          save_audio=None, save_plot=None, analyze_quantum_states=False,
                          save_quantum_plot=None):
    """
    Comprehensive evaluation and visualization of a trained model.
    
    Args:
        model: Trained DQNN model
        dataset: AudioQuantumDataset instance
        model_name: Name of the model for display
        seconds_to_show: Number of seconds to visualize
        save_audio: Optional path to save audio output (None to skip saving)
        save_plot: Optional path to save waveform plot (None to skip saving)
        analyze_quantum_states: If True, analyze the quantum states
    
    Returns:
        predictions, ground_truth, mse, psnr
    """
    # Get original ground truth data
    ground_truth = dataset.get_original_audio()

    # Generate predictions and collect quantum states if needed later
    predictions, target_states, output_states = evaluate_dqnn(model, dataset, collect_quantum_states=True)
    
    # Check for NaN or Inf values
    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        raise ValueError(f"Invalid values detected in model predictions: NaN: {np.any(np.isnan(predictions))}, Inf: {np.any(np.isinf(predictions))}")
    
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
    
    print(f"\n=== {model_name} Evaluation (Classical Domain) ===")
    print(f"Classical MSE: {mse:.5e}")
    print(f"Classical PSNR: {psnr:.2f} dB")
    print(f"Note: Training used quantum fidelity but evaluation uses classical metrics")
    
    # Visualize waveform
    plot_title = f"{model_name}"
    plot_audio_waveform(
        predictions, 
        ground_truth, 
        dataset.sample_rate, 
        plot_title, 
        seconds_to_show,
        save_path=save_plot
    )

    # Analyze quantum states if needed *
    if analyze_quantum_states:
        quantum_state_analysis(model_name, target_states, output_states, dataset, save_plot=save_quantum_plot)
        print()
    
    # Play audio clips
    if seconds_to_show is None:
        samples_to_play = len(ground_truth)  # Play entire audio
    else:
        samples_to_play = min(int(seconds_to_show * dataset.sample_rate), len(ground_truth))
    
    # Play ground truth audio
    display(play_audio(ground_truth[:samples_to_play], dataset.sample_rate, "Ground Truth Audio"))
    
    # Play predicted audio
    display(play_audio(predictions[:samples_to_play], dataset.sample_rate, f"{model_name} Predicted Audio"))
    
    # Save audio if path provided
    if save_audio:
        sf.write(save_audio, predictions, dataset.sample_rate)
        print(f"Saved predicted audio to {save_audio}")
    
    return predictions, ground_truth, mse, psnr


def quantum_state_analysis(model_name, target_states, output_states, dataset, max_samples=None, save_plot=None):
    """
    Analyze quantum states produced by the DQNN and compare with target states.
    
    Args:
        target_states: List of target quantum states
        output_states: List of output quantum states from DQNN
        dataset: AudioQuantumDataset instance for decoding
        max_samples: Maximum number of samples to display on plots
        save_plot: Optional path to save the analysis plot
        
    Returns:
        Dictionary with analysis results
    """
    print(f"\n=== {model_name} Evaluation (Quantum Domain) ===")
    
    # Extract Bloch vectors and calculate statistics
    target_bloch = []
    output_bloch = []
    output_purities = []
    angles_between = []
    target_amplitudes = []
    output_amplitudes = []
    fidelities = []
    
    # Limit samples if necessary
    if max_samples is None or max_samples >= len(output_states):
        sample_indices = np.arange(len(output_states), dtype=int)
    else:
        np.linspace(0, len(output_states)-1, max_samples, dtype=int)

    # Analyze each quantum state
    for i in sample_indices:
        target = target_states[i]
        output = output_states[i]
        
        # Convert to density matrix if needed
        if target.type == 'ket':
            target_dm = target * target.dag()
        else:
            target_dm = target
            
        # Calculate target Bloch vector
        r_x_t = float(2 * np.real((target_dm[0,1]).full()[0,0] if isinstance(target_dm[0,1], qt.Qobj) else target_dm[0,1]))
        r_y_t = float(2 * np.imag((target_dm[0,1]).full()[0,0] if isinstance(target_dm[0,1], qt.Qobj) else target_dm[0,1]))
        r_z_t = float(np.real(((target_dm[0,0] - target_dm[1,1])).full()[0,0] if isinstance(target_dm[0,0], qt.Qobj) else (target_dm[0,0] - target_dm[1,1])))
        target_bloch.append([r_x_t, r_y_t, r_z_t])
        
        # Calculate output Bloch vector
        r_x_o = float(2 * np.real((output[0,1]).full()[0,0] if isinstance(output[0,1], qt.Qobj) else output[0,1]))
        r_y_o = float(2 * np.imag((output[0,1]).full()[0,0] if isinstance(output[0,1], qt.Qobj) else output[0,1]))
        r_z_o = float(np.real(((output[0,0] - output[1,1])).full()[0,0] if isinstance(output[0,0], qt.Qobj) else (output[0,0] - output[1,1])))
        output_bloch.append([r_x_o, r_y_o, r_z_o])
        
        # Calculate output purity
        purity = float(np.real((output * output).tr()))
        output_purities.append(purity)
        
        # Calculate angle between Bloch vectors (if output is not at origin)
        output_length = np.sqrt(r_x_o**2 + r_y_o**2 + r_z_o**2)
        if output_length > 1e-6:
            # Normalize both vectors
            target_norm = np.array([r_x_t, r_y_t, r_z_t]) / np.sqrt(r_x_t**2 + r_y_t**2 + r_z_t**2)
            output_norm = np.array([r_x_o, r_y_o, r_z_o]) / output_length
            
            # Dot product
            cos_angle = np.clip(np.dot(target_norm, output_norm), -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi  # in degrees
            angles_between.append(angle)
        else:
            angles_between.append(np.nan)
            
        # Calculate fidelity between target and output states
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
                target_for_fid = target * target.dag()
            else:
                target_for_fid = target
            if output.type != 'oper':
                output_for_fid = output * output.dag()
            else:
                output_for_fid = output
            fidelity = float(qt.fidelity(target_for_fid, output_for_fid))
        
        fidelities.append(fidelity)
            
        # Decode to classical amplitudes
        target_amplitude = dataset.decode_amplitude(target_dm)
        output_amplitude = dataset.decode_amplitude(output)
        target_amplitudes.append(target_amplitude)
        output_amplitudes.append(output_amplitude)
    
    # Convert to arrays
    target_bloch = np.array(target_bloch)
    output_bloch = np.array(output_bloch)
    output_purities = np.array(output_purities)
    angles_between = np.array(angles_between)
    target_amplitudes = np.array(target_amplitudes)
    output_amplitudes = np.array(output_amplitudes)
    fidelities = np.array(fidelities)
    
    # Calculate Bloch vector lengths
    output_bloch_lengths = np.sqrt(np.sum(output_bloch**2, axis=1))
    
    # Print summary statistics
    print(f"Average purity: {np.mean(output_purities):.4f}")
    print(f"Average Bloch vector length: {np.mean(output_bloch_lengths):.4f}")
    print(f"Pure states (purity > 0.99): {np.sum(output_purities > 0.99)/len(output_purities)*100:.2f}%")
    print(f"Average angle between target and output Bloch vectors: {np.nanmean(angles_between):.2f}°")
    print(f"Average fidelity: {np.mean(fidelities):.4f}")
    
    # Create visualization with new layout
    fig = plt.figure(figsize=(12, 8))
    
    # 1. Top left: Plot purity distribution
    ax1 = fig.add_subplot(221)
    ax1.hist(output_purities, bins=20, alpha=0.7)
    ax1.axvline(x=1.0, color='r', linestyle='--', label='Pure State')
    ax1.set_xlabel('Quantum State Purity')
    ax1.set_ylabel('Count')
    ax1.set_title('DQNN Output State Purity Distribution')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Top right: 2D visualization of Bloch vectors projected onto xz-plane
    ax2 = fig.add_subplot(222)
    # Draw unit circle representing pure states
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    
    # Plot target states
    target_x = target_bloch[:, 0]
    target_z = target_bloch[:, 2]
    ax2.scatter(target_x, target_z, c='b', marker='o', alpha=0.5, label='Target States')
    
    # Plot output states
    output_x = output_bloch[:, 0]
    output_z = output_bloch[:, 2]
    ax2.scatter(output_x, output_z, c='r', marker='x', alpha=0.5, label='Output States')
    
    # Draw some lines connecting target and output
    # Only draw a subset to avoid cluttering
    for i in range(len(sample_indices)):
        ax2.plot([target_x[i], output_x[i]], [target_z[i], output_z[i]], 'g-', alpha=0.17)
    
    ax2.set_xlabel('x-component of Bloch Vector')
    ax2.set_ylabel('z-component of Bloch Vector')
    ax2.set_title('Bloch Vector Projections (xz-plane)')
    ax2.grid(True)
    ax2.legend()
    ax2.axis('equal')
    
    # 3. Bottom left: Plot angle distribution between target and output
    ax3 = fig.add_subplot(223)
    valid_angles = angles_between[~np.isnan(angles_between)]
    if len(valid_angles) > 0:
        ax3.hist(valid_angles, bins=20, alpha=0.7)
        ax3.set_xlabel('Angle Between Target and Output Bloch Vectors (degrees)')
        ax3.set_ylabel('Count')
        ax3.set_title('Alignment Between Target and DQNN Output States')
        ax3.grid(True)
    else:
        ax3.text(0.5, 0.5, 'No valid angles to display', ha='center', va='center')
    
    # 4. Bottom right: Plot fidelity distribution
    ax4 = fig.add_subplot(224)
    ax4.hist(fidelities, bins=20, alpha=0.7)
    ax4.axvline(x=1.0, color='r', linestyle='--', label='Perfect Fidelity')
    ax4.set_xlabel('Fidelity Between Target and Output States')
    ax4.set_ylabel('Count')
    ax4.set_title('DQNN Output Fidelity Distribution')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_plot:
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')
        print(f"Saved quantum state analysis plot to {save_plot}\n")
        
    plt.show()
    
    # Return analysis results
    return {
        'purities': output_purities,
        'bloch_lengths': output_bloch_lengths,
        'angles_between': angles_between,
        'fidelities': fidelities,
        'target_bloch': target_bloch,
        'output_bloch': output_bloch,
        'target_amplitudes': target_amplitudes,
        'output_amplitudes': output_amplitudes
    }


def create_dqnn_trained_model(audio_file, output_path=None, qnn_arch=None, hidden_size=2, num_layers=2, 
                             lda=1.0, ep=0.01, num_epochs=500, batch_size=64, training_rounds=1,
                             use_full_dataset=False, max_data_points=None, seconds_to_show=None,
                             verify_encoding=False, analyze_quantum_states=False):
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
        verify_encoding: If True, verify the quantum encoding/decoding process
        analyze_quantum_states: If True, analyze the quantum states produced by the model
        
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
    
    # Verify encoding/decoding if requested
    if verify_encoding:
        save_verification_plot = (f"{output_path}/quantum_encoding_verification.png"
                                  if output_path else None)
        verification_results = verify_quantum_encoding(
            dataset, 
            seconds_to_show=seconds_to_show,
            save_plot=save_verification_plot
        )
    
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
        plt.title("DQNN", fontsize=22)
        plt.xlabel("Round", fontsize=16)
    else:
        plt.title("DQNN", fontsize=22)
        plt.xlabel("Epoch", fontsize=16)
        
    plt.ylabel("Fidelity", fontsize=16)
    plt.grid(True)
    plt.show()
    
    if output_path:
        fig.savefig(f"{output_path}/dqnn_loss.png", dpi=300, bbox_inches='tight')
    
    # Evaluate model
    print("\nEvaluating DQNN model...")
    save_audio = f"{output_path}/dqnn_audio.wav" if output_path else None
    save_plot = f"{output_path}/dqnn_waveform.png" if output_path else None
    save_quantum_plot = (f"{output_path}/dqnn_quantum_state_analysis.png"
                         if output_path and analyze_quantum_states else None)
    
    predictions, ground_truth, mse, psnr = evaluate_and_visualize(
        model,
        dataset,
        model_name="DQNN",
        seconds_to_show=seconds_to_show,
        save_audio=save_audio,
        save_plot=save_plot,
        analyze_quantum_states=analyze_quantum_states,
        save_quantum_plot=save_quantum_plot
    )
    
    return model, dataset, (predictions, ground_truth, mse, psnr)
    

################################################################################

def inference_arbitrary_dqnn(model, input_coords, model_name="DQNN", save_path=None):
    """
    Performs inference on DQNN model using arbitrary input coordinates [0, 1].
    Uses static methods from AudioQuantumDataset for encoding/decoding.
    Plots the normalized model output against the input coordinates.

    Args:
        model (DQNN): Trained DQNN model.
        input_coords (list or np.ndarray): Input coordinates in range [0, 1].
        model_name (str): Name of the model for the plot title. Defaults to "DQNN".
        save_path (str, optional): Path to save the waveform plot. Defaults to None.
    """
    # Validate and prepare input coordinates
    coords_np = np.array(input_coords)
    if np.any(coords_np < 0) or np.any(coords_np > 1):
        raise ValueError("Input coordinates must be within the range [0, 1].")
    if len(np.unique(coords_np)) != len(coords_np):
        print("Warning: Input coordinates contain duplicates. Using unique values and sorting.")
        coords_np = np.unique(coords_np)
    if not np.all(np.diff(coords_np) >= 0):
        print("Warning: Input coordinates are not sorted. Sorting them now.")
        coords_np = np.sort(coords_np)

    # Perform inference point-by-point (due to quantum state handling)
    print(f"Performing {model_name} inference on {len(coords_np)} arbitrary coordinates...")
    normalized_predictions = []
    
    for coord in tqdm(coords_np, desc=f"{model_name} Arbitrary Inference"):
        time_state = AudioQuantumDataset.encode_time(coord)
        output_state = model.predict(time_state)
        norm_amp = AudioQuantumDataset.decode_amplitude(output_state)
        normalized_predictions.append(norm_amp)
        
    normalized_predictions = np.array(normalized_predictions)
    print("Inference and decoding complete.\n")

    # Plot normalized output vs input coordinates
    plot_title = f"{model_name} Inference: {len(coords_np)} samples in [{coords_np.min():.3f}, {coords_np.max():.3f}]"
    fig = plt.figure(figsize=(10, 4))
    plt.plot(coords_np, normalized_predictions, marker='.', linestyle='',
             markersize=4, alpha=0.75, color='mediumseagreen')
    plt.title(plot_title, fontsize=18) # Use auto-generated title
    plt.xlabel("Input", fontsize=14)
    plt.ylabel("Output", fontsize=14)
    plt.ylim(-1.1, 1.1)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved arbitrary inference plot to {save_path}")
    plt.show()


def inference_audio_dqnn(model, dataset, inference_sr, inference_seconds,
                              model_name="DQNN", save_audio_path=None, save_plot_path=None):
    """
    Performs DQNN inference and compares output waveform with processed ground truth.
    Uses static and dataset methods for encoding/decoding and denormalization.
    Generates input coordinates based on the dataset's desired audio duration
    and the desired inference_sr. Resamples/truncates the ground truth audio
    to match inference_sr and inference_seconds for comparison. Plots the model's
    generated waveform against the processed ground truth. Plays audio clips. 

    Args:
        model (DQNN): Trained DQNN model.
        dataset (AudioQuantumDataset): Dataset instance for the original audio.
        inference_sr (int): Target sample rate for processing and playback.
        inference_seconds (float): Target duration for comparison.
        model_name (str): Name of the model for the plot title and audio labels. Defaults to "DQNN".
        save_audio_path (str, optional): Path to save predicted audio. Defaults to None.
        save_plot_path (str, optional): Path to save waveform plot. Defaults to None.
    """
    
    print(f"\nAttempting {model_name} inference for {inference_seconds:.2f} s at {inference_sr} Hz")
    
    # --- Process Ground Truth ---
    original_data = dataset.original_data
    orig_sr = dataset.sample_rate
    orig_duration = len(original_data) / orig_sr

    # Resample ground truth to inference SR
    try:
        resampled_truth = librosa.resample(original_data, orig_sr=orig_sr, target_sr=inference_sr)
    except Exception as e:
        print(f"Error during resampling: {e}")
        return

    # Truncate resampled ground truth to inference duration
    num_samples_target = int(inference_seconds * inference_sr)
    current_duration = inference_seconds
    if num_samples_target > len(resampled_truth):
        print(f"Warning: Requested duration ({inference_seconds:.2f}s) exceeds the original audio's duration ({orig_duration:.2f}s)")
        print(f"\tUsing the original audio's duration while resampling ({len(resampled_truth)} samples at {inference_sr} Hz)")
        num_samples_target = len(resampled_truth)
        current_duration = len(resampled_truth) / inference_sr # Use actual duration
    processed_truth = resampled_truth[:num_samples_target] # Renamed variable
    print(f"Ground truth adjusted for comparison: {num_samples_target} samples "
          f"at {inference_sr} Hz ({current_duration:.2f} s)")

    # --- Generate Model Input & Predict ---
    print(f"Generating input coordinates and running inference...")
    # Coords based on *original duration* and *inference SR*, then truncated using *current duration*
    num_coords = int(orig_duration * inference_sr)
    coords_full = np.linspace(0, 1, num_coords)
    input_coords = coords_full[:num_samples_target]
    
    # Predict normalized output point-by-point
    normalized_predictions = []
    for coord in tqdm(input_coords, desc=f"{model_name} Audio Inference"):
        time_state = AudioQuantumDataset.encode_time(coord)             # Encode time coord
        output_state = model.predict(time_state)                       # Predict output quantum state
        norm_amp = AudioQuantumDataset.decode_amplitude(output_state)  # Decode to normalized amplitude
        normalized_predictions.append(norm_amp)

    # Denormalize using dataset parameters
    denormalized_predictions = dataset.amplitude_to_original(np.array(normalized_predictions))
    print("Inference and decoding and denormalization complete.")

    # Ensure lengths match
    if len(denormalized_predictions) != len(processed_truth):
        min_len = min(len(denormalized_predictions), len(processed_truth))
        denormalized_predictions = denormalized_predictions[:min_len]
        processed_truth = processed_truth[:min_len]
        print(f"\nWarning: Length mismatch between prediction length ({len(denormalized_predictions)})"
              f"and adjusted ground truth length ({len(processed_truth)}), truncating to smaller length ({min_len})")

    # --- Plotting (uses plot_audio_waveform from this file) ---
    print("Plotting waveform comparison...")
    formatted_sr = f"{inference_sr} Hz" if inference_sr < 1000 else f"{(inference_sr / 1000.0):.1f} kHz"
    plot_title = f"{model_name} Inference: {current_duration:.2f} s at {formatted_sr}"
    # Use local function
    plot_audio_waveform(
        predictions=denormalized_predictions,
        ground_truth=processed_truth,
        sample_rate=inference_sr,
        title=plot_title, # Pass auto-generated title
        seconds=current_duration,
        save_path=save_plot_path,
        color_prediction='mediumseagreen',
        color_original='mediumpurple',  
    )

    # --- Audio Playback (uses play_audio from this file) ---
    print("Playing audio clips...")
    display(play_audio(processed_truth, inference_sr,
                       f"Ground Truth Adjusted ({formatted_sr})"))
    display(play_audio(denormalized_predictions, inference_sr,
                       f"{model_name} Inference Audio ({formatted_sr})"))

    # --- Optional: Save Predicted Audio ---
    if save_audio_path:
        try:
            sf.write(save_audio_path, denormalized_predictions, inference_sr)
            print(f"Saved predicted audio to {save_audio_path}")
        except Exception as e:
            print(f"Error saving audio: {e}")
