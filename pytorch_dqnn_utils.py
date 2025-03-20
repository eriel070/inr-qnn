import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf
from typing import List, Tuple, Optional, Dict, Any, Union

from pytorch_dqnn import PyTorchDQNN

class DQNNAudioDataset(Dataset):
    """
    Dataset for training DQNN models on audio signals.
    Maps time coordinates to amplitude values.
    """
    def __init__(self, 
                audio_path: str, 
                max_seconds: Optional[float] = None, 
                normalize_time: bool = True, 
                normalize_amplitude: bool = True):
        """
        Initialize dataset from audio file.
        
        Args:
            audio_path: Path to audio file
            max_seconds: Maximum number of seconds to use (None = use all)
            normalize_time: If True, normalize time indices to [0, 1]
            normalize_amplitude: If True, normalize amplitudes to [-1, 1]
        """
        # Load audio file
        data, self.sample_rate = sf.read(audio_path, dtype='float32')
        
        # Convert to mono if stereo
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
            
        # Convert to torch tensors
        self.time_indices_tensor = torch.tensor(self.time_indices_normalized, dtype=torch.float32).unsqueeze(1)
        self.data_tensor = torch.tensor(data_normalized, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return self.data_length
    
    def __getitem__(self, idx):
        return self.time_indices_tensor[idx], self.data_tensor[idx]
    
    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a random batch of samples"""
        indices = np.random.choice(len(self), batch_size)
        times = self.time_indices_tensor[indices]
        amplitudes = self.data_tensor[indices]
        return times, amplitudes
    
    def normalize_time_value(self, time_value: float) -> float:
        """Convert a time value to normalized time space"""
        if self.normalize_time:
            return time_value / (self.data_length - 1)
        return time_value
        
    def denormalize_time_value(self, normalized_time: float) -> float:
        """Convert a normalized time value back to original time space"""
        if self.normalize_time:
            return normalized_time * (self.data_length - 1)
        return normalized_time
    
    def normalize_amplitude_value(self, amplitude: float) -> float:
        """Convert an amplitude value to normalized space"""
        if self.normalize_amplitude:
            if self.amplitude_range < 1e-6:
                abs_max = np.max(np.abs(self.original_data))
                return amplitude / (abs_max if abs_max > 0 else 1.0)
            else:
                return 2.0 * (amplitude - self.amplitude_min) / self.amplitude_range - 1.0
        return amplitude
    
    def denormalize_amplitude_value(self, normalized_amplitude: float) -> float:
        """Convert a normalized amplitude back to original space"""
        if self.normalize_amplitude:
            if self.amplitude_range < 1e-6:
                abs_max = np.max(np.abs(self.original_data))
                return normalized_amplitude * (abs_max if abs_max > 0 else 1.0)
            else:
                return (normalized_amplitude + 1.0) * self.amplitude_range / 2.0 + self.amplitude_min
        return normalized_amplitude
    
    def get_original_audio(self) -> np.ndarray:
        """Get the original unprocessed audio data"""
        return self.original_data
    
    def generate_time_points(self, num_points: Optional[int] = None) -> torch.Tensor:
        """Generate evenly spaced time points in normalized space for inference"""
        if num_points is None:
            num_points = self.data_length
            
        if self.normalize_time:
            return torch.linspace(0, 1, num_points).unsqueeze(1)
        else:
            return torch.linspace(0, self.data_length-1, num_points).unsqueeze(1)


def train_dqnn_model(
    model: PyTorchDQNN, 
    dataset: DQNNAudioDataset, 
    num_epochs: int = 1000, 
    batch_size: int = 512, 
    learning_rate: float = 0.01,
    device: Optional[str] = None,
    log_interval: int = 10
) -> Tuple[PyTorchDQNN, List[float]]:
    """
    Train a DQNN model on an audio dataset.
    
    Args:
        model: The DQNN model to train
        dataset: DQNNAudioDataset instance
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device to train on ('cpu' or 'cuda')
                If None, uses the model's device
        log_interval: Interval for logging progress
        
    Returns:
        Trained model and history of losses
    """
    if device is None:
        device = model.device
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs on {device}...")
    losses = model.train(dataloader, num_epochs, learning_rate, progress_bar=True)
    
    print(f"Training completed. Final loss: {losses[-1]:.6f}")
    return model, losses


def evaluate_dqnn_model(
    model: PyTorchDQNN, 
    dataset: DQNNAudioDataset, 
    device: Optional[str] = None
) -> np.ndarray:
    """
    Evaluate a trained DQNN model by generating audio samples for the entire signal.
    
    Args:
        model: Trained DQNN model
        dataset: DQNNAudioDataset instance
        device: Device to use for inference
    
    Returns:
        Generated audio samples (denormalized) at the original sample rate
    """
    if device is None:
        device = model.device
        
    model.eval()
    
    # Generate time points in the normalized space expected by the model
    time_points = dataset.generate_time_points().to(device)
    
    # Generate predictions in small batches to avoid memory issues
    batch_size = 1000
    num_batches = (len(time_points) + batch_size - 1) // batch_size
    
    normalized_predictions = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(time_points))
        
        batch_time_points = time_points[start_idx:end_idx]
        
        with torch.no_grad():
            batch_predictions = model(batch_time_points).cpu().numpy().flatten()
            
        normalized_predictions.append(batch_predictions)
    
    normalized_predictions = np.concatenate(normalized_predictions)
    
    # Denormalize predictions back to original amplitude range
    predictions = np.array([
        dataset.denormalize_amplitude_value(amp) for amp in normalized_predictions
    ])
    
    return predictions


def evaluate_and_visualize_dqnn(
    model: PyTorchDQNN, 
    dataset: DQNNAudioDataset, 
    model_name: str = "PyTorch DQNN", 
    seconds_to_show: int = 5,
    save_audio: Optional[str] = None, 
    save_plot: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Comprehensive evaluation and visualization of a trained DQNN model.
    
    Args:
        model: Trained DQNN model
        dataset: DQNNAudioDataset instance
        model_name: Name of the model for display
        seconds_to_show: Number of seconds to visualize
        save_audio: Optional path to save audio output (None to skip saving)
        save_plot: Optional path to save waveform plot (None to skip saving)
    
    Returns:
        predictions, ground_truth, mse, psnr
    """
    # Get original ground truth data
    ground_truth = dataset.get_original_audio()
    
    # Generate predictions and denormalize them
    predictions = evaluate_dqnn_model(model, dataset)
    
    # Make sure predictions match the length of ground truth
    if len(predictions) != len(ground_truth):
        print(f"Warning: prediction length ({len(predictions)}) doesn't match ground truth ({len(ground_truth)})")
        # Adjust lengths to match for comparison purposes
        min_length = min(len(predictions), len(ground_truth))
        predictions = predictions[:min_length]
        ground_truth = ground_truth[:min_length]
    
    # Calculate metrics
    mse = np.mean((predictions - ground_truth)**2)
    max_amp = np.max(np.abs(ground_truth))
    if max_amp > 0:
        psnr = 20 * np.log10(max_amp / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    print(f"=== {model_name} Evaluation ===")
    print(f"MSE: {mse:.6f}")
    print(f"PSNR: {psnr:.2f} dB")
    
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
        print(f"Saved waveform plot to {save_plot}")
        
    plt.show()
    
    # Save audio if path provided
    if save_audio:
        sf.write(save_audio, predictions, dataset.sample_rate)
        print(f"Saved predicted audio to {save_audio}")
    
    return predictions, ground_truth, mse, psnr


def plot_training_curve(losses: List[float], model_name: str = "PyTorch DQNN", save_path: Optional[str] = None) -> None:
    """
    Plot the training loss curve.
    
    Args:
        losses: List of loss values from training
        model_name: Name of the model for display
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.title(f"{model_name} Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training loss plot to {save_path}")
        
    plt.show()


def get_dqnn_model(input_qubits: int = 1, 
                  hidden_qubits: List[int] = [2, 2], 
                  output_qubits: int = 1, 
                  device: Optional[str] = None) -> PyTorchDQNN:
    """
    Create a DQNN model configured for audio representation.
    
    Args:
        input_qubits: Number of qubits in input layer (typically 1 for time)
        hidden_qubits: List of the number of qubits in each hidden layer
        output_qubits: Number of qubits in output layer (typically 1 for amplitude)
        device: Device to run the model on ('cuda' or 'cpu')
               If None, automatically selects cuda if available
        
    Returns:
        Initialized DQNN model
    """
    qnn_arch = [input_qubits] + hidden_qubits + [output_qubits]
    model = PyTorchDQNN(qnn_arch, device=device)
    
    # Print model details
    num_params = model.get_num_parameters()
    print(f"DQNN architecture: {qnn_arch}")
    print(f"DQNN parameters: {num_params:,}")
    print(f"DQNN running on: {model.device}")
    
    return model