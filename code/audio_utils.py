import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import soundfile as sf
import librosa
from tqdm.notebook import tqdm

class AudioSignalDataset(Dataset):
    """
    Dataset for training INR models on audio signals.
    Maps time indices to amplitude values.
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
            
        # Convert to torch tensors
        self.time_indices_tensor = torch.tensor(self.time_indices_normalized, dtype=torch.float32).unsqueeze(1)
        self.data_tensor = torch.tensor(data_normalized, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return self.data_length
    
    def __getitem__(self, idx):
        return self.time_indices_tensor[idx], self.data_tensor[idx]
    
    def get_batch(self, batch_size):
        """Get a random batch of samples"""
        indices = np.random.choice(len(self), batch_size)
        times = self.time_indices_tensor[indices]
        amplitudes = self.data_tensor[indices]
        return times, amplitudes
    
    def normalize_time_value(self, time_value):
        """Convert a time value to normalized time space"""
        if self.normalize_time:
            return time_value / (self.data_length - 1)
        return time_value
        
    def denormalize_time_value(self, normalized_time):
        """Convert a normalized time value back to original time space"""
        if self.normalize_time:
            return normalized_time * (self.data_length - 1)
        return normalized_time
    
    def normalize_amplitude_value(self, amplitude):
        """Convert an amplitude value to normalized space"""
        if self.normalize_amplitude:
            if self.amplitude_range < 1e-6:
                abs_max = np.max(np.abs(self.original_data))
                return amplitude / (abs_max if abs_max > 0 else 1.0)
            else:
                return 2.0 * (amplitude - self.amplitude_min) / self.amplitude_range - 1.0
        return amplitude
    
    def denormalize_amplitude_value(self, normalized_amplitude):
        """Convert a normalized amplitude back to original space"""
        if self.normalize_amplitude:
            if self.amplitude_range < 1e-6:
                abs_max = np.max(np.abs(self.original_data))
                return normalized_amplitude * (abs_max if abs_max > 0 else 1.0)
            else:
                return (normalized_amplitude + 1.0) * self.amplitude_range / 2.0 + self.amplitude_min
        return normalized_amplitude
    
    def get_original_audio(self):
        """Get the original unprocessed audio data"""
        return self.original_data
    
    def generate_time_points(self, num_points=None):
        """Generate evenly spaced time points in normalized space for inference"""
        if num_points is None:
            num_points = self.data_length
            
        if self.normalize_time:
            return torch.linspace(0, 1, num_points).unsqueeze(1)
        else:
            return torch.linspace(0, self.data_length-1, num_points).unsqueeze(1)


def train_inr_model(model, dataset, num_epochs=1000, batch_size=1024, lr=1e-4, 
                   device='cuda', log_interval=50):
    """
    Train an INR model on an audio dataset.
    
    Args:
        model: The model to train
        dataset: AudioSignalDataset instance
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        log_interval: Interval for logging progress
        
    Returns:
        Trained model and history of losses
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100
    )
    criterion = nn.MSELoss()
    
    losses = []
    best_loss = float('inf')
    
    print(f"Starting training for {num_epochs} epochs...")
    
    # Using tqdm.notebook.tqdm for better display in Jupyter
    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        # Get random batch
        time_indices, amplitudes = dataset.get_batch(batch_size)
        time_indices = time_indices.to(device)
        amplitudes = amplitudes.to(device)
        
        # Forward pass
        outputs = model(time_indices)
        loss = criterion(outputs, amplitudes)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update learning rate based on loss
        scheduler.step(loss)
        
        # Store loss
        current_loss = loss.item()
        losses.append(current_loss)
        
        # Track best model
        if current_loss < best_loss:
            best_loss = current_loss
        
        # Update progress bar description
        if epoch % log_interval == 0 or epoch == num_epochs - 1:
            pbar.set_description(f"Training: loss={current_loss:.6f}, best={best_loss:.6f}")
    
    print(f"Training completed. Final loss: {losses[-1]:.5e}, Best loss: {best_loss:.5e}")
    return model, losses


def evaluate_audio_model(model, dataset, device='cuda'):
    """
    Evaluate a trained INR model by generating audio samples for the entire signal.
    
    Args:
        model: Trained INR model
        dataset: AudioSignalDataset instance
        device: Device to use for inference
    
    Returns:
        Generated audio samples (denormalized) at the original sample rate
    """
    model.eval()
    
    # Generate time points in the normalized space expected by the model
    time_points = dataset.generate_time_points().to(device)
    
    # Generate predictions
    with torch.no_grad():
        normalized_predictions = model(time_points).cpu().numpy().flatten()
    
    # Denormalize predictions back to original amplitude range
    predictions = dataset.denormalize_amplitude_value(normalized_predictions)
    
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
        seconds: Number of seconds to show
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
    
    return fig


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


def evaluate_and_visualize(model, dataset, device, model_name,
                           seconds_to_show=None, save_audio=None, save_plot=None):
    """
    Comprehensive evaluation and visualization of a trained model.
    
    Args:
        model: Trained INR model
        dataset: AudioSignalDataset instance
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
    predictions = evaluate_audio_model(model, dataset, device)
    
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
    
    print(f"=== {model_name} Evaluation (Denormalized)===")
    print(f"MSE: {mse:.5e}")
    print(f"PSNR: {psnr:.2f} dB")
    
    # Use plot_audio_waveform for visualization
    plot_title = f"{model_name}"
    plot_audio_waveform(
        predictions, 
        ground_truth, 
        dataset.sample_rate, 
        plot_title, 
        seconds_to_show,
        save_path=save_plot
    )
    
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


#################################################################################

def inference_arbitrary(model, input_coords, device, model_name="Classical INR", save_path=None):
    """
    Performs inference using arbitrary input coordinates [0, 1].
    Plots the normalized model output against the input coordinates.
    
    Args:
        model: Trained classical INR model (e.g., SIREN, GaborMFN).
        input_coords (list or np.ndarray): Input coordinates in range [0, 1].
        device: PyTorch device ('cuda' or 'cpu').
        model_name (str): Name of the model for the plot title.
        save_path (str, optional): Path to save the waveform plot. Defaults to None.
    """
    model.eval()
    model.to(device)

    # Validate and prepare input coordinates
    coords_np = np.array(input_coords, dtype=np.float32)
    
    if np.any(coords_np < 0) or np.any(coords_np > 1):
        raise ValueError("Input coordinates must be within the range [0, 1].")
    if len(np.unique(coords_np)) != len(coords_np):
        print("Warning: Input coordinates contain duplicates. Using unique values and sorting.")
        coords_np = np.unique(coords_np)
    if not np.all(np.diff(coords_np) >= 0):
        print("Warning: Input coordinates are not sorted. Sorting them now.")
        coords_np = np.sort(coords_np)

    coords_tensor = torch.tensor(coords_np, dtype=torch.float32).unsqueeze(1).to(device)

    # Perform inference
    print(f"Performing {model_name} inference on {len(coords_tensor)} arbitrary coordinates...")
    with torch.no_grad():
        # Output is expected in the normalized range used during training
        normalized_predictions = model(coords_tensor).cpu().numpy().flatten()
    print("Inference complete.\n")

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


def inference_audio(model, dataset, inference_sr, inference_seconds, device,
                         model_name="Classical INR", save_audio_path=None, save_plot_path=None):
    """
    Performs inference and compares output waveform with processed ground truth.
    Generates input coordinates based on the dataset's desired audio duration
    and the desired inference_sr. Resamples/truncates the ground truth audio
    to match inference_sr and inference_seconds for comparison. Plots the model's
    generated waveform against the processed ground truth. Plays audio clips. 
    
    Args:
        model: Trained classical INR model (e.g., SIREN, GaborMFN).
        dataset (AudioSignalDataset): Dataset instance for the original audio.
        inference_sr (int): Target sample rate for processing and playback.
        inference_seconds (float): Target duration for comparison.
        device: PyTorch device ('cuda' or 'cpu').
        model_name (str): Name of the model for the plot title and audio labels.
        save_audio_path (str, optional): Path to save predicted audio. Defaults to None.
        save_plot_path (str, optional): Path to save waveform plot. Defaults to None.
    """
    model.eval()
    model.to(device)

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
    coords_full = torch.linspace(0, 1, num_coords).unsqueeze(1)
    input_coords = coords_full[:num_samples_target].to(device)

    # Predict normalized output
    with torch.no_grad():
        normalized_predictions = model(input_coords).cpu().numpy().flatten()

    # Denormalize using dataset parameters
    denormalized_predictions = dataset.denormalize_amplitude_value(normalized_predictions)
    print("Inference and denormalization complete.")

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
