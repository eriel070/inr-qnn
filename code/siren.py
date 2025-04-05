import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    """
    Sine activation layer with custom initialization
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer initialization - uniform within range
                self.linear.weight.uniform_(-1 / self.in_features, 
                                            1 / self.in_features)      
            else:
                # Hidden layer initialization - preserves variance of activations
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                            np.sqrt(6 / self.in_features) / self.omega_0)
                
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """
    SIREN model for representing signals like audio, images, etc.
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, 
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        # Network architecture
        self.net = []
        
        # First layer
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        
        # Hidden layers
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                     is_first=False, omega_0=hidden_omega_0))
        
        # Final layer - can be linear or sine
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            # Initialize last layer with specific distribution
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                            np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        # Convert to Sequential model
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        """Forward pass
        Args:
            x: Input coordinates (time steps for audio)
               Expected shape: (batch_size, 1) for audio
        Returns:
            Output values (amplitudes for audio)
        """
        return self.net(x)


def get_siren_model(hidden_size=256, hidden_layers=5, first_omega_0=30):
    """
    Creates a SIREN model configured for audio representation
    
    Args:
        hidden_size: Width of hidden layers
        hidden_layers: Number of hidden layers
        first_omega_0: Frequency multiplier for first layer
        
    Returns:
        Initialized SIREN model
    """
    model = SIREN(
        in_features=1,            # Time parameter (t)
        hidden_features=hidden_size,
        hidden_layers=hidden_layers,
        out_features=1,           # Audio amplitude
        outermost_linear=True,    # Linear output for unrestricted range
        first_omega_0=first_omega_0, # Controls how fine details are captured
        hidden_omega_0=30.0       # Standard for hidden layers
    )
    return model