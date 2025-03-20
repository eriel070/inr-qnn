import torch
import torch.nn as nn
import numpy as np

class GaborMFN(nn.Module):
    """
    Multiplicative Filter Network with Gabor filters for signal representation.
    """
    def __init__(self, in_features=1, hidden_features=256, out_features=1, num_layers=4, 
                 weight_scale=1.0, alpha=6.0, beta=1.0, omega_0=30.0):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_features = hidden_features
        
        # First layer parameters
        self.first_linear = nn.Linear(in_features, hidden_features)
        self.first_linear.weight.data.uniform_(-np.sqrt(3 * weight_scale / in_features), 
                                              np.sqrt(3 * weight_scale / in_features))
        self.first_linear.bias.data.uniform_(-np.sqrt(1 / hidden_features), 
                                            np.sqrt(1 / hidden_features))
        
        # Create linear layers
        self.linears = nn.ModuleList([nn.Linear(hidden_features, hidden_features) 
                                     for _ in range(num_layers - 2)])
        # Final output layer
        self.linears.append(nn.Linear(hidden_features, out_features))
        
        # Initialize weights with uniform distribution
        for linear in self.linears:
            linear.weight.data.uniform_(-np.sqrt(3 * weight_scale / hidden_features), 
                                       np.sqrt(3 * weight_scale / hidden_features))
            linear.bias.data.uniform_(-np.sqrt(1 / hidden_features), 
                                     np.sqrt(1 / hidden_features))
        
        # Gabor filter parameters
        # omegas: frequencies of sinusoidal components
        self.omegas = nn.ParameterList([nn.Parameter(torch.empty(hidden_features)) 
                                       for _ in range(num_layers - 1)])
        # phis: phase shifts
        self.phis = nn.ParameterList([nn.Parameter(torch.empty(hidden_features)) 
                                     for _ in range(num_layers - 1)])
        # mus: means of Gaussian envelopes (centers)
        self.mus = nn.ParameterList([nn.Parameter(torch.empty(hidden_features)) 
                                    for _ in range(num_layers - 1)])
        # gammas: scales of Gaussian envelopes (inverse width)
        self.gammas = nn.ParameterList([nn.Parameter(
            torch.distributions.Gamma(alpha / (i + 1), beta).sample((hidden_features,))
        ) for i in range(num_layers - 1)])
        
        # Initialize parameters
        for i in range(num_layers - 1):
            # Initialize frequencies with decreasing magnitudes at deeper layers
            self.omegas[i].data.uniform_(-1, 1).mul_(2 * np.pi * omega_0 * self.gammas[i])
            # Random phase initialization
            self.phis[i].data.uniform_(-np.pi, np.pi)
            # Center Gaussians within input domain
            self.mus[i].data.uniform_(-1, 1)
            
    def forward(self, x):
        """
        Forward pass through the MFN.
        
        Args:
            x: Input tensor of shape (batch_size, 1) for audio representation
            
        Returns:
            Signal values at the input coordinates
        """
        # First Gabor filter
        g = torch.exp(-0.5 * self.gammas[0] * torch.square(x - self.mus[0])) * \
            torch.sin(self.omegas[0] * x + self.phis[0])
        
        # Apply first linear layer
        z = self.first_linear(x) * g
        
        # Apply remaining layers with multiplicative Gabor filters
        for i in range(1, self.num_layers - 1):
            # Compute Gabor filter response
            g = torch.exp(-0.5 * self.gammas[i] * torch.square(x - self.mus[i])) * \
                torch.sin(self.omegas[i] * x + self.phis[i])
            
            # Multiplicative update - key difference from traditional networks
            z = self.linears[i - 1](z) * g
            
        # Final linear layer
        return self.linears[-1](z)


def get_gabor_mfn_model(hidden_size=256, num_layers=5, omega_0=30.0):
    """
    Creates a Gabor MFN model configured for audio representation
    
    Args:
        hidden_size: Width of hidden layers
        num_layers: Total number of layers including input and output
        omega_0: Base frequency scale
        
    Returns:
        Initialized Gabor MFN model
    """
    model = GaborMFN(
        in_features=1,            # Time parameter (t)
        hidden_features=hidden_size,
        out_features=1,           # Audio amplitude
        num_layers=num_layers,
        omega_0=omega_0           # Base frequency scale
    )
    return model