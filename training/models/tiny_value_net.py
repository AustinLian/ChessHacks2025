import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyValueNet(nn.Module):
    """
    Minimal value-only network for quick training and debugging.
    
    Much smaller than full policy-value network.
    Can be used for:
    - Fast prototyping
    - Distillation target
    - Baseline comparison
    """
    
    def __init__(self, input_planes=27, hidden_size=256):
        super().__init__()
        
        self.input_planes = input_planes
        
        # Simple conv layers
        self.conv1 = nn.Conv2d(input_planes, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        # Value head
        self.fc1 = nn.Linear(32 * 8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Input: (batch, 27, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Value output
        x = F.relu(self.fc1(x))
        value = torch.tanh(self.fc2(x))
        
        return value
    
    def predict(self, x):
        """Convenience method for inference."""
        with torch.no_grad():
            return self.forward(x)


def create_tiny_model(config=None):
    """Factory function to create tiny model."""
    if config is None:
        config = {
            'input_planes': 27,
            'hidden_size': 256,
        }
    
    return TinyValueNet(**config)
