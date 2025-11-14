import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with two conv layers."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class PolicyHead(nn.Module):
    """Policy head for move prediction."""
    
    def __init__(self, in_channels, policy_channels=32, policy_size=4672):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, policy_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(policy_channels)
        self.fc = nn.Linear(policy_channels * 8 * 8, policy_size)
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ValueHead(nn.Module):
    """Value head for position evaluation."""
    
    def __init__(self, in_channels, value_channels=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, value_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(value_channels)
        self.fc1 = nn.Linear(value_channels * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class ResNetPolicyValue(nn.Module):
    """
    ResNet-based policy-value network for chess.
    
    Architecture:
    - Initial conv layer
    - N residual blocks
    - Policy head (move probabilities)
    - Value head (position evaluation)
    """
    
    def __init__(self, num_blocks=10, channels=128, input_planes=27,
                 policy_channels=32, value_channels=32, policy_size=4672):
        super().__init__()
        
        self.input_planes = input_planes
        self.num_blocks = num_blocks
        self.channels = channels
        
        # Initial convolution
        self.conv_input = nn.Conv2d(input_planes, channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResBlock(channels) for _ in range(num_blocks)
        ])
        
        # Output heads
        self.policy_head = PolicyHead(channels, policy_channels, policy_size)
        self.value_head = ValueHead(channels, value_channels)
        
    def forward(self, x):
        # Input: (batch, 27, 8, 8)
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Heads
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
    
    def predict(self, x):
        """Convenience method for inference."""
        with torch.no_grad():
            policy, value = self.forward(x)
            policy = F.softmax(policy, dim=-1)
            return policy, value


def create_model(config=None):
    """Factory function to create model from config."""
    if config is None:
        config = {
            'num_blocks': 10,
            'channels': 128,
            'input_planes': 27,
            'policy_channels': 32,
            'value_channels': 32,
            'policy_size': 4672,
        }
    
    return ResNetPolicyValue(**config)
