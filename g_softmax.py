import torch
import torch.nn as nn
import torch.nn.functional as F

class GSoftmax(nn.Module):
    def __init__(self, temperature=1.0, alpha=0.1):
        super(GSoftmax, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, logits, labels):
        # Compute the traditional softmax
        softmax = F.softmax(logits / self.temperature, dim=1)
        
        # Compute the Gaussian regularization term
        num_classes = logits.size(1)
        one_hot_labels = F.one_hot(labels, num_classes).float()
        class_means = torch.sum(one_hot_labels.unsqueeze(2) * logits.unsqueeze(1), dim=0) / (torch.sum(one_hot_labels, dim=0).unsqueeze(1) + 1e-8)
        distances = torch.cdist(logits, class_means, p=2)
        gaussian_term = torch.exp(-self.alpha * distances)
        
        # Combine the softmax and Gaussian term
        g_softmax = softmax * gaussian_term
        g_softmax = g_softmax / g_softmax.sum(dim=1, keepdim=True)
        
        return g_softmax

# Example usage in a neural network model
class ExampleModel(nn.Module):
    def __init__(self, num_classes):
        super(ExampleModel, self).__init__()
        self.fc = nn.Linear(512, num_classes)
        self.g_softmax = GSoftmax()

    def forward(self, x, labels):
        logits = self.fc(x)
        g_softmax_output = self.g_softmax(logits, labels)
        return g_softmax_output
    

