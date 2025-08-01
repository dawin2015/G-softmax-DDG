import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the GSoftmax activation class
class GSoftmax(nn.Module):
    def __init__(self, temperature=1.0, alpha=0.1):
        super(GSoftmax, self).__init__()
        # Temperature parameter for scaling logits
        self.temperature = temperature
        # Alpha parameter for Gaussian regularization strength
        self.alpha = alpha

    def forward(self, logits, labels):
        # Compute the traditional softmax over logits, scaled by temperature
        softmax = F.softmax(logits / self.temperature, dim=1)
        
        # Compute the Gaussian regularization term
        num_classes = logits.size(1)
        # Convert labels to one-hot encoding
        one_hot_labels = F.one_hot(labels, num_classes).float()
        # Calculate the mean logits for each class
        class_means = torch.sum(one_hot_labels.unsqueeze(2) * logits.unsqueeze(1), dim=0) / (torch.sum(one_hot_labels, dim=0).unsqueeze(1) + 1e-8)
        # Calculate the Euclidean distance from each sample to each class mean
        distances = torch.cdist(logits, class_means, p=2)
        # Compute the Gaussian term based on distances
        gaussian_term = torch.exp(-self.alpha * distances)
        
        # Combine the softmax output and the Gaussian regularization term
        g_softmax = softmax * gaussian_term
        # Normalize the result so each row sums to 1
        g_softmax = g_softmax / g_softmax.sum(dim=1, keepdim=True)
        
        return g_softmax

# Example usage of GSoftmax in a neural network model
class ExampleModel(nn.Module):
    def __init__(self, num_classes):
        super(ExampleModel, self).__init__()
        # Linear layer to produce logits
        self.fc = nn.Linear(512, num_classes)
        # Instantiate the GSoftmax activation
        self.g_softmax = GSoftmax()

    def forward(self, x, labels):
        # Compute logits from input features
        logits = self.fc(x)
        # Apply GSoftmax activation
        g_softmax_output = self.g_softmax(logits, labels)
        return g_softmax_output
