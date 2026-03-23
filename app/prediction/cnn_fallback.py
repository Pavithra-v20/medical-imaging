import torch
import torch.nn as nn
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)

# A simple 3D CNN for fallback classification.
# In a real-world scenario, this would be a pre-trained model.
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=2): # e.g., 0: No Lesion, 1: Lesion
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7 * 7, 128) # Placeholder shape, will be adapted
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.adap_pool = nn.AdaptiveAvgPool3d((7, 7, 7))

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.adap_pool(x)
        x = x.view(-1, 16 * 7 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

async def call_cnn_fallback(volume: np.ndarray) -> dict:
    """
    Runs a simple, pre-defined 3D CNN as a fallback when MedGemma is unavailable.
    This provides a baseline prediction directly from the image volume.

    Returns a standardized prediction dict.
    """
    logger.info("medgemma_unavailable_fallback_cnn")
    
    # 1. Initialize model and load data
    # In a real app, you would load pre-trained weights, e.g., model.load_state_dict(...)
    model = Simple3DCNN()
    model.eval()

    # Convert numpy array to torch tensor
    # The preprocessor already gives us the correct (1, 1, D, H, W) shape
    volume_tensor = torch.from_numpy(volume.astype(np.float32))

    # 2. Run inference
    with torch.no_grad():
        output = model(volume_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    # 3. Format to standard response
    disease_label = "Lesion Detected" if predicted_class.item() == 1 else "No Lesion Detected"
    
    result = {
        "disease_label": disease_label,
        "confidence": confidence.item(),
        "reasoning": "Prediction from local fallback CNN model due to MedGemma unavailability.",
        "model_used": "cnn_fallback",
        "raw_response": str(probabilities.numpy()),
    }
    
    logger.info("cnn_fallback_prediction", disease_label=result["disease_label"], confidence=result["confidence"])
    
    return result
