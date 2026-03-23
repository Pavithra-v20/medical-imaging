import os
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityRanged, ToTensord
from app.utils.logger import get_logger
from app.utils.exceptions import SegmentationError

logger = get_logger(__name__)

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "mri_unet.pth"

# VISTA-3D equivalent spacing for consistancy or model specific
TARGET_SPACING = (1.5, 1.5, 1.5)

async def run_mri_segmentation(nifti_path: Path, session_id: str) -> np.ndarray:
    """
    Run local MRI segmentation using a pretrained model from the models folder.
    """
    if not MODEL_PATH.exists():
        logger.error("mri_model_missing", path=str(MODEL_PATH))
        # For demonstration purposes, if the model is missing, we raise an error
        # rather than returning a dummy, as the user explicitly asked to load weights.
        raise SegmentationError(
            f"MRI model weights not found at {MODEL_PATH}. "
            "Please place the 'mri_unet.pth' file in the 'models' directory."
        )

    try:
        # 1. Initialize Model (Example: 3D UNet)
        # In a real scenario, the architecture should match the weights.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=35, # matching VISTA-3D or custom
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)

        # 2. Load Weights
        model.load_state_code(torch.load(MODEL_PATH, map_location=device))
        model.eval()

        # 3. Preprocess & Inference
        # In a real app, we'd use MONAI Transforms
        img = nib.load(str(nifti_path))
        data = img.get_fdata()
        # simplified preprocessing for now
        input_tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        logger.info(
            "mri_segmentation_complete",
            session_id=session_id,
            mask_shape=mask.shape,
            unique_labels=int(np.unique(mask).size),
        )

        return mask.astype(np.int16)

    except Exception as e:
        logger.error("mri_inference_failed", error=str(e))
        raise SegmentationError(f"MRI local inference failed: {e}")
