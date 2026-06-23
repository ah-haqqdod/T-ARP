import os
from pathlib import Path

import numpy as np
from jax import numpy as jnp
from PIL import Image


def png_to_tensor(image_path: str) -> jnp.ndarray:
    """
    Load a PNG image and convert it to a JAX tensor.

    Args:
        image_path: Path to the PNG image file

    Returns:
        JAX tensor of shape (height, width, channels) with float32 values in [0, 1]
    """
    # Load image using PIL
    img = Image.open(image_path)

    # Convert to RGB if necessary
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Convert to numpy array and then to JAX array
    img_array = jnp.array(img, dtype=jnp.float32)

    # Normalize to [0, 1]
    img_array = img_array / 255.0

    return img_array


# dataloader
# Dataloader implementation
class KodakDataLoader:
    """Simple dataloader for PNG images from a directory."""

    def __init__(
        self,
        data_dir: str = "datasets/kodak",
        image_extensions: tuple = (".png", ".jpg", ".jpeg"),
    ):
        """
        Initialize the dataloader.

        Args:
            data_dir: Directory containing images
            image_extensions: Tuple of valid image extensions
        """
        self.data_dir = Path(data_dir)
        self.image_extensions = image_extensions
        self.image_paths = []

        # Collect all image paths
        if self.data_dir.exists() and self.data_dir.is_dir():
            for ext in image_extensions:
                self.image_paths.extend(list(self.data_dir.glob(f"*{ext}")))
                self.image_paths.extend(list(self.data_dir.glob(f"*{ext.upper()}")))

        # Sort for consistent ordering
        self.image_paths.sort()

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.image_paths)

    def __iter__(self):
        """Iterate over images."""
        self.current_idx = 0
        return self

    def __next__(self) -> jnp.ndarray:
        """Get the next image as a tensor."""
        if self.current_idx >= len(self.image_paths):
            raise StopIteration

        img_path = self.image_paths[self.current_idx]
        self.current_idx += 1

        return png_to_tensor(str(img_path))
