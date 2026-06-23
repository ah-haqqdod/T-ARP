import subprocess

import numpy as np
import yuvio
from jax import numpy as jnp

# TODO: add verifications


def read_yuv_tensor(yuv_path, width, height, pix_fmt="yuv420p"):
    """
    Reads an entire YUV file into a JAX/NumPy tensor.
    Returns: A tensor with shape (Frames, Height, Width, 3) and dtype uint8.
    """

    # 1. Read frames using yuvio
    frames = yuvio.mimread(yuv_path, width, height, pix_fmt)

    # 2. Prepare a list to hold each frame's full-resolution data
    frame_tensors = []

    for frame in frames:
        # frame.y is full resolution (height, width)
        # frame.u and frame.v are HALF resolution for yuv420p (height//2, width//2)
        y_plane = frame.y

        # UPSAMPLE U and V to full resolution to create (H, W, 3) tensor
        # This is necessary because your tensor format expects all 3 channels at same size
        u_plane_full = np.repeat(np.repeat(frame.u, 2, axis=0), 2, axis=1)
        v_plane_full = np.repeat(np.repeat(frame.v, 2, axis=0), 2, axis=1)

        # Stack planes to create (Height, Width, 3) array
        frame_full = np.stack([y_plane, u_plane_full, v_plane_full], axis=-1)
        frame_tensors.append(frame_full)

    # 3. Stack all frames along the first dimension
    video_tensor = np.stack(frame_tensors, axis=0)

    print(f"Read {len(frames)} frames. Tensor shape: {video_tensor.shape}")
    return jnp.array(video_tensor)  # Convert to JAX array


def write_tensor_to_yuv(tensor, yuv_path, width, height, pix_fmt="yuv420p"):
    """
    Writes a tensor to a YUV file.
    Accepts: A tensor with shape (Frames, Height, Width, 3) and dtype uint8.
    """
    # Convert JAX array to NumPy if needed
    if isinstance(tensor, jnp.ndarray):
        tensor = np.array(tensor)

    num_frames = tensor.shape[0]
    frames_list = []

    for i in range(num_frames):
        # Extract full-resolution planes from tensor
        y_plane = tensor[i, :, :, 0]  # Shape: (height, width)
        u_plane_full = tensor[i, :, :, 1]  # Full resolution U
        v_plane_full = tensor[i, :, :, 2]  # Full resolution V

        # DOWNSAMPLE U and V for yuv420p format
        # Average every 2x2 block - this is the reverse of the upsampling in read function
        u_plane = (
            u_plane_full[0::2, 0::2].astype(np.uint16)
            + u_plane_full[0::2, 1::2]
            + u_plane_full[1::2, 0::2]
            + u_plane_full[1::2, 1::2]
        ) // 4

        v_plane = (
            v_plane_full[0::2, 0::2].astype(np.uint16)
            + v_plane_full[0::2, 1::2]
            + v_plane_full[1::2, 0::2]
            + v_plane_full[1::2, 1::2]
        ) // 4

        # Create yuvio frame object with correctly shaped planes
        frame = yuvio.frame(
            (y_plane, u_plane.astype(np.uint8), v_plane.astype(np.uint8)), pix_fmt
        )
        frames_list.append(frame)

    # Write all frames to file
    yuvio.mimwrite(yuv_path, frames_list)
    print(f"Saved {num_frames} frames to {yuv_path}")


def yuv_to_mp4_ffmpeg(yuv_path, mp4_path, width, height, framerate, pix_fmt="yuv420p"):
    """
    Convert a raw YUV video file to MP4 using FFmpeg.

    Args:
        yuv_path (str): Path to the input .yuv file.
        mp4_path (str): Path for the output .mp4 file.
        width (int): Frame width in pixels.
        height (int): Frame height in pixels.
        framerate (int): Frame rate of the video (e.g., 30 for 30 fps).
        pix_fmt (str): YUV pixel format. Must match the file (e.g., 'yuv420p', 'nv12').
    """
    # Build the FFmpeg command using the recommended options for raw video[citation:3]
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file without asking
        "-f",
        "rawvideo",  # Force input format
        "-video_size",
        f"{width}x{height}",  # Specify frame size[citation:3]
        "-framerate",
        str(framerate),  # Input frame rate
        "-pixel_format",
        pix_fmt,  # Pixel format[citation:3]
        "-i",
        str(yuv_path),  # Input file
        "-c:v",
        "libx264",  # Use the H.264 video codec
        "-preset",
        "slow",  # Encoding speed/compression trade-off
        "-crf",
        "23",  # Quality (18-28 is typical, lower = better quality)
        "-pix_fmt",
        "yuv420p",  # Output pixel format (widely compatible)
        str(mp4_path),
    ]

    try:
        # Execute the command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully converted '{yuv_path}' to '{mp4_path}'")
    except FileNotFoundError:
        print(
            "Error: 'ffmpeg' command not found. Please ensure FFmpeg is installed and in your system PATH."
        )
        raise
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg conversion failed with error code {e.returncode}:")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise
