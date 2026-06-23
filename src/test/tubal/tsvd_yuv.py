import subprocess
from functools import partial
from typing import Optional

import chex
import jax
import numpy as np

# from PIL import Image
import yuvio
from jax import numpy as jnp

from t_arp.matrix import RSVD
from t_arp.tubal import TARP, TCSSBaselines, TMatrix, TMatrixTOnly, t_cross

# METHOD_NAME = "orth_proj_pinv"
# METHOD_NAME = "orth_proj_normalized"
METHOD_NAME = "householder"
N_SLICES = 50
# VIDEO_PATH = "datasets/yuv_video/Video/waterfall_cif.yuv"
VIDEO_PATH = "datasets/yuv_video/Video/stefan_cif.yuv"
WIDTH = 352  # QCIF width
HEIGHT = 288  # QCIF height
FRAMES = 90
FRAMERATE = 25
SAVE_PATH = "src/test/tubal/"
# SAVE_PATH = "src/t_arp/tubal/"


def read_yuv_to_tensor(yuv_path, width, height, pix_fmt="yuv420p"):
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


try:
    tensor = read_yuv_to_tensor(VIDEO_PATH, WIDTH, HEIGHT)
    print(f"Success! Tensor shape: {tensor.shape}")  # e.g., (5, 144, 176, 3)
    print(f"Data type: {tensor.dtype}")
    print(f"Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
except FileNotFoundError:
    print(f"Error: File not found at {VIDEO_PATH}. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")

yuv_to_mp4_ffmpeg(
    VIDEO_PATH, SAVE_PATH + "video_org.mp4", WIDTH, HEIGHT, FRAMERATE, pix_fmt="yuv420p"
)


def t_svd(M: TMatrix) -> TMatrixTOnly:
    svd_fn = partial(jnp.linalg.svd, full_matrices=False)
    t_U, t_S, t_V = M.facewise_operation(svd_fn)

    return t_V


def t_rsvd(key: jax.random.PRNGKey, M: TMatrix) -> TMatrixTOnly:
    # rsvd = RSVD(
    #     rank=min(M.shape[0], M.shape[1]) // 4, n_oversamples=5, n_subspace_iters=1
    # )
    rsvd = RSVD(
        rank=min(N_SLICES * 2, min(M.shape[0], M.shape[1])),
        # rank=min(N_SLICES * 3, min(M.shape[0], M.shape[1])),
        n_oversamples=5,
        n_subspace_iters=1,
    )
    rsvd_fn = lambda A: rsvd(key=key, A=A)
    # svd_fn = partial(jnp.linalg.svd, full_matrices=False)
    _, _, t_V = M.facewise_operation(rsvd_fn)

    return t_V


def func(tensor, V):
    rank = N_SLICES
    key = jax.random.PRNGKey(0)
    M = TMatrix(tensor)

    print("Finding columns J")

    key, subkey = jax.random.split(key)

    A_J, W, A_I = t_cross(
        subkey,
        M,
        V,
        partial_tcss_module_constructor=partial(TARP, method=METHOD_NAME),
        n_vert_slices=rank,
        # use_intersection=True,
    )

    # J = t_arp(subkey, V)
    # # A_J = TMatrix(M.data[:, J, :])
    # A_J = TMatrix(jnp.take(M.data, J, axis=1))
    # jax.block_until_ready(A_J)

    # print("QR decomposition of A(J, :, ...)")
    # Q_J, _ = A_J.facewise_operation(partial(jnp.linalg.qr, mode="reduced"))
    # Q_J = Q_J.create_t_matrix(dtype=jnp.float16)
    # jax.block_until_ready(Q_J)

    # print("Finding rows I")
    # key, subkey = jax.random.split(key)
    # I = t_arp(subkey, Q_J)
    # # A_I = TMatrix(M.data[I, :, :])
    # A_I = TMatrix(jnp.take(M.data, I, axis=0))

    # W = (
    #     A_J.facewise_operation(jnp.linalg.pinv)
    #     @ M
    #     @ A_I.facewise_operation(jnp.linalg.pinv)
    # )
    print(A_J.shape, W.shape, A_I.shape)
    M_recon = A_J @ W @ A_I
    M_recon = (M_recon.data * 255.0).astype(jnp.uint8)

    M_recon = jnp.clip(M_recon, 0, 255)

    return M_recon


print("view or image", tensor[0, :, :, 0])

tensor_HWCF = jnp.moveaxis(tensor, source=0, destination=-1)
tensor_HWCF = tensor_HWCF.astype(jnp.float32) / 255.0
# tensor_HWCF = tensor_HWCF.astype(jnp.float16)
print("tensor_HWCF dtype", tensor_HWCF.dtype)

M = TMatrix(tensor_HWCF)
print(M._shape)

key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
t_V = t_rsvd(key, M)
# t_V = t_svd(M)

V = t_V.T.create_t_matrix(jnp.float32)

print("Computing V")
jax.block_until_ready(V)
print("Transform shape", t_V.t_data.shape)
print("Data shape", V.shape)

# M_recon = eqx.filter_jit(func)(V)
M_recon = func(tensor_HWCF.astype(jnp.float32), V)

# img_pil = Image.fromarray(np.asarray(M_recon.data * 255).astype(np.uint8))
# img_pil.save(SAVE_PATH + f"t_arp_rank={rank}.png")


rec_tensor = jnp.moveaxis(M_recon, -1, 0)

print(rec_tensor[0, :, :, :])

# save yuv
write_tensor_to_yuv(rec_tensor, SAVE_PATH + "rec.yuv", width=WIDTH, height=HEIGHT)

yuv_to_mp4_ffmpeg(
    SAVE_PATH + "rec.yuv",
    SAVE_PATH + "video_rec.mp4",
    WIDTH,
    HEIGHT,
    FRAMERATE,
    pix_fmt="yuv420p",
)
