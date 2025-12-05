"""
SAM2 Video Segmentation Backend

Flask-based backend server for interactive video segmentation using SAM2.
Supports video upload, annotation, propagation, and review workflows.
"""

import os
import sys
import json
import pickle
import subprocess
import threading
import tempfile
import shutil
import queue
import base64
import io
import time
import re
from datetime import datetime, timedelta

import numpy as np
import torch
import cv2
import pandas as pd
import requests
import psutil
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib.colors as mcolors

# PyTorch optimization flags
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["TORCHDYNAMO_DISABLE_CUDAGRAPHS"] = "1"

# Add SAM2 to path
sys.path.append("./sam2")
SAM2_DIR = "./sam2"

# Import SAM2
try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    print("Error: Unable to import sam2 module. Make sure it's in your PYTHONPATH.")
    sys.exit(1)

# Constants
DEFAULT_FPS = 30
DEFAULT_FRAME_QUALITY = 2  # FFmpeg quality parameter (1-31, lower is better)
SESSION_TIMEOUT_SECONDS = 1800  # 30 minutes
CLEANUP_CHECK_INTERVAL = 300  # 5 minutes
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB
SAM2_CHECKPOINT = os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_large.pt")
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GOOGLE_SPREADSHEET_ID = "1h0D54fn1ZkdUgBaA63Nm4vvGWtpG7uA2JJo7nmaZ4aU"
SUPPORTED_VIDEO_FORMATS = (".mp4", ".mov", ".avi", ".MP4", ".MOV", ".AVI")
SUPPORTED_IMAGE_FORMATS = (".jpg", ".jpeg")

# Global variables
offline_queue = queue.Queue()
propagation_process = None
propagation_status = {
    "running": False,
    "current_object": None,
    "progress": 0
}
sessions = {}
device = None
predictor = None
session_last_active = {}  # Track when sessions were last accessed

# Flask application setup
app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)


def initialize_model():
    """
    Initialize the SAM2 model for video segmentation.
    
    Automatically selects the best available device (CUDA, MPS, or CPU)
    and configures device-specific optimizations.
    """
    global device, predictor
    
    # Select device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Configure device-specific settings
    if device.type == "cuda":
        torch.set_float32_matmul_precision('high')
        # Enable TF32 for Ampere GPUs (compute capability >= 8.0)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print("Note: MPS support is preliminary. Results may vary from CUDA.")
    
    # Load SAM2 model
    print(f"Loading SAM2 model from {SAM2_CHECKPOINT}...")
    predictor = build_sam2_video_predictor(
        MODEL_CONFIG,
        SAM2_CHECKPOINT,
        device=device,
        vos_optimized=False,
    )
    print("SAM2 model loaded successfully")


def print_gpu_memory_usage():
    """Print current GPU memory usage for debugging."""
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(current_device) / (1024 * 1024)
        cached = torch.cuda.memory_reserved(current_device) / (1024 * 1024)
        print(f"GPU Memory: {allocated:.2f} MB allocated, {cached:.2f} MB cached")
    else:
        print("CUDA not available")


# Mask processing utilities
def mask_to_rle_pytorch(tensor: torch.Tensor):
    """
    Encode binary masks to uncompressed RLE (Run-Length Encoding) format.
    
    Args:
        tensor: Binary mask tensor of shape (batch, height, width)
        
    Returns:
        List of dictionaries containing RLE encoding for each mask
    """
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)
    
    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()
    
    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat([
            torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
            cur_idxs + 1,
            torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
        ])
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out

def coco_encode_rle(uncompressed_rle):
    """
    Convert uncompressed RLE to COCO format.
    
    Args:
        uncompressed_rle: Dictionary with 'size' and 'counts' keys
        
    Returns:
        COCO-formatted RLE dictionary
    """
    from pycocotools import mask as mask_utils
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def mask_to_base64_image(mask, obj_id=None):
    """
    Convert binary mask to base64-encoded PNG for web display.
    
    Args:
        mask: Binary mask array
        obj_id: Object ID for consistent coloring (default: 0)
        
    Returns:
        Base64-encoded PNG image string
    """
    # Create a colormap
    cmap = plt.get_cmap("tab10")
    colors = list(cmap.colors)  # This creates a list copy we can modify
    colors[7] = tuple(i * 0.5 for i in colors[7])
    modified_cmap = mcolors.ListedColormap(colors)
    
    # Important: Use the original object ID for consistent coloring
    # obj_id here is the original_obj_idx, which starts from 0
    cmap_idx = 0 if obj_id is None else (obj_id % 10)
    color = np.array([*modified_cmap(cmap_idx)[:3], 0.6])
    
    # Create colored mask image
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    # Convert to 8-bit RGBA
    mask_image = (mask_image * 255).astype(np.uint8)
    
    # Create PIL image
    pil_img = Image.fromarray(mask_image, 'RGBA')
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    
    # Convert to base64
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

def save_video_segments(session_id, video_idx, video_segments, fps=DEFAULT_FPS, output_suffix=""):
    """
    Create video with segmentation masks overlaid on original frames.
    
    Args:
        session_id: Session identifier
        video_idx: Index of the video (1 or 2)
        video_segments: Dictionary mapping frame indices to object masks
        fps: Frames per second for output video
        output_suffix: Optional suffix for output filename
        
    Returns:
        Path to saved video file
    """
    session = sessions[session_id]
    frame_names = session[f'video{video_idx}_frame_names']
    video_dir = session[f'video{video_idx}_dir']
    
    # Create session-specific output directory
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id)
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"video{video_idx}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    # Get the first frame to determine video dimensions
    first_frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
    height, width = first_frame.shape[:2]
    
    # Use H.264 codec for better browser compatibility
    if sys.platform == 'darwin':  # macOS
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 encoding
    else:  # Windows/Linux
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Try more compatible MPEG-4 encoding
    
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Get color map for visualization
    cmap = plt.get_cmap("tab10")
    colors = list(cmap.colors)  # This creates a list copy we can modify
    colors[7] = tuple(i * 0.5 for i in colors[7])
    modified_cmap = mcolors.ListedColormap(colors)
    
    # Process each frame
    print(f"Creating video {video_idx} at {output_path}...")
    for frame_idx, frame_name in enumerate(frame_names):
        # Read the original frame
        frame = cv2.imread(os.path.join(video_dir, frame_name))
        # Convert from BGR to RGB for consistent coloring
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # If this frame has segmentation masks, overlay them
        if frame_idx in video_segments:
            overlay = np.zeros_like(frame_rgb, dtype=np.float32)
            
            # Process each object's mask in this frame
            for obj_id, mask in video_segments[frame_idx].items():
                # Get original object ID for coloring
                # The obj_id in video_segments is the SAM2 obj_id which starts from 1
                # For consistent coloring, subtract 1 to get 0-based index
                original_obj_id = int(obj_id) - 1
                
                # Use the same color mapping as mask_to_base64_image
                color_rgb = np.array(to_rgb(modified_cmap(original_obj_id % 10)))
                
                # Add colored mask to overlay
                for c in range(3):
                    overlay[:,:,c] += mask[0] * color_rgb[c] * 0.5  # 50% opacity
            
            # Combine original frame with overlay
            combined = np.clip(frame_rgb + overlay * 255, 0, 255).astype(np.uint8)
            
            # Convert back to BGR for OpenCV
            output_frame = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        else:
            # If no masks for this frame, use original frame
            output_frame = frame
        
        # Write frame to video
        video_writer.write(output_frame)
        
    # Release video writer
    video_writer.release()
    print(f"Video {video_idx} saved to {output_path}, size: {os.path.getsize(output_path)} bytes")
    return output_path

def process_video_to_frames(video_path, output_dir):
    """
    Extract individual frames from video using ffmpeg.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        
    Returns:
        True if successful, False otherwise
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        'ffmpeg', 
        '-i', video_path, 
        '-q:v', str(DEFAULT_FRAME_QUALITY), 
        '-r', str(DEFAULT_FPS), 
        '-start_number', '0',
        os.path.join(output_dir, '%05d.jpg')
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing video: {e}")
        return False



# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    """Serve the main annotation interface."""
    return render_template('index.html')


@app.route('/start_offline_propagation', methods=['POST'])
def start_offline_propagation():
    """Start offline segmentation propagation process."""
    global propagation_process, propagation_status
    
    if propagation_status["running"]:
        return jsonify({
            "success": False, 
            "error": "A propagation job is already running"
        })
    
    try:
        # Clean GPU memory first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleaned before starting offline propagation")
        
        # Start the propagation process without capturing output
        # Redirect stdout and stderr to /dev/null
        with open(os.devnull, 'w') as devnull:
            propagation_process = subprocess.Popen(
                ["python", "propagate_offline.py"],
                stdout=devnull,
                stderr=devnull
            )
        
        # Update status
        propagation_status = {
            "running": True,
            "current_object": "Running...",
            "progress": 0,
            "start_time": time.time()
        }
        
        # Start a thread to periodically check process status
        def monitor_process():
            global propagation_status
            
            while propagation_process.poll() is None:
                # Process is still running
                # Update elapsed time as a form of progress
                elapsed = time.time() - propagation_status["start_time"]
                propagation_status["progress"] = min(99, elapsed / 60 * 10)  # Rough estimate: 10% per minute, max 99%
                
                # Sleep before checking again
                time.sleep(5)
            
            # Process has ended
            propagation_status = {
                "running": False,
                "current_object": None,
                "progress": 0
            }
            print("Offline propagation process has ended")
        
        # Start monitoring thread
        threading.Thread(target=monitor_process, daemon=True).start()
        
        return jsonify({"success": True})
    
    except Exception as e:
        print(f"Error starting propagation: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/cancel_offline_propagation', methods=['POST'])
def cancel_offline_propagation():
    """Cancel the currently running offline propagation process."""
    global propagation_process, propagation_status
    
    if not propagation_status["running"]:
        return jsonify({
            "success": False, 
            "error": "No propagation job is currently running"
        })
    
    try:
        # Kill the process and all its children
        parent = psutil.Process(propagation_process.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
        
        # Update status
        propagation_status = {
            "running": False,
            "current_object": None,
            "progress": 0
        }
        
        return jsonify({"success": True})
    
    except Exception as e:
        print(f"Error cancelling propagation: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/check_offline_propagation_status')
def check_offline_propagation_status():
    """Get the current status of offline propagation process."""
    global propagation_status
    return jsonify(propagation_status)


@app.route('/upload_videos', methods=['POST'])
def upload_videos():
    """
    Handle video upload and initialize annotation session.
    
    Expects:
        - video1: First video file
        - video2: Second video file
        - scene_type: Description of the scene
        
    Returns:
        Session ID and video metadata
    """
    if 'video1' not in request.files or 'video2' not in request.files:
        return jsonify({'error': 'Both videos must be provided'}), 400
    
    video1_file = request.files['video1']
    video2_file = request.files['video2']
    scene_type = request.form.get('scene_type', '').strip()
    
    if video1_file.filename == '' or video2_file.filename == '':
        return jsonify({'error': 'Both videos must be selected'}), 400
    
    if not scene_type:
        return jsonify({'error': 'Scene type is required'}), 400
    
    # Extract the base filenames without extensions
    video1_base = os.path.splitext(os.path.basename(video1_file.filename))[0]
    video2_base = os.path.splitext(os.path.basename(video2_file.filename))[0]
    
    # Clean the filenames to make them suitable for directory names
    # Remove special characters and replace spaces with underscores
    video1_clean = re.sub(r'[^\w\s-]', '', video1_base).strip().replace(' ', '_')
    video2_clean = re.sub(r'[^\w\s-]', '', video2_base).strip().replace(' ', '_')
    
    # Generate session ID using the combination of the video names 
    # Add a short timestamp suffix to ensure uniqueness
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()) 
    session_id = f"{video1_clean}_{video2_clean}_{timestamp}"
    
    # Create session directory
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    # Process video 1
    video1_filename = secure_filename(video1_file.filename)
    video1_ext = os.path.splitext(video1_filename)[1]
    video1_path = os.path.join(session_dir, f"video1{video1_ext}")
    video1_file.save(video1_path)
    
    video1_frames_dir = os.path.join(session_dir, 'video1_frames')
    if not process_video_to_frames(video1_path, video1_frames_dir):
        return jsonify({'error': 'Failed to process video 1'}), 500
    
    # Process video 2
    video2_filename = secure_filename(video2_file.filename)
    video2_ext = os.path.splitext(video2_filename)[1]
    video2_path = os.path.join(session_dir, f"video2{video2_ext}")
    video2_file.save(video2_path)
    
    video2_frames_dir = os.path.join(session_dir, 'video2_frames')
    if not process_video_to_frames(video2_path, video2_frames_dir):
        return jsonify({'error': 'Failed to process video 2'}), 500
    
    # Get frame names for both videos
    video1_frame_names = [
        p for p in os.listdir(video1_frames_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    video1_frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    video2_frame_names = [
        p for p in os.listdir(video2_frames_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    video2_frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    # Initialize SAM2 predictor states
    video1_inference_state = predictor.init_state(video_path=video1_frames_dir)
    video2_inference_state = predictor.init_state(video_path=video2_frames_dir)
    
    # Store session data
    sessions[session_id] = {
        'scene_type': scene_type,
        'video1_path': video1_path,
        'video1_dir': video1_frames_dir,
        'video1_frame_names': video1_frame_names,
        'video1_inference_state': video1_inference_state,
        'video2_path': video2_path,
        'video2_dir': video2_frames_dir,
        'video2_frame_names': video2_frame_names,
        'video2_inference_state': video2_inference_state,
        'object_configs': [],
        'video1_objects': {},  # Will contain {obj_idx: {frame_idx, status, etc.}}
        'video2_objects': {},
        'video1_completed_objects': set(),
        'video2_completed_objects': set(),
        'video1_segments': {},
        'video2_segments': {}
    }
    
    update_session_activity(session_id)
    
    return jsonify({
        'session_id': session_id,
        'video1_frame_count': len(video1_frame_names),
        'video1_fps': 30,  # Assuming 30fps
        'video2_frame_count': len(video2_frame_names),
        'video2_fps': 30   # Assuming 30fps
    })

@app.route('/set_object_config', methods=['POST'])
def set_object_config():
    """Store object configuration for a session."""
    data = request.json
    session_id = data.get('session_id')
    object_configs = data.get('object_configs', [])
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    session = sessions[session_id]
    session['object_configs'] = object_configs
    
    return jsonify({'success': True})


@app.route('/get_frame', methods=['GET'])
def get_frame():
    """Retrieve a specific frame from a video."""
    session_id = request.args.get('session_id')
    video_idx = int(request.args.get('video_idx', 1))
    frame_idx = int(request.args.get('frame_idx', 0))
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    session = sessions[session_id]
    
    # Get video-specific data
    frame_names = session[f'video{video_idx}_frame_names']
    video_dir = session[f'video{video_idx}_dir']
    
    if frame_idx >= len(frame_names):
        return jsonify({'error': 'Invalid frame index'}), 400
    
    # Get frame path
    frame_path = os.path.join(video_dir, frame_names[frame_idx])
    
    # Convert to base64 for sending to client
    with open(frame_path, 'rb') as img_file:
        img_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    update_session_activity(session_id)
    
    return jsonify({
        'frame_data': f"data:image/jpeg;base64,{img_data}",
        'frame_idx': frame_idx
    })

@app.route('/set_object_frame', methods=['POST'])
def set_object_frame():
    """Set the frame index for a specific object in a video."""
    data = request.json
    session_id = data.get('session_id')
    video_idx = int(data.get('video_idx', 1))
    obj_idx = int(data.get('obj_idx', 0))
    original_obj_idx = int(data.get('original_obj_idx', 0))
    frame_idx = int(data.get('frame_idx', 0))
    status = data.get('status', 'moved')
    view_idx = data.get('view_idx', 0)  # Get the view index
    total_views = data.get('total_views', 1)  # Get total views
    is_reannotation = data.get('is_reannotation', False)
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    session = sessions[session_id]
    
    # Calculate SAM2 object ID - for re-annotation, use the original if available
    # For multiple views, use a derived ID from the original object ID and view index
    base_sam_obj_id = original_obj_idx + 1  # Base ID
    sam_obj_id = base_sam_obj_id + (view_idx * 100)  # Add offset for different views
    
    # Update object frame information
    video_objects = session[f'video{video_idx}_objects']
    
    # Generate a unique key for this object+view combination
    obj_view_key = f"{obj_idx}_{view_idx}"
    
    # Store object info
    video_objects[obj_view_key] = {
        'frame_idx': frame_idx,
        'status': status,
        'original_obj_idx': original_obj_idx,
        'sam_obj_id': sam_obj_id,  # Store the SAM object ID
        'view_idx': view_idx,
        'total_views': total_views
    }
    
    return jsonify({'success': True})

@app.route('/process_clicks', methods=['POST'])
def process_clicks():
    """Process user clicks to generate segmentation masks using SAM2."""
    data = request.json
    session_id = data.get('session_id')
    video_idx = int(data.get('video_idx', 1))
    frame_idx = int(data.get('frame_idx', 0))
    obj_idx = int(data.get('obj_idx', 0))
    original_obj_idx = int(data.get('original_obj_idx', 0))
    points = np.array(data.get('points'), dtype=np.float32)
    labels = np.array(data.get('labels'), dtype=np.int32)
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    session = sessions[session_id]
    
    if len(points) == 0:
        return jsonify({'error': 'No points provided'}), 400
    
    # Get video-specific inference state
    inference_state = session[f'video{video_idx}_inference_state']
    
    # Get the correct SAM object ID
    video_objects = session[f'video{video_idx}_objects']
    sam_obj_id = original_obj_idx + 1  # Default
    
    if obj_idx in video_objects and 'sam_obj_id' in video_objects[obj_idx]:
        sam_obj_id = video_objects[obj_idx]['sam_obj_id']
    
    # Process points with SAM2
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=sam_obj_id,
        points=points,
        labels=labels,
        clear_old_points=True
    )
    
    # Find the correct mask index for our object
    mask_idx = -1
    for i, obj_id in enumerate(out_obj_ids):
        if obj_id == sam_obj_id:
            mask_idx = i
            break
    
    if mask_idx == -1:
        return jsonify({'error': 'Failed to find generated mask'}), 500
    
    # Get binary mask for our specific object
    mask = (out_mask_logits[mask_idx] > 0.0).cpu().numpy()
    
    # Convert mask to base64 image for display
    mask_img = mask_to_base64_image(mask, original_obj_idx)
    
    return jsonify({
        'success': True,
        'mask_img': mask_img
    })

@app.route('/complete_object', methods=['POST'])
def complete_object():
    """Mark an object as completed for a specific video."""
    data = request.json
    session_id = data.get('session_id')
    video_idx = int(data.get('video_idx', 1))
    obj_idx = int(data.get('obj_idx', 0))
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    session = sessions[session_id]
    session[f'video{video_idx}_completed_objects'].add(obj_idx)
    
    return jsonify({'success': True})


@app.route('/finish_annotation', methods=['POST'])
def finish_annotation():
    """
    Finish the annotation process and save inference data.
    
    For new annotations: Saves inference data for offline propagation
    For reannotations: Updates existing inference data with new annotations
    """
    data = request.json
    session_id = data.get('session_id')
    is_reannotation = data.get('is_reannotation', False)
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    session = sessions[session_id]
    
    # Create session-specific output directory
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id)
    os.makedirs(output_dir, exist_ok=True)

    objects_data = []
    
    for i, obj_config in enumerate(session['object_configs']):
        obj_name = obj_config['name']
        obj_number = obj_config['number']
        
        # Find this object in video 1
        video1_obj_idx = -1
        video1_frame_idx = -1
        video1_status = None
        
        for idx, obj_data in session['video1_objects'].items():
            if obj_data['original_obj_idx'] == i:
                video1_obj_idx = idx
                video1_frame_idx = obj_data['frame_idx']
                video1_status = obj_data['status']
                break
        
        # Find this object in video 2
        video2_obj_idx = -1
        video2_frame_idx = -1
        video2_status = None
        
        for idx, obj_data in session['video2_objects'].items():
            if obj_data['original_obj_idx'] == i:
                video2_obj_idx = idx
                video2_frame_idx = obj_data['frame_idx']
                video2_status = obj_data['status']
                break
        
        deformability = obj_config.get('deformability', None)

        # Add to objects data
        objects_data.append({
            'label': f"{obj_name}_{obj_number}",
            'in_video1': obj_config['inVideo1'],
            'in_video2': obj_config['inVideo2'],
            'video1_frame_idx': video1_frame_idx,
            'video1_status': video1_status,
            'video2_frame_idx': video2_frame_idx,
            'video2_status': video2_status,
            'original_obj_idx': i,
            'deformability': deformability,
        })

    # Handle reannotation scenario
    if is_reannotation:
        # Load existing inference data
        inference_data_path = os.path.join(output_dir, "inference_data.pkl")
        
        if not os.path.exists(inference_data_path):
            return jsonify({'error': 'Original inference data not found for reannotation'}), 404
        
        try:
            with open(inference_data_path, 'rb') as f:
                existing_inference_data = pickle.load(f)
            
            # Update the point inputs for reannotated objects
            # Video 1
            existing_objects_data = existing_inference_data['objects']
            current_objects_data = objects_data
            existing_video1_idx_to_id = existing_inference_data['video1']['obj_idx_to_id']
            existing_video1_point_inputs_per_obj = existing_inference_data['video1']['point_inputs_per_obj']
            current_video1_point_inputs_per_obj = session['video1_inference_state']['point_inputs_per_obj']
            current_video1_idx_to_id = session['video1_inference_state']['obj_idx_to_id']
            if 'video1' in existing_inference_data and 'point_inputs_per_obj' in existing_inference_data['video1']:
                for obj_idx, point_inputs in current_video1_point_inputs_per_obj.items():
                    current_obj_id_int = current_video1_idx_to_id[obj_idx] - 1
                    current_obj_name = current_objects_data[current_obj_id_int]['label']

                    # For reannotated objects, concatenate the point inputs with existing ones
                    for existing_obj_idx, existing_point_inputs in existing_video1_point_inputs_per_obj.items():
                        # Get existing point inputs
                        existing_obj_id_int = existing_video1_idx_to_id[existing_obj_idx] - 1
                        existing_obj_name = existing_objects_data[existing_obj_id_int]['label']

                        if current_obj_name == existing_obj_name:
                            # Concatenate or merge frame by frame
                            for frame_idx, inputs in point_inputs.items():
                                # Convert to int if it's a string key
                                frame_idx_int = int(frame_idx) if isinstance(frame_idx, str) else frame_idx
                                
                                if frame_idx_int in existing_point_inputs:
                                    # Merge inputs for the same frame
                                    # This is a simplistic approach - in reality you may want 
                                    # more sophisticated merging logic
                                    print(f"Merging point inputs for object {obj_idx} frame {frame_idx}")
                                    existing_point_inputs[frame_idx_int]['point_coords'] = torch.cat([existing_point_inputs[frame_idx_int]['point_coords'], inputs['point_coords']], dim=-2)
                                    existing_point_inputs[frame_idx_int]['point_labels'] = torch.cat([existing_point_inputs[frame_idx_int]['point_labels'], inputs['point_labels']], dim=-1)
                                else:
                                    # Add or update frame inputs
                                    existing_point_inputs[frame_idx_int] = inputs
                        
                            # Update the entry in the existing data
                            existing_inference_data['video1']['point_inputs_per_obj'][existing_obj_idx] = existing_point_inputs
                            break # Only one object per object name
            
            # Video 2
            if 'video2' in existing_inference_data and 'point_inputs_per_obj' in existing_inference_data['video2']:
                existing_video2_idx_to_id = existing_inference_data['video2']['obj_idx_to_id']
                existing_video2_point_inputs_per_obj = existing_inference_data['video2']['point_inputs_per_obj']
                current_video2_point_inputs_per_obj = session['video2_inference_state']['point_inputs_per_obj']
                current_video2_idx_to_id = session['video2_inference_state']['obj_idx_to_id']
                for obj_idx, point_inputs in current_video2_point_inputs_per_obj.items():
                    current_obj_id_int = current_video2_idx_to_id[obj_idx] - 1
                    current_obj_name = current_objects_data[current_obj_id_int]['label']

                    # For reannotated objects, concatenate the point inputs with existing ones
                    for existing_obj_idx, existing_point_inputs in existing_video2_point_inputs_per_obj.items():
                        # Get existing point inputs
                        existing_obj_id_int = existing_video2_idx_to_id[existing_obj_idx] - 1
                        existing_obj_name = existing_objects_data[existing_obj_id_int]['label']
                        
                        if current_obj_name == existing_obj_name:
                            # Concatenate or merge frame by frame
                            for frame_idx, inputs in point_inputs.items():
                                # Convert to int if it's a string key
                                frame_idx_int = int(frame_idx) if isinstance(frame_idx, str) else frame_idx
                                
                                if frame_idx_int in existing_point_inputs:
                                    # Merge inputs for the same frame
                                    # This is a simplistic approach - in reality you may want 
                                    # more sophisticated merging logic
                                    print(f"Merging point inputs for object {obj_idx} frame {frame_idx}")
                                    existing_point_inputs[frame_idx_int]['point_coords'] = torch.cat([existing_point_inputs[frame_idx_int]['point_coords'], inputs['point_coords']], dim=-2)
                                    existing_point_inputs[frame_idx_int]['point_labels'] = torch.cat([existing_point_inputs[frame_idx_int]['point_labels'], inputs['point_labels']], dim=-1)
                                else:
                                    # Add or update frame inputs
                                    existing_point_inputs[frame_idx_int] = inputs

                            # Update the entry in the existing data
                            existing_inference_data['video2']['point_inputs_per_obj'][existing_obj_idx] = existing_point_inputs

                            break # Only one object per object name
            # Save updated inference data
            with open(inference_data_path, 'wb') as f:
                pickle.dump(existing_inference_data, f)
            
            print(f"Updated inference data for reannotation: {session_id}")

            # Remove the status from the review_meta.json
            review_meta_path = os.path.join(output_dir, "review_meta.json")
            if os.path.exists(review_meta_path):
                review_meta = json.load(open(review_meta_path, "r"))
                if 'status' in review_meta:
                    del review_meta['status']
                with open(review_meta_path, "w") as f:
                    json.dump(review_meta, f)            
            
            return jsonify({
                'success': True,
                'message': 'Reannotation data saved successfully',
                'inference_data_path': inference_data_path
            })
        
        except Exception as e:
            print(f"Error updating inference data for reannotation: {e}")
            return jsonify({'error': f'Error updating inference data: {str(e)}'}), 500
    else:
        # Extract relevant inference state for offline processing
        inference_data = {
            'video1': {
                'obj_idx_to_id': session['video1_inference_state'].get('obj_idx_to_id', {}),
                'point_inputs_per_obj': session['video1_inference_state'].get('point_inputs_per_obj', {}),
                'frames_dir': session['video1_dir'],
                'frame_names': session['video1_frame_names']
            },
            'video2': {
                'obj_idx_to_id': session['video2_inference_state'].get('obj_idx_to_id', {}),
                'point_inputs_per_obj': session['video2_inference_state'].get('point_inputs_per_obj', {}),
                'frames_dir': session['video2_dir'],
                'frame_names': session['video2_frame_names']
            },
            'objects': objects_data,
            'scene_type': session['scene_type'],
            'comments': session.get('comments', {})
        }
        
        # Save inference data for offline processing
        inference_data_path = os.path.join(output_dir, "inference_data.pkl")
        with open(inference_data_path, 'wb') as f:
            pickle.dump(inference_data, f)
        
        # Generate timestamp for the log
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Update the JSON log file with session information
        log_data = {
            'session_id': session_id,
            'scene_type': session['scene_type'],
            'timestamp': timestamp,
            'is_propagated': False,
            'is_reviewed': False,
            'video1_path': session['video1_path'],
            'video2_path': session['video2_path'],
            'object_count': len(session['object_configs'])
        }
        
        # Load existing log or create new one
        log_file_path = os.path.join(app.config['RESULTS_FOLDER'], "annotation_log.json")
        if os.path.exists(log_file_path):
            try:
                with open(log_file_path, 'r') as f:
                    log_entries = json.load(f)
            except json.JSONDecodeError:
                log_entries = []
        else:
            log_entries = []
        
        # Add new entry or update existing
        entry_updated = False
        for i, entry in enumerate(log_entries):
            if entry.get('session_id') == session_id:
                log_entries[i] = log_data
                entry_updated = True
                break
        
        if not entry_updated:
            log_entries.append(log_data)
        
        # Save updated log
        with open(log_file_path, 'w') as f:
            json.dump(log_entries, f, indent=4)
        
        # Return success without waiting for propagation
        print(f"Annotation data saved for offline processing: {session_id}")
        
        return jsonify({
            'success': True,
            'inference_data_path': inference_data_path,
            'is_propagated': False,
            'message': 'Annotation data stored for offline processing'
        })



@app.route('/results/<session_id>/<filename>')
def serve_result_file(session_id, filename):
    print(f"Serving file: {session_id}/{filename}")  # Debug log
    result_dir = os.path.join("results", session_id)
    
    # If file doesn't exist in standard results, check offline results
    if not os.path.exists(os.path.join(result_dir, filename)):
        print(f"File not found in standard results, checking offline results")
        result_dir = os.path.join("results_offline_processing", session_id)
    
    # Check if file exists before trying to serve it
    file_path = os.path.join(result_dir, filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return f"File not found: {filename}", 404
    
    # Log file size for debugging
    print(f"File size: {os.path.getsize(file_path)} bytes")
    
    try:
        return send_from_directory(result_dir, filename)
    except Exception as e:
        print(f"Error serving file {filename}: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/debug_files/<session_id>')
def debug_files(session_id):
    """Debug endpoint to check files in session directory"""
    session_dir = os.path.join("results", session_id)
    
    if not os.path.exists(session_dir):
        return jsonify({
            "exists": False,
            "message": f"Directory {session_dir} does not exist"
        })
    
    files = os.listdir(session_dir)
    file_info = {}
    
    for f in files:
        full_path = os.path.join(session_dir, f)
        file_info[f] = {
            "size": os.path.getsize(full_path) if os.path.isfile(full_path) else "N/A",
            "is_dir": os.path.isdir(full_path),
            "is_file": os.path.isfile(full_path)
        }
    
    return jsonify({
        "exists": True,
        "path": session_dir,
        "files": file_info
    })

@app.route('/results/<path:filename>')
def results(filename):
    """Serve files from results folder"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/cleanup_session', methods=['POST'])
def cleanup_session():
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session ID'}), 400
        
    session = sessions[session_id]
    
    try:
        import shutil
        # Get frame directories for both videos
        video1_frames_dir = session.get('video1_dir')
        video2_frames_dir = session.get('video2_dir')
        
        # Print debug information
        print(f"Cleaning up session frames: {session_id}")
        
        # Remove video1 frames directory if it exists
        if video1_frames_dir and os.path.exists(video1_frames_dir):
            print(f"Removing video1 frames directory: {video1_frames_dir}")
            shutil.rmtree(video1_frames_dir)
        
        # Remove video2 frames directory if it exists
        if video2_frames_dir and os.path.exists(video2_frames_dir):
            print(f"Removing video2 frames directory: {video2_frames_dir}")
            shutil.rmtree(video2_frames_dir)
        
        # Clear GPU memory for this session
        if 'video1_inference_state' in session:
            # Delete tensors from inference state
            clear_inference_state(session['video1_inference_state'])
            session['video1_inference_state'] = None
            
        if 'video2_inference_state' in session:
            # Delete tensors from inference state
            clear_inference_state(session['video2_inference_state'])
            session['video2_inference_state'] = None
            
        # Clean up temporary files if we know their location
        try:
            if 'video1_dir' in session and session['video1_dir'] and os.path.exists(session['video1_dir']):
                shutil.rmtree(session['video1_dir'])
                
            if 'video2_dir' in session and session['video2_dir'] and os.path.exists(session['video2_dir']):
                shutil.rmtree(session['video2_dir'])
        except Exception as e:
            print(f"Error cleaning files for session {session_id}: {e}")
        
        # Remove from sessions dict
        del sessions[session_id]
        
        # Remove from activity tracker
        if session_id in session_last_active:
            del session_last_active[session_id]
        
        # Clear GPU cache after cleaning sessions
        if torch.cuda.is_available():
            print(f"Freeing GPU memory after cleaning sessions")
            torch.cuda.empty_cache()
            print_gpu_memory_usage()
            
        return jsonify({'success': True, 'message': 'Session resources cleaned up successfully'})
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        return jsonify({'error': f'Error during cleanup: {str(e)}'}), 500

def clear_inference_state(inference_state):
    """
    Recursively free memory used by tensors in inference state.
    
    Args:
        inference_state: Dictionary or nested structure containing tensors
    """
    if inference_state is None:
        return
        
    try:
        # For dictionary-based inference states (most common format)
        if isinstance(inference_state, dict):
            for key, value in list(inference_state.items()):
                if hasattr(value, 'detach'):
                    inference_state[key] = None
                elif isinstance(value, dict):
                    clear_inference_state(value)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if hasattr(item, 'detach'):
                            value[i] = None
                        elif isinstance(item, dict):
                            clear_inference_state(item)
    except Exception as e:
        print(f"Warning: Error clearing inference state: {str(e)}")

@app.route('/get_objects_data', methods=['GET'])
def get_objects_data():
    """Get object configuration data for a session."""
    session_id = request.args.get('session_id')
    
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    session = sessions[session_id]
    objects_data = []
    
    for i, obj_config in enumerate(session['object_configs']):
        obj_name = obj_config['name']
        obj_number = obj_config['number']
        
        # Find this object in video 1
        video1_obj_idx = -1
        video1_frame_idx = -1
        video1_status = None
        
        for idx, obj_data in session['video1_objects'].items():
            if obj_data['original_obj_idx'] == i:
                video1_obj_idx = idx
                video1_frame_idx = obj_data['frame_idx']
                video1_status = obj_data['status']
                break
        
        # Find this object in video 2
        video2_obj_idx = -1
        video2_frame_idx = -1
        video2_status = None
        
        for idx, obj_data in session['video2_objects'].items():
            if obj_data['original_obj_idx'] == i:
                video2_obj_idx = idx
                video2_frame_idx = obj_data['frame_idx']
                video2_status = obj_data['status']
                break
        
        # Add to objects data
        objects_data.append({
            'label': obj_name,
            'number': obj_number,
            'in_video1': obj_config.get('inVideo1', False),
            'in_video2': obj_config.get('inVideo2', False),
            'video1_frame_idx': video1_frame_idx,
            'video1_status': video1_status,
            'video2_frame_idx': video2_frame_idx,
            'video2_status': video2_status,
            'original_obj_idx': i,
            'original_index': i,  # Added for compatibility with some client code
            'views_in_video1': obj_config.get('viewsInVideo1', 1),
            'views_in_video2': obj_config.get('viewsInVideo2', 1),
            'current_view_video1': obj_config.get('currentViewVideo1', 0),
            'current_view_video2': obj_config.get('currentViewVideo2', 0),
            'completed_views_video1': obj_config.get('completedViewsVideo1', set()),
            'completed_views_video2': obj_config.get('completedViewsVideo2', set())
        })
    
    # Include video frame counts and fps for reannotation mode
    response_data = {
        'success': True,
        'objects': objects_data,
        'video1_frame_count': session.get('video1_frame_count', len(session.get('video1_frame_names', []))),
        'video1_fps': session.get('video1_fps', 30),
        'video2_frame_count': session.get('video2_frame_count', len(session.get('video2_frame_names', []))),
        'video2_fps': session.get('video2_fps', 30),
        'reannotation_config': session.get('reannotation_config', []),
        'scene_type': session.get('scene_type', 'reannotation')
    }
    
    return jsonify(response_data)

@app.route('/save_comments', methods=['POST'])
def save_comments():
    """Save reviewer comments for a session."""
    data = request.json
    session_id = data.get('session_id')
    comments = data.get('comments', {})
    
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    session = sessions[session_id]
    session['comments'] = comments
    return jsonify({'success': True})


# ============================================================================
# Review Interface Routes
# ============================================================================

@app.route('/review')
def review_dashboard():
    """Serve the review dashboard page."""
    return render_template('review.html')


@app.route('/review/session/<session_id>')
def review_session(session_id):
    """Serve the review session page for a specific session."""
    return render_template('review_session.html')


@app.route('/review/upload')
def upload_review():
    """Serve the upload for review page."""
    return render_template('upload_review.html')


@app.route('/api/sessions')
def get_sessions():
    """
    Get a list of all available sessions for review.
    
    Scans both standard results and offline processing folders
    for completed segmentation sessions.
    """
    valid_sessions = []
    
    # Get absolute path to the application directory
    app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define folders to scan with absolute paths
    folders_to_scan = [
        app.config['RESULTS_FOLDER'],
        os.path.join(app_dir, 'results_offline_processing')  # Use the actual folder name
    ]
    
    for folder_path in folders_to_scan:
        print(f"Scanning folder: {folder_path}")  # For debugging
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist")
            continue
        
        # List all directories in the folder
        try:
            session_dirs = [d for d in os.listdir(folder_path) 
                           if os.path.isdir(os.path.join(folder_path, d))]
            
            # Process each directory that contains required files
            for session_id in session_dirs:
                session_path = os.path.join(folder_path, session_id)
                video1_path = os.path.join(session_path, 'video1.mp4')
                video2_path = os.path.join(session_path, 'video2.mp4')
                pkl_path = os.path.join(session_path, 'segments.pkl')
                meta_path = os.path.join(session_path, 'review_meta.json')
                
                # Only include sessions with videos and data
                if (os.path.exists(video1_path) and 
                    os.path.exists(video2_path) and 
                    os.path.exists(pkl_path)):
                    
                    # Get modify timestamp
                    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', 
                                 time.gmtime(os.path.getmtime(pkl_path)))
                    
                    # Check if this session has been reviewed
                    status = 'reviewed' if os.path.exists(meta_path) else 'needs_review'
                    
                    # Check if any objects need reannotation
                    needs_reannotation = False
                    finish_reannotation = False
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, 'r') as f:
                                review_data = json.load(f)
                                if 'objectReviews' in review_data:
                                    for obj_review in review_data['objectReviews']:
                                        if obj_review.get('needsReannotation', False):
                                            needs_reannotation = True
                                            break
                                if 'status' in review_data and review_data['status'] == 'finish_reannotation':
                                    finish_reannotation = True
                        except Exception as e:
                            print(f"Error checking review meta data: {e}")
                    
                    valid_sessions.append({
                        'id': session_id,
                        'status': status,
                        'timestamp': timestamp,
                        'date': timestamp.split('_')[0],
                        'time': timestamp.split('_')[1].replace('-', ':'),
                        'source': 'standard' if folder_path == app.config['RESULTS_FOLDER'] else 'offline',
                        'video1_url': f"/{'results' if folder_path == app.config['RESULTS_FOLDER'] else 'results_offline'}/{session_id}/video1.mp4",
                        'video2_url': f"/{'results' if folder_path == app.config['RESULTS_FOLDER'] else 'results_offline'}/{session_id}/video2.mp4",
                        'needsReannotation': needs_reannotation,
                        'finish_reannotation': finish_reannotation
                    })
        except Exception as e:
            print(f"Error scanning directory {folder_path}: {e}")
    
    # Sort sessions by timestamp (newest first)
    valid_sessions.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return jsonify(valid_sessions)


@app.route('/debug/check_session/<session_id>')
def debug_check_session(session_id):
    """Debug endpoint to check if a session exists."""
    folders_to_check = [
        app.config['RESULTS_FOLDER'],
        'results_offline_processing'
    ]
    
    results = {}
    for folder in folders_to_check:
        path = os.path.join(folder, session_id)
        exists = os.path.exists(path)
        if exists:
            files = os.listdir(path)
        else:
            files = []
        
        results[folder] = {
            'exists': exists,
            'path': path,
            'files': files
        }
    
    return jsonify(results)

@app.route('/api/sessions/<session_id>')
def get_session_data(session_id):
    """
    Get detailed data for a specific session.
    
    Returns session metadata, video URLs, object data, and review information.
    """
    print(f"Fetching session data for: {session_id}")  # Debug log
    
    # Check both folders for the session
    folders_to_check = [
        app.config['RESULTS_FOLDER'],
        'results_offline_processing'
    ]
    
    session_path = None
    source_folder = None
    
    for folder in folders_to_check:
        potential_path = os.path.join(folder, session_id)
        print(f"Checking path: {potential_path}, exists: {os.path.exists(potential_path)}")
        if os.path.exists(potential_path):
            session_path = potential_path
            source_folder = 'standard' if folder == app.config['RESULTS_FOLDER'] else 'offline'
            break
    
    if not session_path:
        print(f"Session not found: {session_id}")
        return jsonify({'error': 'Session not found'}), 404
    
    # Check for various video file possibilities
    video_files = os.listdir(session_path)
    print(f"Files in session directory: {video_files}")
    
    # Try different possible video filenames
    video1_path = None
    video2_path = None
    
    # Check for video1 options in order of preference
    if 'video1_compressed.mp4' in video_files:
        video1_path = 'video1_compressed.mp4'
    elif 'video1.mp4' in video_files:
        ffmpeg_path = "/usr/bin/ffmpeg"
        subprocess.run([
            ffmpeg_path, '-y', '-i', os.path.join(session_path, 'video1.mp4'),
            '-vcodec', 'libx264', '-crf', '28',
            '-preset', 'fast', os.path.join(session_path, 'video1_compressed.mp4')
        ], check=True, stderr=subprocess.PIPE)
        video1_path = 'video1_compressed.mp4'
    elif 'fixed1.mp4' in video_files:
        video1_path = 'fixed1.mp4'
    else:
        # Try to find any video file that might be video1
        for file in video_files:
            if file.lower().startswith('video1') and file.lower().endswith(('.mp4', '.avi', '.mov')):
                video1_path = file
                break
    
    # Check for video2 options in order of preference
    if 'video2_compressed.mp4' in video_files:
        video2_path = 'video2_compressed.mp4'
    elif 'video2.mp4' in video_files:
        subprocess.run([
            ffmpeg_path, '-y', '-i', os.path.join(session_path, 'video2.mp4'),
            '-vcodec', 'libx264', '-crf', '28',
            '-preset', 'fast', os.path.join(session_path, 'video2_compressed.mp4')
        ], check=True, stderr=subprocess.PIPE)
        video2_path = 'video2_compressed.mp4'
    elif 'fixed2.mp4' in video_files:
        video2_path = 'fixed2.mp4'
    else:
        # Try to find any video file that might be video2
        for file in video_files:
            if file.lower().startswith('video2') and file.lower().endswith(('.mp4', '.avi', '.mov')):
                video2_path = file
                break
    
    print(f"Video paths found: video1={video1_path}, video2={video2_path}")
    
    # Check for segments.pkl
    pkl_path = os.path.join(session_path, 'segments.pkl')
    if not os.path.exists(pkl_path):
        print(f"segments.pkl not found in {session_path}")
        
        # Try to find any .pkl file
        pkl_files = [f for f in video_files if f.endswith('.pkl')]
        if pkl_files:
            pkl_path = os.path.join(session_path, pkl_files[0])
            print(f"Found alternative .pkl file: {pkl_files[0]}")
        else:
            return jsonify({'error': 'Session data incomplete: segments.pkl not found'}), 404
    
    # Check if we found both videos
    if not video1_path or not video2_path:
        print(f"One or both videos not found in {session_path}")
        return jsonify({'error': 'Session data incomplete: video files not found'}), 404
    
    # Get creation timestamp
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', 
                 time.gmtime(os.path.getctime(pkl_path)))
    
    # Check if this session has been reviewed
    meta_path = os.path.join(session_path, 'review_meta.json')
    status = 'reviewed' if os.path.exists(meta_path) else 'needs_review'
    
    # Construct video URLs based on source folder
    if source_folder == 'offline':
        video1_url = f'/results_offline/{session_id}/{video1_path}'
        video2_url = f'/results_offline/{session_id}/{video2_path}'
    else:
        video1_url = f'/results/{session_id}/{video1_path}'
        video2_url = f'/results/{session_id}/{video2_path}'
    
    # Load object data from pickle file
    try:
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        
        # Handle different structures of segments.pkl
        if isinstance(pkl_data, dict):
            if 'objects' in pkl_data:
                objects_data = pkl_data['objects']
            elif 'object_configs' in pkl_data:
                objects_data = pkl_data.get('object_configs', [])
            else:
                # Try to reconstruct objects from what's available
                objects_data = []
                
                # If video objects are available, try to extract object data
                if 'video1_objects' in pkl_data or 'video2_objects' in pkl_data:
                    # This is a more complex case - we'd need to extract from the RLE format
                    print("Found video objects, but no object list. Using empty object list.")
        else:
            objects_data = []
        
        scene_type = pkl_data.get('scene_type', 'unknown') if isinstance(pkl_data, dict) else 'unknown'
    except Exception as e:
        print(f"Error loading session data: {e}")
        objects_data = []
        scene_type = 'unknown'
    
    # Load review metadata if it exists
    reviewer_data = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                reviewer_data = json.load(f)
        except Exception as e:
            print(f"Error loading review metadata: {e}")
    
    # Combine all data
    session_data = {
        'id': session_id,
        'status': status,
        'source': source_folder,
        'timestamp': timestamp,
        'date': timestamp.split('_')[0],
        'time': timestamp.split('_')[1].replace('-', ':'),
        'video1_url': video1_url,
        'video2_url': video2_url,
        'pkl_path': f'/results/{session_id}/segments.pkl',
        'scene_type': scene_type,
        'objects': objects_data,
        'review': reviewer_data
    }
    
    return jsonify(session_data)

@app.route('/debug_session/<session_id>')
def debug_session(session_id):
    """Debug endpoint to check session data"""
    results = {}
    
    # Check both folders
    folders_to_check = [
        app.config['RESULTS_FOLDER'],
        'results_offline_processing'
    ]
    
    for folder in folders_to_check:
        folder_path = os.path.join(folder, session_id)
        
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            
            # Detailed file info
            file_info = {}
            for file in files:
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    file_info[file] = {
                        'size': os.path.getsize(file_path),
                        'modified': time.ctime(os.path.getmtime(file_path)),
                        'created': time.ctime(os.path.getctime(file_path))
                    }
            
            # Check for segments.pkl
            pkl_path = os.path.join(folder_path, 'segments.pkl')
            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, 'rb') as f:
                        pkl_data = pickle.load(f)
                    pkl_keys = list(pkl_data.keys()) if isinstance(pkl_data, dict) else 'Not a dictionary'
                except Exception as e:
                    pkl_keys = f"Error loading: {str(e)}"
            else:
                pkl_keys = 'File not found'
            
            # Test URL access
            video_files = [f for f in files if f.endswith('.mp4')]
            video_urls = []
            for video in video_files:
                if folder == app.config['RESULTS_FOLDER']:
                    video_urls.append(f'/results/{session_id}/{video}')
                else:
                    video_urls.append(f'/results_offline/{session_id}/{video}')
            
            results[folder] = {
                'exists': True,
                'path': folder_path,
                'files': files,
                'file_details': file_info,
                'pkl_keys': pkl_keys,
                'video_urls': video_urls
            }
        else:
            results[folder] = {
                'exists': False,
                'path': folder_path
            }
    
    # Add server info
    results['server_info'] = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results_folder': app.config['RESULTS_FOLDER'],
        'upload_folder': app.config['UPLOAD_FOLDER']
    }
    
    return jsonify(results)


@app.route('/api/reviews', methods=['POST'])
def submit_review():
    """
    Submit a review for a session.
    
    Saves review metadata including object reviews and reannotation flags.
    """
    data = request.json
    
    # Validate required fields
    required_fields = ['sessionId', 'reviewerName', 'timestamp', 'objectReviews']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    session_id = data['sessionId']
    source_hint = data.get('source', 'standard')
    
    # Try the hinted folder first, then check the other
    base_folders = []
    if source_hint == 'offline':
        base_folders = ['results_offline_processing', app.config['RESULTS_FOLDER']]
    else:
        base_folders = [app.config['RESULTS_FOLDER'], 'results_offline_processing']
    
    # Find the actual folder where the session exists
    folder_path = None
    for folder in base_folders:
        potential_path = os.path.join(folder, session_id)
        if os.path.exists(potential_path):
            folder_path = potential_path
            break
    
    if not folder_path:
        return jsonify({'error': f'Session {session_id} not found in any results folder'}), 404
    
    # Save review metadata
    meta_path = os.path.join(folder_path, 'review_meta.json')
    try:
        with open(meta_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Review metadata saved to {meta_path}")
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error saving review: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/upload-review', methods=['POST'])
def upload_for_review():
    """
    Upload segmented videos and data for review.
    
    Accepts video files and pickled segmentation data,
    stores them in the results folder for review.
    """
    try:
        session_id = request.form.get('session_id')
        video1 = request.files.get('video1')
        video2 = request.files.get('video2')
        data_file = request.files.get('data')
        
        if not session_id or not video1 or not video2 or not data_file:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Clean session ID for filesystem safety
        session_id = secure_filename(session_id)
        
        # Create session directory
        session_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save the files with consistent naming
        video1_path = os.path.join(session_dir, 'video1.mp4')
        video2_path = os.path.join(session_dir, 'video2.mp4')
        data_path = os.path.join(session_dir, 'segments.pkl')
        
        video1.save(video1_path)
        video2.save(video2_path)
        data_file.save(data_path)
        
        print(f"Files saved for session {session_id}")
        
        # Try to compress videos if ffmpeg is available
        try:
            def compress_video(input_path, output_path):
                subprocess.run([
                    'ffmpeg', '-y', '-i', input_path,
                    '-vcodec', 'libx264', '-qscale:v', '3',
                    '-preset', 'fast', output_path
                ], check=True, stderr=subprocess.PIPE)
                
            video1_compressed_path = os.path.join(session_dir, 'video1_compressed.mp4')
            video2_compressed_path = os.path.join(session_dir, 'video2_compressed.mp4')
            
            compress_video(video1_path, video1_compressed_path)
            compress_video(video2_path, video2_compressed_path)
            
            print(f"Videos compressed for session {session_id}")
        except Exception as e:
            print(f"Warning: Could not compress videos: {e}")
            # Continue without compression - original videos will be used
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Files uploaded successfully'
        })
        
    except Exception as e:
        print(f"Error in upload_for_review: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/results_offline/<path:filename>')
def results_offline(filename):
    """Serve files from offline results folder."""
    return send_from_directory('results_offline_processing', filename)


@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session from the results folder."""
    # Check both folders
    folders_to_check = [
        app.config['RESULTS_FOLDER'],
        'results_offline_processing'
    ]
    
    session_path = None
    for folder in folders_to_check:
        potential_path = os.path.join(folder, session_id)
        if os.path.exists(potential_path):
            session_path = potential_path
            break
    
    if not session_path:
        return jsonify({'error': 'Session not found'}), 404
    
    try:
        # Delete the directory and all its contents
        import shutil
        shutil.rmtree(session_path)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    


# ============================================================================
# Session Management and Cleanup
# ============================================================================

def update_session_activity(session_id):
    """
    Update timestamp of when a session was last active.
    
    Args:
        session_id: Session identifier to update
    """
    if session_id:
        session_last_active[session_id] = datetime.now()


def start_cleanup_thread():
    """
    Start a background thread to automatically clean up inactive sessions.
    
    Monitors session activity and removes sessions that have been
    inactive for longer than SESSION_TIMEOUT_SECONDS.
    """
    def cleanup_worker():
        while True:
            try:
                current_time = datetime.now()
                # Find inactive sessions
                inactive_sessions = []
                for session_id, last_active in list(session_last_active.items()):
                    elapsed = (current_time - last_active).total_seconds()
                    if elapsed > SESSION_TIMEOUT_SECONDS:
                        inactive_sessions.append(session_id)
                
                # Clean up each inactive session
                for session_id in inactive_sessions:
                    print(f"Auto-cleaning inactive session: {session_id}")
                    if session_id in sessions:
                        # Clean up GPU resources
                        session = sessions[session_id]
                        
                        # Clear inference states
                        if 'video1_inference_state' in session:
                            clear_inference_state(session['video1_inference_state'])
                            session['video1_inference_state'] = None
                            
                        if 'video2_inference_state' in session:
                            clear_inference_state(session['video2_inference_state'])
                            session['video2_inference_state'] = None
                        
                        # Clean up temporary files if we know their location
                        try:
                            import shutil
                            if 'video1_dir' in session and session['video1_dir'] and os.path.exists(session['video1_dir']):
                                shutil.rmtree(session['video1_dir'])
                                
                            if 'video2_dir' in session and session['video2_dir'] and os.path.exists(session['video2_dir']):
                                shutil.rmtree(session['video2_dir'])
                        except Exception as e:
                            print(f"Error cleaning files for session {session_id}: {e}")
                        
                        # Remove from sessions dict
                        del sessions[session_id]
                    
                    # Remove from activity tracker
                    if session_id in session_last_active:
                        del session_last_active[session_id]
                
                # Clear GPU cache after cleaning sessions
                if inactive_sessions and torch.cuda.is_available():
                    print(f"Freeing GPU memory after cleaning {len(inactive_sessions)} sessions")
                    torch.cuda.empty_cache()
                    print_gpu_memory_usage()
            except Exception as e:
                print(f"Error in cleanup thread: {e}")
                
            # Sleep for a while before checking again
            time.sleep(300)  # Check every 5 minutes
    
    # Start the background thread
    thread = threading.Thread(target=cleanup_worker, daemon=True)
    thread.start()
    print("Started session cleanup background thread")


# ============================================================================
# Reannotation Routes
# ============================================================================

@app.route('/api/start_reannotation', methods=['POST'])
def start_reannotation():
    """
    Start the reannotation process for selected objects.
    
    Initializes a new session for reannotating specific objects
    that were flagged during review.
    """
    data = request.json
    session_id = data.get('session_id')
    reannotation_config = data.get('reannotation_config', [])
    
    if not session_id:
        return jsonify({'error': 'Missing session ID'}), 400
    
    if not reannotation_config:
        return jsonify({'error': 'No objects selected for reannotation'}), 400
    
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    video_files = sorted([x for x in os.listdir(session_dir) if x.endswith(".mp4") or x.endswith(".mov") or x.endswith(".avi") or x.endswith(".MP4") or x.endswith(".MOV") or x.endswith(".AVI")])
    video1_path = os.path.join(session_dir, video_files[0])
    video2_path = os.path.join(session_dir, video_files[1])
    if not os.path.exists(video1_path) or not os.path.exists(video2_path):
        return jsonify({'error': 'Session not found'}), 404
    
    # Load existing inference data
    try:
        # Create a new session for reannotation
        if session_id in sessions:
            # Clear existing session if it exists
            clear_inference_state(sessions[session_id].get('video1_inference_state'))
            clear_inference_state(sessions[session_id].get('video2_inference_state'))
            del sessions[session_id]
        
        # Create a new session
        sessions[session_id] = {
            'scene_type': 'reannotation',  # Mark as reannotation
            'video1_inference_state': None,
            'video2_inference_state': None,
            'video1_objects': {},
            'video2_objects': {},
            'video1_completed_objects': set(),
            'video2_completed_objects': set()
        }
        
        
        # Process videos to frames if necessary
        video1_frames_dir = os.path.join(session_dir, 'video1_frames')
        video2_frames_dir = os.path.join(session_dir, 'video2_frames')
        
        # If frame directories don't exist, create them
        if not video1_frames_dir or not os.path.exists(video1_frames_dir):
            os.makedirs(video1_frames_dir, exist_ok=True)
            # Process frames
            if not process_video_to_frames(video1_path, video1_frames_dir):
                return jsonify({'error': 'Failed to extract frames from video 1'}), 500
            
            # Count frames in the directory
            video1_frames = [f for f in os.listdir(video1_frames_dir) if f.endswith('.jpg')]
            video1_frame_count = len(video1_frames)
            video1_fps = 30  # Default
        else:
            # Count frames in existing directory
            video1_frames = [f for f in os.listdir(video1_frames_dir) if f.endswith('.jpg')]
            video1_frame_count = len(video1_frames)
            video1_fps = 30  # Default
        
        if not video2_frames_dir or not os.path.exists(video2_frames_dir):
            os.makedirs(video2_frames_dir, exist_ok=True)
            # Process frames
            if not process_video_to_frames(video2_path, video2_frames_dir):
                return jsonify({'error': 'Failed to extract frames from video 2'}), 500
                
            # Count frames in the directory
            video2_frames = [f for f in os.listdir(video2_frames_dir) if f.endswith('.jpg')]
            video2_frame_count = len(video2_frames)
            video2_fps = 30  # Default
        else:
            # Count frames in existing directory
            video2_frames = [f for f in os.listdir(video2_frames_dir) if f.endswith('.jpg')]
            video2_frame_count = len(video2_frames)
            video2_fps = 30  # Default
        
        # Store paths in session
        sessions[session_id]['video1_path'] = video1_path
        sessions[session_id]['video2_path'] = video2_path
        sessions[session_id]['video1_dir'] = video1_frames_dir
        sessions[session_id]['video2_dir'] = video2_frames_dir
        sessions[session_id]['video1_frame_count'] = video1_frame_count
        sessions[session_id]['video2_frame_count'] = video2_frame_count
        sessions[session_id]['video1_fps'] = video1_fps
        sessions[session_id]['video2_fps'] = video2_fps
        
        # Extract object configs from inference data
        object_configs = []
        # for obj in inference_data.get('objects', []):
        #     object_configs.append({
        #         'name': obj['label'].split('_')[0],
        #         'number': int(obj['label'].split('_')[1]) if '_' in obj['label'] and obj['label'].split('_')[1].isdigit() else 1,
        #         'inVideo1': obj['in_video1'],
        #         'inVideo2': obj['in_video2'],
        #         'originalIndex': obj['original_obj_idx']
        #     })


        for obj in reannotation_config:
            obj_name = '_'.join(obj['name'].split('_')[:-1])
            obj_number = int(obj['name'].split('_')[-1]) if '_' in obj['name'] and obj['name'].split('_')[-1].isdigit() else 1            
            object_configs.append({
                'name': obj_name,
                'number': obj_number,
                'inVideo1': obj['inVideo1'],
                'inVideo2': obj['inVideo2'],
                'originalIndex': obj['originalIndex'],
                'viewsInVideo1': obj['viewsInVideo1'],
                'viewsInVideo2': obj['viewsInVideo2'],
                'currentViewVideo1': obj['currentViewVideo1'],
                'currentViewVideo2': obj['currentViewVideo2'],
                'completedViewsVideo1': obj['completedViewsVideo1'],
                'completedViewsVideo2': obj['completedViewsVideo2']
            })
        # Store object configs
        sessions[session_id]['object_configs'] = object_configs
        
        # Initialize inference state
        sessions[session_id]['video1_inference_state'] = predictor.init_state(video_path=video1_frames_dir)
        sessions[session_id]['video2_inference_state'] = predictor.init_state(video_path=video2_frames_dir)
        
        # Store reannotation config
        sessions[session_id]['reannotation_config'] = reannotation_config
        
        # Set frame names (needed for finish_annotation)
        video1_frames.sort(key=lambda p: int(os.path.splitext(p)[0]))
        video2_frames.sort(key=lambda p: int(os.path.splitext(p)[0]))
        sessions[session_id]['video1_frame_names'] = video1_frames
        sessions[session_id]['video2_frame_names'] = video2_frames
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Reannotation session prepared'
        })
        
    except Exception as e:
        print(f"Error preparing reannotation session: {e}")
        return jsonify({'error': f'Error preparing reannotation: {str(e)}'}), 500

@app.route('/api/session_objects/<session_id>')
def get_session_objects(session_id):
    """Retrieve object data for a specific session from stored files."""
    try:
        session_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id)
        if not os.path.exists(session_dir):
            return jsonify({"error": "Session not found"}), 404
            
        # First try to load from inference_data.pkl
        inference_data_path = os.path.join(session_dir, "inference_data.pkl")
        if os.path.exists(inference_data_path):
            with open(inference_data_path, 'rb') as f:
                inference_data = pickle.load(f)
            
            # Return the objects data
            if isinstance(inference_data, dict) and 'objects' in inference_data:
                return jsonify({
                    "objects": inference_data['objects']
                })
        
        # If inference_data.pkl doesn't exist or doesn't have objects data,
        # try to get it from segments.pkl
        segments_path = os.path.join(session_dir, "segments.pkl")
        if os.path.exists(segments_path):
            with open(segments_path, 'rb') as f:
                segments_data = pickle.load(f)
            
            # Try to find objects in segments data
            if isinstance(segments_data, dict) and 'objects' in segments_data:
                return jsonify({
                    "objects": segments_data['objects']
                })
        
        # If we still couldn't find objects data, try loading from the session directly
        session_data = get_session_data(session_id).get_json()
        if session_data and 'objects' in session_data:
            return jsonify({
                "objects": session_data['objects']
            })
            
        # If we couldn't find any objects data
        return jsonify({"error": "No objects data found for this session"}), 404
        
    except Exception as e:
        app.logger.error(f"Error retrieving session objects: {str(e)}")
        return jsonify({"error": f"Failed to retrieve session objects: {str(e)}"}), 500

@app.route('/cleanup_gpu_memory', methods=['POST'])
def cleanup_gpu_memory():
    """Explicitly clean GPU memory cache."""
    try:
        # Clear any cached models or temporary tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared explicitly")
            
            # Print memory usage after cleaning
            print_gpu_memory_usage()
            
            return jsonify({"success": True, "message": "GPU memory cleared"})
        else:
            return jsonify({"success": True, "message": "No CUDA device available"})
    except Exception as e:
        print(f"Error cleaning GPU memory: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/restart_server', methods=['POST'])
def restart_server():
    """Restart the Flask server (for maintenance purposes)."""
    try:
        # Get the current Python executable path and script path
        python_path = sys.executable
        script_path = os.path.abspath(__file__)
        
        # Prepare the restart command
        restart_cmd = f"{python_path} {script_path}"
        
        # Use a separate thread to restart the server after a short delay
        def restart_after_delay():
            time.sleep(1)  # Wait for response to be sent
            print("Restarting server...")
            # Start the new server process
            subprocess.Popen(restart_cmd, shell=True)
            # Exit the current process
            os._exit(0)
        
        # Start the restart thread
        threading.Thread(target=restart_after_delay, daemon=True).start()
        
        return jsonify({"success": True, "message": "Server restarting..."})
    except Exception as e:
        print(f"Error restarting server: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
@app.route('/api/get_objects_from_csv', methods=['GET'])
def get_objects_from_csv():
    """
    Read objects from the CSV file and find the matching row for a session.
    
    Parses the objects database CSV to automatically populate object
    configurations based on video names.
    """
    try:
        # Get the current session ID from query parameters
        session_id = request.args.get('session_id', '')
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
            
        # Attempt to read the CSV file from the root directory
        csv_path = "objects.csv"  # Assuming the file is named objects.csv
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Get the column names
        video1_col = next((col for col in df.columns if 'video 1' in col.lower()), None)
        video2_col = next((col for col in df.columns if 'video 2' in col.lower()), None)
        removed_col = next((col for col in df.columns if 'removed' in col.lower()), None)
        changed_col = next((col for col in df.columns if 'changed' in col.lower()), None)
        added_col = next((col for col in df.columns if 'added' in col.lower()), None)
        
        if not all([video1_col, video2_col, removed_col, changed_col, added_col]):
            return jsonify({'error': 'CSV must contain Video 1, Video 2, Removed, Added, and Changed columns'}), 400
        
        # Find the row where both video1 and video2 names are contained in the session_id
        
        matching_row = None
        for idx, row in df.iterrows():
            video1_name = str(row[video1_col])
            video2_name = str(row[video2_col])
            
            # Skip if either video name is NaN or empty
            if pd.isna(video1_name) or pd.isna(video2_name) or video1_name.strip() == '' or video2_name.strip() == '':
                continue
            
            # Clean the video names the same way they're cleaned when creating session_id
            # Remove special characters and replace spaces with underscores
            video1_clean = re.sub(r'[^\w\s-]', '', video1_name).strip().replace(' ', '_')
            video2_clean = re.sub(r'[^\w\s-]', '', video2_name).strip().replace(' ', '_')
            
            # Create the possible prefix patterns (both orderings)
            pattern1 = f"{video1_clean}_{video2_clean}_"  # video1 followed by video2
            pattern2 = f"{video2_clean}_{video1_clean}_"  # video2 followed by video1
            
            # Check if session_id starts with either pattern
            if session_id.startswith(pattern1) or session_id.startswith(pattern2):
                matching_row = row
                break
        
        if matching_row is None:
            return jsonify({'error': 'No matching row found for this session'}), 404
        
        # Helper function to convert string to list of objects
        def str_to_list(x):
            if pd.isna(x):
                return []
            
            # Handle different formats - CSV export might show lists as "[item1, item2]"
            x = str(x)
            if x.startswith('[') and x.endswith(']'):
                x = x[1:-1]  # Remove brackets
                
            return [item.strip() for item in x.split(',') if item.strip()]
        
        # Process objects from the matching row - simplified format
        objects = []
        
        # Process removed objects (Video 1 only)
        removed_objects = str_to_list(matching_row[removed_col])
        for obj in removed_objects:
            if obj:  # Only add non-empty objects
                objects.append({
                    'name': obj,
                    'inVideo1': True,
                    'inVideo2': False
                })
        
        # Process changed objects (Both videos)
        changed_objects = str_to_list(matching_row[changed_col])
        for obj in changed_objects:
            if obj:  # Only add non-empty objects
                objects.append({
                    'name': obj,
                    'inVideo1': True,
                    'inVideo2': True
                })
        
        # Process added objects (Video 2 only)
        added_objects = str_to_list(matching_row[added_col])
        for obj in added_objects:
            if obj:  # Only add non-empty objects
                objects.append({
                    'name': obj,
                    'inVideo1': False,
                    'inVideo2': True
                })
        
        return jsonify({
            'success': True, 
            'objects': objects
        })
    
    except FileNotFoundError:
        return jsonify({'error': 'CSV file not found in the root directory'}), 404
    except Exception as e:
        return jsonify({'error': f'Error reading CSV: {str(e)}'}), 500

@app.route('/download_object_database', methods=['POST'])
def download_object_database():
    """
    Download the object database from Google Spreadsheet.
    
    Fetches the latest version of the objects CSV from Google Sheets
    and saves it locally for automatic object configuration.
    """
    try:
        # Google Spreadsheet configuration
        gid = "0"
        download_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SPREADSHEET_ID}/export?format=csv&gid={gid}"
        
        print(f"Downloading object database from: {download_url}")
        
        # Download the CSV content
        response = requests.get(download_url)
        
        # Check if request was successful
        if response.status_code != 200:
            error_msg = f'Failed to download spreadsheet: HTTP {response.status_code}'
            print(error_msg)
            return jsonify({'success': False, 'error': error_msg}), 500
        
        # Save the content to objects.csv
        csv_content = response.content
        
        # Ensure file is removed if it exists (to prevent permission issues)
        csv_path = "objects.csv"
        if os.path.exists(csv_path):
            os.remove(csv_path)
        
        # Write the new content
        with open(csv_path, 'wb') as f:
            f.write(csv_content)
            
        print(f"Successfully downloaded and saved object database to {csv_path}")
        
        # Return success response
        return jsonify({
            'success': True,
            'message': 'Object database downloaded successfully'
        })
    
    except Exception as e:
        error_msg = f"Error downloading object database: {str(e)}"
        print(error_msg)
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


# ============================================================================
# Application Startup
# ============================================================================

if __name__ == '__main__':
    # Initialize SAM2 model
    initialize_model()
    
    # Start the cleanup thread (uncomment to enable automatic cleanup)
    # start_cleanup_thread()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5001, debug=False)
