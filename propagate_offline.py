"""
SAM2 Offline Video Segmentation Propagation

This script performs offline propagation of segmentation masks across video frames
using the Segment Anything Model 2 (SAM2). It processes video pairs, propagates
user-annotated points, and generates visualization videos with segmentation overlays.
"""

import os
import sys
import json
import pickle
import subprocess
import shutil
import argparse
from pathlib import Path

# Set matplotlib to use a non-GUI backend
os.environ["MPLBACKEND"] = "Agg"

# PyTorch optimization flags
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["TORCHDYNAMO_DISABLE_CUDAGRAPHS"] = "1"

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib.colors as mcolors

# Add SAM2 to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sam2'))
from sam2.build_sam import build_sam2_video_predictor
SAM2_DIR = os.path.join(os.path.dirname(__file__), 'sam2')

# Constants
DEFAULT_FPS = 30
DEFAULT_SAM2_CHECKPOINT = os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_large.pt")
DEFAULT_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SUPPORTED_VIDEO_FORMATS = (".mp4", ".mov", ".avi", ".MP4", ".MOV", ".AVI")
FRAME_QUALITY = 2  # FFmpeg quality parameter (1-31, lower is better)

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
        '-q:v', str(FRAME_QUALITY), 
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SAM2 offline video segmentation propagation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--base_dir', 
        type=str, 
        default='results',
        help='Base directory containing the session folders with results'
    )
    parser.add_argument(
        '--upload_dir', 
        type=str, 
        default='uploads',
        help='Upload directory containing the source videos'
    )
    return parser.parse_args()

def save_video_segments(session_dir, video_idx, video_segments, frames_dir, fps=DEFAULT_FPS):
    """
    Create video with segmentation masks overlaid on original frames.
    
    Args:
        session_dir: Directory for the current session
        video_idx: Index of the video (1 or 2)
        video_segments: Dictionary mapping frame indices to object masks
        frames_dir: Directory with extracted video frames
        fps: Frames per second for output video
        
    Returns:
        Path to saved video file
    """
    # Create output filename
    output_filename = f"video{video_idx}.mp4"
    output_path = os.path.join(session_dir, output_filename)
    
    # Get frame names from the extracted frames directory
    frame_names = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    # Get the first frame to determine video dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frame_names[0]))
    height, width = first_frame.shape[:2]
    
    # Use H.264 codec for better browser compatibility
    if sys.platform == 'darwin':  # macOS
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 encoding
    else:  # Windows/Linux
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Try more compatible MPEG-4 encoding
    
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Get color map for visualization
    cmap = plt.get_cmap("tab10")
    colors = list(cmap.colors)
    colors[7] = tuple(i * 0.5 for i in colors[7])  # Darken color index 7
    modified_cmap = mcolors.ListedColormap(colors)
    
    # Process each frame
    print(f"Creating video {video_idx} with segmentation overlays at {output_path}...")
    for frame_idx, frame_name in enumerate(frame_names):
        # Read the original frame
        frame = cv2.imread(os.path.join(frames_dir, frame_name))
        # Convert from BGR to RGB for consistent coloring
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # If this frame has segmentation masks, overlay them
        if frame_idx in video_segments:
            overlay = np.zeros_like(frame_rgb, dtype=np.float32)
            
            # Process each object's mask in this frame
            combined_mask = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=int)
            for obj_id, mask in video_segments[frame_idx].items():
                mask = mask[0]
                
                # Check for overlap with existing combined mask
                overlap = (combined_mask > 0) & (mask > 0)
                if overlap.any():
                    # Handle overlap by splitting overlapping regions
                    non_overlap = (mask > 0) & ~(combined_mask > 0)
                    split_overlap = np.zeros_like(overlap)
                    
                    # Get bounding box of overlap region
                    y_indices, x_indices = np.where(overlap)
                    if len(y_indices) > 0 and len(x_indices) > 0:
                        min_y, max_y = np.min(y_indices), np.max(y_indices)
                        min_x, max_x = np.min(x_indices), np.max(x_indices)
                        
                        # Split strategy depends on which objects are already present
                        if (combined_mask[overlap] == 1).any():
                            # Horizontal split: bottom half to current object
                            mid_y = (min_y + max_y) // 2
                            for y, x in zip(y_indices, x_indices):
                                if y > mid_y:
                                    split_overlap[y, x] = True
                        elif (combined_mask[overlap] == 2).any():
                            # Vertical split: right half to current object
                            mid_x = (min_x + max_x) // 2
                            for y, x in zip(y_indices, x_indices):
                                if x > mid_x:
                                    split_overlap[y, x] = True
                        else:
                            # Quadrant split: bottom-right quadrant to current object
                            mid_x = (min_x + max_x) // 2
                            mid_y = (min_y + max_y) // 2
                            for y, x in zip(y_indices, x_indices):
                                if x > mid_x and y > mid_y:
                                    split_overlap[y, x] = True
                    
                    # Final mask is non-overlapping part plus split overlapping part
                    effective_mask = non_overlap | split_overlap
                    combined_mask[effective_mask] = combined_mask[effective_mask] + 1
                else:
                    # No overlap, use the full mask
                    effective_mask = (mask > 0)
                    combined_mask[effective_mask] = combined_mask[effective_mask] + 1
                
                # Get original object ID for consistent coloring across videos
                original_obj_id = int(obj_id) - 1
                color_rgb = np.array(to_rgb(modified_cmap(original_obj_id % 10)))
                
                # Apply color to overlay for effective mask region
                overlay[effective_mask] = color_rgb
            
            # Apply the overlay with alpha blending (50% transparency)
            visible_mask = (combined_mask > 0)
            alpha = 0.5
            frame_rgb[visible_mask] = np.clip(
                (1 - alpha) * frame_rgb[visible_mask] + alpha * overlay[visible_mask] * 255,
                0, 255
            ).astype(np.uint8)
            
            # Convert back to BGR for OpenCV
            output_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            # No masks for this frame, use original
            output_frame = frame
        
        # Write frame to video
        video_writer.write(output_frame)
        
    # Release video writer
    video_writer.release()
    print(f"Video {video_idx} saved to {output_path}, size: {os.path.getsize(output_path)} bytes")
    return output_path

def propagate_offline_segmentation(base_dir, upload_dir):
    """
    Process all pending segmentation sessions and propagate annotations.
    
    Args:
        base_dir: Base directory containing session result folders
        upload_dir: Directory containing uploaded source videos
    """
    # Get all session directories sorted by modification time
    all_session_dirs = list(Path(base_dir).iterdir())
    all_session_dirs.sort(key=lambda x: x.stat().st_mtime)
    copy_all_session_dirs = all_session_dirs.copy()
    
    # Filter sessions that need processing
    for session_dir in copy_all_session_dirs:
        # Skip already processed sessions unless they need reannotation
        if (session_dir / "video1.mp4").exists() and \
           (session_dir / "video2.mp4").exists() and \
           (session_dir / "segments.pkl").exists():
            
            if not (session_dir / "review_meta.json").exists():
                all_session_dirs.remove(session_dir) if session_dir in all_session_dirs else None
                continue
                
            # Check if reannotation is needed
            review_json = session_dir / "review_meta.json"
            with open(review_json, "r") as f:
                review = json.load(f)
                
            need_repropagate = any(
                obj.get('needsReannotation', False) 
                for obj in review.get('objectReviews', [])
            )
            
            if review.get('status') == "finish_reannotation":
                need_repropagate = False
                
            if not need_repropagate:
                all_session_dirs.remove(session_dir) if session_dir in all_session_dirs else None
    
    # Additional filtering
    all_session_dirs = [x for x in all_session_dirs if x.is_dir()]
    all_session_dirs = [x for x in all_session_dirs if 'save' not in x.name]
    all_session_dirs = [
        x for x in all_session_dirs 
        if (x / "inference_data.pkl").exists()
    ]
    
    print(f"Found {len(all_session_dirs)} session directories to process")

    # Load SAM2 model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_checkpoint = DEFAULT_SAM2_CHECKPOINT
    model_cfg = DEFAULT_MODEL_CONFIG
    
    print(f"Loading SAM2 model from {sam2_checkpoint} on {device}...")
    predictor = build_sam2_video_predictor(
        model_cfg,
        sam2_checkpoint,
        device=device,
        vos_optimized=False,
    )

    # Process each session
    for scene_dir in all_session_dirs:
        print(f"\n{'='*60}")
        print(f"Processing session: {scene_dir.name}")
        print(f"{'='*60}")
        
        session_id = scene_dir.name
        
        # Load inference data
        with open(scene_dir / "inference_data.pkl", "rb") as f:
            inference_data = pickle.load(f)

        # Locate source videos
        source_video_dir = os.path.join(upload_dir, session_id)
        video_files = sorted([
            f for f in os.listdir(source_video_dir) 
            if f.endswith(SUPPORTED_VIDEO_FORMATS)
        ])
        
        if len(video_files) < 2:
            print(f"Warning: Expected 2 videos, found {len(video_files)}. Skipping...")
            continue
            
        video_1_path = os.path.join(source_video_dir, video_files[0])
        video_2_path = os.path.join(source_video_dir, video_files[1])
        video1_frames_dir = os.path.join(source_video_dir, "video1_frames")
        video2_frames_dir = os.path.join(source_video_dir, "video2_frames")

        # Clean up existing frame directories
        for frames_dir in [video1_frames_dir, video2_frames_dir]:
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)

        # Extract frames from videos
        print("Extracting frames from videos...")
        if not process_video_to_frames(video_1_path, video1_frames_dir):
            print(f"Failed to extract frames from video 1. Skipping...")
            continue
        if not process_video_to_frames(video_2_path, video2_frames_dir):
            print(f"Failed to extract frames from video 2. Skipping...")
            continue

        # Initialize inference states for both videos
        print("Initializing SAM2 inference states...")
        video1_inference_state = predictor.init_state(video_path=video1_frames_dir)
        video_H = video1_inference_state["video_height"]
        video_W = video1_inference_state["video_width"]

        # Extract saved annotation data
        video1_saved_dict = inference_data["video1"]
        video2_saved_dict = inference_data["video2"]

        video1_obj_idx_to_id = video1_saved_dict["obj_idx_to_id"]
        video2_obj_idx_to_id = video2_saved_dict["obj_idx_to_id"]

        video1_point_inputs_per_obj = video1_saved_dict["point_inputs_per_obj"]
        video2_point_inputs_per_obj = video2_saved_dict["point_inputs_per_obj"]

        # Add point annotations for video 1
        print(f"Adding {len(video1_point_inputs_per_obj)} objects to video 1...")
        for idx_1, video1_point_inputs in video1_point_inputs_per_obj.items():
            video1_obj_id = video1_obj_idx_to_id[idx_1]

            for frame_idx, points_dict in video1_point_inputs.items():
                # Scale points to video dimensions
                points = points_dict["point_coords"][0].cpu().numpy()
                points = points / predictor.image_size * np.array([video_W, video_H])
                labels = points_dict["point_labels"][0].cpu().numpy()
                
                predictor.add_new_points_or_box(
                    inference_state=video1_inference_state,
                    frame_idx=frame_idx,
                    obj_id=video1_obj_id,
                    points=points,
                    labels=labels,
                    clear_old_points=True
                )

        # Initialize inference state for video 2
        video2_inference_state = predictor.init_state(video_path=video2_frames_dir)
        video_H = video2_inference_state["video_height"]
        video_W = video2_inference_state["video_width"]

        # Add point annotations for video 2
        print(f"Adding {len(video2_point_inputs_per_obj)} objects to video 2...")
        for idx_2, video2_point_inputs in video2_point_inputs_per_obj.items():
            video2_obj_id = video2_obj_idx_to_id[idx_2]
            
            for frame_idx, points_dict in video2_point_inputs.items():
                # Scale points to video dimensions
                points = points_dict["point_coords"][0].cpu().numpy()
                points = points / predictor.image_size * np.array([video_W, video_H])
                labels = points_dict["point_labels"][0].cpu().numpy()
                
                predictor.add_new_points_or_box(
                    inference_state=video2_inference_state,
                    frame_idx=frame_idx,
                    obj_id=video2_obj_id,
                    points=points,
                    labels=labels,
                    clear_old_points=True
                )

        # Propagate segmentation masks across all frames for video 1
        print("Propagating segmentation masks for video 1...")
        video1_segments = {}
        try:
            if len(video1_inference_state['obj_idx_to_id']) > 0:
                outputs = predictor.propagate_in_video(
                    inference_state=video1_inference_state,
                    start_frame_idx=0
                )
                
                for out_frame_idx, out_obj_ids, out_mask_logits in outputs:
                    if out_frame_idx not in video1_segments:
                        video1_segments[out_frame_idx] = {}
                    
                    for j, out_obj_id in enumerate(out_obj_ids):
                        # Only store masks with non-zero pixels
                        if (out_mask_logits[j] > 0.0).sum() > 0:
                            mask = (out_mask_logits[j] > 0.0).cpu().numpy()
                            video1_segments[out_frame_idx][str(out_obj_id)] = mask
        except Exception as e:
            print(f"Error during video 1 propagation: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Convert segments to object-centric format and encode as RLE
        print("Converting video 1 masks to RLE format...")
        video1_objects_rle = {}
        for frame_idx, frame_masks in video1_segments.items():
            for obj_id, mask in frame_masks.items():
                true_obj_id = int(obj_id) - 1
                if true_obj_id not in video1_objects_rle:
                    video1_objects_rle[true_obj_id] = {}
                
                rles = mask_to_rle_pytorch(torch.tensor(mask))
                video1_objects_rle[true_obj_id][str(frame_idx)] = coco_encode_rle(rles[0])

        # Clean up to free memory
        del video1_inference_state
        torch.cuda.empty_cache()

        # Propagate segmentation masks across all frames for video 2
        print("Propagating segmentation masks for video 2...")
        video2_segments = {}
        try:
            if len(video2_inference_state['obj_idx_to_id']) > 0:
                outputs = predictor.propagate_in_video(
                    inference_state=video2_inference_state,
                    start_frame_idx=0
                )
                
                for out_frame_idx, out_obj_ids, out_mask_logits in outputs:
                    if out_frame_idx not in video2_segments:
                        video2_segments[out_frame_idx] = {}
                    
                    for j, out_obj_id in enumerate(out_obj_ids):
                        # Only store masks with non-zero pixels
                        if (out_mask_logits[j] > 0.0).sum() > 0:
                            mask = (out_mask_logits[j] > 0.0).cpu().numpy()
                            video2_segments[out_frame_idx][str(out_obj_id)] = mask
        except Exception as e:
            print(f"Error during video 2 propagation: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Convert segments to object-centric format and encode as RLE
        print("Converting video 2 masks to RLE format...")
        video2_objects_rle = {}
        for frame_idx, frame_masks in video2_segments.items():
            for obj_id, mask in frame_masks.items():
                true_obj_id = int(obj_id) - 1
                if true_obj_id not in video2_objects_rle:
                    video2_objects_rle[true_obj_id] = {}
                
                rles = mask_to_rle_pytorch(torch.tensor(mask))
                video2_objects_rle[true_obj_id][str(frame_idx)] = coco_encode_rle(rles[0])
        
        # Clean up to free memory
        del video2_inference_state
        torch.cuda.empty_cache()
        
        # Prepare output data structure
        objects = inference_data.get("objects", [])
        save_dict = {
            'scene_type': inference_data.get('scene_type', ''),
            'video1_objects': video1_objects_rle,
            'video2_objects': video2_objects_rle,
            'objects': objects,
            'comments': inference_data.get('comments', {})
        }
        
        # Save segmentation results
        output_file = scene_dir / "segments.pkl"
        print(f"Saving propagated segments to {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(save_dict, f)
        
        # Create visualization videos with segmentation overlays
        print("Creating visualization videos...")
        save_video_segments(scene_dir, 1, video1_segments, video1_frames_dir)
        save_video_segments(scene_dir, 2, video2_segments, video2_frames_dir)
        
        # Clean up temporary files and directories
        print("Cleaning up temporary files...")
        shutil.rmtree(video1_frames_dir, ignore_errors=True)
        shutil.rmtree(video2_frames_dir, ignore_errors=True)

        # Remove compressed videos if they exist
        for compressed_video in ["video1_compressed.mp4", "video2_compressed.mp4"]:
            compressed_path = scene_dir / compressed_video
            if compressed_path.exists():
                compressed_path.unlink()

        # Update review status if review exists
        review_meta_path = scene_dir / "review_meta.json"
        if review_meta_path.exists():
            with open(review_meta_path, "r") as f:
                review = json.load(f)
            review['status'] = "finish_reannotation"
            with open(review_meta_path, "w") as f:
                json.dump(review, f, indent=2)
        
        print(f"✓ Session {session_id} processed successfully!")


def main():
    """Main entry point for offline segmentation propagation."""
    args = parse_args()
    
    print("="*60)
    print("SAM2 Offline Video Segmentation Propagation")
    print("="*60)
    print(f"Base directory: {args.base_dir}")
    print(f"Upload directory: {args.upload_dir}")
    print()

    propagate_offline_segmentation(args.base_dir, args.upload_dir)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == "__main__":
    main()