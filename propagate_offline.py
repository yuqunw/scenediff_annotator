import os
import sys

# Set matplotlib to use a non-GUI backend
os.environ["MPLBACKEND"] = "Agg"

os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["TORCHDYNAMO_DISABLE_CUDAGRAPHS"] = "1"  # New critical flag

import uuid
import json
import pickle
import subprocess
import threading
import tempfile
import shutil
import queue
import numpy as np
import torch
import cv2
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import matplotlib.pyplot as plt
import base64
import io
import time
from matplotlib.colors import to_rgb
import matplotlib.colors as mcolors
import re
from datetime import datetime, timedelta

def mask_to_rle_pytorch(tensor: torch.Tensor):
    """Encodes masks to an uncompressed RLE format"""
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
    """Converts uncompressed RLE to COCO format"""
    from pycocotools import mask as mask_utils
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def process_video_to_frames(video_path, output_dir):
    """Process video into individual frames using ffmpeg"""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        'ffmpeg', 
        '-i', video_path, 
        '-q:v', '2', 
        '-r', '30', 
        '-start_number', '0',
        os.path.join(output_dir, '%05d.jpg')
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing video: {e}")
        return False

from sam2.build_sam import build_sam2_video_predictor
from pathlib import Path

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='SAM2 offline propagation')
    parser.add_argument('--base_dir', type=str, required=False, default='results',
                      help='Base directory containing the session folders')
    parser.add_argument('--upload_dir', type=str, required=False, default='uploads',
                      help='Upload directory containing the session folders')

    return parser.parse_args()

def save_video_segments(session_dir, video_idx, video_segments, frames_dir, fps=30):
    """Create video with segmentation masks overlaid
    
    Args:
        session_dir: Directory for the current session
        video_idx: Index of the video (1 or 2)
        video_segments: Dictionary of segmentation masks
        frames_dir: Directory with extracted video frames
        fps: Frames per second for output video
        
    Returns:
        Path to saved video
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
    colors = list(cmap.colors)  # This creates a list copy we can modify
    colors[7] = tuple(i * 0.5 for i in colors[7])
    modified_cmap = mcolors.ListedColormap(colors)

    # cmap = plt.get_cmap('tab20')    
    # colors = list(cmap.colors)  # This creates a list copy we can modify
    # colors[1] = tuple(i * 0.5 for i in colors[1])
    # colors[15] = tuple(i * 0.5 for i in colors[15])
    # modified_cmap = mcolors.ListedColormap(colors)    
    
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
            combined_mask = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=bool)
            for obj_id, mask in video_segments[frame_idx].items():
                mask = mask[0]
                
                # Check for overlap with existing combined mask
                overlap = (combined_mask > 0) & (mask > 0)
                if overlap.any():
                    # If there's overlap, use only non-overlapping parts of the mask
                    # plus a simple continuous region split of the overlapping area
                    non_overlap = (mask > 0) & ~(combined_mask > 0)
                    split_overlap = np.zeros_like(overlap)
                    
                    # Get bounding box of overlap region    
                    y_indices, x_indices = np.where(overlap)
                    if len(y_indices) > 0 and len(x_indices) > 0:
                        min_y, max_y = np.min(y_indices), np.max(y_indices)
                        min_x, max_x = np.min(x_indices), np.max(x_indices)
                        
                        if (combined_mask[overlap] == 1).any():
                            # Simple split: divide the overlap region horizontally
                            mid_y = (min_y + max_y) // 2
                            for y, x in zip(y_indices, x_indices):
                                if y > mid_y:  # Bottom half goes to current object
                                    split_overlap[y, x] = True
                        elif (combined_mask[overlap] == 2).any():
                            # Simple split: divide the overlap region horizontally
                            mid_x = (min_x + max_x) // 2
                            for y, x in zip(y_indices, x_indices):
                                if x > mid_x:  # Right half goes to current object
                                    split_overlap[y, x] = True
                        else: 
                            # Simple split: divide the overlap region horizontally
                            mid_x = (min_x + max_x) // 2
                            mid_y = (min_y + max_y) // 2
                            for y, x in zip(y_indices, x_indices):
                                if x > mid_x and y > mid_y:  # Bottom right half goes to current object
                                    split_overlap[y, x] = True
                        # else: # > 4 overlap
                        #     # Simple split: divide the overlap region horizontally
                        #     mid_x = (min_x + max_x) // 2
                        #     mid_y = (min_y + max_y) // 2
                        #     for y, x in zip(y_indices, x_indices):
                        #         if x < mid_x and y < mid_y:  # Top left half goes to current object
                        #             split_overlap[y, x] = True
                    
                    # Final mask is non-overlapping part plus split overlapping part
                    effective_mask = non_overlap | split_overlap
                    # Update the combined mask with an integer indicating this is region is taken
                    # Using ints for the combined mask helps track which regions are occupied
                    combined_mask[effective_mask] = combined_mask[effective_mask] + 1
                else:
                    # No overlap, use the full mask
                    effective_mask = (mask > 0)
                    combined_mask[effective_mask] = combined_mask[effective_mask] + 1
                
                # Get original object ID for coloring (consistent across videos)
                original_obj_id = int(obj_id) - 1
                
                # Direct color mapping based on object ID (consistent since IDs are consistent)
                color_rgb = np.array(to_rgb(modified_cmap(original_obj_id % 10)))
                
                # Apply mask to overlay - important to apply only to the effective mask
                overlay[effective_mask] = color_rgb
            
            # Convert combined mask to boolean for final overlay
            visible_mask = (combined_mask > 0)
            # Apply the overlay with better blending
            frame_rgb[visible_mask] = np.clip(0.5 * frame_rgb[visible_mask] + 0.5 * overlay[visible_mask] * 255, 0, 255).astype(np.uint8)
            
            # Convert back to BGR for OpenCV
            output_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            # If no masks for this frame, use original frame
            output_frame = frame
        
        # Write frame to video
        video_writer.write(output_frame)
        
    # Release video writer
    video_writer.release()
    print(f"Video {video_idx} saved to {output_path}, size: {os.path.getsize(output_path)} bytes")
    return output_path

def propagate_offline_segmentation(base_dir, upload_dir):
    all_session_dirs = list(Path(base_dir).iterdir())
    # sort by time
    all_session_dirs.sort(key=lambda x: x.stat().st_mtime)
    copy_all_session_dirs = all_session_dirs.copy()
    # Remove the one with video1.mp4, video2.mp4, segments.pkl
    # Check whether the dir contains video1.mp4, video2.mp4, segments.pkl
    for session_dir in copy_all_session_dirs:
        if (session_dir / "video1.mp4").exists() and (session_dir / "video2.mp4").exists() and (session_dir / "segments.pkl").exists():
            if not (session_dir / "review_meta.json").exists():
                all_session_dirs.remove(session_dir) if session_dir in all_session_dirs else None
                continue
            review_json = os.path.join(session_dir, "review_meta.json")
            review = json.load(open(review_json, "r"))
            need_repropagate = False
            for obj in review['objectReviews']:
                if obj['needsReannotation']:
                    need_repropagate = True
                    break
            if 'status' in review and review['status'] == "finish_reannotation":
                need_repropagate = False
            if not need_repropagate:
                all_session_dirs.remove(session_dir) if session_dir in all_session_dirs else None
                
    all_session_dirs = [x for x in all_session_dirs if x.is_dir()]
    all_session_dirs = [x for x in all_session_dirs if not 'save' in x.name]
    # Remove the one without inference_data.pkl
    for session_dir in copy_all_session_dirs:
        if not (session_dir / "inference_data.pkl").exists():
            all_session_dirs.remove(session_dir) if session_dir in all_session_dirs else None
    print(f"Found {len(all_session_dirs)} session directories")

    # Load sam2 model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    print(f"Loading SAM2 model from {sam2_checkpoint}...")
    predictor = build_sam2_video_predictor(
        model_cfg,
        sam2_checkpoint,
        device=device,
        vos_optimized=False,  # Keep SAM2's native optimizations
    )

    for scene_dir in all_session_dirs:
        print(f"Processing scene {scene_dir}")
        session_id = scene_dir.name
        inference_data = pickle.load(open(os.path.join(scene_dir, "inference_data.pkl"), "rb"))

        source_video_dir = os.path.join(upload_dir, session_id)
        video_files = sorted([x for x in os.listdir(source_video_dir) if x.endswith(".mp4") or x.endswith(".mov") or x.endswith(".avi") or x.endswith(".MP4") or x.endswith(".MOV") or x.endswith(".AVI")])
        video_1_path = os.path.join(source_video_dir, video_files[0])
        video_2_path = os.path.join(source_video_dir, video_files[1])
        video1_frames_dir = os.path.join(upload_dir, session_id, "video1_frames")    
        video2_frames_dir = os.path.join(upload_dir, session_id, "video2_frames")

        # Direct rm the video1_frames_dir, video2_frames_dir if exists
        if os.path.exists(video1_frames_dir):
            shutil.rmtree(video1_frames_dir)
        if os.path.exists(video2_frames_dir):
            shutil.rmtree(video2_frames_dir)

        process_video_to_frames(video_1_path, video1_frames_dir)
        process_video_to_frames(video_2_path, video2_frames_dir)

    
        video1_inference_state = predictor.init_state(video_path=video1_frames_dir)   
        # load inference_data.pkl
        video_H = video1_inference_state["video_height"]
        video_W = video1_inference_state["video_width"]

        video1_saved_dict = inference_data["video1"]
        video2_saved_dict = inference_data["video2"]

        video1_obj_idx_to_id = video1_saved_dict["obj_idx_to_id"]
        video2_obj_idx_to_id = video2_saved_dict["obj_idx_to_id"]

        video1_point_inputs_per_obj = video1_saved_dict["point_inputs_per_obj"]
        video2_point_inputs_per_obj = video2_saved_dict["point_inputs_per_obj"]

        for idx_1, video1_point_inputs in video1_point_inputs_per_obj.items():
            video1_obj_id = video1_obj_idx_to_id[idx_1]

            for frame_idx, points_dict in video1_point_inputs.items():
                points = points_dict["point_coords"][0].cpu().numpy()
                points = points / predictor.image_size
                points = points * np.array([video_W, video_H])
                labels = points_dict["point_labels"][0].cpu().numpy()
                _,  out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=video1_inference_state,
                    frame_idx=frame_idx,
                    obj_id=video1_obj_id,
                    points=points,
                    labels=labels,
                    clear_old_points=True
                )
        # # Save the out_mask_logits and video1_frames_dir / out_frame_idx.jpg
        # rgb_path = os.path.join(video1_frames_dir, f"{frame_idx:05d}.jpg")
        # rgb = cv2.imread(rgb_path)
        # # Add points visualization
        # # Draw points on the RGB image
        # for point_idx, (point, label) in enumerate(zip(points, labels)):
        #     # Convert point coordinates to integers
        #     x, y = point.astype(int)
        #     # Draw positive points in green, negative in red
        #     color = (0, 255, 0) if label == 1 else (255, 0, 0)
        #     # Draw circle for each point
        #     cv2.circle(rgb, (x, y), 5, color, -1)
            
        # # Overlay mask on RGB image
        # mask_overlay = np.zeros_like(rgb, dtype=np.float32)
        # mask_bool = (out_mask_logits[0] > 0.0).cpu().numpy()[0]
        # mask_overlay[mask_bool] = [0, 0, 255]  # Blue overlay for mask
        # rgb = cv2.addWeighted(rgb, 1.0, mask_overlay.astype(np.uint8), 0.5, 0)
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        # plt.imsave('rgb.png', rgb)

        video2_inference_state = predictor.init_state(video_path=video2_frames_dir) 

        video_H = video2_inference_state["video_height"]
        video_W = video2_inference_state["video_width"]

        for idx_2, video2_point_inputs in video2_point_inputs_per_obj.items():
            video2_obj_id = video2_obj_idx_to_id[idx_2]
            for frame_idx, points_dict in video2_point_inputs.items():
                points = points_dict["point_coords"][0].cpu().numpy()
                points = points / predictor.image_size
                points = points * np.array([video_W, video_H])
                labels = points_dict["point_labels"][0].cpu().numpy()
                _, _, _ = predictor.add_new_points_or_box(
                    inference_state=video2_inference_state,
                    frame_idx=frame_idx,
                    obj_id=video2_obj_id,
                    points=points,
                    labels=labels,
                    clear_old_points=True
                )

        # Propagate the 
        print("Propagating objects for video 1...")
        video1_segments = {}
        try:
            if len(video1_inference_state['obj_idx_to_id']) > 0:
                outputs = predictor.propagate_in_video(
                    inference_state=video1_inference_state,
                    start_frame_idx=0)
                
                for out_frame_idx, out_obj_ids, out_mask_logits in outputs:
                    # Initialize frame dictionary if it doesn't exist
                    if out_frame_idx not in video1_segments:
                        video1_segments[out_frame_idx] = {}
                    
                    # Update masks for each object ID
                    for j, out_obj_id in enumerate(out_obj_ids):
                        if (out_mask_logits[j] > 0.0).sum() > 0:
                            video1_segments[out_frame_idx][str(out_obj_id)] = (out_mask_logits[j] > 0.0).cpu().numpy()
        except Exception as e:
            print(f"Error during video 1 propagation: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

        # Convert segments to object-centric format for storage
        video1_objects_rle = {}

        # Reorganize video1 segments from frame->object to object->frame
        for frame_idx, frame_masks in video1_segments.items():
            for obj_id, mask in frame_masks.items():
                # Initialize object dict if not exist
                true_obj_id = int(obj_id) - 1
                if true_obj_id not in video1_objects_rle:
                    video1_objects_rle[true_obj_id] = {}
                
                # Store binary mask
                rles = mask_to_rle_pytorch(torch.tensor(mask))
                video1_objects_rle[true_obj_id][str(frame_idx)] = coco_encode_rle(rles[0])

        # Delete video1_inference_state
        del video1_inference_state
        torch.cuda.empty_cache()

        print("Propagating objects for video 2...")
        video2_segments = {}
        try:
            if len(video2_inference_state['obj_idx_to_id']) > 0:
                outputs = predictor.propagate_in_video(
                    inference_state=video2_inference_state,
                    start_frame_idx=0)
                
                for out_frame_idx, out_obj_ids, out_mask_logits in outputs:
                    # Initialize frame dictionary if it doesn't exist
                    if out_frame_idx not in video2_segments:
                        video2_segments[out_frame_idx] = {}
                    
                    # Update masks for each object ID
                    for j, out_obj_id in enumerate(out_obj_ids):
                        if (out_mask_logits[j] > 0.0).sum() > 0:
                            video2_segments[out_frame_idx][str(out_obj_id)] = (out_mask_logits[j] > 0.0).cpu().numpy()
        except Exception as e:
            print(f"Error during video 2 propagation: {str(e)}")
            import traceback
            traceback.print_exc()
            continue


        video2_objects_rle = {}
        
            
        # Reorganize video2 segments from frame->object to object->frame
        for frame_idx, frame_masks in video2_segments.items():
            for obj_id, mask in frame_masks.items():
                true_obj_id = int(obj_id) - 1
                # Initialize object dict if not exist
                if true_obj_id not in video2_objects_rle:
                    video2_objects_rle[true_obj_id] = {}
                
                # Store binary mask
                rles = mask_to_rle_pytorch(torch.tensor(mask))
                video2_objects_rle[true_obj_id][str(frame_idx)] = coco_encode_rle(rles[0])
        
        # Delete video2_inference_state
        del video2_inference_state
        torch.cuda.empty_cache()
        
        # Get objects data from inference data
        objects = inference_data.get("objects", [])
        
        # Create save dictionary with reorganized structure
        save_dict = {
            'scene_type': inference_data.get('scene_type', ''),
            'video1_objects': video1_objects_rle,
            'video2_objects': video2_objects_rle,
            'objects': objects,
            'comments': inference_data.get('comments', {})
        }
        
        # Save as pickle file
        output_file = os.path.join(scene_dir, "segments.pkl")
        print(f"Saving propagated segments to {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(save_dict, f)
        
        # Before cleaning up, save videos with segmentation overlays
        print("Creating visualization videos with segmentation overlays...")
        video1_path = save_video_segments(scene_dir, 1, video1_segments, video1_frames_dir)
        video2_path = save_video_segments(scene_dir, 2, video2_segments, video2_frames_dir)
        
        # Clean up temporary frame directories
        shutil.rmtree(video1_frames_dir, ignore_errors=True)
        shutil.rmtree(video2_frames_dir, ignore_errors=True)

        # Remove the video1_compressed.mp4, video2_compressed.mp4
        os.remove(os.path.join(scene_dir, "video1_compressed.mp4")) if os.path.exists(os.path.join(scene_dir, "video1_compressed.mp4")) else None
        os.remove(os.path.join(scene_dir, "video2_compressed.mp4")) if os.path.exists(os.path.join(scene_dir, "video2_compressed.mp4")) else None

        if (scene_dir / "review_meta.json").exists():
            # Change the status to "Finished Reannotation"
            review_json = os.path.join(scene_dir, "review_meta.json")
            review = json.load(open(review_json, "r"))
            review['status'] = "finish_reannotation"
            with open(review_json, "w") as f:
                json.dump(review, f, indent=2)


def main():
    args = parse_args()
    base_dir = args.base_dir
    upload_dir = args.upload_dir
    print(f"Starting offline propagation from {base_dir}")

    propagate_offline_segmentation(base_dir, upload_dir)


if __name__ == "__main__":
    main()