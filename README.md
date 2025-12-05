# SceneDiff Annotator

A video annotation tool for logging down changes between paired video sequences. 

🔗 Check out the [project page](https://yuqunw.github.io/SceneDiff) for more details.

## Overview

The SceneDiff Annotator is built on top of [SAM2](https://github.com/facebookresearch/sam2) and provides a complete workflow for annotating changes between video pairs:

- **Upload & Configure**: Upload video pairs and specify object attributes (deformability, change type, multiplicity)
- **Interactive Annotation**: Provide sparse point prompts on selected frames with an intuitive click-based interface
- **Offline Propagation**: Automatically propagate masks throughout both videos
- **Review & Refine**: Visualize annotated videos, refine annotations, and verify results

### Demo

📹 **[Watch the annotation demo video](videos/annotation_visualization.mov)** (click to download and view)

*The demo shows the complete annotation workflow in action.*

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)

### Setup Instructions

1. **Clone the repository** (including SAM2 submodule):
   ```bash
   git clone --recursive https://github.com/yuqunw/scenediff_annotator
   cd scenediff_annotator
   ```

2. **Install SAM2 and dependencies**:
   ```bash
   cd sam2
   pip install -e .
   ```

3. **Download SAM2 checkpoints**:
   ```bash
   cd checkpoints
   bash download_ckpts.sh
   cd ../..
   ```

For detailed SAM2 installation instructions, refer to the [official SAM2 repository](https://github.com/facebookresearch/sam2).

## Usage

### Starting the Application

1. **Launch the backend server**:
   ```bash
   python backend.py
   ```

2. **Open the web interface**:
   Navigate to `http://localhost:5000` in your web browser.

### Annotation Workflow

1. **Upload Videos**: Upload a pair of videos, fill in the Scene Type, and wait for the initialization.

2. **Configure Object Attributes**: 
   - Specify the number of changed objects
   - Specify the name, multiplicity index
   - Specify the appearance in video 1 and video 2 and number of frames for annotation
   - Specify the deformabiltiy

3. **Provide Annotations**:
   - Select the frame in the video
   - Click to add point prompts

4. **Run Offline Propagation**:
   - After all annotation, click `Start Offline Job` to begin mask propagation
   - SAM2 will propagate objects throughout both videos offline. Could close the page.

5. **Review & Refine**:
   - Navigate to `Review Sessions` to visualize results
   - Review or refine annotations if needed

## Output Format

The uploaded videos and generated outputs are saved at `./uploads` and `./results`. The annotation tool generates two primary output files:

- **`inference_data.pkl`**: Stores initial prompt inputs used for mask propagation
- **`segments.pkl`**: Contains the final segmentation masks and metadata

### Segments Structure

The `segments.pkl` file follows this hierarchical structure:

```python
segments = {
    'scenetype': str,                    # Type of scene change
    'video1_objects': {
        'object_id': {
            'frame_id': {
                'mask': RLE_Mask         # Run-length encoded mask
            }
        }
    },
    'video2_objects': {
        'object_id': {
            'frame_id': {
                'mask': RLE_Mask
            }
        }
    },
    'objects': {
        'object_1': {
            'label': str,                # Object label/name
            'in_video1': bool,           # Present in video 1
            'in_video2': bool,           # Present in video 2
            'deformability': str         # 'rigid' or 'deformable'
        }
    }
}
```

### Loading Masks

To convert RLE masks back to tensors:

```python
import torch
from pycocotools import mask as mask_utils

# Load and decode RLE mask
tensor_mask = torch.tensor(mask_utils.decode(rle_mask))
```

<!-- ## Citation

If you use this annotation tool in your research, please cite the SceneDiff project:

```bibtex
@misc{scenediff2024,
  title={SceneDiff: Scene Change Detection and Analysis},
  author={Your Name},
  year={2024},
  howpublished={\url{https://yuqunw.github.io/SceneDiff}}
}
``` -->

## Acknowledgements

This project is built upon the excellent [SAM2 repository](https://github.com/facebookresearch/sam2) (Segment Anything Model 2). We gratefully acknowledge their contributions to the computer vision community.

## License

See [LICENSE](LICENSE) for more information.