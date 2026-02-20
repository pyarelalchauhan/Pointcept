"""
Scene-Level HDF5 Dataset Loader for Bhopal MLS Data
Compatible with Pointcept framework and PTv3/Concerto/Sonata

This loader is for SEMANTIC SEGMENTATION (per-point labels) using
scene-level data created by 19b_create_scene_V7_dataset.py.

Key Differences from Instance-Level Dataset:
- Instance-level: Object classification (1 label per point cloud)
- Scene-level: Semantic segmentation (1 label PER POINT)

HDF5 Structure (Scene V7):
    scenes/points              [N, max_pts, 3]    float32  - XYZ (scene-centered)
    scenes/intensity           [N, max_pts]       float32  - Raw values (0-65535)
    scenes/gps_time            [N, max_pts]       float32  - GPS timestamp
    scenes/labels              [N, max_pts]       int32    - Per-point class (0-61, -1=padding)
    scenes/global_ids          [N, max_pts]       int32    - Per-point instance ID (-1=padding)
    scenes/num_points          [N]                uint32   - Valid point count
    scenes/original_num_points [N]                uint32   - Original before downsampling
    metadata/scene_names       [N]                str      - Scene name (e.g., S1_0001)
    metadata/parent_scene      [N]                str      - Original V5 scene name
    metadata/bhopal_paths      [N]                uint8    - Bhopal path (1-8)
    metadata/segment_size_m    [N]                float32  - 5.0, 2.5, 1.25, or 0.625
    splits/train_indices, val_indices, test_indices

V7 Features:
- Adaptive subdivision: 5m → 2.5m → 1.25m → 0.625m segments
- Building class sampling: 90% retention (classes 8-12)
- Min threshold: 3,000 points; Max cap: 1,000,000 points
- Total: 2,340 scenes, ~969M points

Date: 2026-01-24
"""

import h5py
import numpy as np
from pathlib import Path

from .builder import DATASETS
from .transform import Compose, TRANSFORMS


# =============================================================================
# CLASS-BALANCED SAMPLING TRANSFORM
# =============================================================================
# This transform addresses Bhopal's severe class imbalance (Building = 65%)
# by downsampling dominant classes BEFORE voxelization.
#
# Why before voxelization?
# - Voxelization (GridSample) selects one point per voxel cell
# - If 65% of points are Building, ~65% of voxels will be Building-dominated
# - By reducing Building points first, we get more balanced voxel distribution
#
# Usage in config:
#   dict(type="ClassBalancedSample",
#        class_sample_rates={2: 0.4},  # Keep 40% of Building (class 2)
#        min_points=10000)
# =============================================================================

@TRANSFORMS.register_module()
class ClassBalancedSample:
    """
    Class-balanced point sampling to address class imbalance.

    Applies different sampling rates to different classes before voxelization.
    Designed to reduce dominant classes (e.g., Building at 65%) while preserving
    rare classes (e.g., Fencing at 0.26%).

    Args:
        class_sample_rates: Dict mapping class_id → retention_rate (0.0-1.0)
                           Classes not in dict keep all points (rate=1.0)
                           Example: {2: 0.4} keeps 40% of Building (class 2)
        min_points: Minimum total points to keep (prevents over-sampling)
        seed: Random seed for reproducibility (None = different each call)

    Example for Bhopal 12-class:
        # Building is 65% of data, reduce to ~26% effective contribution
        ClassBalancedSample(class_sample_rates={2: 0.4}, min_points=10000)
    """

    def __init__(self, class_sample_rates=None, min_points=10000, seed=None):
        self.class_sample_rates = class_sample_rates or {}
        self.min_points = min_points
        self.seed = seed

        # Validate rates
        for cls, rate in self.class_sample_rates.items():
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"Sample rate for class {cls} must be in [0, 1], got {rate}")

    def __call__(self, data_dict):
        """
        Apply class-balanced sampling to point cloud.

        Args:
            data_dict: Must contain 'segment' (class labels) and 'coord'

        Returns:
            data_dict: With sampled points (all relevant arrays filtered)
        """
        if "segment" not in data_dict or len(self.class_sample_rates) == 0:
            return data_dict

        segment = data_dict["segment"]
        n_points = len(segment)

        if n_points == 0:
            return data_dict

        # Create keep mask (start with all True)
        keep_mask = np.ones(n_points, dtype=bool)

        # Set random seed if specified
        rng = np.random.RandomState(self.seed) if self.seed is not None else np.random

        # Apply class-specific sampling
        for class_id, sample_rate in self.class_sample_rates.items():
            if sample_rate >= 1.0:
                continue  # Keep all points for this class

            # Find points of this class
            class_mask = (segment == class_id)
            n_class = class_mask.sum()

            if n_class == 0:
                continue

            # Determine how many to drop
            n_keep = int(n_class * sample_rate)
            n_drop = n_class - n_keep

            if n_drop > 0:
                # Get indices of this class
                class_indices = np.where(class_mask)[0]
                # Randomly select indices to drop
                drop_indices = rng.choice(class_indices, size=n_drop, replace=False)
                keep_mask[drop_indices] = False

        # Ensure minimum points
        n_kept = keep_mask.sum()
        if n_kept < self.min_points and n_points >= self.min_points:
            # Need to restore some dropped points
            n_restore = self.min_points - n_kept
            dropped_indices = np.where(~keep_mask)[0]
            if len(dropped_indices) > 0:
                restore_indices = rng.choice(
                    dropped_indices,
                    size=min(n_restore, len(dropped_indices)),
                    replace=False
                )
                keep_mask[restore_indices] = True

        # Apply mask to all relevant arrays
        # List of keys that should be filtered (if present)
        # V9: Includes 'color' (RGB) and 'normal' from HDF5
        filter_keys = ["coord", "segment", "instance", "color", "normal", "intensity",
                       "strength", "origin_coord", "origin_segment", "index",
                       "rgb", "normals"]  # Alternative key names for compatibility

        for key in filter_keys:
            if key in data_dict and data_dict[key] is not None:
                arr = data_dict[key]
                if isinstance(arr, np.ndarray) and len(arr) == n_points:
                    data_dict[key] = arr[keep_mask]

        return data_dict


# =============================================================================
# FLEXIBLE CLASS CONFIGURATION
# =============================================================================
#
# Two formats are supported for defining class mappings (similar to SemanticKITTI):
#
# 1. learning_map (dict) - SemanticKITTI style:
#    learning_map = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, ...}
#    Keys are original labels, values are target class indices.
#    Use ignore_index (default -1) for classes to ignore.
#
# 2. class_groups (list of lists) - Bhopal style:
#    class_groups = [[0,1,2,3], [4,5,6,7], ...]
#    Index = target class, values = original classes that map to it.
#
# Both formats produce the same internal representation.
# =============================================================================


def get_default_learning_map():
    """
    Get the default 62-to-12 class learning map for Bhopal MLS dataset.

    This is the standard mapping from 62 fine-grained classes to 12 parent classes.
    Similar to SemanticKITTI's get_learning_map() static method.

    Returns:
        dict: Mapping from original label (0-61) to target label (0-11)
    """
    learning_map = {}

    # 0: Ground (original 0-3)
    for i in range(0, 4):
        learning_map[i] = 0

    # 1: Vegetation (original 4-7)
    for i in range(4, 8):
        learning_map[i] = 1

    # 2: Building (original 8-12)
    for i in range(8, 13):
        learning_map[i] = 2

    # 3: Road (original 13)
    learning_map[13] = 3

    # 4: Wire (original 14)
    learning_map[14] = 4

    # 5: Pole and Tower (original 15-18)
    for i in range(15, 19):
        learning_map[i] = 5

    # 6: Wire-Structure Connector (original 19)
    learning_map[19] = 6

    # 7: Vehicle (original 20-34)
    for i in range(20, 35):
        learning_map[i] = 7

    # 8: Board (original 35-37)
    for i in range(35, 38):
        learning_map[i] = 8

    # 9: Fencing (original 38-42)
    for i in range(38, 43):
        learning_map[i] = 9

    # 10: Human and Animal (original 43-49)
    for i in range(43, 50):
        learning_map[i] = 10

    # 11: Road Side Object (original 50-61)
    for i in range(50, 62):
        learning_map[i] = 11

    return learning_map


def get_default_learning_map_inv():
    """
    Get the inverse mapping from 12 target classes to representative original labels.

    Useful for visualization and result submission.

    Returns:
        dict: Mapping from target label to a representative original label
    """
    return {
        0: 0,    # Ground -> original 0
        1: 4,    # Vegetation -> original 4
        2: 8,    # Building -> original 8
        3: 13,   # Road -> original 13
        4: 14,   # Wire -> original 14
        5: 15,   # Pole and Tower -> original 15
        6: 19,   # Wire-Structure Connector -> original 19
        7: 20,   # Vehicle -> original 20
        8: 35,   # Board -> original 35
        9: 38,   # Fencing -> original 38
        10: 43,  # Human and Animal -> original 43
        11: 50,  # Road Side Object -> original 50
    }


def get_default_class_names():
    """
    Get the default 12 class names for Bhopal MLS dataset.

    Returns:
        list: List of 12 class names
    """
    return [
        "Ground",                    # 0
        "Vegetation",                # 1
        "Building",                  # 2
        "Road",                      # 3
        "Wire",                      # 4
        "Pole and Tower",            # 5
        "Wire-Structure Connector",  # 6
        "Vehicle",                   # 7
        "Board",                     # 8
        "Fencing",                   # 9
        "Human and Animal",          # 10
        "Road Side Object",          # 11
    ]


def build_class_mapping_from_groups(class_groups):
    """
    Build a learning_map dict from class_groups format.

    Args:
        class_groups: List of lists, where class_groups[new_label] = [old_label1, old_label2, ...]

    Returns:
        dict: Mapping from original label to new label (learning_map format)

    Example:
        class_groups = [[0,1,2], [3,4,5]]  # 2 classes
        -> {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    """
    mapping = {}
    for new_label, old_labels in enumerate(class_groups):
        for old_label in old_labels:
            mapping[old_label] = new_label
    return mapping


# Legacy format for backward compatibility
DEFAULT_12CLASS_CONFIG = {
    "num_classes": 12,
    "class_names": get_default_class_names(),
    "class_groups": [
        list(range(0, 4)),     # 0: Ground (0-3)
        list(range(4, 8)),     # 1: Vegetation (4-7)
        list(range(8, 13)),    # 2: Building (8-12)
        [13],                  # 3: Road (13)
        [14],                  # 4: Wire (14)
        list(range(15, 19)),   # 5: Pole and Tower (15-18)
        [19],                  # 6: Wire-Structure Connector (19)
        list(range(20, 35)),   # 7: Vehicle (20-34)
        list(range(35, 38)),   # 8: Board (35-37)
        list(range(38, 43)),   # 9: Fencing (38-42)
        list(range(43, 50)),   # 10: Human and Animal (43-49)
        list(range(50, 62)),   # 11: Road Side Object (50-61)
    ],
}

PARENT_CLASSES = {i: name for i, name in enumerate(DEFAULT_12CLASS_CONFIG["class_names"])}


@DATASETS.register_module()
class BhopalSceneHDF5Dataset:
    """
    Bhopal MLS Scene-Level HDF5 Dataset for Semantic Segmentation.

    Compatible with Pointcept/PTv3/Concerto/Sonata frameworks.

    Loads scene-level data from HDF5 created by 19b_create_scene_V7_dataset.py:
    - Path-based geographic splits (train/val/test)
    - Per-point semantic labels (62 classes or remapped to 12)
    - Up to 1M points per scene (adaptive subdivision with max cap)
    - Adaptive segment sizes: 5m → 2.5m → 1.25m → 0.625m
    """

    # ================================================================
    # STATIC METHODS (SemanticKITTI API compatibility)
    # ================================================================

    @staticmethod
    def get_learning_map():
        """
        Get the default 62-to-12 class learning map.

        Returns:
            dict: {original_label: target_label} mapping
        """
        return get_default_learning_map()

    @staticmethod
    def get_learning_map_inv():
        """
        Get the inverse mapping from target to representative original labels.

        Returns:
            dict: {target_label: original_label} mapping
        """
        return get_default_learning_map_inv()

    @staticmethod
    def get_class_names():
        """
        Get the default 12 class names.

        Returns:
            list: List of class names
        """
        return get_default_class_names()

    # ================================================================
    # INITIALIZATION
    # ================================================================

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        num_points: int = None,  # None = use all points (no subsampling)
        transform=None,
        test_mode: bool = False,
        test_cfg=None,
        loop: int = 1,
        use_intensity: bool = True,
        use_gps_time: bool = False,
        use_normals: bool = False,
        use_rgb: bool = False,       # V9: Whether to load RGB from HDF5
        cache_data: bool = False,
        remap_classes: bool = True,  # Default True: remap to target classes
        learning_map: dict = None,   # SemanticKITTI style: {old_label: new_label, ...}
        class_groups: list = None,   # Bhopal style: [[old1, old2], [old3, old4], ...]
        class_names: list = None,    # Names for target classes
        class_mapping: dict = None,  # DEPRECATED: use learning_map instead
        ignore_index: int = -1,
        load_instance: bool = False,  # Instance seg: load global_ids from HDF5
    ):
        """
        Args:
            h5_path: Path to HDF5 file (e.g., bhopal_scene_v7_train_1234_val_56_test_78.h5)
            split: Dataset split ('train', 'val', or 'test')
            num_points: Target number of points per sample. None = use ALL points (no subsampling)
            transform: Pointcept transform pipeline
            test_mode: Whether in test mode
            test_cfg: Test configuration
            loop: Number of times to loop over dataset (for training)
            use_intensity: Whether to use intensity as feature (normalized to [0, 1])
            use_gps_time: Whether to use GPS time as feature (z-score normalized)
            use_normals: Whether to load/compute normals
            use_rgb: Whether to load RGB from HDF5 (V9+) - normalized to [0, 1]
                     ~35-40% coverage (Bhopal_2/8 have 0%), missing RGB = zeros
            cache_data: Whether to cache data in memory
            remap_classes: Whether to remap original classes to target classes
            learning_map: SemanticKITTI style - direct {old_label: new_label} dict
                          Example: {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, ...}
            class_groups: Bhopal style - List of lists defining class combinations
                          Example: [[0,1,2,3], [4,5,6,7], ...] means:
                            - Original classes 0,1,2,3 -> target class 0
                            - Original classes 4,5,6,7 -> target class 1
            class_names: Names for target classes (must match num_classes)
            class_mapping: DEPRECATED - use learning_map instead
            ignore_index: Label to ignore in loss computation (default: -1)
        """
        super().__init__()
        self.h5_path = Path(h5_path)
        self.split = split
        self.num_points = num_points
        self.transform = Compose(transform) if transform is not None else None
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        # Build test-time transforms if in test mode
        # Following official Pointcept DefaultDataset pattern
        if self.test_mode:
            if self.test_cfg:
                # Build transforms from test_cfg (supports both dict and Config object access)
                voxelize_cfg = self.test_cfg.get('voxelize') if hasattr(self.test_cfg, 'get') else getattr(self.test_cfg, 'voxelize', None)
                crop_cfg = self.test_cfg.get('crop') if hasattr(self.test_cfg, 'get') else getattr(self.test_cfg, 'crop', None)
                post_cfg = self.test_cfg.get('post_transform') if hasattr(self.test_cfg, 'get') else getattr(self.test_cfg, 'post_transform', None)
                aug_cfg = self.test_cfg.get('aug_transform') if hasattr(self.test_cfg, 'get') else getattr(self.test_cfg, 'aug_transform', None)

                self.test_voxelize = TRANSFORMS.build(voxelize_cfg) if voxelize_cfg else None
                self.test_crop = TRANSFORMS.build(crop_cfg) if crop_cfg else None
                self.post_transform = Compose(post_cfg) if post_cfg else None
                self.aug_transform = [Compose(aug) for aug in aug_cfg] if aug_cfg else []
            else:
                # test_mode without test_cfg - use defaults
                self.test_voxelize = None
                self.test_crop = None
                self.post_transform = None
                self.aug_transform = []

        self.loop = loop
        self.use_intensity = use_intensity
        self.use_gps_time = use_gps_time
        self.use_normals = use_normals
        self.use_rgb = use_rgb
        self.cache_data = cache_data
        self.remap_classes = remap_classes
        self.ignore_index = ignore_index
        self.load_instance = load_instance

        # ================================================================
        # FLEXIBLE CLASS MAPPING SETUP
        # ================================================================
        # Priority: learning_map > class_groups > class_mapping (deprecated) > default
        self._setup_class_mapping(learning_map, class_groups, class_names, class_mapping)

        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"HDF5 file not found: {self.h5_path}\n"
                f"Please create it using 19b_create_scene_V7_dataset.py at:\n"
                f"  /DATA/pyare/Accessibility/data/Bhopal/19b_create_scene_V7_dataset.py"
            )

        # Load dataset info and indices
        self._load_dataset_info()

        # Print configuration
        features = ["XYZ"]
        if use_rgb:
            features.append("RGB(normalized)")
        if use_intensity:
            features.append("intensity(normalized)")
        if use_gps_time:
            features.append("gps_time(normalized)")
        if use_normals:
            features.append("normals")

        print(f"Loaded Bhopal Scene HDF5 dataset for split '{split}':")
        print(f"  - Scenes: {len(self.indices):,}")
        if self.num_points is None:
            print(f"  - Points: ALL (no subsampling)")
        else:
            print(f"  - Target points: {self.num_points:,}")
        print(f"  - Max points/scene: {self.max_points_per_scene:,}")
        print(f"  - Classes: {self.num_classes} ({self._class_config_source})")
        print(f"  - Class names: {self.target_class_names}")
        print(f"  - Features: {' + '.join(features)}")
        print(f"  - Total channels: {self.get_num_channels()}")
        print(f"  - Ignore index: {self.ignore_index}")
        if self.load_instance:
            print(f"  - Instance loading: ENABLED (from scenes/global_ids)")

    def _setup_class_mapping(self, learning_map, class_groups, class_names, class_mapping):
        """
        Setup class mapping from various input formats.

        Priority:
        1. learning_map (SemanticKITTI style dict) - most compatible
        2. class_groups (Bhopal style list of lists)
        3. class_mapping (deprecated, same as learning_map)
        4. Default 12-class configuration
        """
        if not self.remap_classes:
            # No remapping - use original 62 classes
            self.class_mapping = None
            self.num_classes = 62
            self.target_class_names = [f"Class_{i}" for i in range(62)]
            self._class_config_source = "original 62 fine-grained"
            return

        if learning_map is not None:
            # SemanticKITTI style: direct {old_label: new_label} dict
            self.class_mapping = learning_map
            # Count unique target classes (excluding ignore_index if present)
            target_values = [v for v in learning_map.values() if v != self.ignore_index]
            self.num_classes = len(set(target_values))

            if class_names is not None:
                if len(class_names) != self.num_classes:
                    raise ValueError(
                        f"class_names length ({len(class_names)}) must match "
                        f"num_classes ({self.num_classes})"
                    )
                self.target_class_names = class_names
            else:
                self.target_class_names = [f"Class_{i}" for i in range(self.num_classes)]

            self._class_config_source = f"custom {self.num_classes}-class learning_map"

        elif class_groups is not None:
            # Bhopal style: list of lists
            self.class_mapping = build_class_mapping_from_groups(class_groups)
            self.num_classes = len(class_groups)

            if class_names is not None:
                if len(class_names) != len(class_groups):
                    raise ValueError(
                        f"class_names length ({len(class_names)}) must match "
                        f"class_groups length ({len(class_groups)})"
                    )
                self.target_class_names = class_names
            else:
                self.target_class_names = [f"Class_{i}" for i in range(self.num_classes)]

            self._class_config_source = f"custom {self.num_classes}-class groups"

        elif class_mapping is not None:
            # DEPRECATED: same as learning_map
            self.class_mapping = class_mapping
            target_values = [v for v in class_mapping.values() if v != self.ignore_index]
            self.num_classes = len(set(target_values))
            self.target_class_names = [f"Class_{i}" for i in range(self.num_classes)]
            self._class_config_source = f"custom {self.num_classes}-class mapping (deprecated)"

        else:
            # Default: 12-class Bhopal configuration
            self.class_mapping = build_class_mapping_from_groups(
                DEFAULT_12CLASS_CONFIG["class_groups"]
            )
            self.num_classes = DEFAULT_12CLASS_CONFIG["num_classes"]
            self.target_class_names = DEFAULT_12CLASS_CONFIG["class_names"]
            self._class_config_source = "default 12-class Bhopal"

        # Build label_to_name for compatibility
        self.label_to_name = {i: name for i, name in enumerate(self.target_class_names)}

    def _load_dataset_info(self):
        """Load dataset information and split indices from HDF5."""
        with h5py.File(self.h5_path, 'r') as h5f:
            # Support both old naming (train/val/test) and new naming (train_indices/val_indices/test_indices)
            def get_split_key(name):
                return f'splits/{name}_indices' if f'splits/{name}_indices' in h5f else f'splits/{name}'

            # Load split indices
            if self.split == 'train':
                indices = h5f[get_split_key('train')][:]
            elif self.split == 'val':
                indices = h5f[get_split_key('val')][:]
            elif self.split == 'test':
                test_key = get_split_key('test')
                if test_key not in h5f:
                    raise ValueError("Test split not found in HDF5 file")
                indices = h5f[test_key][:]
            else:
                raise ValueError(f"Invalid split: {self.split}")

            # Filter out scenes with too few points (prevents empty batch errors)
            MIN_POINTS_PER_SCENE = 1000  # Minimum points to be usable
            if 'scenes/num_points' in h5f:
                num_points_all = h5f['scenes/num_points'][:]
                num_points_split = num_points_all[indices]
                valid_mask = num_points_split >= MIN_POINTS_PER_SCENE
                n_filtered = (~valid_mask).sum()
                if n_filtered > 0:
                    print(f"  WARNING: Filtered {n_filtered} scenes with <{MIN_POINTS_PER_SCENE} points")
                self.indices = indices[valid_mask]
            else:
                self.indices = indices

            # Load dataset metadata (for reference, not used for class mapping)
            # V7 default: 1,000,000 (was 15M in V2)
            self.max_points_per_scene = h5f.attrs.get('max_points_per_scene', h5f.attrs.get('max_points', 1000000))
            self.num_classes_original = h5f.attrs.get('num_classes', 62)

            # Note: num_classes and label_to_name are set in _setup_class_mapping()
            # We keep class_names as alias for target_class_names for compatibility
            self.class_names = self.target_class_names

    def get_num_channels(self):
        """Get total number of feature channels (excluding XYZ coord)."""
        channels = 0
        if self.use_rgb:
            channels += 3
        if self.use_intensity:
            channels += 1
        if self.use_gps_time:
            channels += 1
        if self.use_normals:
            channels += 3
        return channels

    def __len__(self):
        """Return dataset size."""
        return len(self.indices) * self.loop

    def _normalize_intensity(self, intensity):
        """Normalize intensity from [0, 65535] to [0, 1]."""
        return intensity / 65535.0

    def _normalize_gps_time(self, gps_time, valid_mask):
        """Normalize GPS time using z-score (only on valid points)."""
        valid_gps = gps_time[valid_mask]
        if len(valid_gps) == 0:
            return gps_time

        mean = valid_gps.mean()
        std = valid_gps.std()
        if std < 1e-6:
            return gps_time - mean

        normalized = (gps_time - mean) / std
        # Set padding points to 0
        normalized[~valid_mask] = 0
        return normalized

    def _remap_labels(self, labels):
        """
        Remap 62 fine-grained labels to 12 parent classes.

        Args:
            labels: Per-point labels array (62-class, -1=padding)

        Returns:
            Remapped labels (12-class, -1=padding preserved)
        """
        remapped = labels.copy()
        # Remap valid labels (padding -1 won't match any key, stays as -1)
        for old_label, new_label in self.class_mapping.items():
            remapped[labels == old_label] = new_label
        return remapped

    def get_data(self, idx):
        """
        Load raw data for a scene (without transforms).

        Args:
            idx: Scene index (after handling looping)

        Returns:
            data_dict: Dictionary with raw scene data
        """
        # Handle looping
        idx = idx % len(self.indices)
        global_idx = self.indices[idx]

        with h5py.File(self.h5_path, 'r') as h5f:
            # Load actual point count
            num_pts = int(h5f['scenes/num_points'][global_idx])

            # Load coordinates (already scene-centered in HDF5)
            coord = h5f['scenes/points'][global_idx, :num_pts].astype(np.float32)

            # Load intensity
            intensity_raw = h5f['scenes/intensity'][global_idx, :num_pts].astype(np.float32)

            # Load GPS time if needed
            if self.use_gps_time and 'scenes/gps_time' in h5f:
                gps_time_raw = h5f['scenes/gps_time'][global_idx, :num_pts].astype(np.float32)
            else:
                gps_time_raw = None

            # V9: Load RGB if needed (from scenes/rgb dataset)
            if self.use_rgb and 'scenes/rgb' in h5f:
                rgb_raw = h5f['scenes/rgb'][global_idx, :num_pts].astype(np.float32)
            else:
                rgb_raw = None

            # V8: Load normals if needed (from scenes/normals dataset)
            if self.use_normals and 'scenes/normals' in h5f:
                normals = h5f['scenes/normals'][global_idx, :num_pts].astype(np.float32)
            else:
                normals = None

            # Load per-point labels (CRITICAL for semantic segmentation!)
            segment = h5f['scenes/labels'][global_idx, :num_pts].astype(np.int64)

            # Load per-point instance IDs if needed (for instance segmentation)
            if self.load_instance and 'scenes/global_ids' in h5f:
                instance = h5f['scenes/global_ids'][global_idx, :num_pts].astype(np.int32)
            else:
                instance = None

            # Load scene name
            scene_name = h5f['metadata/scene_names'][global_idx]
            if isinstance(scene_name, bytes):
                scene_name = scene_name.decode('utf-8')

            # Load bhopal path (for debugging)
            bhopal_path_key = 'metadata/bhopal_paths' if 'metadata/bhopal_paths' in h5f else 'metadata/source_path'
            bhopal_path = int(h5f[bhopal_path_key][global_idx])

        # ================================================================
        # PREPROCESSING
        # ================================================================

        # 1. Remap 62 classes to 12 parent classes if enabled
        if self.remap_classes:
            segment = self._remap_labels(segment)

        # 2. Normalize intensity to [0, 1]
        intensity = self._normalize_intensity(intensity_raw)

        # 3. Normalize GPS time if used
        if self.use_gps_time and gps_time_raw is not None:
            valid_mask = segment >= 0  # Use segment to identify valid points
            gps_time = self._normalize_gps_time(gps_time_raw, valid_mask)
        else:
            gps_time = None

        # 4. Normalize RGB to [0, 1] if used (V9: from uint8 0-255)
        if self.use_rgb and rgb_raw is not None:
            rgb = rgb_raw / 255.0  # [N, 3] in range [0, 1]
        else:
            rgb = None

        # 5. Normals are already unit vectors (from HDF5), no normalization needed

        # ================================================================
        # POINT SAMPLING (optional - skip if num_points is None)
        # ================================================================

        if self.num_points is not None:
            # Subsample to fixed number of points
            if num_pts > self.num_points:
                # Random sampling without replacement
                sample_indices = np.random.choice(num_pts, self.num_points, replace=False)
            elif num_pts < self.num_points:
                # Random sampling with replacement (upsample)
                sample_indices = np.random.choice(num_pts, self.num_points, replace=True)
            else:
                sample_indices = np.arange(num_pts)

            coord = coord[sample_indices]
            segment = segment[sample_indices]
            intensity = intensity[sample_indices]

            if gps_time is not None:
                gps_time = gps_time[sample_indices]

            if rgb is not None:
                rgb = rgb[sample_indices]

            if normals is not None:
                normals = normals[sample_indices]

            if instance is not None:
                instance = instance[sample_indices]
        # else: use ALL points (no subsampling)

        # ================================================================
        # BUILD FEATURE VECTOR
        # ================================================================

        feat_list = []
        if self.use_intensity:
            feat_list.append(intensity[:, None])  # [N, 1]
        if gps_time is not None:
            feat_list.append(gps_time[:, None])   # [N, 1]

        if len(feat_list) > 0:
            feat = np.concatenate(feat_list, axis=1).astype(np.float32)
        else:
            feat = np.zeros((len(coord), 0), dtype=np.float32)

        # ================================================================
        # BUILD DATA DICT
        # ================================================================

        data_dict = {
            'coord': coord,                               # [N, 3] - XYZ
            'feat': feat,                                 # [N, C] - features
            'segment': segment,                           # [N] - per-point labels
            'name': f"{scene_name}_bhopal{bhopal_path}",  # Scene identifier
        }

        # For Pointcept compatibility (Collect transform uses these keys)
        if self.use_intensity:
            data_dict['strength'] = intensity[:, None]  # [N, 1] - 'strength' is Pointcept convention for LiDAR intensity

        # V9: Add RGB as 'color' for Pointcept compatibility
        # IMPORTANT: Always add 'color' key when use_rgb=True to ensure ZeroColor transform works
        # If RGB data is missing from HDF5, default to zeros (geometry-only mode)
        if self.use_rgb:
            if rgb is not None:
                data_dict['color'] = rgb.astype(np.float32)  # [N, 3] - already normalized to [0, 1]
            else:
                data_dict['color'] = np.zeros((len(coord), 3), dtype=np.float32)  # Zeros if missing

        # V8: Add normals as 'normal' for Pointcept compatibility
        # IMPORTANT: Always add 'normal' key when use_normals=True to ensure ZeroNormal transform works
        # If normals data is missing from HDF5, default to zeros (geometry-only mode)
        if self.use_normals:
            if normals is not None:
                data_dict['normal'] = normals.astype(np.float32)  # [N, 3] - unit vectors
            else:
                data_dict['normal'] = np.zeros((len(coord), 3), dtype=np.float32)  # Zeros if missing

        # Add grid_size for PTv3 serialization
        data_dict['grid_size'] = 0.02

        # Instance segmentation: add instance IDs
        if instance is not None:
            data_dict['instance'] = instance

        return data_dict

    def get_data_name(self, idx):
        """Get scene name for a given index."""
        idx = idx % len(self.indices)
        global_idx = self.indices[idx]

        with h5py.File(self.h5_path, 'r') as h5f:
            scene_name = h5f['metadata/scene_names'][global_idx]
            if isinstance(scene_name, bytes):
                scene_name = scene_name.decode('utf-8')
            bhopal_path_key = 'metadata/bhopal_paths' if 'metadata/bhopal_paths' in h5f else 'metadata/source_path'
            bhopal_path = int(h5f[bhopal_path_key][global_idx])
        return f"{scene_name}_bhopal{bhopal_path}"

    def prepare_train_data(self, idx):
        """Prepare data for training mode."""
        data_dict = self.get_data(idx)
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        """
        Prepare data for test mode with fragment-based inference.

        This follows the Pointcept DefaultDataset pattern for precise evaluation:
        1. Load raw data and apply base transforms
        2. Create augmented copies (rotations)
        3. Voxelize each augmented copy into fragments
        4. Return fragment_list for inference aggregation
        """
        from copy import deepcopy

        # Load data and apply base transform
        data_dict = self.get_data(idx)
        if self.transform is not None:
            data_dict = self.transform(data_dict)

        # Extract segment and name for result dict
        result_dict = dict(
            segment=data_dict.pop("segment"),
            name=data_dict.pop("name")
        )

        # Handle instance data for instance segmentation evaluation
        if "instance" in data_dict:
            result_dict["instance"] = data_dict.pop("instance")

        # Handle origin_segment for proper evaluation on original points
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict, "inverse mapping required with origin_segment"
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        # Create augmented copies using test-time augmentation transforms
        data_dict_list = []
        if self.aug_transform:
            for aug in self.aug_transform:
                data_dict_list.append(aug(deepcopy(data_dict)))
        else:
            # No augmentation - use data as-is
            data_dict_list.append(data_dict)

        # Create fragments through voxelization
        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                # No voxelization - add index and use whole point cloud
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]

            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        # Apply post-transform to each fragment
        if self.post_transform is not None:
            for i in range(len(fragment_list)):
                fragment_list[i] = self.post_transform(fragment_list[i])

        result_dict["fragment_list"] = fragment_list
        return result_dict

    def __getitem__(self, idx):
        """
        Load and preprocess a scene.

        Returns:
            - In train mode: data_dict with coord, feat, segment, name
            - In test mode: result_dict with segment, name, fragment_list
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def get_class_name(self, label):
        """Get class name from label."""
        return self.label_to_name.get(label, f"Unknown_{label}")

    def get_scene_info(self, idx):
        """Get metadata for a scene."""
        idx = idx % len(self.indices)
        global_idx = self.indices[idx]

        with h5py.File(self.h5_path, 'r') as h5f:
            scene_name = h5f['metadata/scene_names'][global_idx]
            if isinstance(scene_name, bytes):
                scene_name = scene_name.decode('utf-8')

            bhopal_path_key = 'metadata/bhopal_paths' if 'metadata/bhopal_paths' in h5f else 'metadata/source_path'

            info = {
                'scene_name': scene_name,
                'bhopal_path': int(h5f[bhopal_path_key][global_idx]),
                'num_points': int(h5f['scenes/num_points'][global_idx]),
            }
            # Optional: num_instances may not exist in all HDF5 files
            if 'metadata/num_instances' in h5f:
                info['num_instances'] = int(h5f['metadata/num_instances'][global_idx])
            return info


# ================================================================
# VALIDATION FUNCTION
# ================================================================

def validate_scene_dataset(h5_path, split='train', num_samples=3):
    """
    Validate that scene dataset is loaded correctly.

    Checks:
    1. Intensity is normalized to [0, 1]
    2. Per-point labels are valid
    3. Class remapping works correctly
    """
    print("=" * 80)
    print("SCENE DATASET VALIDATION")
    print("=" * 80)

    dataset = BhopalSceneHDF5Dataset(
        h5_path=h5_path,
        split=split,
        num_points=20000,
        use_intensity=True,
        use_gps_time=False,
        remap_classes=True,  # Test 62->12 remapping
    )

    print(f"\nDataset info:")
    print(f"  Total scenes: {len(dataset)}")
    print(f"  Num classes: {dataset.num_classes}")
    print(f"  Num channels: {dataset.get_num_channels()}")

    print(f"\nValidating {num_samples} random samples...")

    for i in range(num_samples):
        idx = np.random.randint(len(dataset))
        data = dataset[idx]

        # Check intensity normalization
        if 'intensity' in data:
            intensity = data['intensity']
            assert intensity.min() >= 0.0, f"Intensity min < 0: {intensity.min()}"
            assert intensity.max() <= 1.0, f"Intensity max > 1: {intensity.max()}"

        # Check labels are in valid range (0-11 for 12-class, or -1 for ignore)
        segment = data['segment']
        valid_labels = segment[segment >= 0]
        if len(valid_labels) > 0:
            assert valid_labels.max() < dataset.num_classes, \
                f"Label {valid_labels.max()} >= num_classes {dataset.num_classes}"

        # Check unique classes
        unique_classes = np.unique(segment[segment >= 0])

        print(f"  Sample {i+1}: {data['name']}")
        print(f"    Coord shape: {data['coord'].shape}")
        print(f"    Feat shape: {data['feat'].shape}")
        print(f"    Segment shape: {segment.shape}")
        print(f"    Unique classes: {len(unique_classes)}")
        print(f"    Intensity range: [{intensity.min():.3f}, {intensity.max():.3f}]")
        print(f"    ✓ Passed")

    print("\n" + "=" * 80)
    print("ALL VALIDATION TESTS PASSED!")
    print("=" * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', type=str, required=True, help='Path to scene HDF5 file')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    args = parser.parse_args()

    validate_scene_dataset(args.h5, args.split)
