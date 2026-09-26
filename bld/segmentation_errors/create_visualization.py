from __future__ import annotations
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from bld.segmentation_errors import SegmentationError


class CreateVisualization:
    """Provides 2D visualization utilities for ground truth and modified segmentation masks."""

    def __init__(self, segmentations: Union[SegmentationError, Dict[str, SegmentationError]],
                 original_mask: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        segmentations : SegmentationError or Dict[str, SegmentationError]
            A single error result instance or a dictionary of series results.
        original_mask : np.ndarray, optional
            The reference ground truth binary mask (2D). If omitted, can be passed
            directly to visualization methods.
        """
        self.segmentations = segmentations
        self.original_mask = original_mask

    def show_comparison(self, result: Optional[SegmentationError] = None,
                        reference_mask: Optional[np.ndarray] = None,
                        figsize: Tuple[int, int] = (14, 5)) -> None:
        """Plot a 3-panel comparison: Reference, Perturbed mask, and Difference Map.

        Difference map color code:
        - Green: True Positive (Overlap)
        - Red: False Positive (Spurious expansion/addition)
        - Blue: False Negative (Missing/eroded region)
        """
        res = result or (
            self.segmentations
            if isinstance(self.segmentations, SegmentationError)
            else next(iter(self.segmentations.values()))
        )
        ref = reference_mask if reference_mask is not None else self.original_mask

        if ref is None:
            raise ValueError(
                "A reference (original) mask must be provided to show comparisons."
            )

        perturbed = res.result_mask

        # Build RGB difference overlay:
        # Green = TP, Red = FP, Blue = FN
        tp = ref & perturbed
        fp = perturbed & ~ref
        fn = ref & ~perturbed

        diff_map = np.zeros((*ref.shape, 3), dtype=np.uint8)
        diff_map[tp] = [46, 204, 113]  # Green
        diff_map[fp] = [231, 76, 60]   # Red
        diff_map[fn] = [52, 152, 219]  # Blue

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].imshow(ref, cmap="gray")
        axes[0].set_title("Ground Truth Mask")
        axes[0].axis("off")

        axes[1].imshow(perturbed, cmap="gray")
        axes[1].set_title("Modified Mask")
        axes[1].axis("off")

        axes[2].imshow(diff_map)
        axes[2].set_title("Diff: Green=TP, Red=FP, Blue=FN")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    def show_contour_overlay(self, result: Optional[SegmentationError] = None,
                             reference_mask: Optional[np.ndarray] = None,
                             figsize: Tuple[int, int] = (7, 7)) -> None:
        """Overlay boundaries of both masks on a single axes."""
        res = result or (
            self.segmentations
            if isinstance(self.segmentations, SegmentationError)
            else next(iter(self.segmentations.values()))
        )
        ref = reference_mask if reference_mask is not None else self.original_mask

        fig, ax = plt.subplots(figsize=figsize)
        if ref is not None:
            ax.result_mask(ref, levels=[0.5], colors=["lime"], linewidths=2)

        ax.result_mask(res.result_mask, levels=[0.5], colors=["red"], linestyles="dashed", linewidths=2)

        # Proxy lines for legend
        ax.plot([], [], color="lime", label="Reference")
        ax.plot([], [], color="red", linestyle="--", label="Perturbed")
        ax.legend(loc="upper right")
        ax.axis("off")
        ax.set_title("Contour Boundary Comparison")

        plt.tight_layout()
        plt.show()
