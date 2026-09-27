from __future__ import annotations
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from bld.segmentation_errors import SegmentationError


class CreateVisualization:
    """Provides 2D visualization utilities for ground truth and modified segmentation masks."""

    def __init__(self, segmentations: Union[SegmentationError, Dict[str, SegmentationError]],
                 original_mask: np.ndarray):
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
        """Plot a 3-panel comparison: Reference, Perturbed mask, and Difference Map."""
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

        # 1. Ensure 2D shapes and cast explicitly to boolean
        ref_b = np.squeeze(ref) > 0
        pert_b = np.squeeze(res.result_mask) > 0

        if ref_b.shape != pert_b.shape:
            raise ValueError(f"Shape mismatch: reference {ref_b.shape} vs perturbed {pert_b.shape}")

        # 2. Compute boolean TP, FP, FN regions
        tp = ref_b & pert_b
        fp = pert_b & (~ref_b)
        fn = ref_b & (~pert_b)

        # 3. Create RGB difference map
        diff_map = np.zeros((*ref_b.shape, 3), dtype=np.uint8)
        diff_map[tp] = [46, 204, 113]  # Green: True Positive
        diff_map[fp] = [231, 76, 60]  # Red: False Positive
        diff_map[fn] = [52, 152, 219]  # Blue: False Negative

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].imshow(ref_b, cmap="gray")
        axes[0].set_title("Ground Truth Mask")
        axes[0].axis("off")

        axes[1].imshow(pert_b, cmap="gray")
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
            ax.contour(ref, levels=[0.5], colors=["lime"], linewidths=2)

        ax.contour(res.result_mask, levels=[0.5], colors=["red"], linestyles="dashed", linewidths=2)

        # Proxy lines for legend
        ax.plot([], [], color="lime", label="Reference")
        ax.plot([], [], color="red", linestyle="--", label="Perturbed")
        ax.legend(loc="upper right")
        ax.axis("off")
        ax.set_title("Contour Boundary Comparison")

        plt.tight_layout()
        plt.show()
