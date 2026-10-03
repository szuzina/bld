"""
Controlled generation of segmentation errors.

The `CreateSegmentationError` class creates reproducible, physically scaled perturbations of binary
2D segmentation masks.  Distances are specified in millimetres, so the generated error is independent of voxel size.

The class is intended for controlled segmentation-error experiments, e.g. relating a known contour perturbation to
segmentation metrics and downstream radiotherapy dose metrics.
"""

from .segmentation_error import SegmentationError
from bld.segmentation_errors.create_segmentation_error import CreateSegmentationError
from .segmentation_error_patient import SegmentationErrorPatient
from .generate_series import GenerateSeries
from .create_visualization import CreateVisualization
from .error_metrics_evaluator import ErrorMetricsEvaluator
from .error_analyzer import ErrorMetricsAnalyzer
