from __future__ import annotations
from typing import Iterable, Any, Dict

from bld.segmentation_errors import SegmentationErrorPatient


class GenerateSeries:
    """Generate a reproducible series of errors.
    Results: a dictionary keyed as "<error_type>_<magnitude>mm", values: SegmentationError class objects."""

    def __init__(self, error_types: Iterable[str], magnitudes_mm: Iterable[float]):
        self.error_types = error_types
        self.magnitudes_mm = magnitudes_mm

        self.results = self.generate_series()

    def generate_series(self, **kwargs: Any) -> Dict[str, SegmentationErrorPatient]:
        results = {}
        for error in self.error_types:
            for magnitude in self.magnitudes_mm:
                result = SegmentationErrorPatient(
                    error_type=error,
                    magnitude_mm=magnitude,
                    **kwargs
                )
                key = f"{error}_{self._format_magnitude(magnitude)}mm"
                results[key] = result
        return results

    @staticmethod
    def _format_magnitude(value: float) -> str:
        return f"{float(value):g}"
