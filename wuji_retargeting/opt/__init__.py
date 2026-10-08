"""Optimizers for hand retargeting.

AdaptiveOptimizerAnalytical - Recommended optimizer using Huber loss + analytical gradients + NLopt SLSQP.
Uses adaptive blending between TipDirVec and FullHandVec based on pinch distance.

VectorOptimizer - Generic key-vector matching optimizer. Minimizes distances between
configurable (origin_link -> task_link) robot vectors and corresponding MediaPipe
keypoint vectors. Configurable via `retarget.key_vectors` in YAML.

All parameters are read from YAML configuration files.
"""

from .base import (
    BaseOptimizer,
    LPFilter,
    TimingStats,
    M_TO_CM,
    CM_TO_M,
)
from .adaptive_analytical import AdaptiveOptimizerAnalytical
from .l20_retarget_v2 import L20RetargetV2
from .l20_feature_retarget import L20FeatureRetargeter
from .l20_semantic_v4 import L20SemanticV4
from .l20_command_retarget import L20CommandRetargeter
from .l20_multi_pinch_retarget import L20MultiPinchRetargeter
from .vector import VectorOptimizer


__all__ = [
    "BaseOptimizer",
    "AdaptiveOptimizerAnalytical",
    "L20RetargetV2",
    "L20FeatureRetargeter",
    "L20SemanticV4",
    "L20CommandRetargeter",
    "L20MultiPinchRetargeter",
    "VectorOptimizer",
    "LPFilter",
    "TimingStats",
    "M_TO_CM",
    "CM_TO_M",
]
