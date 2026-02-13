"""
Data classes for ZoMBI-Hop configuration and state.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
import torch


@dataclass
class ZoMBIHopConfig:
    """
    Configuration for ZoMBI-Hop optimization.

    Contains all hyperparameters and settings for the optimization algorithm.
    Parameters set to None will be auto-computed during optimization.
    """

    # Core optimization parameters
    max_zooms: int = 3
    max_iterations: int = 10
    top_m_points: Optional[int] = None  # Auto-computed as max(d + 1, 4) if None
    n_restarts: int = 30
    raw: int = 500

    # Penalization parameters
    penalization_threshold: float = 1e-3
    penalty_num_directions: Optional[int] = None  # Auto-computed as 10 * d if None
    penalty_max_radius: float = 0.3
    penalty_radius_step: Optional[float] = None  # Auto-computed from input noise if None

    # Convergence parameters (PI + noise-based Y/X thresholds)
    convergence_pi_threshold: float = 0.01
    input_noise_threshold_mult: float = 2.0
    output_noise_threshold_mult: float = 2.0

    # GP parameters
    max_gp_points: int = 3000

    # Acquisition parameters
    repulsion_lambda: Optional[float] = None  # Auto-computed dynamically if None

    # Device and dtype
    device: str = 'cuda'
    dtype: str = 'float64'

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ZoMBIHopConfig':
        """Create config from dictionary."""
        # Filter to only include known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)

    def get_torch_dtype(self) -> torch.dtype:
        """Get torch dtype from string representation."""
        dtype_map = {
            'float32': torch.float32,
            'float64': torch.float64,
            'float16': torch.float16,
        }
        return dtype_map.get(self.dtype, torch.float64)

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.max_zooms > 0, "max_zooms must be positive"
        assert self.max_iterations > 0, "max_iterations must be positive"
        # top_m_points can be None (auto-computed) or positive
        assert self.top_m_points is None or self.top_m_points > 0, "top_m_points must be None or positive"
        assert self.n_restarts > 0, "n_restarts must be positive"
        assert self.raw > 0, "raw must be positive"
        assert self.penalization_threshold > 0, "penalization_threshold must be positive"
        # penalty_num_directions can be None (auto-computed) or positive
        assert self.penalty_num_directions is None or self.penalty_num_directions > 0, "penalty_num_directions must be None or positive"
        assert self.penalty_max_radius > 0, "penalty_max_radius must be positive"
        # penalty_radius_step can be None (auto-computed from input noise) or positive
        assert self.penalty_radius_step is None or self.penalty_radius_step > 0, "penalty_radius_step must be None or positive"
        assert 0 <= self.convergence_pi_threshold <= 1, "convergence_pi_threshold must be in [0, 1]"
        assert self.input_noise_threshold_mult > 0, "input_noise_threshold_mult must be positive"
        assert self.output_noise_threshold_mult > 0, "output_noise_threshold_mult must be positive"
        assert self.max_gp_points > 0, "max_gp_points must be positive"
        # repulsion_lambda can be None (auto-computed) or positive
        assert self.repulsion_lambda is None or self.repulsion_lambda > 0, "repulsion_lambda must be None or positive"


@dataclass
class Checkpoint:
    """
    Checkpoint metadata (legacy - kept for compatibility).

    Note: This is a placeholder. Actual checkpoint data is stored
    in DataHandler and saved to disk as separate files.
    """
    run_uuid: str
    d: int
    max_zooms: int
    max_iterations: int
    top_m_points: Optional[int]  # Can be None (auto-computed)
    n_restarts: int
    raw: int
    penalization_threshold: float
    penalty_num_directions: Optional[int]  # Can be None (auto-computed)
    penalty_max_radius: float
    penalty_radius_step: Optional[float]  # Can be None (auto-computed)
    convergence_pi_threshold: float
    input_noise_threshold_mult: float
    output_noise_threshold_mult: float
    max_gp_points: int
    device: str
    dtype: str
    timestamp: Optional[str] = None
    version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
