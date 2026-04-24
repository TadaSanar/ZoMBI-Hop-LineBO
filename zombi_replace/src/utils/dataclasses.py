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

    # Convergence parameters (PI + noise-based Y/X thresholds)
    convergence_pi_threshold: float = 0.01
    input_noise_threshold_mult: float = 2.0
    output_noise_threshold_mult: float = 2.0
    n_consecutive_converged: int = 2  # Require this many consecutive converged iterations before declaring needle

    # GP parameters
    max_gp_points: int = 3000

    # Acquisition parameters
    repulsion_lambda: Optional[float] = None  # Auto-computed dynamically if None
    acquisition_type: str = "ucb"  # "ucb" or "ei"; both use repulsion
    ucb_beta: float = 0.1  # Exploration weight for UCB (only when acquisition_type=="ucb")
    nat_grad_step: float = 0.02  # Natural-gradient ascent step on the simplex
    nat_grad_max_steps: int = 50  # Max ascent steps per acquisition restart

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
        assert 0 <= self.convergence_pi_threshold <= 1, "convergence_pi_threshold must be in [0, 1]"
        assert self.input_noise_threshold_mult > 0, "input_noise_threshold_mult must be positive"
        assert self.output_noise_threshold_mult > 0, "output_noise_threshold_mult must be positive"
        assert self.n_consecutive_converged >= 1, "n_consecutive_converged must be >= 1"
        assert self.max_gp_points > 0, "max_gp_points must be positive"
        # repulsion_lambda can be None (auto-computed) or positive
        assert self.repulsion_lambda is None or self.repulsion_lambda > 0, "repulsion_lambda must be None or positive"
        assert self.nat_grad_step > 0, "nat_grad_step must be positive"
        assert self.nat_grad_max_steps >= 1, "nat_grad_max_steps must be >= 1"


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
    convergence_pi_threshold: float
    input_noise_threshold_mult: float
    output_noise_threshold_mult: float
    n_consecutive_converged: int = 2
    max_gp_points: int = 3000
    device: str = 'cuda'
    dtype: str = 'float64'
    timestamp: Optional[str] = None
    version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
