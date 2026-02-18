"""Quick test that UCB and EI acquisition types work in GPSimplex."""
from src.utils.gp_simplex import GPSimplex
from src.utils.datahandler import DataHandler
import torch

# Use one device for everything (GPU if available) to avoid cuda/cpu mismatch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test_acquisition_type_param():
    dh = DataHandler(directory=None, device=DEVICE, d=3)
    # DataHandler needs bounds for _compute_repulsion_lambda; set via save_init
    X = torch.tensor([[0.3, 0.3, 0.4], [0.5, 0.3, 0.2]], dtype=torch.float64, device=DEVICE)
    Y = torch.tensor([[1.0], [1.2]], dtype=torch.float64, device=DEVICE)
    bounds = torch.zeros(2, 3, dtype=torch.float64, device=DEVICE)
    bounds[0] = 0.0
    bounds[1] = 1.0
    dh.save_init(X, X, Y, bounds)
    # Default is UCB
    gp_ucb = GPSimplex(data_handler=dh, acquisition_type="ucb", ucb_beta=0.1, device=DEVICE)
    assert gp_ucb.acquisition_type == "ucb"
    assert gp_ucb.ucb_beta == 0.1
    # EI
    gp_ei = GPSimplex(data_handler=dh, acquisition_type="ei", device=DEVICE)
    assert gp_ei.acquisition_type == "ei"
    # Fit minimal data and create acquisition (X, Y already on DEVICE)
    gp_ucb.fit(X, Y)
    acq_ucb = gp_ucb.create_acquisition()
    assert acq_ucb is not None
    gp_ei.fit(X, Y)
    acq_ei = gp_ei.create_acquisition(best_f=1.2)
    assert acq_ei is not None
    # compute_log_ei_at_point: UCB returns acq value, EI returns log EI
    x = torch.tensor([0.4, 0.35, 0.25], dtype=torch.float64, device=DEVICE)
    val_ucb = gp_ucb.compute_log_ei_at_point(x, best_f=1.2)
    val_ei = gp_ei.compute_log_ei_at_point(x, best_f=1.2)
    assert isinstance(val_ucb, float)
    assert isinstance(val_ei, float)
    print("test_acquisition_type_param passed")

if __name__ == "__main__":
    test_acquisition_type_param()
