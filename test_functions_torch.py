import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Slider
# import ternary

class MultiMinimaAckley:
    def __init__(self, minima_locations, amplitudes=None, sharpness=None,
                 offsets=None, global_scale=20.0, exp_scale=0.2):
        """
        Create a multi-minima function based on modified Ackley functions.

        Args:
            minima_locations: torch.Tensor of shape (num_minima, d) - locations of minima
            amplitudes: torch.Tensor of shape (num_minima,) - depth of each minimum
            sharpness: torch.Tensor of shape (num_minima,) - sharpness of each minimum
            offsets: torch.Tensor of shape (num_minima,) - vertical offset for each minimum
            global_scale: float - global scaling parameter (default 20.0)
            exp_scale: float - exponential scaling parameter (default 0.2)
        """
        self.minima_locations = minima_locations.clone()
        self.num_minima = minima_locations.shape[0]
        self.d = minima_locations.shape[1]

        # Default parameters if not provided
        if amplitudes is None:
            self.amplitudes = torch.ones(self.num_minima)
        else:
            self.amplitudes = amplitudes.clone()

        if sharpness is None:
            self.sharpness = torch.ones(self.num_minima) * 5.0
        else:
            self.sharpness = sharpness.clone()

        if offsets is None:
            self.offsets = torch.zeros(self.num_minima)
        else:
            self.offsets = offsets.clone()

        self.global_scale = global_scale
        self.exp_scale = exp_scale

    def single_ackley(self, x, center, amplitude, sharp, offset):
        """
        Compute a single modified Ackley function centered at 'center'.
        Modified to work well in [0,1] space and avoid numerical issues.
        """
        # Compute distance from center
        diff = x - center.unsqueeze(0)  # (n, d)

        # Modified Ackley with bounded terms
        # Term 1: RMS distance term with careful scaling
        term1 = -self.global_scale * torch.exp(
            -self.exp_scale * torch.sqrt(
                torch.clamp(torch.mean(diff**2, dim=-1), min=1e-8)
            ) * sharp
        )

        # Term 2: Cosine term with damping to avoid oscillations
        # Scale down the cosine term in [0,1] space
        cos_scale = 2.0 * np.pi * sharp / 5.0  # Reduced frequency
        term2 = -torch.exp(
            torch.clamp(
                torch.mean(torch.cos(cos_scale * diff), dim=-1),
                min=-1.0, max=1.0
            )
        )

        # Combine terms with amplitude scaling and offset
        # Add constants to ensure the function is positive
        value = amplitude * (term1 + term2 + self.global_scale + np.e) + offset

        return value

    def evaluate(self, x):
        """
        Evaluate the combined multi-minima function at points x.

        Args:
            x: torch.Tensor of shape (n, d) - points to evaluate

        Returns:
            torch.Tensor of shape (n,) - function values
        """
        # Initialize with a base value to ensure positivity
        result = torch.ones(x.shape[0]) * 0.1

        # Compute contribution from each minimum
        for i in range(self.num_minima):
            contribution = self.single_ackley(
                x,
                self.minima_locations[i],
                self.amplitudes[i],
                self.sharpness[i],
                self.offsets[i]
            )

            # Use minimum to create distinct basins
            if i == 0:
                result = contribution
            else:
                result = torch.minimum(result, contribution)

        # Ensure no NaN or Inf values
        result = torch.nan_to_num(result, nan=1e6, posinf=1e6, neginf=0.0)

        return result

    def plot_2d_interactive(self, resolution=100, figsize=(12, 10)):
        """Interactive 2D plot with sliders."""
        if self.d != 2:
            raise ValueError("2D plotting only works for d=2")

        # Create figure and axis
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.4)  # Make room for sliders

        # Initial plot
        x = torch.linspace(0, 1, resolution)
        y = torch.linspace(0, 1, resolution)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        Z = self.evaluate(points).reshape(resolution, resolution)

        surf = [ax.plot_surface(X.numpy(), Y.numpy(), Z.numpy(),
                               cmap=cm.viridis, alpha=0.8)]

        # Plot minima markers
        markers = []
        for i in range(self.num_minima):
            loc = self.minima_locations[i].numpy()
            marker = ax.scatter(loc[0], loc[1], 0, color='red', s=100, marker='x')
            markers.append(marker)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Function Value')
        ax.set_title('Multi-Minima Ackley Function - Interactive')

        # Create sliders
        slider_height = 0.03
        slider_spacing = 0.04
        current_bottom = 0.02

        sliders = []

        # Global parameters
        ax_global = plt.axes([0.15, current_bottom, 0.3, slider_height])
        slider_global = Slider(ax_global, 'Global Scale', 5.0, 50.0,
                              valinit=self.global_scale)
        sliders.append(('global_scale', slider_global))
        current_bottom += slider_spacing

        ax_exp = plt.axes([0.15, current_bottom, 0.3, slider_height])
        slider_exp = Slider(ax_exp, 'Exp Scale', 0.05, 0.5,
                           valinit=self.exp_scale)
        sliders.append(('exp_scale', slider_exp))
        current_bottom += slider_spacing

        # Per-minimum parameters
        for i in range(self.num_minima):
            # Amplitude
            ax_amp = plt.axes([0.15, current_bottom, 0.3, slider_height])
            slider_amp = Slider(ax_amp, f'Amplitude {i+1}', 0.1, 2.0,
                               valinit=self.amplitudes[i].item())
            sliders.append((f'amp_{i}', slider_amp))
            current_bottom += slider_spacing

            # Sharpness
            ax_sharp = plt.axes([0.55, current_bottom - slider_spacing, 0.3, slider_height])
            slider_sharp = Slider(ax_sharp, f'Sharpness {i+1}', 1.0, 10.0,
                                 valinit=self.sharpness[i].item())
            sliders.append((f'sharp_{i}', slider_sharp))

            # Offset
            ax_offset = plt.axes([0.15, current_bottom, 0.3, slider_height])
            slider_offset = Slider(ax_offset, f'Offset {i+1}', -10.0, 10.0,
                                  valinit=self.offsets[i].item())
            sliders.append((f'offset_{i}', slider_offset))
            current_bottom += slider_spacing

            # X position
            ax_x = plt.axes([0.15, current_bottom, 0.3, slider_height])
            slider_x = Slider(ax_x, f'X pos {i+1}', 0.0, 1.0,
                             valinit=self.minima_locations[i, 0].item())
            sliders.append((f'x_{i}', slider_x))
            current_bottom += slider_spacing

            # Y position
            ax_y = plt.axes([0.55, current_bottom - slider_spacing, 0.3, slider_height])
            slider_y = Slider(ax_y, f'Y pos {i+1}', 0.0, 1.0,
                             valinit=self.minima_locations[i, 1].item())
            sliders.append((f'y_{i}', slider_y))
            current_bottom += slider_spacing

        def update(val):
            # Update parameters
            self.global_scale = sliders[0][1].val
            self.exp_scale = sliders[1][1].val

            slider_idx = 2
            for i in range(self.num_minima):
                self.amplitudes[i] = sliders[slider_idx][1].val
                self.sharpness[i] = sliders[slider_idx + 1][1].val
                self.offsets[i] = sliders[slider_idx + 2][1].val
                self.minima_locations[i, 0] = sliders[slider_idx + 3][1].val
                self.minima_locations[i, 1] = sliders[slider_idx + 4][1].val
                slider_idx += 5

            # Recompute surface
            Z_new = self.evaluate(points).reshape(resolution, resolution)

            # Clear and redraw
            ax.clear()
            ax.plot_surface(X.numpy(), Y.numpy(), Z_new.numpy(),
                           cmap=cm.viridis, alpha=0.8)

            # Redraw markers
            for i in range(self.num_minima):
                loc = self.minima_locations[i].numpy()
                ax.scatter(loc[0], loc[1], 0, color='red', s=100, marker='x')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Function Value')
            ax.set_title('Multi-Minima Ackley Function - Interactive')

            fig.canvas.draw_idle()

        # Connect sliders to update function
        for name, slider in sliders:
            slider.on_changed(update)

        plt.show()
        return fig, sliders

    def plot_ternary_interactive(self, scale=100, figsize=(14, 10), resolution=100):
        """Interactive ternary plot with sliders using native matplotlib for smooth updates."""
        if self.d != 3:
            raise ValueError("Ternary plotting only works for d=3")

        import warnings

        # Create figure
        fig = plt.figure(figsize=figsize)

        # Create main plot axis
        ax = plt.axes([0.1, 0.4, 0.8, 0.55])

        # Create coordinate transformation functions
        def ternary_to_cartesian(a, b, c):
            """Convert ternary coordinates to cartesian."""
            x = 0.5 * a + b
            y = np.sqrt(3) / 2 * a
            return x, y

        def create_ternary_grid_data():
            """Create grid data for ternary plot."""
            # Create a finer grid for smoother visualization
            grid_points = []
            grid_values = []

            step = 1.0 / resolution
            for i in range(resolution + 1):
                for j in range(resolution + 1 - i):
                    a = i * step
                    b = j * step
                    c = 1.0 - a - b
                    if c >= 0 and c <= 1:
                        x, y = ternary_to_cartesian(a, b, c)
                        grid_points.append([x, y])

                        # Evaluate function
                        coords = torch.tensor([a, b, c], dtype=torch.float32)
                        value = self.evaluate(coords.unsqueeze(0)).item()
                        grid_values.append(value)

            return np.array(grid_points), np.array(grid_values)

        # Initial data
        points, values = create_ternary_grid_data()

        # Create scatter plot for heatmap effect
        scatter = ax.scatter(points[:, 0], points[:, 1], c=values,
                           cmap='viridis', s=20, edgecolors='none')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Function Value')

        # Draw ternary triangle boundary
        triangle = plt.Polygon([(0, 0), (1, 0), (0.5, np.sqrt(3)/2)],
                              fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(triangle)

        # Add gridlines
        for i in range(1, 10):
            # Horizontal lines (constant A)
            a = i / 10
            x1, y1 = ternary_to_cartesian(a, 0, 1-a)
            x2, y2 = ternary_to_cartesian(a, 1-a, 0)
            ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=0.5)

            # Lines from bottom-left to right (constant B)
            b = i / 10
            x1, y1 = ternary_to_cartesian(0, b, 1-b)
            x2, y2 = ternary_to_cartesian(1-b, b, 0)
            ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=0.5)

            # Lines from bottom-right to left (constant C)
            c = i / 10
            x1, y1 = ternary_to_cartesian(0, 1-c, c)
            x2, y2 = ternary_to_cartesian(1-c, 0, c)
            ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=0.5)

        # Plot minima markers
        minima_markers = []
        for i in range(self.num_minima):
            loc = self.minima_locations[i].numpy()
            x, y = ternary_to_cartesian(loc[0], loc[1], loc[2])
            marker = ax.scatter(x, y, marker='x', color='red', s=100, zorder=5)
            minima_markers.append(marker)

        # Set axis properties
        ax.set_aspect('equal')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
        ax.axis('off')
        ax.set_title('Multi-Minima Ackley Function on Ternary Domain', fontsize=16)

        # Add axis labels
        ax.text(0.5, -0.05, 'Component B', ha='center', fontsize=12)
        ax.text(-0.05, np.sqrt(3)/4, 'Component A', ha='center', rotation=60, fontsize=12)
        ax.text(1.05, np.sqrt(3)/4, 'Component C', ha='center', rotation=-60, fontsize=12)

        # Create sliders
        slider_height = 0.02
        slider_spacing = 0.025
        current_bottom = 0.02

        sliders = []

        # Global parameters
        ax_global = plt.axes([0.15, current_bottom, 0.3, slider_height])
        slider_global = Slider(ax_global, 'Global Scale', 5.0, 50.0,
                              valinit=self.global_scale)
        sliders.append(('global_scale', slider_global))
        current_bottom += slider_spacing

        ax_exp = plt.axes([0.15, current_bottom, 0.3, slider_height])
        slider_exp = Slider(ax_exp, 'Exp Scale', 0.05, 0.5,
                           valinit=self.exp_scale)
        sliders.append(('exp_scale', slider_exp))
        current_bottom += slider_spacing

        # Per-minimum parameters
        for i in range(self.num_minima):
            # Amplitude
            ax_amp = plt.axes([0.15, current_bottom, 0.3, slider_height])
            slider_amp = Slider(ax_amp, f'Amplitude {i+1}', 0.1, 2.0,
                               valinit=self.amplitudes[i].item())
            sliders.append((f'amp_{i}', slider_amp))

            # Sharpness
            ax_sharp = plt.axes([0.55, current_bottom, 0.3, slider_height])
            slider_sharp = Slider(ax_sharp, f'Sharpness {i+1}', 1.0, 10.0,
                                 valinit=self.sharpness[i].item())
            sliders.append((f'sharp_{i}', slider_sharp))
            current_bottom += slider_spacing

            # Offset
            ax_offset = plt.axes([0.15, current_bottom, 0.3, slider_height])
            slider_offset = Slider(ax_offset, f'Offset {i+1}', -10.0, 10.0,
                                  valinit=self.offsets[i].item())
            sliders.append((f'offset_{i}', slider_offset))
            current_bottom += slider_spacing

            # Ternary coordinates
            ax_a = plt.axes([0.15, current_bottom, 0.2, slider_height])
            slider_a = Slider(ax_a, f'A{i+1}', 0.0, 1.0,
                             valinit=self.minima_locations[i, 0].item())
            sliders.append((f'a_{i}', slider_a))

            ax_b = plt.axes([0.4, current_bottom, 0.2, slider_height])
            slider_b = Slider(ax_b, f'B{i+1}', 0.0, 1.0,
                             valinit=self.minima_locations[i, 1].item())
            sliders.append((f'b_{i}', slider_b))

            ax_c = plt.axes([0.65, current_bottom, 0.2, slider_height])
            slider_c = Slider(ax_c, f'C{i+1}', 0.0, 1.0,
                             valinit=self.minima_locations[i, 2].item())
            sliders.append((f'c_{i}', slider_c))
            current_bottom += slider_spacing

        def update(val):
            # Update parameters
            self.global_scale = sliders[0][1].val
            self.exp_scale = sliders[1][1].val

            slider_idx = 2
            for i in range(self.num_minima):
                self.amplitudes[i] = sliders[slider_idx][1].val
                self.sharpness[i] = sliders[slider_idx + 1][1].val
                self.offsets[i] = sliders[slider_idx + 2][1].val

                # Get ternary coordinates and normalize
                a = sliders[slider_idx + 3][1].val
                b = sliders[slider_idx + 4][1].val
                c = sliders[slider_idx + 5][1].val
                total = a + b + c
                if total > 0:
                    self.minima_locations[i, 0] = a / total
                    self.minima_locations[i, 1] = b / total
                    self.minima_locations[i, 2] = c / total

                slider_idx += 6

            # Update only the data
            _, new_values = create_ternary_grid_data()

            # Update scatter plot colors
            scatter.set_array(new_values)
            scatter.set_clim(vmin=new_values.min(), vmax=new_values.max())

            # Update minima markers
            for i, marker in enumerate(minima_markers):
                loc = self.minima_locations[i].numpy()
                x, y = ternary_to_cartesian(loc[0], loc[1], loc[2])
                marker.set_offsets([[x, y]])

            # Redraw
            fig.canvas.draw_idle()

        # Connect sliders to update function
        for name, slider in sliders:
            slider.on_changed(update)

        plt.show()
        return fig, sliders

# Example usage and testing
if __name__ == "__main__":
    print("Interactive 2D Example:")
    print("Use the sliders to adjust parameters and see the function change in real-time!\n")

    # 2D interactive example
    minima_2d = torch.tensor([
        [0.2, 0.3],
        [0.7, 0.6],
        [0.2, 0.6]
    ])

    amplitudes_2d = torch.tensor([1.0, 0.8, 1.0])
    sharpness_2d = torch.tensor([5.0, 7.0, 5.0])

    func_2d = MultiMinimaAckley(minima_2d, amplitudes_2d, sharpness_2d)
    func_2d.plot_2d_interactive()

    print("\nInteractive 3D Ternary Example:")
    print("Use the sliders to adjust parameters. Note: A, B, C coordinates are automatically normalized to sum to 1.\n")

    # 3D interactive ternary example
    minima_3d = torch.tensor([
        [0.6, 0.3, 0.1],
        [0.1, 0.7, 0.2]
    ])

    # Normalize to ensure they sum to 1 (for ternary)
    minima_3d = minima_3d / minima_3d.sum(dim=1, keepdim=True)

    amplitudes_3d = torch.tensor([1.0, 0.9, 1.1])
    sharpness_3d = torch.tensor([6.0, 5.0, 7.0])

    func_3d = MultiMinimaAckley(minima_3d, amplitudes_3d, sharpness_3d)
    func_3d.plot_ternary_interactive()

    # Test higher dimensions (non-interactive)
    print("\n5D Example (non-interactive):")
    minima_5d = torch.rand(6, 5)  # 6 minima in 5D space
    minima_5d = minima_5d / minima_5d.max()  # Normalize to [0,1]

    func_5d = MultiMinimaAckley(minima_5d)
    test_points_5d = torch.rand(100, 5)
    values_5d = func_5d.evaluate(test_points_5d)

    print(f"5D Function statistics:")
    print(f"Min: {values_5d.min():.4f}, Max: {values_5d.max():.4f}")
    print(f"Mean: {values_5d.mean():.4f}, Std: {values_5d.std():.4f}")

    # Verify no NaN or Inf
    print(f"Contains NaN: {torch.isnan(values_5d).any()}")
    print(f"Contains Inf: {torch.isinf(values_5d).any()}")
