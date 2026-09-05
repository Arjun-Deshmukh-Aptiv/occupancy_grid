import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.widgets import Slider, RadioButtons, TextBox
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection


class VehicleGridVisualizer:
    def __init__(self):
        # Vehicle parameters
        self.L = 4.0  # Vehicle length
        self.W = 2.0  # Vehicle width

        # Grid parameters
        self.grid_side_length_m = 40.0
        self.resolution = 1.0
        self.grid_size = int(self.grid_side_length_m/self.resolution)

        # Initial State [x, y, heading]
        self.state = np.array([self.grid_side_length_m/2, self.grid_side_length_m/2, 0.0])

        # Trajectory prediction parameters
        self.pred_time = 6.0
        self.pred_steps = 60
        self.dt = 0.1  # Simulation time step

        self.setup_plot()

        self.is_paused = False

    def on_key(self, event):
        if event.key == ' ':  # Press Spacebar to toggle pause/resume
            if self.anim.running:
                self.anim.event_source.stop()
                self.anim.running = False
            else:
                self.anim.event_source.start()
                self.anim.running = True

    def setup_plot(self):
        # Setup Figure and Subplots
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_axes([0.05, 0.4, 0.9, 0.55])

        # Setup GUI Control Axes
        ax_color = 'lightgoldenrodyellow'
        self.ax_v = self.fig.add_axes([0.15, 0.30, 0.65, 0.03], facecolor=ax_color)
        self.ax_steer = self.fig.add_axes([0.15, 0.25, 0.65, 0.03], facecolor=ax_color)
        self.ax_decay_s = self.fig.add_axes([0.15, 0.20, 0.65, 0.03], facecolor=ax_color)
        self.ax_decay_lat = self.fig.add_axes([0.15, 0.15, 0.65, 0.03], facecolor=ax_color)
        self.ax_thresh = self.fig.add_axes([0.15, 0.10, 0.65, 0.03], facecolor=ax_color)
        self.ax_radio = self.fig.add_axes([0.85, 0.10, 0.12, 0.20], facecolor=ax_color)
        self.ax_res = self.fig.add_axes([0.15, 0.05, 0.65, 0.03], facecolor=ax_color)

        # Sliders
        self.sl_v = Slider(self.ax_v, 'Speed', -5.0, 15.0, valinit=5.0)
        self.sl_steer = Slider(self.ax_steer, 'Steering', -0.5, 0.5, valinit=0.1)
        self.sl_decay_s = Slider(self.ax_decay_s, 'Long. Decay', 0.01, 0.5, valinit=0.02)
        self.sl_decay_lat = Slider(self.ax_decay_lat, 'Lat. Decay', 0.01, 1.0, valinit=0.1)
        self.sl_thresh = Slider(self.ax_thresh, 'Reachability Threshold', 0.01, 0.99, valinit=0.6)
        # Resolution increment factor for area-of-interest (enter integer)
        self.res_text = TextBox(self.ax_res, 'Res Increment Factor', initial='2')

        # Radio Buttons for Grid Mode
        self.radio = RadioButtons(self.ax_radio, ('World Fixed', 'Host Translating'))

        # Animation hook
        self.anim = FuncAnimation(self.fig, self.update, interval=50, blit=False, cache_frame_data=False)

    def predict_trajectory(self, state, v, steer):
        """ Predicts the path ahead based on current speed and steering. """
        x, y, theta = state
        traj_x, traj_y = [x], [y]

        for _ in range(self.pred_steps):
            theta += (v / self.L) * np.tan(steer) * (self.pred_time / self.pred_steps)
            x += v * np.cos(theta) * (self.pred_time / self.pred_steps)
            y += v * np.sin(theta) * (self.pred_time / self.pred_steps)
            traj_x.append(x)
            traj_y.append(y)

        return np.array(traj_x), np.array(traj_y)

    def compute_reachability(self, X, Y, traj_x, traj_y, v):
        """ Computes the heat map for the grid. """
        # Flatten grid for vectorized distance calculation
        x_flat = X.flatten()
        y_flat = Y.flatten()

        # Compute distances from every grid cell to every point in the trajectory
        dx = x_flat[:, np.newaxis] - traj_x[np.newaxis, :]
        dy = y_flat[:, np.newaxis] - traj_y[np.newaxis, :]
        dist_sq = dx ** 2 + dy ** 2

        # Find closest trajectory point for each grid cell
        min_idx = np.argmin(dist_sq, axis=1)
        min_dist = np.sqrt(np.take_along_axis(dist_sq, min_idx[:, np.newaxis], axis=1).flatten())

        # Compute longitudinal distance (s) along the path
        # Multiply by approx step distance
        step_dist = v * (self.pred_time / self.pred_steps) if v != 0 else 0
        s_dist = min_idx * step_dist

        # Compute effective lateral distance (ignoring distance inside vehicle width)
        lat_eff = np.maximum(0, min_dist - (self.W / 2))

        # Compute heat with Exponential decay
        k_s = self.sl_decay_s.val
        k_lat = self.sl_decay_lat.val

        heat_flat = np.exp(-k_s * s_dist) * np.exp(-k_lat * lat_eff ** 2)

        # Hard cut-off if behind the vehicle or beyond prediction horizon
        heat_flat[s_dist > (self.pred_time * abs(v))] = 0.0

        return heat_flat.reshape(X.shape)

    def update(self, frame):
        if hasattr(self, 'fine_grid_lines') and self.fine_grid_lines is not None:
            self.fine_grid_lines.remove()
            self.fine_grid_lines = None

        v = self.sl_v.val
        steer = self.sl_steer.val
        thresh = self.sl_thresh.val
        mode = self.radio.value_selected

        # 1. Kinematic Update (update host vehicle position)
        self.state[2] += (v / self.L) * np.tan(steer) * self.dt
        self.state[0] += v * np.cos(self.state[2]) * self.dt
        self.state[1] += v * np.sin(self.state[2]) * self.dt

        # Wrap around for World Fixed mode to prevent driving off completely
        if mode == 'World Fixed':
            self.state[0] = self.state[0] % 100
            self.state[1] = self.state[1] % 100

        x, y, theta = self.state

        # 2. Determine Grid Extents based on Mode
        if mode == 'World Fixed':
            grid_x_min, grid_x_max = 0, self.grid_side_length_m
            grid_y_min, grid_y_max = 0, self.grid_side_length_m
        else:
            # Translating grid (no rotation)
            grid_x_min, grid_x_max = x - self.grid_side_length_m/2, x + self.grid_side_length_m/2
            grid_y_min, grid_y_max = y - self.grid_side_length_m/2, y + self.grid_side_length_m/2

        x_lin = np.linspace(grid_x_min, grid_x_max, self.grid_size)
        y_lin = np.linspace(grid_y_min, grid_y_max, self.grid_size)
        X, Y = np.meshgrid(x_lin, y_lin)

        # 3. Predict Trajectory & Compute Heatmap
        traj_x, traj_y = self.predict_trajectory(self.state, v, steer)
        R = self.compute_reachability(X, Y, traj_x, traj_y, v)

        reachable_cells = np.sum(R >= thresh)

        total_cells = R.size

        ref_factor = int(float(self.res_text.text))

        adaptive_cells = (
            total_cells
            - reachable_cells
            + reachable_cells * ref_factor**2
        )

        full_fine_cells = total_cells * ref_factor**2

        # CSI
        csi = (
            1.0 -
            adaptive_cells / full_fine_cells
        ) * 100.0

        # ARG
        arg = adaptive_cells / total_cells

        # DRC
        drc = reachable_cells / total_cells * 100.0

        # Value Gain per Cell
        vgc_adaptive = reachable_cells / adaptive_cells
        vgc_fullfine = reachable_cells / full_fine_cells

        vgc_gain = (
            vgc_adaptive /
            max(vgc_fullfine, 1e-6)
        )

        metrics = (
            f"CSI : {csi:.1f}%\n"
            f"ARG : {arg:.2f}x\n"
            f"DRC : {drc:.1f}%\n"
            f"VGC : {vgc_gain:.2f}x"
        )

        
        

        # 4. Draw Everything
        self.ax.clear()
        # Show pause/continue hint

        self.ax.text(
                    0.02,
                    0.95,
                    metrics,
                    transform=self.ax.transAxes,
                    color='white',
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(facecolor='black', alpha=0.7)
                )

        
        # self.ax.text(
        #     0.01,
        #     0.99,
        #     "Space bar: pause/continue",
        #     transform=self.ax.transAxes,
        #     va='top',
        #     ha='left',
        #     color='white',
        #     fontsize=10,
        #     bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=4),
        #     zorder=10,
        # )
        self.ax.set_title("Vehicle Reachability Grid Prediction")
        self.ax.xaxis.set_major_locator(MultipleLocator(self.resolution))
        self.ax.yaxis.set_major_locator(MultipleLocator(self.resolution))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        # Create a copy of R and mask out values that are nearly 0 (e.g., less than 0.01)
        epsilon = 0.01
        R_masked = np.ma.masked_where(np.abs(R) < epsilon, R)

        # Configure the colormap to render masked/transparent values cleanly
        cmap = plt.get_cmap('jet').copy()
        cmap.set_bad(color='none')  # Makes masked values completely transparent

        # Plot Heatmap (Blue to Red)
        pcm = self.ax.pcolormesh(X, Y, R_masked, cmap='jet', vmin=0.0, vmax=1.0, edgecolors='red', shading='auto', linewidth=0.2)

        # Plot Threshold Boundary & highlight "Double Resolution" zone using hatching
        if R.min() < thresh < R.max():
            # Solid Yellow Boundary
            self.ax.contour(X, Y, R, levels=[thresh], colors='yellow', linewidths=2.5)
            rows, cols = R.shape
            try:
                resolution_increment_factor = int(float(self.res_text.text))
                if resolution_increment_factor < 1:
                    resolution_increment_factor = 1
            except Exception:
                resolution_increment_factor = 2

            for i in range(rows-1):
                for j in range(cols-1):
                    if R[i, j] >= thresh:
                        # Get the 4 corners of this cell
                        x0, x1 = X[i, j] - self.resolution/2.0, X[i, j] + self.resolution/2.0
                        y0, y1 = Y[i, j] - self.resolution/2.0, Y[i, j] + self.resolution/2.0

                        # Create minigrid inside the cell
                        x_mini_lin = np.linspace(x0, x1, resolution_increment_factor + 1)
                        y_mini_lin = np.linspace(y0, y1, resolution_increment_factor + 1)

                        X_mini, Y_mini = np.meshgrid(x_mini_lin, y_mini_lin)

                        # Plot horizontal grid lines of the mini-grid
                        for r in range(X_mini.shape[0]):
                            self.ax.plot(X_mini[r, :], Y_mini[r, :], color='black', linewidth=1.0, alpha=1.0)

                        # Plot vertical grid lines of the mini-grid
                        for c in range(X_mini.shape[1]):
                            self.ax.plot(X_mini[:, c], Y_mini[:, c], color='black', linewidth=1.0, alpha=1.0)


        # Plot Vehicle Body
        # Compute bottom-left corner from center
        bl_x = x - (self.L / 2) * np.cos(theta) + (self.W / 2) * np.sin(theta)
        bl_y = y - (self.L / 2) * np.sin(theta) - (self.W / 2) * np.cos(theta)

        rect = Rectangle((bl_x, bl_y), self.L, self.W, angle=np.degrees(theta),
                         edgecolor='white', facecolor='black', lw=2, zorder=5)
        self.ax.add_patch(rect)

        # Plot Trajectory Line
        self.ax.plot(traj_x, traj_y, color='white', linestyle='--', linewidth=2, zorder=4)

        # Set limits
        self.ax.set_xlim(grid_x_min, grid_x_max)
        self.ax.set_ylim(grid_y_min, grid_y_max)
        self.ax.set_aspect('equal')
        self.ax.grid(True, color='white', alpha=0.2)

        self.anim.running = True
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)


if __name__ == '__main__':
    viz = VehicleGridVisualizer()

    plt.show()