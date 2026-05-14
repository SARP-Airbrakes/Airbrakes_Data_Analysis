# This class can be used to plot the different fields of flight data
# post-implementation of the Kalman filter.

from matplotlib import pyplot as plt
from tests.analysis_v1 import FlightDataAnalyzer


class FlightDataAnalyzerV2(FlightDataAnalyzer):

  def __init__(self, data_file_path, output_dir, time_range=None):
    super().__init__(data_file_path, output_dir, time_range)

  def plot(self):
    self.plot_filtered_accelerometer()
    self.plot_filtered_altitude_residuals()
    self.plot_filtered_velocity_and_altitude()
    super().plot()

  def plot_filtered_accelerometer(self):
    time = self.df['time_s']
    accel_x = self.df['estimated_accel_x_mps2']
    accel_y = self.df['estimated_accel_y_mps2']
    accel_z = self.df['estimated_accel_z_mps2']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(time, accel_x, color='red', linestyle='-', linewidth=2, label='Accel X')
    ax.plot(time, accel_y, color='green', linestyle='-', linewidth=2, label='Accel Y')
    ax.plot(time, accel_z, color='blue', linestyle='-', linewidth=2, label='Accel Z')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Filtered acceleration (m/s²)', fontsize=12)
    ax.set_title('Filtered Accelerometer Data Over Time', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(self.output_dir / 'filtered_accelerometer.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

  def plot_velocity(self):
    time = self.df['time_s']
    upward_velocity = self.df['estimated_upward_velocity_mps']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(time, upward_velocity, color='orange', linestyle='-', linewidth=2, label='Filtered velocity (Z)')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Filtered upward velocity (m/s)', fontsize=12)
    ax.set_title('Filtered Velocity Data Over Time', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(self.output_dir / 'filtered_velocity.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
  def plot_altitude(self):
    time = self.df['time_s']
    filtered_altitude = self.df['estimated_altitude_m']
    baro_altitude = self.df['baro_altitude_m']
    pressure = self.df['pressure_pascals']

    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot motor data on left y-axis
    ax1.plot(time, baro_altitude, color='red', linestyle='-', linewidth=2, label='Barometer measurement')
    ax1.plot(time, filtered_altitude, color='orange', linestyle='-', linewidth=2, label='Filtered altitude')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Altitude (m)', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Create right y-axis for pressure
    ax2 = ax1.twinx()
    ax2.plot(time, pressure, color='darkviolet', linestyle='-', linewidth=2, label='Pressure')
    ax2.set_ylabel('Pressure (Pa)', fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left')
    
    ax1.set_title('Altitude vs Filtered altitude vs Pressure', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(self.output_dir / 'altitude.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

  def plot_filtered_altitude_residuals(self):
    time = self.df['time_s']
    altitude_residuals = self.df['estimated_altitude_m'] - self.df['baro_altitude_m']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(time, altitude_residuals, color='green', linestyle='-', linewidth=2, label='Residual')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Residual between filtered altitude and measured (m)', fontsize=12)
    ax.set_title('Residuals of Filtered Altitude over Time', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(self.output_dir / 'altitude_residuals.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

  def plot_filtered_velocity_and_altitude(self):
    time = self.df['time_s']
    filtered_altitude = self.df['estimated_altitude_m']
    velocity = self.df['estimated_upward_velocity_mps']

    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot motor data on left y-axis
    ax1.plot(time, filtered_altitude, color='orange', linestyle='-', linewidth=2, label='Filtered altitude')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Altitude (m)', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Create right y-axis for pressure
    ax2 = ax1.twinx()
    ax2.plot(time, velocity, color='darkviolet', linestyle='-', linewidth=2, label='Upward velocity')
    ax2.set_ylabel('Upward velocity (m/s)', fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left')
    
    ax1.set_title('Altitude vs Velocity', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(self.output_dir / 'altitude_and_velocity.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

  def plot_mega_plot(self):
    pass
  

