import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np


class FlightDataAnalyzer():
  
    def __init__(self, data_file_path, time_range=None):
        self.df = pd.read_csv(data_file_path)
        self.file_path = data_file_path
        self.time_range = time_range
        
        # Filter data by time range if provided
        if time_range is not None:
            start_time, end_time = time_range
            self.df = self.df[(self.df['time_s'] >= start_time) & (self.df['time_s'] <= end_time)]
        
        self.output_dir = self._setup_output_directory(data_file_path, time_range)

    def _setup_output_directory(self, file_path, time_range):
        """Create output directory under figs/ based on the data filename"""
        file_name = Path(file_path).stem
        
        # Add time range suffix if filtering
        if time_range is not None:
            file_name = f"{file_name}_{int(time_range[0])}-{int(time_range[1])}"
        
        output_dir = Path('figs') / file_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def plot(self):
        self.plot_accelerometer()
        self.plot_temperature()
        self.plot_pressure()
        self.plot_velocity()
        self.plot_current_state()
        self.plot_altitude()
        self.plot_motor_data()
        self.plot_motor_and_pressure_data()
        self.plot_motor_and_acceleration_data()
        self.plot_mega_plot()
        print(f"All figures saved to: {self.output_dir}")

    def plot_accelerometer(self):
        """Plot accelerometer data (x, y, z) over time"""
        time = self.df['time_s']
        accel_x = self.df['accel_x_mps2']
        accel_y = self.df['accel_y_mps2']
        accel_z = self.df['accel_z_mps2']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(time, accel_x, color='red', linestyle='-', linewidth=2, label='Accel X')
        ax.plot(time, accel_y, color='green', linestyle='-', linewidth=2, label='Accel Y')
        ax.plot(time, accel_z, color='blue', linestyle='-', linewidth=2, label='Accel Z')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Acceleration (m/s²)', fontsize=12)
        ax.set_title('Accelerometer Data Over Time', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(self.output_dir / 'accelerometer.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_temperature(self):
        """Plot temperature over time"""
        time = self.df['time_s']
        temperature = self.df['temperature_c']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(time, temperature, color='orange', linestyle='-', linewidth=2, label='Temperature')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title('Temperature Over Time', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(self.output_dir / 'temperature.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_pressure(self):
        """Plot pressure over time"""
        time = self.df['time_s']
        pressure = self.df['pressure_pascals']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(time, pressure, color='darkviolet', linestyle='-', linewidth=2, label='Pressure')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Pressure (Pa)', fontsize=12)
        ax.set_title('Pressure Over Time', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(self.output_dir / 'pressure.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_velocity(self):
        """Plot fused velocity over time"""
        time = self.df['time_s']
        velocity = self.df['fused_velocity_mps']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(time, velocity, color='purple', linestyle='-', linewidth=2, label='Fused Velocity')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Velocity (m/s)', fontsize=12)
        ax.set_title('Fused Velocity Over Time', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(self.output_dir / 'velocity.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_current_state(self):
        """Plot current state over time"""
        time = self.df['time_s']
        state = self.df['current_state']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(time, state, color='brown', linewidth=2, label='Current State')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('State', fontsize=12)
        ax.set_title('Current State Over Time', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(self.output_dir / 'current_state.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_altitude(self):
        """Plot altitude measurements over time"""
        time = self.df['time_s']
        acc_altitude = self.df['acc_altitude_m'] +  self.df['reference_altitude_m']  # Apply accelerometer altitude offset
        baro_altitude = self.df['baro_altitude_m']
        agl_altitude = self.df['agl_altitude_m']
        
        # Calculate maximum altitude
        max_acc = acc_altitude.max()
        max_baro = baro_altitude.max()
        max_agl = agl_altitude.max()
        max_altitude_ft = max_agl * 3.281  # Convert max altitude to feet
        
        # Get the time at which maximum altitude occurs
        if max_acc >= max_baro:
            max_altitude_time = time.iloc[acc_altitude.argmax()]
        else:
            max_altitude_time = time.iloc[baro_altitude.argmax()]
        
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        ax[0].plot(time, acc_altitude, color='green', linestyle='-', linewidth=2, label='Accelerometer Altitude')
        ax[0].plot(time, baro_altitude, color='blue', linestyle='-', linewidth=2, label='Barometric Altitude')
        ax[1].plot(time, agl_altitude, color='magenta', linestyle='-', linewidth=2, label='AGL Altitude')
        
        # Add vertical line for maximum altitude
        ax[1].axvline(x=max_altitude_time, color='red', linestyle='--', linewidth=2, label='Maximum Altitude')
        
        # Annotate the maximum altitude value
        ax[1].annotate(f'Max: {int(max_altitude_ft)} ft', 
                    xy=(max_altitude_time, max_agl),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'))
        
        ax[1].set_xlabel('Time (s)', fontsize=12)
        ax[0].set_ylabel('ASL Altitude (m)', fontsize=12)
        ax[1].set_ylabel('AGL Altitude (m)', fontsize=12)
        ax[0].set_title('Altitude Over Time', fontsize=14)
        for a in ax:
            a.legend(fontsize=11)
            a.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(self.output_dir / 'altitude.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _motor_angle_to_flap_deflection(self, motor_angle):
        """Convert motor angle readings to flap deflection angle."""
        motor_angle = -np.asarray(motor_angle)
        flap_angle = 45.0 * np.sin(motor_angle * np.pi / 8280.0)
        return flap_angle

    def plot_motor_data(self):
        """Plot motor target and actual flap angle over time"""
        time = self.df['time_s']
        motor_target = self.df['motor_target_degrees']
        motor_actual = self.df['motor_actual_degrees']
        flap_target = self._motor_angle_to_flap_deflection(motor_target)
        flap_actual = self._motor_angle_to_flap_deflection(motor_actual)
        motor_power = self.df['motor_commanded_power']
        
        fig, ax = plt.subplots(2,1,figsize=(12, 6),sharex=True)
        
        ax[0].plot(time, flap_target, color='blue', linestyle='-', linewidth=2, label='Target Flap Angle')
        ax[0].plot(time, flap_actual, color='red', linestyle='-', linewidth=2, label='ActualFlap Angle')
        ax[1].plot(time, motor_power, color='green', linestyle='-', linewidth=2, label='Commanded Motor Power')

        ax[0].set_ylabel('Degrees', fontsize=12)
        ax[0].set_title('Motor Target vs Flap Angle Over Time', fontsize=14)
        ax[1].set_xlabel('Time (s)', fontsize=12)
        ax[1].set_ylabel('Power', fontsize=12)

        for a in ax:
            a.grid(True, alpha=0.3)
            a.legend(fontsize=11)

        fig.tight_layout()
        fig.savefig(self.output_dir / 'motor_data.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_motor_and_pressure_data(self):
        """Plot motor data on left y-axis and pressure data on right y-axis"""
        time = self.df['time_s']
        motor_actual = self.df['motor_actual_degrees']
        flap_actual = self._motor_angle_to_flap_deflection(motor_actual)
        pressure = self.df['pressure_pascals']

        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot motor data on left y-axis
        ax1.plot(time, flap_actual, color='red', linestyle='-', linewidth=2, label='Flap Angle')
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Degrees', fontsize=12, color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        
        # Create right y-axis for pressure
        ax2 = ax1.twinx()
        ax2.plot(time, pressure, color='darkviolet', linestyle='-', linewidth=2, label='Pressure')
        ax2.set_ylabel('Pressure (Pa)', fontsize=12, color='darkviolet')
        ax2.tick_params(axis='y', labelcolor='darkviolet')
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left')
        
        ax1.set_title('Motor Data vs Pressure Over Time', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(self.output_dir / 'motor_and_pressure_data.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _air_density_from_pressure(self, pressure, temperature):
        """Calculate air density from pressure using the ideal gas law."""
        # Assuming standard temperature (293.15 K) and specific gas constant for air (287 J/(kg·K))
        specific_gas_constant = 287.0
        air_density = pressure / (specific_gas_constant * temperature)
        return air_density

    def _cd_from_accel(self, mass, area, accel_z, velocity, pressure, temperature):
        """Calculate drag coefficient (Cd) from vertical acceleration and velocity"""
        air_density = self._air_density_from_pressure(pressure, temperature)
        
        # Use the magnitude of the net vertical acceleration for drag force
        drag_force = mass * np.abs(accel_z)
        
        # Calculate velocity magnitude
        velocity_magnitude = np.abs(velocity)
        
        # Avoid division by zero
        velocity_magnitude[velocity_magnitude == 0] = 1e-6
        
        # Calculate Cd using the drag equation: F = 0.5 * Cd * A * rho * v²
        cd = (2 * drag_force) / (area * air_density * velocity_magnitude**2)
        
        return cd
    
    def _get_predicted_apogee(self, mass, altitude, velocity, cd, area, pressure, temperature):
        rho = self._air_density_from_pressure(pressure, temperature)
        k = 0.5 * rho * cd * area
        g = 9.81
        apogee = altitude + mass / (2*k) * np.log(1 + (k * velocity**2) / (mass * g))
        return apogee

    def plot_motor_and_acceleration_data(self):
        """Plot motor data on left y-axis and acceleration data on right y-axis"""
        time = self.df['time_s']
        motor_actual = self.df['motor_actual_degrees']
        flap_actual = self._motor_angle_to_flap_deflection(motor_actual)
        accel_z = self.df['accel_z_mps2'] - 9.81  # Subtract gravity to get net acceleration in z-axis

        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot motor data on left y-axis
        ax1.plot(time, flap_actual, color='red', linestyle='-', linewidth=2, label='Flap Angle')
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Flap Angle (Degrees)', fontsize=12, color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        
        # Create right y-axis for acceleration
        ax2 = ax1.twinx()
        ax2.plot(time, accel_z, color='purple', linestyle='-', linewidth=1.5, label='Accel Z')
        ax2.set_ylabel('Z-Acceleration (m/s²)', fontsize=12, color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left')
        
        ax1.set_title('Motor Data vs Acceleration Over Time', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(self.output_dir / 'motor_and_acceleration_data.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    def plot_mega_plot(self):
        """Plot AGL altitude, velocity, flap position, drag coefficient, and predicted apogee in one figure."""
        time = self.df['time_s']
        agl_altitude = self.df['agl_altitude_m']
        velocity = self.df['fused_velocity_mps']
        flap_actual = self._motor_angle_to_flap_deflection(self.df['motor_actual_degrees'])
        pressure = self.df['pressure_pascals']
        temperature = self.df['temperature_c'] + 273.15
        accel_z = self.df['accel_z_mps2'] - 9.81

        mass_kg = 40.0 * 0.45359237
        frontal_area_m2 = np.pi * 0.0762**2 # 6in diameter rocket
        drag_coeff = self._cd_from_accel(mass_kg, frontal_area_m2, accel_z, velocity, pressure, temperature)
        predicted_apogee = self._get_predicted_apogee(mass_kg, agl_altitude, velocity, drag_coeff, frontal_area_m2, pressure, temperature)

        fig, axs = plt.subplots(5, 1, figsize=(14, 22), sharex=True)

        axs[0].plot(time, agl_altitude, color='magenta', linewidth=2, label='AGL Altitude')
        axs[0].set_ylabel('AGL Altitude (m)', fontsize=12)
        axs[0].legend(fontsize=11)
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(time, velocity, color='purple', linewidth=2, label='Velocity')
        axs[1].set_ylabel('Velocity (m/s)', fontsize=12)
        axs[1].legend(fontsize=11)
        axs[1].grid(True, alpha=0.3)

        axs[2].plot(time, flap_actual, color='red', linewidth=2, label='Flap Angle')
        axs[2].set_ylabel('Flap Angle (deg)', fontsize=12)
        axs[2].legend(fontsize=11)
        axs[2].grid(True, alpha=0.3)

        axs[3].plot(time, drag_coeff, color='darkgreen', linewidth=2, label='Drag Coefficient')
        axs[3].set_ylabel('Cd', fontsize=12)
        axs[3].set_ylim(0, 10)  # Set y-axis limits for better visualization
        axs[3].legend(fontsize=11)
        axs[3].grid(True, alpha=0.3)

        axs[4].plot(time, predicted_apogee, color='teal', linewidth=2, label='Predicted Apogee')
        axs[4].set_xlabel('Time (s)', fontsize=12)
        axs[4].set_ylabel('Predicted Apogee (m)', fontsize=12)
        axs[4].legend(fontsize=11)
        axs[4].grid(True, alpha=0.3)

        apogee_time = time.iloc[np.nanargmax(agl_altitude.values)]
        for ax in axs:
            ax.axvline(x=apogee_time, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

        fig.suptitle('Mega Plot: AGL Altitude, Velocity, Flap Angle, Drag Coefficient, and Predicted Apogee (40 lb rocket)', fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        fig.savefig(self.output_dir / 'mega_plot.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
def main():
  # Analyze epic_data.csv for time range 1550-1625 seconds
  flight_analyzer = FlightDataAnalyzer('data/epic_data.csv', time_range=(1550, 1625))
  flight_analyzer.plot()


if __name__ == '__main__':
  main()