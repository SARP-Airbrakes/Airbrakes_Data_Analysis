
from tests.analysis_v1 import FlightDataAnalyzer
from pathlib import Path
import os

def main():
  script_path = Path(os.path.abspath(__file__))
  data_path = script_path.parent / 'data.csv'

  output_dir = Path(f'generated/fsm-20260410')

  flight_analyzer = FlightDataAnalyzer(data_path, output_dir)
  flight_analyzer.plot_accelerometer()
  flight_analyzer.plot_current_state()
  flight_analyzer.plot_motor_data()
  flight_analyzer.plot_motor_and_acceleration_data()
  flight_analyzer.plot_temperature()
  flight_analyzer.plot_pressure()

if __name__ == '__main__':
  main()
