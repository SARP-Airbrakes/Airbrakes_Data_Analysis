
from tests.analysis_v2 import FlightDataAnalyzerV2
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import os

# data.csv starts at 2026-05-12T03:20:28.948Z
# truth.csv starts at 2026-05-12T03:21:01.653Z
def main():
  data_start = pd.Timestamp('2026-05-12T03:20:28.948Z', tz='UTC')
  truth_start = pd.Timestamp('2026-05-12T03:21:01.653Z', tz='UTC')

  difference = truth_start - data_start

  script_path = Path(os.path.abspath(__file__))
  data_path = script_path.parent / 'data.csv'
  truth_path = script_path.parent / 'truth.csv'

  truth_file = pd.read_csv(truth_path)

  output_dir = Path(f'generated/cartest-20260508')

  flight_analyzer = FlightDataAnalyzerV2(data_path, output_dir)
  flight_analyzer.plot()

  data_times = np.add(flight_analyzer.df['time_s'], -difference.total_seconds())
  truth_times = (pd.to_datetime(truth_file['time'], unit='s') - truth_start).astype('timedelta64[s]')

  data_altitude = flight_analyzer.df['estimated_altitude_m']
  truth_altitude = truth_file['altitude']
  
  fig, ax = plt.subplots(figsize=(12, 6))
  ax.plot(data_times, data_altitude, color='red', linestyle='-', linewidth=2, label='Filtered altitude')
  ax.plot(truth_times, truth_altitude, color='blue', linestyle='-', linewidth=2, label='Phone GPS altitude')

  ax.set_xlabel('Time (s)', fontsize=12)
  ax.set_ylabel('Altitude ASL (m)', fontsize=12)
  ax.set_title('Altitude Data Over time', fontsize=14)
  ax.legend(fontsize=11)
  ax.grid(True, alpha=0.3)

  fig.tight_layout()
  fig.savefig(output_dir / 'truth_altitude.png', dpi=150, bbox_inches='tight')
  plt.close(fig)

if __name__ == '__main__':
  main()
