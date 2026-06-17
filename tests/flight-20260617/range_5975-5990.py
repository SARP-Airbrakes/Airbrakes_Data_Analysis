
from tests.analysis_v2 import FlightDataAnalyzerV2
from pathlib import Path
import os

def main():
  script_path = Path(os.path.abspath(__file__))
  data_path = script_path.parent / 'data.csv'

  time_range = (5975, 5990)
  output_dir = Path(f'generated/flight-20260617/{int(time_range[0])}-{int(time_range[1])}')

  flight_analyzer = FlightDataAnalyzerV2(data_path, output_dir, time_range)
  flight_analyzer.plot()

if __name__ == '__main__':
  main()
