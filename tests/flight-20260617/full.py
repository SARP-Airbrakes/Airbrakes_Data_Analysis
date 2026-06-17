
from tests.analysis_v2 import FlightDataAnalyzerV2
from pathlib import Path
import os

def main():
  script_path = Path(os.path.abspath(__file__))
  data_path = script_path.parent / 'data.csv'

  output_dir = Path(f'generated/flight-20260617/full')

  flight_analyzer = FlightDataAnalyzerV2(data_path, output_dir)
  flight_analyzer.plot()

if __name__ == '__main__':
  main()
