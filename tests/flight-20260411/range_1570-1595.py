
from . import FlightDataAnalyzer
from pathlib import Path
import os

def main():
  script_path = Path(os.path.abspath(__file__))
  data_path = script_path.parent / 'data.csv'

  flight_analyzer = FlightDataAnalyzer(data_path, time_range=(1570, 1595))
  flight_analyzer.plot()

if __name__ == '__main__':
  main()
