# Airbrakes data and analysis repository
This repository houses the data extracted from the airbrakes system during tests
(including flights, HILs, etc.) as well as corresponding generated figures for
analysis.

## Usage
To run the scripts:
```sh
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the generators as modules.
python3 -m tests.flight-20260411.range_1500-1700
python3 -m tests.fsm-20260410
```

