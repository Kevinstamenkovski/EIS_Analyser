# EIS Analysis Tool (GUI)

Python GUI for EIS data loading, Nyquist plotting, equivalent-circuit fitting, and Excel export.

## Install

Requires Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas matplotlib customtkinter openpyxl xlrd impedance
```

## Run

```bash
python main.py
```

## What it does

- Load `.xlsx` / `.xls` EIS files
- Select material and resistivity
- Plot Nyquist data
- Enter a circuit formula
- Enter initial fit values
- Fit with `impedance.models.circuits.CustomCircuit`
- Export results to Excel

## Circuit example

```text
L0-R0-p(R1,CPE1)-p(R2,CPE2)
```

## Notes

- `p(...)` means parallel
- `CPE` needs 2 fit parameters
- Parameter order follows the circuit from left to right
- Changing material / resistivity / circuit resets downstream steps

## Repositories used

- `impedance` library: GitHub repo available here. :contentReference[oaicite:0]{index=0}
- `hybrid-drt`: `jdhuang-csm/hybrid-drt` on GitHub. :contentReference[oaicite:1]{index=1}
- DRT tool repo: `ciuccislab/pyDRTtools` on GitHub. :contentReference[oaicite:2]{index=2}

## Output

- Fitted parameters
- Nyquist plot with data + fit
- Exportable Excel output
```
