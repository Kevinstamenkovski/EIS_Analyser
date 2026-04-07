# EIS Analysis Tool (GUI)

Python GUI for loading EIS Excel data, plotting Nyquist graphs, entering equivalent circuits, fitting with `impedance`, and exporting results.

## Requirements

- Python 3.9+
- `venv`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install pandas numpy scipy matplotlib customtkinter openpyxl xlrd impedance
```

## Run

```bash
python main.py
```

## What the app does

- Load `.xlsx` / `.xls` EIS files
- Select material and resistivity
- Plot Nyquist data
- Enter a circuit formula
- Enter initial fit values
- Fit with `impedance.models.circuits.CustomCircuit`
- Show data, circuit, initial values, fit, and output in separate tabs
- Export fit results to Excel

## Workflow

1. Click **Select Data**
2. Choose the Excel file
3. Select **material**
4. Select **resistivity**
5. Click **Plot Nyquist**
6. Click **Get Circuit**
7. Enter a circuit formula
8. Click **Get Init. Vals.**
9. Enter initial values for all fit parameters
10. Click **Fit Graph**
11. Click **Get Output**
12. Export the result if needed

## Circuit example

```text
L0-R0-p(R1,CPE1)-p(R2,CPE2)
```

## Notes

- `p(...)` means parallel
- `CPE` uses 2 fit parameters
- Parameter order follows the circuit from left to right
- Changing material, resistivity, or circuit resets downstream steps

## Python packages used

- `pandas`
- `numpy`
- `scipy`
- `matplotlib`
- `customtkinter`
- `openpyxl`
- `xlrd`
- `impedance`

## Repositories

- `impedance`: `ECSHackWeek/impedance.py`
- `hybdrt`: `jdhuang-csm/hybrid-drt`
- `pyDRTtools`: `ciuccislab/pyDRTtools`

## Output

- Fitted parameters
- Nyquist plot with measured data and fitted curve
- Exportable Excel output with fit data
