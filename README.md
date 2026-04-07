EIS Analysis Tool (GUI)

A Python-based GUI application for analyzing Electrochemical Impedance Spectroscopy (EIS) data, fitting equivalent circuits, and visualizing results.

Features
Load Excel EIS datasets
Select material and resistivity dynamically
Plot Nyquist graphs
Define custom equivalent circuit models
Input initial parameter guesses
Fit data using impedance
Visualize fitted vs measured data
Export results to Excel
Step-based workflow (FSM-driven UI)
Workflow
Select Data
Load .xlsx or .xls file
Choose material and resistivity
Plot Nyquist
Visualize raw impedance data

Get Circuit

Input circuit formula

Example:

L0-R0-p(R1,CPE1)-p(R2,CPE2)
Get Initial Values
Enter initial guesses for each fit parameter
Parameters are derived automatically from the circuit
Fit Graph
Performs nonlinear fitting using CustomCircuit
Displays fitted curve over Nyquist plot
Get Output
View fitted parameters
Export results to Excel
Installation
Option 1 — Script (recommended)
chmod +x run.sh
./run.sh

This will:

create a virtual environment
install dependencies
run the app
Option 2 — Manual setup
python3 -m venv .venv
source .venv/bin/activate

pip install pandas matplotlib customtkinter openpyxl xlrd impedance

Run:

python main.py
Dependencies
Python 3.9+
pandas
matplotlib
customtkinter
impedance
openpyxl
xlrd
Input Data Format

The Excel file must contain:

Columns
Frequency
Z'
-Z''
Z
Phase
Time
Structure
Data organized in blocks per material and resistivity
Each block must start with a "Frequency" header
Circuit Syntax

Supported examples:

R0
R0-C0
R0-p(R1,C0)
L0-R0-p(R1,CPE1)-p(R2,CPE2)
Notes
p(...) denotes parallel elements
CPE elements require two parameters
Parameter order is derived left-to-right from the circuit string
Initial guesses must match this order
Fitting Engine

Uses:

impedance.models.circuits.CustomCircuit
impedance.visualization.plot_nyquist

Core flow:

circuit = CustomCircuit(circuit_string, initial_guess=initial_guess)
circuit.fit(frequency, impedance)
Z_fit = circuit.predict(frequency)
Output

The application generates:

Fitted parameters (circuit.parameters_)
Fitted impedance (Z_fit)
Nyquist plot (data vs fit)
Exportable .xlsx file containing:
Frequency
Measured impedance
Fitted impedance
UI Architecture
Left panel
Dynamic input (selectors, circuit, parameters)
Right panel (tabbed)
Data
Circuit
Initial Values
Fit
Output
Bottom panel
FSM-controlled workflow buttons
State Management
Finite State Machine (FSM)
Each step unlocks the next
Re-running a step invalidates downstream states
Notes
Changing material/resistivity resets:
circuit
initial values
fit results
Changing circuit resets:
initial values
fit
output
Parameter parsing is regex-based → extendable for new element types
