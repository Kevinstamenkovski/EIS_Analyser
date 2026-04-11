# EIS App User Documentation

This app is for loading EIS data from an Excel file, choosing a circuit, fitting the data, and exporting results.

This document is written for a first-time user.

---

## 1. What this app does

The app can:

- load one Excel file
- show Nyquist plots
- let you choose one material or many materials
- let you choose one resistivity or many resistivities
- fit an equivalent circuit to the selected data
- export fitted results to Excel
- generate a batch report for all datasets in the file

The app does **not** decide whether your circuit is scientifically correct.
It only fits the circuit you give it.

---

## 2. What the buttons mean

### Select Data
Open your Excel file.

### Plot Nyquist
Draw the Nyquist plot for the current selection.
Use this to check if the data looks normal.

### Get Circuit
Type the circuit formula.
Example:

`L0-R0-p(R1,CPE1)`

### Get Init. Vals.
Shows the initial values used for fitting.
Usually you can stay in **AUTO** mode.

### Fit Graph
Fits the currently selected dataset and shows data vs fit.

### Get Output
Shows fitted parameters and fit-quality numbers for the current dataset.

### Compute All
Fits all datasets in the current selection.
Example:
- one material + all resistivities
- all materials + all resistivities

### Compare Graphs
Shows all computed fits together on one graph.

### Batch Fit + Report
Fits every dataset in the whole file and creates a big report table.

### Export Batch Report
Saves the batch report to Excel.

---

## 3. Normal workflow

Use this order:

1. Click **Select Data**
2. Choose **Material**
3. Choose **Resistivity**
4. Click **Plot Nyquist**
5. Click **Get Circuit**
6. Enter the circuit formula
7. Leave initial values in **AUTO** mode unless you really need manual values
8. Click **Fit Graph** or **Compute All**
9. Click **Get Output** if you want the table for one result
10. Export if needed

---

## 4. Important behavior after the FSM fix

This was changed on purpose.

When you change **material** or **resistivity** inside the **same loaded file**:

- the app now **keeps your circuit**
- the app now **keeps your initial-value mode**
- old fit results are cleared
- you can fit the new selection immediately

This is the correct behavior.

Why?
Because changing the selected dataset does **not** mean you forgot the circuit.
It only means the old fit result is no longer valid for the new dataset.

So after changing material or resistivity, the app should go back to:

- **Data selected**
- **Circuit still valid**
- **Initial values still valid**
- **Fit/output need to be recomputed**

That is how it works now.

---

## 5. AUTO vs MANUAL initial values

### AUTO mode
Use this by default.
The app estimates starting values from the selected dataset.

This is best when:

- you are not an electrochemistry expert
- you want quick fitting
- you are switching between many datasets

### MANUAL mode
Use this only if the researcher tells you exact values to enter.

Important:

- manual values are reused for every fit
- bad manual values can make fitting fail
- AUTO mode is usually safer for non-experts

---

## 6. What happens when selection changes

### If you change only material/resistivity
The app keeps:

- loaded file
- circuit
- initial-value mode

The app clears:

- fitted graph
- fitted parameters
- output table
- batch results

This is normal.

### If you load a new file
The app clears everything.
This is also normal.

---

## 7. How to read the tabs

### Data tab
Shows what file and dataset are selected.

### Circuit tab
Shows the circuit formula and parameter order.

### Initial Values tab
Shows the values used as the starting point for fitting.

### Fit tab
Shows fit summary text.
Graphs appear in a separate plot window.

### Output tab
Shows export-ready information.

---

## 8. Common mistakes

### Mistake 1: Wrong circuit
A fit can converge even when the circuit is physically wrong.
Always let the researcher approve the circuit.

### Mistake 2: Using MANUAL values for everything
One set of manual values may work for one dataset and fail for another.
AUTO mode is usually better.

### Mistake 3: Forgetting that old fits are cleared after selection change
If you change material or resistivity, you must fit again.
The old fit belonged to the old dataset.

### Mistake 4: Bad Excel layout
The file reader expects a specific Excel structure.
If the sheet layout changes a lot, loading may fail.

---

## 9. What fit-quality numbers mean

You will see values like:

- `chi_square`
- `reduced_chi_square`
- `chi_square_modulus`
- `reduced_chi_square_modulus`
- `RMSE_real`
- `RMSE_imag`
- `R2_real`
- `R2_imag`
- `R2_combined`

For EIS work, the modulus-weighted chi-square values are usually more useful than R².
Do not judge the science only from one number.
Always inspect the graph too.

---

## 10. Simple recommended workflow for non-experts

Do this unless the researcher says otherwise:

1. Load file
2. Pick one material
3. Pick one resistivity
4. Plot Nyquist
5. Enter circuit
6. Keep AUTO initial values
7. Fit Graph
8. Check the fit visually
9. Get Output
10. Repeat for other materials/resistivities

For many datasets:

1. Load file
2. Enter circuit once
3. Keep AUTO initial values
4. Select the group you want
5. Click Compute All
6. Use Compare Graphs
7. Export results

---

## 11. Troubleshooting

### “No valid EIS data blocks were found”
The Excel file format probably does not match what the app expects.

### Fit fails
Possible reasons:

- wrong circuit
- noisy data
- manual initial values are bad
- dataset is very different from the others

Try this first:

- use AUTO initial values
- start with a simpler circuit
- check the Nyquist plot

### Output is empty
You probably changed selection and did not fit again yet.
Run **Fit Graph** or **Compute All**.

---

## 12. One-sentence rule to remember

**Changing the dataset keeps the circuit, but clears the old fit.**
