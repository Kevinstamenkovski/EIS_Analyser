import pandas as pd
from pprint import pprint
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from hybdrt.models import DRT
from hybdrt import plotting as hplt


def file_read():
    df = pd.read_excel("EIS_data_DRT.xlsx", sheet_name=0, header=None)

    result = {}
    rows, cols = df.shape
    col_names = ["Frequency", "Z'", "-Z''", "Z", "Phase", "Time"]

    for start in range(cols - 5):
        header = str(df.iloc[2, start]).strip()

        if not header.startswith("Frequency"):
            continue

        meta_col = start + 2

        section_raw = df.iloc[0, meta_col]
        res_raw = df.iloc[1, meta_col]

        if pd.isna(section_raw) or pd.isna(res_raw):
            continue

        section = str(section_raw).strip()
        res = float(str(res_raw).replace("A/cm2", "").strip())

        j = 3
        data_rows = []

        while j < rows:
            freq_val = df.iloc[j, start]

            if pd.isna(freq_val):
                break

            row_vals = df.iloc[j, start:start + 6].tolist()
            data_rows.append(row_vals)
            j += 1

        if not data_rows:
            continue

        data = pd.DataFrame(data_rows, columns=col_names)
        data = data.apply(pd.to_numeric, errors="coerce")

        block_dict = {
            "res": res,
            "Frequency": data["Frequency"].dropna().tolist(),
            "Z'": data["Z'"].dropna().tolist(),
            "-Z''": data["-Z''"].dropna().tolist(),
            "Z": data["Z"].dropna().tolist(),
            "Phase": data["Phase"].dropna().tolist(),
            "Time": data["Time"].dropna().tolist(),
        }

        result.setdefault(section, []).append(block_dict)

    return result


def access_data(result, section_name, resistivity):
    section = result.get(section_name, [])

    for block in section:
        if block.get("res") == resistivity:
            return block

    return None


def prepare_eis_arrays(data):
    freq = np.array(data["Frequency"], dtype=float)
    z_real = np.array(data["Z'"], dtype=float)
    neg_z_imag = np.array(data["-Z''"], dtype=float)

    # keep only valid points
    mask = np.isfinite(freq) & np.isfinite(z_real) & np.isfinite(neg_z_imag)
    freq = freq[mask]
    z_real = z_real[mask]
    neg_z_imag = neg_z_imag[mask]

    # optional: keep only positive frequencies
    mask = freq > 0
    freq = freq[mask]
    z_real = z_real[mask]
    neg_z_imag = neg_z_imag[mask]

    # hybdrt expects complex impedance: Z = Z' + jZ''
    # your sheet stores -Z'', so actual imag part is negative of that column
    z = z_real - 1j * neg_z_imag

    return freq, z


def run_drt(data, title="EIS + DRT"):
    freq, z = prepare_eis_arrays(data)

    print("Points used:", len(freq))
    print("Freq min/max:", freq.min(), freq.max())

    # raw EIS plot
    plt.figure(figsize=(6, 5))
    hplt.plot_eis((freq, z))
    plt.title(f"{title} - Raw EIS")
    plt.tight_layout()
    plt.show()

    # DRT fit
    drt = DRT()
    drt.fit_eis(freq, z, lambda_0=1)

    # DRT result plot
    drt.plot_results()
    plt.suptitle(f"{title} - DRT Fit")
    plt.tight_layout()
    plt.show()

    return drt


def main():
    result = file_read()

    while True:
        mat = input("What EOC material it is: ").strip()
        res = input("What resistivity is it: ").strip()

        try:
            data = access_data(result, mat, float(res))
        except ValueError:
            print("Resistivity HAS TO BE FLOAT!")
            print(f"Res: {res}, Mat: {mat}")
            continue

        if data is None:
            print("No matching data found.")
            continue

        pprint(data)

        title = f"{mat} | {float(res)} A/cm2"

        try:
            run_drt(data, title=title)
        except Exception as e:
            print("DRT fit failed:")
            print(e)


main()