import pandas as pd
from pprint import pprint
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import customtkinter as ctk
from tkinter import filedialog


def file_read(filepath):
    df = pd.read_excel(filepath, sheet_name=0, header=None)

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

        try:
            res = float(str(res_raw).replace("A/cm2", "").strip())
        except ValueError:
            continue

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


def generate_nyquist_graph(ax, data):
    ax.plot(data["Z'"], data["-Z''"])
    ax.set_title("Nyquist Plot")
    ax.set_xlabel("Z'")
    ax.set_ylabel("-Z''")
    ax.grid(True)


def generate_bode_graph(ax, data):
    ax.plot(data["Frequency"], data["Z"])
    ax.set_title("Bode Plot")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Z")
    ax.grid(True)


def process_file(filepath):
    result = file_read(filepath)

    data = access_data(result, "EOC", 0.07)

    pprint(data)

    if data is None:
        print("No matching data found.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    generate_nyquist_graph(ax, data)
    # generate_bode_graph(ax, data)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def select_file():
    filepath = filedialog.askopenfilename(
        title="Select Excel File",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )

    if filepath:
        process_file(filepath)


def create_ui():
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    app = ctk.CTk()
    app.title("EIS File Loader")
    app.geometry("1200x800")

    button = ctk.CTkButton(
        app,
        text="Select Excel File",
        command=select_file
    )
    button.pack(expand=True)

    return app


def main():
    app = create_ui()
    app.mainloop()


# if __name__ == "__main__":
main()