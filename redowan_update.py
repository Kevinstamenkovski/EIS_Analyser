import pandas as pd
from pprint import pprint
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from impedance.visualization import plot_nyquist
from impedance.models.circuits import CustomCircuit


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


def generate_nyquist_graph(ax, data):
    ax.plot(data["Z'"], data["-Z''"])
    ax.set_title("Nyquist Plot")
    ax.set_xlabel("Z'")
    ax.set_ylabel("-Z''")


def generate_bode_graph(ax, data):
    ax.plot(data["Frequency"], data["Z"])
    ax.set_title("Bode Plot")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Z")


def main():
    result = file_read()

    # mat = input("What EOC material it is: ").strip()
    # res = input("What resistivity is it: ").strip()

    # try:
    #     data = access_data(result, mat, float(res))
    # except ValueError:
    #     print("Resistivity HAS TO BE FLOAT!")
    #     print(f"Res: {res}, Mat: {mat}")


    data = access_data(result, "EOC", 0.07)

    #pprint(data)

    # if data is None:
    #     print("No matching data found.")
    #     continue

###########################################################################
    circuit = 'L0-R0-p(R1,CPE1)-p(R2,CPE2)'

    ##
    # L0 ->>> L0     ````   starting number: 0.005 _}}
    # R0 ->>> L0-R0     `````  Starting number: 100
    # p(R1,C1) _>>>> L0-R0-p(R1,C1)
    ##
    initial_guess = [0.0001, .01, .01, 10, .1, 1, 1, .1]

    circuit = CustomCircuit(circuit, initial_guess=initial_guess)

    circuit.fit(data["Frequency"], data["Z"])

    pprint(circuit.parameters_)
    print(circuit)  # this is the holy grail of our code

    Z_fit = circuit.predict(data["Frequency"])


###########################################################################

    fig, ax = plt.subplots()

    generate_nyquist_graph(ax, data)

    print("#################################")

    plot_nyquist(Z_fit, fmt='-', scale=10, ax=ax)

    plt.legend(["Data", "Fit"])
    plt.tight_layout()
    plt.show()
    plt.close(fig)


main()