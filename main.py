import pandas as pd
from pprint import pprint

df = pd.read_excel("row_reduced.xlsx", sheet_name=0, header=None)

result = {}
rows, cols = df.shape

col_names = ["Frequency", "Z_real", "Z_imag", "Z", "Phase", "Time"]


for start in range(cols - 5):
    header = str(df.iloc[2, start]).strip()

    # each dataset starts where the header row says Frequency
    if not header.startswith("Frequency"):
        continue

    # in your file, section + resistivity are stored above the 3rd col of the 6-col block
    meta_col = start + 2

    section_raw = df.iloc[0, meta_col]
    res_raw = df.iloc[1, meta_col]

    if pd.isna(section_raw) or pd.isna(res_raw):
        continue

    section = str(section_raw).strip()
    res = float(str(res_raw).replace("A/cm2", "").strip())

    # read data until blank in the Frequency column
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
        "Z_real": data["Z_real"].dropna().tolist(),
        "Z_imag": data["Z_imag"].dropna().tolist(),
        "Z": data["Z"].dropna().tolist(),
        "Phase": data["Phase"].dropna().tolist(),
        "Time": data["Time"].dropna().tolist(),
    }

    result.setdefault(section, []).append(block_dict)

# pprint(result, width=140, sort_dicts=False)


# Searches For section_name > resistivity in result data set
# OUTPUT  {"res": res, "frequency": [frequencies], ...}
def access_data(result, section_name, resistivity):
    section = result.get(section_name, [])

    for block in section:
        if block.get("res") == resistivity:
            return block

    return None


# pprint(access_data(result, "EOC", 1))


def main():          
    return