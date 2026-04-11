import re
import numpy as np
import pandas as pd
import customtkinter as ctk
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from tkinter import filedialog
from impedance.models.circuits import CustomCircuit


ALL_MATERIALS = "__ALL_MATERIALS__"
ALL_RESISTIVITIES = "__ALL_RESISTIVITIES__"
EPS = 1e-30


class StepFSM:
    def __init__(self):
        self.completed = set()
        self.buttons = {}

        self.dependencies = {
            "select_data": [],
            "plot_nyquist": ["select_data"],
            "get_circuit": ["select_data"],
            "get_init_vals": ["get_circuit"],
            "fit_graph": ["get_init_vals"],
            "get_output": ["get_init_vals"],
            "compute_all": ["get_init_vals"],
            "compare_graphs": ["get_init_vals"],
            "batch_fit_report": ["get_init_vals"],
            "export_batch_report": ["batch_fit_report"],
        }

        self.downstream = {
            "select_data": {
                "plot_nyquist",
                "get_circuit",
                "get_init_vals",
                "fit_graph",
                "get_output",
                "compute_all",
                "compare_graphs",
                "batch_fit_report",
                "export_batch_report",
            },
            "get_circuit": {
                "get_init_vals",
                "fit_graph",
                "get_output",
                "compute_all",
                "compare_graphs",
                "batch_fit_report",
                "export_batch_report",
            },
            "get_init_vals": {
                "fit_graph",
                "get_output",
                "compute_all",
                "compare_graphs",
                "batch_fit_report",
                "export_batch_report",
            },
            "fit_graph": set(),
            "get_output": set(),
            "compute_all": set(),
            "compare_graphs": set(),
            "plot_nyquist": set(),
            "batch_fit_report": {"export_batch_report"},
            "export_batch_report": set(),
        }

    def can_run(self, step_name):
        required = self.dependencies.get(step_name, [])
        return all(dep in self.completed for dep in required)

    def invalidate_downstream(self, step_name):
        for step in self.downstream.get(step_name, set()):
            self.completed.discard(step)

    def mark_done(self, step_name):
        self.completed.add(step_name)
        self.refresh_buttons()

    def rerun_step(self, step_name):
        self.invalidate_downstream(step_name)
        self.completed.discard(step_name)
        self.refresh_buttons()

    def refresh_buttons(self):
        for step_name, button in self.buttons.items():
            button.configure(state="normal" if self.can_run(step_name) else "disabled")


class AppData:
    def __init__(self):
        self.filepath = None
        self.result = None

        self.selected_material = None
        self.selected_resistivity = None
        self.selected_data = None

        self.circuit_formula = None
        self.parameter_names = []

        self.use_auto_initial_guess = True
        self.initial_guess = []
        self.manual_initial_guess = []

        self.fit_parameters = None
        self.fit_prediction = None
        self.fit_circuit_string = None

        self.output_df = None
        self.params_df = None
        self.stats_df = None

        self.batch_results = {}
        self.available_plot_keys = []
        self.selected_plot_key = None

        self.batch_fit_rows = []
        self.batch_report_df = None


def natural_sort_key(text):
    text = str(text)
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def prettify_report_columns(df):
    rename_map = {}
    for col in df.columns:
        if isinstance(col, str):
            if col.startswith("CPE") and col.endswith("_T"):
                rename_map[col] = col.replace("_T", "_Y0")
            elif col.startswith("CPE") and col.endswith("_P"):
                rename_map[col] = col.replace("_P", "_n")
    return df.rename(columns=rename_map)


def build_batch_report_df(batch_fit_rows):
    if not batch_fit_rows:
        return pd.DataFrame()

    df = pd.DataFrame(batch_fit_rows)

    fixed_cols = [
        "EIS_ID",
        "Material",
        "Resistivity",
        "Circuit",
        "n_points",
        "n_params",
        "degrees_of_freedom",
    ]

    metric_cols = [
        "chi_square",
        "reduced_chi_square",
        "chi_square_modulus",
        "reduced_chi_square_modulus",
        "RMSE_real",
        "RMSE_imag",
        "RMSE_modulus",
        "MAE_real",
        "MAE_imag",
        "MAE_modulus",
        "R2_real",
        "R2_imag",
        "R2_combined",
    ]

    special_cols = set(fixed_cols + metric_cols)
    param_cols = [c for c in df.columns if c not in special_cols]
    param_cols = sorted(param_cols, key=natural_sort_key)

    ordered_cols = (
        [c for c in fixed_cols if c in df.columns]
        + param_cols
        + [c for c in metric_cols if c in df.columns]
    )

    df = df[ordered_cols]
    df = prettify_report_columns(df)
    return df


def append_batch_fit_result(
    batch_rows,
    material,
    resistivity,
    circuit_formula,
    parameter_names,
    fit_parameters,
    stats_dict,
):
    row = {
        "EIS_ID": f"{material} | {resistivity}",
        "Material": material,
        "Resistivity": resistivity,
        "Circuit": circuit_formula,
    }

    for name, value in zip(parameter_names, fit_parameters):
        row[name] = float(value)

    if stats_dict is not None:
        for key, value in stats_dict.items():
            row[key] = value

    batch_rows.append(row)


def compute_fit_statistics(z_data, z_fit, n_params, eps=EPS):
    z_data = np.asarray(z_data, dtype=complex)
    z_fit = np.asarray(z_fit, dtype=complex)

    if len(z_data) != len(z_fit):
        raise ValueError("z_data and z_fit must have the same length.")

    n_points = len(z_data)
    dof = max(2 * n_points - n_params, 1)

    residual = z_data - z_fit

    chi_square = np.sum((residual.real ** 2) + (residual.imag ** 2))
    reduced_chi_square = chi_square / dof

    weights = np.abs(z_data) ** 2
    weights = np.where(weights < eps, eps, weights)
    chi_square_modulus = np.sum(((residual.real ** 2) + (residual.imag ** 2)) / weights)
    reduced_chi_square_modulus = chi_square_modulus / dof

    rmse_real = np.sqrt(np.mean(residual.real ** 2))
    rmse_imag = np.sqrt(np.mean(residual.imag ** 2))
    rmse_modulus = np.sqrt(np.mean(np.abs(residual) ** 2))

    mae_real = np.mean(np.abs(residual.real))
    mae_imag = np.mean(np.abs(residual.imag))
    mae_modulus = np.mean(np.abs(residual))

    ss_res_real = np.sum((z_data.real - z_fit.real) ** 2)
    ss_tot_real = np.sum((z_data.real - np.mean(z_data.real)) ** 2)
    r2_real = np.nan if ss_tot_real < eps else 1 - ss_res_real / ss_tot_real

    y_imag_data = -z_data.imag
    y_imag_fit = -z_fit.imag
    ss_res_imag = np.sum((y_imag_data - y_imag_fit) ** 2)
    ss_tot_imag = np.sum((y_imag_data - np.mean(y_imag_data)) ** 2)
    r2_imag = np.nan if ss_tot_imag < eps else 1 - ss_res_imag / ss_tot_imag

    y_all = np.concatenate([z_data.real, -z_data.imag])
    yhat_all = np.concatenate([z_fit.real, -z_fit.imag])
    ss_res_all = np.sum((y_all - yhat_all) ** 2)
    ss_tot_all = np.sum((y_all - np.mean(y_all)) ** 2)
    r2_combined = np.nan if ss_tot_all < eps else 1 - ss_res_all / ss_tot_all

    return {
        "n_points": n_points,
        "n_params": n_params,
        "degrees_of_freedom": dof,
        "chi_square": chi_square,
        "reduced_chi_square": reduced_chi_square,
        "chi_square_modulus": chi_square_modulus,
        "reduced_chi_square_modulus": reduced_chi_square_modulus,
        "RMSE_real": rmse_real,
        "RMSE_imag": rmse_imag,
        "RMSE_modulus": rmse_modulus,
        "MAE_real": mae_real,
        "MAE_imag": mae_imag,
        "MAE_modulus": mae_modulus,
        "R2_real": r2_real,
        "R2_imag": r2_imag,
        "R2_combined": r2_combined,
    }


def parse_parameter_names(circuit_formula):
    element_tokens = re.findall(r"[A-Za-z]+\d+", circuit_formula)

    param_map = {
        "R": lambda tok: [tok],
        "C": lambda tok: [tok],
        "L": lambda tok: [tok],
        "W": lambda tok: [tok],
        "CPE": lambda tok: [f"{tok}_T", f"{tok}_P"],
        "Wo": lambda tok: [f"{tok}_R", f"{tok}_T"],
        "Ws": lambda tok: [f"{tok}_R", f"{tok}_T"],
        "G": lambda tok: [f"{tok}_Y", f"{tok}_A"],
    }

    parameter_names = []

    for token in element_tokens:
        prefix_match = re.match(r"[A-Za-z]+", token)
        if not prefix_match:
            continue

        prefix = prefix_match.group(0)
        if prefix in param_map:
            parameter_names.extend(param_map[prefix](token))
        else:
            parameter_names.append(token)

    return parameter_names


def extract_numeric(value):
    if pd.isna(value):
        return None
    text = str(value)
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


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
        res = extract_numeric(res_raw)
        if res is None:
            continue

        j = 3
        data_rows = []
        while j < rows:
            freq_val = df.iloc[j, start]
            if pd.isna(freq_val):
                break
            data_rows.append(df.iloc[j, start : start + 6].tolist())
            j += 1

        if not data_rows:
            continue

        data = pd.DataFrame(data_rows, columns=col_names)

        freq = pd.to_numeric(data["Frequency"], errors="coerce").to_numpy(dtype=float)
        z_real = pd.to_numeric(data["Z'"], errors="coerce").to_numpy(dtype=float)
        neg_z_imag = pd.to_numeric(data["-Z''"], errors="coerce").to_numpy(dtype=float)
        z_mag = pd.to_numeric(data["Z"], errors="coerce").to_numpy(dtype=float) if "Z" in data else np.full_like(freq, np.nan)
        phase = pd.to_numeric(data["Phase"], errors="coerce").to_numpy(dtype=float) if "Phase" in data else np.full_like(freq, np.nan)
        time = pd.to_numeric(data["Time"], errors="coerce").to_numpy(dtype=float) if "Time" in data else np.full_like(freq, np.nan)

        z_complex = z_real - 1j * neg_z_imag

        mask = np.isfinite(freq) & np.isfinite(z_real) & np.isfinite(neg_z_imag) & (freq > 0)

        freq = freq[mask]
        z_real = z_real[mask]
        neg_z_imag = neg_z_imag[mask]
        z_complex = z_complex[mask]
        z_mag = z_mag[mask]
        phase = phase[mask]
        time = time[mask]

        if len(freq) < 3:
            continue

        sort_idx = np.argsort(freq)[::-1]
        freq = freq[sort_idx]
        z_real = z_real[sort_idx]
        neg_z_imag = neg_z_imag[sort_idx]
        z_complex = z_complex[sort_idx]
        z_mag = z_mag[sort_idx]
        phase = phase[sort_idx]
        time = time[sort_idx]

        block_dict = {
            "res": float(res),
            "Frequency": freq,
            "Z'": z_real,
            "-Z''": neg_z_imag,
            "Z_complex": z_complex,
            "Z_mag": z_mag,
            "Phase": phase,
            "Time": time,
        }

        result.setdefault(section, []).append(block_dict)

    return result


def access_data(result, section_name, resistivity, rtol=1e-8, atol=1e-12):
    section = result.get(section_name, [])
    for block in section:
        if np.isclose(float(block.get("res")), float(resistivity), rtol=rtol, atol=atol):
            return block
    return None


def generate_nyquist_graph(ax, z_complex, label=None, fmt="-"):
    z_complex = np.asarray(z_complex, dtype=complex)
    ax.plot(z_complex.real, -z_complex.imag, fmt, label=label)
    ax.set_title("Nyquist Plot")
    ax.set_xlabel("Z' (Ohms)")
    ax.set_ylabel("-Z'' (Ohms)")
    ax.grid(True)


def estimate_peak_frequencies(freq, z_complex, max_peaks=3):
    y = -np.asarray(z_complex, dtype=complex).imag
    freq = np.asarray(freq, dtype=float)

    if len(y) < 3:
        return []

    peak_idx = []
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] > 0:
            peak_idx.append(i)

    if not peak_idx:
        peak_idx = [int(np.argmax(y))] if np.max(y) > 0 else []

    peak_idx = sorted(peak_idx, key=lambda i: y[i], reverse=True)[:max_peaks]
    return [float(freq[i]) for i in peak_idx if np.isfinite(freq[i]) and freq[i] > 0]


def auto_guess_initial_values(circuit_formula, parameter_names, data):
    freq = np.asarray(data["Frequency"], dtype=float)
    z = np.asarray(data["Z_complex"], dtype=complex)
    z_real = z.real
    neg_z_imag = -z.imag

    hf_idx = int(np.argmax(freq))
    lf_idx = int(np.argmin(freq))

    r0_est = max(float(z_real[hf_idx]), EPS)
    total_span = max(float(np.max(z_real) - np.min(z_real)), EPS)
    imag_peak = max(float(np.max(neg_z_imag)), EPS)
    resistive_names = [name for name in parameter_names if re.fullmatch(r"R\d+", name)]
    peak_freqs = estimate_peak_frequencies(freq, z, max_peaks=max(1, len(resistive_names)))

    if not peak_freqs:
        mid_freq = float(np.sqrt(max(freq[hf_idx], EPS) * max(freq[lf_idx], EPS)))
        peak_freqs = [mid_freq]

    # Estimate arc sizes. R0 is usually series ohmic resistance.
    extra_r_count = max(sum(name != "R0" for name in resistive_names), 1)
    arc_r_default = max(total_span / extra_r_count, r0_est * 0.2, 1e-3)

    guesses = []
    cpe_counter = 0
    c_counter = 0
    r_counter = 0
    l_counter = 0
    w_counter = 0
    g_counter = 0

    for name in parameter_names:
        if re.fullmatch(r"R\d+", name):
            if name == "R0":
                value = r0_est
            else:
                peak_freq = peak_freqs[min(r_counter, len(peak_freqs) - 1)]
                diameter_guess = max(arc_r_default, 2.0 * imag_peak, 1e-3)
                value = diameter_guess
                r_counter += 1
            guesses.append(float(max(value, EPS)))
            continue

        if re.fullmatch(r"C\d+", name):
            peak_freq = peak_freqs[min(c_counter, len(peak_freqs) - 1)]
            r_char = max(arc_r_default, 1e-3)
            value = 1.0 / max(2 * np.pi * peak_freq * r_char, 1e-12)
            guesses.append(float(np.clip(value, 1e-12, 1.0)))
            c_counter += 1
            continue

        if re.fullmatch(r"L\d+", name):
            high_f = max(float(freq[hf_idx]), EPS)
            high_im = abs(float(z.imag[hf_idx]))
            value = high_im / (2 * np.pi * high_f)
            guesses.append(float(np.clip(max(value, 1e-9), 1e-9, 1.0)))
            l_counter += 1
            continue

        if re.fullmatch(r"W\d+", name):
            guesses.append(1.0)
            w_counter += 1
            continue

        if name.endswith("_T") and name.startswith("CPE"):
            peak_freq = peak_freqs[min(cpe_counter, len(peak_freqs) - 1)]
            r_char = max(arc_r_default, 1e-3)
            n_default = 0.85
            omega_peak = 2 * np.pi * max(peak_freq, EPS)
            value = 1.0 / max(r_char * (omega_peak ** n_default), 1e-12)
            guesses.append(float(np.clip(value, 1e-12, 1.0)))
            continue

        if name.endswith("_P") and name.startswith("CPE"):
            guesses.append(0.85)
            cpe_counter += 1
            continue

        if (name.endswith("_R") and name.startswith("Wo")) or (name.endswith("_R") and name.startswith("Ws")):
            guesses.append(max(arc_r_default, 1e-3))
            continue

        if (name.endswith("_T") and name.startswith("Wo")) or (name.endswith("_T") and name.startswith("Ws")):
            low_f = max(float(freq[lf_idx]), EPS)
            guesses.append(float(np.clip(1.0 / (2 * np.pi * low_f), 1e-6, 1e6)))
            continue

        if name.endswith("_Y") and name.startswith("G"):
            guesses.append(1e-3)
            continue

        if name.endswith("_A") and name.startswith("G"):
            guesses.append(1.0)
            g_counter += 1
            continue

        guesses.append(1.0)

    return [float(max(val, EPS)) for val in guesses]


def create_ui():
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    app = ctk.CTk()
    app.title("EIS File Loader")
    app.geometry("1350x900")

    fsm = StepFSM()
    app_data = AppData()

    margin = 0.01
    gap = 0.02
    usable_width = 1 - (2 * margin)
    usable_height = 1 - (2 * margin) - gap

    top_height = usable_height * 0.74
    bottom_height = usable_height * 0.24

    app_frame = ctk.CTkFrame(master=app)
    app_frame.grid_rowconfigure(0, weight=1)
    app_frame.grid_columnconfigure(0, weight=1)
    app_frame.grid_columnconfigure(1, weight=3)

    user_input_frame = ctk.CTkScrollableFrame(master=app_frame)
    user_output_frame = ctk.CTkTabview(master=app_frame)
    control_frame = ctk.CTkFrame(master=app)

    app_frame.place(relx=margin, rely=margin, relwidth=usable_width, relheight=top_height)
    control_frame.place(relx=margin, rely=margin + top_height + gap, relwidth=usable_width, relheight=bottom_height)

    user_input_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
    user_output_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)

    for row in range(4):
        control_frame.grid_rowconfigure(row, weight=1)
    for col in range(3):
        control_frame.grid_columnconfigure(col, weight=1)

    for tab_name in ["Data", "Circuit", "Initial Values", "Fit", "Output"]:
        user_output_frame.add(tab_name)

    material_var = ctk.StringVar(value="")
    resistivity_var = ctk.StringVar(value="")
    plot_select_var = ctk.StringVar(value="")

    def clear_frame(frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def clear_tab(tab_name):
        tab = user_output_frame.tab(tab_name)
        for widget in tab.winfo_children():
            widget.destroy()

    def clear_tabs_from(step_name):
        if step_name == "select_data":
            for name in ["Data", "Circuit", "Initial Values", "Fit", "Output"]:
                clear_tab(name)
        elif step_name == "get_circuit":
            for name in ["Circuit", "Initial Values", "Fit", "Output"]:
                clear_tab(name)
        elif step_name == "get_init_vals":
            for name in ["Initial Values", "Fit", "Output"]:
                clear_tab(name)
        elif step_name in {"fit_graph", "get_output", "compute_all", "compare_graphs", "batch_fit_report", "export_batch_report"}:
            for name in ["Fit", "Output"]:
                clear_tab(name)

    def show_tab_text(tab_name, text):
        clear_tab(tab_name)
        tab = user_output_frame.tab(tab_name)
        textbox = ctk.CTkTextbox(tab, wrap="word")
        textbox.pack(fill="both", expand=True, padx=10, pady=10)
        textbox.insert("1.0", text)
        textbox.configure(state="disabled")
        user_output_frame.set(tab_name)

    def format_plot_key(material, resistivity):
        return f"{material} | {resistivity}"

    def parse_plot_key(key):
        parts = key.split(" | ")
        if len(parts) != 2:
            raise ValueError(f"Invalid plot key: {key}")
        return parts[0], float(parts[1])

    def reset_selected_block():
        app_data.selected_material = None
        app_data.selected_resistivity = None
        app_data.selected_data = None

        app_data.circuit_formula = None
        app_data.parameter_names = []

        app_data.use_auto_initial_guess = True
        app_data.initial_guess = []
        app_data.manual_initial_guess = []

        app_data.fit_parameters = None
        app_data.fit_prediction = None
        app_data.fit_circuit_string = None

        app_data.output_df = None
        app_data.params_df = None
        app_data.stats_df = None

        app_data.batch_results = {}
        app_data.available_plot_keys = []
        app_data.selected_plot_key = None

        app_data.batch_fit_rows = []
        app_data.batch_report_df = None

    def get_selected_blocks():
        selected = []
        if not app_data.result:
            return selected

        if app_data.selected_material == ALL_MATERIALS:
            for material, blocks in app_data.result.items():
                for block in blocks:
                    selected.append((material, block["res"], block))
            return selected

        if app_data.selected_material is None:
            return selected

        blocks = app_data.result.get(app_data.selected_material, [])

        if app_data.selected_resistivity == ALL_RESISTIVITIES:
            for block in blocks:
                selected.append((app_data.selected_material, block["res"], block))
            return selected

        if app_data.selected_resistivity is not None:
            block = access_data(app_data.result, app_data.selected_material, app_data.selected_resistivity)
            if block is not None:
                selected.append((app_data.selected_material, app_data.selected_resistivity, block))

        return selected

    def get_reference_block():
        selected_blocks = get_selected_blocks()
        if selected_blocks:
            return selected_blocks[0][2]

        if app_data.result:
            first_material = sorted(app_data.result.keys())[0]
            blocks = app_data.result[first_material]
            if blocks:
                return blocks[0]
        return None

    def regenerate_initial_guess_from_reference():
        if app_data.circuit_formula is None or not app_data.parameter_names:
            app_data.initial_guess = []
            return []

        if app_data.use_auto_initial_guess or not app_data.manual_initial_guess:
            ref_block = get_reference_block()
            if ref_block is None:
                app_data.initial_guess = []
            else:
                app_data.initial_guess = auto_guess_initial_values(
                    app_data.circuit_formula,
                    app_data.parameter_names,
                    ref_block,
                )
        else:
            app_data.initial_guess = list(app_data.manual_initial_guess)

        return app_data.initial_guess

    def format_data_summary(extra_text=""):
        selected_blocks = get_selected_blocks()
        if not selected_blocks:
            text = "No selected data."
        elif len(selected_blocks) == 1:
            data = selected_blocks[0][2]
            point_count = len(data["Frequency"])
            first_freq = data["Frequency"][0] if point_count > 0 else "N/A"
            last_freq = data["Frequency"][-1] if point_count > 0 else "N/A"
            material_label = "All Materials" if app_data.selected_material == ALL_MATERIALS else app_data.selected_material
            resistivity_label = "All Resistivities" if app_data.selected_resistivity == ALL_RESISTIVITIES else app_data.selected_resistivity
            text = (
                f"File: {app_data.filepath}\n"
                f"Material: {material_label}\n"
                f"Resistivity: {resistivity_label}\n"
                f"Points loaded: {point_count}\n"
                f"Frequency range: {first_freq} -> {last_freq}\n"
                f"Approx. high-frequency intercept Z': {data['Z_complex'].real[0]:.6g} Ohm"
            )
        else:
            text = (
                f"File: {app_data.filepath}\n"
                f"Selected datasets: {len(selected_blocks)}\n"
                f"Material selection: {'All Materials' if app_data.selected_material == ALL_MATERIALS else app_data.selected_material}\n"
                f"Resistivity selection: {'All Resistivities' if app_data.selected_resistivity == ALL_RESISTIVITIES else app_data.selected_resistivity}"
            )

        if extra_text:
            text += f"\n\n{extra_text}"
        return text

    def format_circuit_summary(extra_text=""):
        if app_data.circuit_formula is None:
            text = "No circuit formula selected."
        else:
            params_text = "\n".join(app_data.parameter_names) if app_data.parameter_names else "No parameters detected."
            text = f"Circuit formula:\n{app_data.circuit_formula}\n\nFit parameters in order:\n{params_text}"
        if extra_text:
            text += f"\n\n{extra_text}"
        return text

    def format_init_vals_summary(extra_text=""):
        regenerate_initial_guess_from_reference()
        if not app_data.initial_guess:
            text = "No initial values available."
        else:
            mode = "AUTO" if app_data.use_auto_initial_guess else "MANUAL"
            vals_text = "\n".join(f"{name} = {value}" for name, value in zip(app_data.parameter_names, app_data.initial_guess))
            text = f"Initial values mode: {mode}\n\nInitial values:\n{vals_text}"
        if extra_text:
            text += f"\n\n{extra_text}"
        return text

    def format_fit_summary(extra_text=""):
        if app_data.fit_parameters is None:
            text = "Fit not run yet."
        else:
            fit_vals = "\n".join(f"{name} = {value}" for name, value in zip(app_data.parameter_names, app_data.fit_parameters))
            stats_text = ""
            if app_data.stats_df is not None and len(app_data.stats_df) == 1:
                row = app_data.stats_df.iloc[0]
                stats_text = (
                    f"\n\nFit statistics:\n"
                    f"n_points = {row['n_points']}\n"
                    f"n_params = {row['n_params']}\n"
                    f"dof = {row['degrees_of_freedom']}\n"
                    f"chi_square = {row['chi_square']:.6g}\n"
                    f"reduced_chi_square = {row['reduced_chi_square']:.6g}\n"
                    f"chi_square_modulus = {row['chi_square_modulus']:.6g}\n"
                    f"reduced_chi_square_modulus = {row['reduced_chi_square_modulus']:.6g}\n"
                    f"RMSE_real = {row['RMSE_real']:.6g}\n"
                    f"RMSE_imag = {row['RMSE_imag']:.6g}\n"
                    f"RMSE_modulus = {row['RMSE_modulus']:.6g}\n"
                    f"R2_real = {row['R2_real']:.6g}\n"
                    f"R2_imag = {row['R2_imag']:.6g}\n"
                    f"R2_combined = {row['R2_combined']:.6g}"
                )
            text = f"Fitted circuit:\n{app_data.fit_circuit_string}\n\nFitted parameters:\n{fit_vals}{stats_text}"
        if extra_text:
            text += f"\n\n{extra_text}"
        return text

    def format_output_summary(extra_text=""):
        if app_data.output_df is None:
            text = "No output generated."
        else:
            text = f"Output ready.\n\nData rows: {len(app_data.output_df)}\nData columns: {list(app_data.output_df.columns)}"
        if extra_text:
            text += f"\n\n{extra_text}"
        return text

    def get_initial_guess_for_dataset(data):
        if app_data.use_auto_initial_guess or not app_data.manual_initial_guess:
            return auto_guess_initial_values(app_data.circuit_formula, app_data.parameter_names, data)
        return list(app_data.manual_initial_guess)

    def fit_single_dataset(material, resistivity, data):
        freq = np.asarray(data["Frequency"], dtype=float)
        z_complex = np.asarray(data["Z_complex"], dtype=complex)
        initial_guess = get_initial_guess_for_dataset(data)

        circuit_obj = CustomCircuit(app_data.circuit_formula, initial_guess=initial_guess)
        try:
            circuit_obj.fit(freq, z_complex, weight_by_modulus=True)
        except TypeError:
            circuit_obj.fit(freq, z_complex)

        z_fit = np.asarray(circuit_obj.predict(freq), dtype=complex)
        fit_parameters = list(circuit_obj.parameters_)
        fit_circuit_string = str(circuit_obj)

        stats_dict = compute_fit_statistics(z_data=z_complex, z_fit=z_fit, n_params=len(fit_parameters))
        stats_df = pd.DataFrame([stats_dict])
        residual = z_complex - z_fit

        output_df = pd.DataFrame(
            {
                "Frequency": freq,
                "Zreal_data": z_complex.real,
                "Neg_Zimag_data": -z_complex.imag,
                "Zreal_fit": z_fit.real,
                "Neg_Zimag_fit": -z_fit.imag,
                "Residual_real": residual.real,
                "Residual_imag": residual.imag,
                "Residual_modulus": np.abs(residual),
            }
        )

        params_df = pd.DataFrame({"parameter": app_data.parameter_names, "value": fit_parameters})

        return {
            "material": material,
            "resistivity": resistivity,
            "data": data,
            "initial_guess_used": initial_guess,
            "fit_parameters": fit_parameters,
            "fit_prediction": z_fit,
            "fit_circuit_string": fit_circuit_string,
            "output_df": output_df,
            "params_df": params_df,
            "stats_df": stats_df,
        }

    def load_result_into_active_fields(result_obj):
        app_data.selected_material = result_obj["material"]
        app_data.selected_resistivity = result_obj["resistivity"]
        app_data.selected_data = result_obj["data"]
        app_data.fit_parameters = result_obj["fit_parameters"]
        app_data.fit_prediction = result_obj["fit_prediction"]
        app_data.fit_circuit_string = result_obj["fit_circuit_string"]
        app_data.output_df = result_obj["output_df"]
        app_data.params_df = result_obj["params_df"]
        app_data.stats_df = result_obj["stats_df"]
        app_data.initial_guess = list(result_obj.get("initial_guess_used", app_data.initial_guess))

    def sync_plot_selector_from_batch_results(default_key=None):
        plot_keys = [format_plot_key(material, resistivity) for material, resistivity in sorted(app_data.batch_results.keys())]
        app_data.available_plot_keys = plot_keys
        if not plot_keys:
            app_data.selected_plot_key = None
            return
        if default_key in plot_keys:
            app_data.selected_plot_key = default_key
        elif app_data.selected_plot_key in plot_keys:
            pass
        else:
            app_data.selected_plot_key = plot_keys[0]

    def ensure_current_single_result():
        selected_blocks = get_selected_blocks()
        if len(selected_blocks) != 1:
            raise ValueError("This action requires exactly one selected dataset, or use Compute All first.")

        material, resistivity, data = selected_blocks[0]
        key = (material, resistivity)
        if key not in app_data.batch_results:
            app_data.batch_results[key] = fit_single_dataset(material, resistivity, data)
        load_result_into_active_fields(app_data.batch_results[key])
        sync_plot_selector_from_batch_results(default_key=format_plot_key(material, resistivity))

    def build_batch_summary():
        if not app_data.batch_results:
            return "No computed results available."
        lines = [f"Computed datasets: {len(app_data.batch_results)}", ""]
        for (material, resistivity), result_obj in sorted(app_data.batch_results.items()):
            row = result_obj["stats_df"].iloc[0]
            lines.append(
                f"{material} | {resistivity} -> R2_combined={row['R2_combined']:.6g}, chi_square={row['chi_square']:.6g}"
            )
        return "\n".join(lines)

    def show_graph_selector():
        if not app_data.batch_results:
            return
        ctk.CTkLabel(user_input_frame, text="Select computed graph", anchor="w").pack(fill="x", padx=10, pady=(10, 5))
        plot_keys = [format_plot_key(material, resistivity) for material, resistivity in sorted(app_data.batch_results.keys())]
        app_data.available_plot_keys = plot_keys
        if not plot_keys:
            return
        if app_data.selected_plot_key in plot_keys:
            plot_select_var.set(app_data.selected_plot_key)
        else:
            app_data.selected_plot_key = plot_keys[0]
            plot_select_var.set(plot_keys[0])

        def on_plot_key_change(choice):
            app_data.selected_plot_key = choice

        selector = ctk.CTkOptionMenu(master=user_input_frame, variable=plot_select_var, values=plot_keys, command=on_plot_key_change)
        selector.pack(fill="x", padx=10, pady=(0, 10))

    def invalidate_after_data_change():
        fsm.invalidate_downstream("select_data")
        fsm.completed.discard("select_data")
        fsm.completed.add("select_data")
        fsm.refresh_buttons()
        clear_tabs_from("select_data")

    def clear_fit_state_only():
        app_data.fit_parameters = None
        app_data.fit_prediction = None
        app_data.fit_circuit_string = None
        app_data.output_df = None
        app_data.params_df = None
        app_data.stats_df = None
        app_data.batch_results = {}
        app_data.available_plot_keys = []
        app_data.selected_plot_key = None
        app_data.batch_fit_rows = []
        app_data.batch_report_df = None

    def invalidate_after_selection_change():
        clear_fit_state_only()
        for step in [
            "fit_graph",
            "get_output",
            "compute_all",
            "compare_graphs",
            "batch_fit_report",
            "export_batch_report",
        ]:
            fsm.completed.discard(step)
        fsm.refresh_buttons()
        clear_tab("Fit")
        clear_tab("Output")

    def show_file_and_selectors():
        clear_frame(user_input_frame)
        if not app_data.result:
            ctk.CTkLabel(user_input_frame, text="No file loaded.", anchor="w").pack(fill="x", padx=10, pady=10)
            return

        materials = [ALL_MATERIALS] + sorted(app_data.result.keys())
        material_labels = ["All Materials"] + sorted(app_data.result.keys())

        ctk.CTkLabel(user_input_frame, text=f"Selected file:\n{app_data.filepath}", justify="left", anchor="w").pack(fill="x", padx=10, pady=(10, 15))
        ctk.CTkLabel(user_input_frame, text="Material", anchor="w").pack(fill="x", padx=10, pady=(0, 5))

        material_display_to_value = dict(zip(material_labels, materials))
        material_value_to_display = {v: k for k, v in material_display_to_value.items()}
        current_material_display = material_value_to_display.get(app_data.selected_material, material_labels[0])
        material_var.set(current_material_display)

        def material_changed(display_choice):
            on_material_change(material_display_to_value[display_choice])

        material_menu = ctk.CTkOptionMenu(master=user_input_frame, variable=material_var, values=material_labels, command=material_changed)
        material_menu.pack(fill="x", padx=10, pady=(0, 10))

        if app_data.selected_material is None:
            on_material_change(ALL_MATERIALS)
            return

        ctk.CTkLabel(user_input_frame, text="Resistivity", anchor="w").pack(fill="x", padx=10, pady=(10, 5))

        if app_data.selected_material == ALL_MATERIALS:
            resistivity_values = [ALL_RESISTIVITIES]
            resistivity_labels = ["All Resistivities"]
        else:
            raw_res_values = sorted([float(block["res"]) for block in app_data.result.get(app_data.selected_material, [])])
            resistivity_values = [ALL_RESISTIVITIES] + raw_res_values
            resistivity_labels = ["All Resistivities"] + [str(val) for val in raw_res_values]

        resistivity_display_to_value = {}
        for label, value in zip(resistivity_labels, resistivity_values):
            resistivity_display_to_value[label] = value

        current_res_value = ALL_RESISTIVITIES if app_data.selected_resistivity == ALL_RESISTIVITIES else app_data.selected_resistivity
        current_res_display = "All Resistivities"
        for display_label, actual_value in resistivity_display_to_value.items():
            if actual_value == ALL_RESISTIVITIES and current_res_value == ALL_RESISTIVITIES:
                current_res_display = display_label
                break
            if actual_value != ALL_RESISTIVITIES and current_res_value is not None and np.isclose(float(actual_value), float(current_res_value)):
                current_res_display = display_label
                break
        resistivity_var.set(current_res_display)

        def resistivity_changed(display_choice):
            on_resistivity_change(resistivity_display_to_value[display_choice])

        resistivity_menu = ctk.CTkOptionMenu(master=user_input_frame, variable=resistivity_var, values=resistivity_labels, command=resistivity_changed)
        resistivity_menu.pack(fill="x", padx=10, pady=(0, 10))

        if app_data.batch_results:
            show_graph_selector()

    def show_circuit_prompt():
        clear_frame(user_input_frame)
        ctk.CTkLabel(user_input_frame, text="Enter circuit formula", anchor="w").pack(fill="x", padx=10, pady=(10, 5))
        ctk.CTkLabel(
            user_input_frame,
            text=(
                "Examples:\n"
                "L0-R0-p(R1,CPE1)\n"
                "L0-R0-p(R1,CPE1)-p(R2,CPE2)\n\n"
                "The app now auto-generates initial values.\n"
                "You can still inspect or override them in the next step."
            ),
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=10, pady=(0, 10))

        circuit_entry = ctk.CTkEntry(master=user_input_frame, placeholder_text="L0-R0-p(R1,CPE1)")
        circuit_entry.pack(fill="x", padx=10, pady=(0, 10))
        if app_data.circuit_formula:
            circuit_entry.insert(0, app_data.circuit_formula)

        def confirm_circuit():
            circuit_formula = circuit_entry.get().strip()
            if not circuit_formula:
                show_tab_text("Circuit", "Circuit formula cannot be empty.")
                return

            parameter_names = parse_parameter_names(circuit_formula)
            if not parameter_names:
                show_tab_text("Circuit", "No fit parameters detected from the circuit formula.")
                return

            app_data.circuit_formula = circuit_formula
            app_data.parameter_names = parameter_names
            app_data.use_auto_initial_guess = True
            app_data.manual_initial_guess = []
            regenerate_initial_guess_from_reference()

            app_data.fit_parameters = None
            app_data.fit_prediction = None
            app_data.fit_circuit_string = None
            app_data.output_df = None
            app_data.params_df = None
            app_data.stats_df = None
            app_data.batch_results = {}
            app_data.available_plot_keys = []
            app_data.selected_plot_key = None
            app_data.batch_fit_rows = []
            app_data.batch_report_df = None

            fsm.invalidate_downstream("get_circuit")
            fsm.completed.discard("get_circuit")
            fsm.completed.discard("get_init_vals")
            fsm.completed.add("get_circuit")
            fsm.completed.add("get_init_vals")
            fsm.refresh_buttons()

            clear_tabs_from("get_circuit")
            show_tab_text("Circuit", format_circuit_summary())
            show_tab_text("Initial Values", format_init_vals_summary("Auto-generated from the selected dataset."))
            show_init_vals_prompt()

        ctk.CTkButton(master=user_input_frame, text="Confirm Circuit", command=confirm_circuit).pack(fill="x", padx=10, pady=(5, 10))
        ctk.CTkButton(master=user_input_frame, text="Back to Data Selectors", command=show_file_and_selectors).pack(fill="x", padx=10, pady=(0, 10))

    def show_init_vals_prompt():
        clear_frame(user_input_frame)
        regenerate_initial_guess_from_reference()

        ctk.CTkLabel(user_input_frame, text="Initial values", anchor="w").pack(fill="x", padx=10, pady=(10, 10))
        if not app_data.parameter_names:
            ctk.CTkLabel(user_input_frame, text="No parameters found. Confirm circuit first.", anchor="w").pack(fill="x", padx=10, pady=10)
            return

        mode_text = "AUTO mode is active. These values are generated from the selected dataset." if app_data.use_auto_initial_guess else "MANUAL mode is active. These values will be used for every fit."
        ctk.CTkLabel(user_input_frame, text=mode_text, justify="left", anchor="w").pack(fill="x", padx=10, pady=(0, 10))

        entries = {}
        for i, param_name in enumerate(app_data.parameter_names):
            ctk.CTkLabel(user_input_frame, text=param_name, anchor="w").pack(fill="x", padx=10, pady=(5, 2))
            entry = ctk.CTkEntry(master=user_input_frame, placeholder_text=f"Value for {param_name}")
            entry.pack(fill="x", padx=10, pady=(0, 5))
            if i < len(app_data.initial_guess):
                entry.insert(0, f"{app_data.initial_guess[i]:.6g}")
            entries[param_name] = entry

        def use_auto_values():
            app_data.use_auto_initial_guess = True
            app_data.manual_initial_guess = []
            regenerate_initial_guess_from_reference()

            clear_fit_state_only()
            fsm.invalidate_downstream("get_init_vals")
            fsm.completed.discard("get_init_vals")
            fsm.completed.add("get_init_vals")
            fsm.refresh_buttons()

            clear_tabs_from("get_init_vals")
            show_tab_text("Initial Values", format_init_vals_summary("Auto initial values enabled. Previous fit results were cleared."))
            show_init_vals_prompt()

        def confirm_manual_init_vals():
            manual_guess = []
            try:
                for param_name in app_data.parameter_names:
                    raw_value = entries[param_name].get().strip()
                    if not raw_value:
                        raise ValueError(f"Missing value for {param_name}")
                    value = float(raw_value)
                    if not np.isfinite(value):
                        raise ValueError(f"Non-finite value for {param_name}")
                    if value <= 0:
                        raise ValueError(f"Initial value must be > 0 for {param_name}")
                    manual_guess.append(value)
            except ValueError as exc:
                show_tab_text("Initial Values", f"Invalid initial values:\n{exc}")
                return

            app_data.use_auto_initial_guess = False
            app_data.manual_initial_guess = manual_guess
            regenerate_initial_guess_from_reference()

            app_data.fit_parameters = None
            app_data.fit_prediction = None
            app_data.fit_circuit_string = None
            app_data.output_df = None
            app_data.params_df = None
            app_data.stats_df = None
            app_data.batch_results = {}
            app_data.available_plot_keys = []
            app_data.selected_plot_key = None
            app_data.batch_fit_rows = []
            app_data.batch_report_df = None

            fsm.invalidate_downstream("get_init_vals")
            fsm.completed.discard("get_init_vals")
            fsm.completed.add("get_init_vals")
            fsm.refresh_buttons()

            clear_tabs_from("get_init_vals")
            show_tab_text("Initial Values", format_init_vals_summary("Manual initial values saved."))

        ctk.CTkButton(master=user_input_frame, text="Use Auto Initial Values", command=use_auto_values).pack(fill="x", padx=10, pady=(10, 5))
        ctk.CTkButton(master=user_input_frame, text="Confirm Manual Initial Values", command=confirm_manual_init_vals).pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkButton(master=user_input_frame, text="Back to Circuit Input", command=show_circuit_prompt).pack(fill="x", padx=10, pady=(0, 10))

        if app_data.batch_results:
            show_graph_selector()

    def show_export_output_prompt():
        clear_frame(user_input_frame)
        ctk.CTkLabel(user_input_frame, text="Output options", anchor="w").pack(fill="x", padx=10, pady=(10, 10))
        if app_data.batch_results:
            show_graph_selector()

        def export_excel():
            if app_data.output_df is None and not app_data.batch_results:
                show_tab_text("Output", "No output available to export.")
                return

            filepath = filedialog.asksaveasfilename(title="Save output as Excel", defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
            if not filepath:
                return

            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                if app_data.batch_results:
                    summary_rows = []
                    for (material, resistivity), result_obj in sorted(app_data.batch_results.items()):
                        row = result_obj["stats_df"].iloc[0].to_dict()
                        row["material"] = material
                        row["resistivity"] = resistivity
                        summary_rows.append(row)

                        safe_sheet = re.sub(r"[^A-Za-z0-9_\-]", "_", f"{material}_{resistivity}")[:25]
                        result_obj["output_df"].to_excel(writer, sheet_name=f"fit_{safe_sheet}"[:31], index=False)
                        result_obj["params_df"].to_excel(writer, sheet_name=f"params_{safe_sheet}"[:31], index=False)

                    pd.DataFrame(summary_rows).to_excel(writer, sheet_name="fit_stats_summary", index=False)
                else:
                    app_data.output_df.to_excel(writer, sheet_name="fit_data", index=False)
                    if app_data.params_df is not None:
                        app_data.params_df.to_excel(writer, sheet_name="fit_parameters", index=False)
                    if app_data.stats_df is not None:
                        app_data.stats_df.to_excel(writer, sheet_name="fit_stats", index=False)

            show_tab_text("Output", format_output_summary(f"Saved to:\n{filepath}"))

        ctk.CTkButton(master=user_input_frame, text="Export Output to Excel", command=export_excel).pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkButton(master=user_input_frame, text="Back to Data Selectors", command=show_file_and_selectors).pack(fill="x", padx=10, pady=(0, 10))

    def on_resistivity_change(choice):
        if not app_data.result:
            return

        if choice == ALL_RESISTIVITIES:
            app_data.selected_resistivity = ALL_RESISTIVITIES
            app_data.selected_data = None
            app_data.fit_parameters = None
            app_data.fit_prediction = None
            app_data.fit_circuit_string = None
            app_data.output_df = None
            app_data.params_df = None
            app_data.stats_df = None
            app_data.batch_results = {}
            app_data.available_plot_keys = []
            app_data.selected_plot_key = None
            app_data.batch_fit_rows = []
            app_data.batch_report_df = None
            if app_data.circuit_formula:
                regenerate_initial_guess_from_reference()
            invalidate_after_selection_change()
            show_tab_text("Data", format_data_summary("All resistivities selected. Circuit kept; previous fit results cleared."))
            return

        if app_data.selected_material == ALL_MATERIALS:
            app_data.selected_resistivity = ALL_RESISTIVITIES
            app_data.selected_data = None
            invalidate_after_selection_change()
            show_tab_text("Data", format_data_summary("All materials selected. Circuit kept; previous fit results cleared."))
            return

        resistivity = float(choice)
        data = access_data(app_data.result, app_data.selected_material, resistivity)
        if data is None:
            app_data.selected_resistivity = None
            app_data.selected_data = None
            show_tab_text("Data", f"No data found for material='{app_data.selected_material}' and resistivity={resistivity}.")
            return

        app_data.selected_resistivity = resistivity
        app_data.selected_data = data
        app_data.fit_parameters = None
        app_data.fit_prediction = None
        app_data.fit_circuit_string = None
        app_data.output_df = None
        app_data.params_df = None
        app_data.stats_df = None
        app_data.batch_results = {}
        app_data.available_plot_keys = []
        app_data.selected_plot_key = None
        app_data.batch_fit_rows = []
        app_data.batch_report_df = None
        if app_data.circuit_formula:
            regenerate_initial_guess_from_reference()
        invalidate_after_selection_change()
        show_tab_text("Data", format_data_summary("Selection changed. Circuit kept; previous fit results cleared."))

    def on_material_change(choice):
        if not app_data.result:
            return

        app_data.selected_material = choice
        app_data.selected_resistivity = ALL_RESISTIVITIES
        app_data.selected_data = None
        app_data.fit_parameters = None
        app_data.fit_prediction = None
        app_data.fit_circuit_string = None
        app_data.output_df = None
        app_data.params_df = None
        app_data.stats_df = None
        app_data.batch_results = {}
        app_data.available_plot_keys = []
        app_data.selected_plot_key = None
        app_data.batch_fit_rows = []
        app_data.batch_report_df = None
        if app_data.circuit_formula:
            regenerate_initial_guess_from_reference()
        show_file_and_selectors()
        invalidate_after_selection_change()
        show_tab_text("Data", format_data_summary("Selection changed. Circuit kept; previous fit results cleared."))

    def run_step(step_name, action):
        if not fsm.can_run(step_name):
            return
        if step_name in fsm.completed:
            fsm.rerun_step(step_name)
            clear_tabs_from(step_name)
        try:
            action()
            if step_name not in {"get_circuit", "get_init_vals"}:
                fsm.mark_done(step_name)
        except Exception as e:
            error_tab = {
                "select_data": "Data",
                "plot_nyquist": "Data",
                "get_circuit": "Circuit",
                "get_init_vals": "Initial Values",
                "fit_graph": "Fit",
                "get_output": "Output",
                "compute_all": "Fit",
                "compare_graphs": "Fit",
                "batch_fit_report": "Output",
                "export_batch_report": "Output",
            }.get(step_name, "Output")
            show_tab_text(error_tab, f"{step_name} failed:\n{e}")

    def select_data():
        filepath = filedialog.askopenfilename(title="Select Excel File", filetypes=[("Excel files", "*.xlsx *.xls")])
        if not filepath:
            raise ValueError("No file selected.")
        result = file_read(filepath)
        if not result:
            raise ValueError("No valid EIS data blocks were found in the selected file.")

        app_data.filepath = filepath
        app_data.result = result
        reset_selected_block()
        show_file_and_selectors()
        show_tab_text("Data", "File loaded. Select material and resistivity.")

    def plot_nyquist_step():
        selected_blocks = get_selected_blocks()
        if not selected_blocks:
            raise ValueError("No selected material/resistivity data.")

        fig, ax = plt.subplots()
        for material, resistivity, data in selected_blocks:
            generate_nyquist_graph(ax, data["Z_complex"], label=f"{material} | {resistivity}", fmt="-")
        if len(selected_blocks) <= 12:
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        show_tab_text("Data", format_data_summary("Nyquist plot generated."))

    def get_circuit_step():
        if not get_selected_blocks():
            raise ValueError("No selected material/resistivity data.")
        show_circuit_prompt()
        if app_data.circuit_formula:
            show_tab_text("Circuit", format_circuit_summary())
        else:
            show_tab_text("Circuit", "Enter a circuit formula in the input panel.\n\nSuggested start:\nL0-R0-p(R1,CPE1)")

    def get_init_vals_step():
        if app_data.circuit_formula is None:
            raise ValueError("Circuit not available.")
        regenerate_initial_guess_from_reference()
        show_init_vals_prompt()
        show_tab_text("Initial Values", format_init_vals_summary("Auto mode is available; manual entry is optional."))

    def compute_all_step():
        if app_data.circuit_formula is None:
            raise ValueError("No circuit formula available.")
        regenerate_initial_guess_from_reference()

        selected_blocks = get_selected_blocks()
        if not selected_blocks:
            raise ValueError("No selected data available.")

        app_data.batch_results = {}
        failures = []
        for material, resistivity, data in selected_blocks:
            try:
                app_data.batch_results[(material, resistivity)] = fit_single_dataset(material, resistivity, data)
            except Exception as exc:
                failures.append(f"{material} | {resistivity}: {exc}")

        if not app_data.batch_results:
            raise ValueError("All fits failed.\n" + "\n".join(failures))

        first_key = sorted(app_data.batch_results.keys())[0]
        load_result_into_active_fields(app_data.batch_results[first_key])
        sync_plot_selector_from_batch_results(default_key=format_plot_key(*first_key))
        show_file_and_selectors()

        msg = build_batch_summary()
        if failures:
            msg += "\n\nFailures:\n" + "\n".join(failures)
        show_tab_text("Fit", msg)

    def fit_graph_step():
        if app_data.batch_results:
            if not app_data.selected_plot_key:
                raise ValueError("No computed graph selected.")
            material, resistivity = parse_plot_key(app_data.selected_plot_key)
            key = (material, resistivity)
            if key not in app_data.batch_results:
                raise ValueError("Selected graph result not found.")

            result_obj = app_data.batch_results[key]
            load_result_into_active_fields(result_obj)
            z_data = np.asarray(result_obj["data"]["Z_complex"], dtype=complex)
            z_fit = np.asarray(result_obj["fit_prediction"], dtype=complex)

            fig, ax = plt.subplots()
            generate_nyquist_graph(ax, z_data, label="Data", fmt="-")
            generate_nyquist_graph(ax, z_fit, label="Fit", fmt="--")
            ax.set_title(f"Nyquist Fit: {material} | {resistivity}")
            ax.legend()
            plt.tight_layout()
            plt.show()
            plt.close(fig)
            show_tab_text("Fit", format_fit_summary(f"Fit graph generated for {material} | {resistivity}."))
            return

        ensure_current_single_result()
        fig, ax = plt.subplots()
        generate_nyquist_graph(ax, app_data.selected_data["Z_complex"], label="Data", fmt="-")
        generate_nyquist_graph(ax, app_data.fit_prediction, label="Fit", fmt="--")
        ax.legend()
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        show_tab_text("Fit", format_fit_summary("Fit graph generated and displayed."))

    def get_output_step():
        if app_data.batch_results:
            if not app_data.selected_plot_key:
                raise ValueError("No computed result selected.")
            material, resistivity = parse_plot_key(app_data.selected_plot_key)
            key = (material, resistivity)
            if key not in app_data.batch_results:
                raise ValueError("Selected output result not found.")

            result_obj = app_data.batch_results[key]
            load_result_into_active_fields(result_obj)
            row = app_data.stats_df.iloc[0]
            output_text = format_output_summary(
                f"Selected result: {material} | {resistivity}\n\n"
                "Fitted parameters:\n"
                + "\n".join(f"{r['parameter']} = {r['value']}" for _, r in app_data.params_df.iterrows())
                + "\n\nFit quality:\n"
                f"chi_square = {row['chi_square']:.6g}\n"
                f"reduced_chi_square = {row['reduced_chi_square']:.6g}\n"
                f"chi_square_modulus = {row['chi_square_modulus']:.6g}\n"
                f"reduced_chi_square_modulus = {row['reduced_chi_square_modulus']:.6g}\n"
                f"R2_real = {row['R2_real']:.6g}\n"
                f"R2_imag = {row['R2_imag']:.6g}\n"
                f"R2_combined = {row['R2_combined']:.6g}"
            )
            show_tab_text("Output", output_text)
            show_export_output_prompt()
            return

        ensure_current_single_result()
        row = app_data.stats_df.iloc[0]
        output_text = format_output_summary(
            "Fitted parameters:\n"
            + "\n".join(f"{r['parameter']} = {r['value']}" for _, r in app_data.params_df.iterrows())
            + "\n\nFit quality:\n"
            f"chi_square = {row['chi_square']:.6g}\n"
            f"reduced_chi_square = {row['reduced_chi_square']:.6g}\n"
            f"chi_square_modulus = {row['chi_square_modulus']:.6g}\n"
            f"reduced_chi_square_modulus = {row['reduced_chi_square_modulus']:.6g}\n"
            f"R2_real = {row['R2_real']:.6g}\n"
            f"R2_imag = {row['R2_imag']:.6g}\n"
            f"R2_combined = {row['R2_combined']:.6g}"
        )
        show_tab_text("Output", output_text)
        show_export_output_prompt()

    def compare_graphs_step():
        if not app_data.batch_results:
            raise ValueError("Run Compute All first to compare graphs.")

        fig, ax = plt.subplots()
        for (material, resistivity), result_obj in sorted(app_data.batch_results.items()):
            z_data = np.asarray(result_obj["data"]["Z_complex"], dtype=complex)
            z_fit = np.asarray(result_obj["fit_prediction"], dtype=complex)
            generate_nyquist_graph(ax, z_data, label=f"Data {material} | {resistivity}", fmt="-")
            generate_nyquist_graph(ax, z_fit, label=f"Fit {material} | {resistivity}", fmt="--")

        ax.set_title("Compare Graphs")
        if len(app_data.batch_results) <= 6:
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        show_tab_text("Fit", build_batch_summary() + "\n\nComparison graph displayed.")

    def fit_all_eis_batch_step():
        if app_data.result is None:
            raise ValueError("No loaded file data found.")
        if app_data.circuit_formula is None:
            raise ValueError("No circuit formula available.")

        app_data.batch_fit_rows = []
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_title("Compare Graphs")
        ax.set_xlabel("Z' (Ohms)")
        ax.set_ylabel("-Z'' (Ohms)")
        ax.grid(True)

        selected_blocks = get_selected_blocks()
        if not selected_blocks:
            raise ValueError("No selected data available.")

        failures = []
        for material, resistivity, block in selected_blocks:
            try:
                result_obj = fit_single_dataset(material, resistivity, block)
                z_complex = np.asarray(block["Z_complex"], dtype=complex)
                z_fit = np.asarray(result_obj["fit_prediction"], dtype=complex)
                fit_parameters = list(result_obj["fit_parameters"])
                stats_dict = result_obj["stats_df"].iloc[0].to_dict()

                append_batch_fit_result(
                    batch_rows=app_data.batch_fit_rows,
                    material=material,
                    resistivity=resistivity,
                    circuit_formula=app_data.circuit_formula,
                    parameter_names=app_data.parameter_names,
                    fit_parameters=fit_parameters,
                    stats_dict=stats_dict,
                )

                ax.plot(z_complex.real, -z_complex.imag, label=f"Data {material} | {resistivity}", lw=1.5)
                ax.plot(z_fit.real, -z_fit.imag, "--", label=f"Fit {material} | {resistivity}", lw=1.5)
            except Exception as exc:
                failures.append(f"{material} | {block.get('res', 'unknown')}: {exc}")

        if len(app_data.batch_fit_rows) <= 6:
            ax.legend(fontsize=8, ncol=1)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

        app_data.batch_report_df = build_batch_report_df(app_data.batch_fit_rows)
        if app_data.batch_report_df.empty:
            msg = "Batch fit finished, but no report rows were created."
            if failures:
                msg += "\n\nFailures:\n" + "\n".join(failures)
            show_tab_text("Output", msg)
            return

        preview_text = app_data.batch_report_df.to_string(index=False)
        if failures:
            preview_text += "\n\nFailures:\n" + "\n".join(failures)
        show_tab_text("Output", preview_text[:30000])

    def export_batch_report_excel():
        if app_data.batch_report_df is None or app_data.batch_report_df.empty:
            show_tab_text("Output", "No batch report available to export.")
            return

        filepath = filedialog.asksaveasfilename(title="Save batch fit report", defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if not filepath:
            return

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            app_data.batch_report_df.to_excel(writer, sheet_name="batch_report", index=False)
        show_tab_text("Output", f"Batch report saved to:\n{filepath}")

    button_defs = [
        ("Select Data", "select_data", select_data),
        ("Plot Nyquist", "plot_nyquist", plot_nyquist_step),
        ("Get Circuit", "get_circuit", get_circuit_step),
        ("Get Init. Vals.", "get_init_vals", get_init_vals_step),
        ("Fit Graph", "fit_graph", fit_graph_step),
        ("Get Output", "get_output", get_output_step),
        ("Compute All", "compute_all", compute_all_step),
        ("Compare Graphs", "compare_graphs", compare_graphs_step),
        ("Batch Fit + Report", "batch_fit_report", fit_all_eis_batch_step),
        ("Export Batch Report", "export_batch_report", export_batch_report_excel),
    ]

    for i, (label, step_name, action) in enumerate(button_defs):
        row = i // 3
        col = i % 3
        btn = ctk.CTkButton(master=control_frame, text=label, command=lambda s=step_name, a=action: run_step(s, a))
        btn.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        fsm.buttons[step_name] = btn

    fsm.refresh_buttons()
    show_file_and_selectors()
    return app


def main():
    app = create_ui()
    app.mainloop()


if __name__ == "__main__":
    main()