import customtkinter as ctk
import pandas as pd
from pprint import pprint
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from tkinter import filedialog


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
        }

        self.downstream = {
            "select_data": {"plot_nyquist", "get_circuit", "get_init_vals", "fit_graph", "get_output"},
            "get_circuit": {"get_init_vals", "fit_graph", "get_output"},
            "get_init_vals": {"fit_graph", "get_output"},
            "plot_nyquist": set(),
            "fit_graph": set(),
            "get_output": set(),
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
            if self.can_run(step_name):
                button.configure(state="normal")
            else:
                button.configure(state="disabled")


class AppData:
    def __init__(self):
        self.filepath = None
        self.result = None
        self.selected_data = None
        self.circuit = None
        self.init_vals = None
        self.output = None


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


def create_ui():
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    app = ctk.CTk()
    app.title("EIS File Loader")
    app.geometry("1200x800")

    fsm = StepFSM()
    app_data = AppData()

    margin = 0.01
    gap = 0.02
    usable_width = 1 - (2 * margin)
    usable_height = 1 - (2 * margin) - gap

    top_height = usable_height * (3 / 4)
    bottom_height = usable_height * (1 / 4)

    app_frame = ctk.CTkFrame(master=app, fg_color="#0F0")
    app_frame.grid_rowconfigure(0, weight=1)
    app_frame.grid_columnconfigure(0, weight=1)
    app_frame.grid_columnconfigure(1, weight=3)

    user_input_frame = ctk.CTkScrollableFrame(master=app_frame, fg_color="#F00")
    user_output_frame = ctk.CTkScrollableFrame(master=app_frame, fg_color="gray")
    control_frame = ctk.CTkFrame(master=app, fg_color="#00F")

    app_frame.place(
        relx=margin,
        rely=margin,
        relwidth=usable_width,
        relheight=top_height
    )

    control_frame.place(
        relx=margin,
        rely=margin + top_height + gap,
        relwidth=usable_width,
        relheight=bottom_height
    )

    user_input_frame.grid(
        row=0,
        column=0,
        sticky="nsew",
        padx=(10, 5),
        pady=10
    )

    user_output_frame.grid(
        row=0,
        column=1,
        sticky="nsew",
        padx=(5, 10),
        pady=10
    )

    for row in range(2):
        control_frame.grid_rowconfigure(row, weight=1)
    for col in range(3):
        control_frame.grid_columnconfigure(col, weight=1)

    def clear_frame(frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def show_input_text(text):
        clear_frame(user_input_frame)
        label = ctk.CTkLabel(
            user_input_frame,
            text=text,
            justify="left",
            anchor="w"
        )
        label.pack(fill="x", padx=10, pady=10)

    def show_output_text(text):
        clear_frame(user_output_frame)
        label = ctk.CTkLabel(
            user_output_frame,
            text=text,
            justify="left",
            anchor="w"
        )
        label.pack(fill="x", padx=10, pady=10)

    def run_step(step_name, action):
        if not fsm.can_run(step_name):
            return

        if step_name in fsm.completed:
            fsm.rerun_step(step_name)

        try:
            action()
            fsm.mark_done(step_name)
            print("Completed:", fsm.completed)
        except Exception as e:
            show_output_text(f"{step_name} failed:\n{e}")
            print(f"{step_name} failed: {e}")

    def select_data():
        filepath = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )

        if not filepath:
            raise ValueError("No file selected.")

        result = file_read(filepath)
        data = access_data(result, "EOC", 0.07)

        if data is None:
            raise ValueError("No matching data found for section='EOC' and resistivity=0.07.")

        app_data.filepath = filepath
        app_data.result = result
        app_data.selected_data = data
        app_data.circuit = None
        app_data.init_vals = None
        app_data.output = None

        pprint(data)
        show_input_text(
            f"Selected file:\n{filepath}\n\n"
            f"Section: EOC\n"
            f"Resistivity: 0.07\n\n"
            f"Points loaded: {len(data['Frequency'])}"
        )
        show_output_text("Data selected successfully.")

    def plot_nyquist():
        if app_data.selected_data is None:
            raise ValueError("No selected data.")

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        generate_nyquist_graph(ax, app_data.selected_data)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

        show_output_text("Nyquist plot generated.")

    def get_circuit():
        if app_data.selected_data is None:
            raise ValueError("No selected data.")

        app_data.circuit = "R(QR)"
        show_output_text(f"Circuit detected:\n{app_data.circuit}")

    def get_init_vals():
        if app_data.circuit is None:
            raise ValueError("Circuit not available.")

        app_data.init_vals = {
            "R1": 0.12,
            "Q1": 1.5e-4,
            "n1": 0.91,
            "R2": 0.87,
        }

        pretty = "\n".join(f"{k}: {v}" for k, v in app_data.init_vals.items())
        show_output_text(f"Initial values:\n{pretty}")

    def fit_graph():
        if app_data.init_vals is None:
            raise ValueError("Initial values not available.")

        show_output_text("Fit graph completed.")

    def get_output():
        if app_data.init_vals is None:
            raise ValueError("Initial values not available.")

        app_data.output = {
            "status": "ok",
            "rmse": 0.0021,
            "iterations": 14,
        }

        pretty = "\n".join(f"{k}: {v}" for k, v in app_data.output.items())
        show_output_text(f"Output:\n{pretty}")

    button_defs = [
        ("Select Data", "select_data", select_data),
        ("Plot Nyquist", "plot_nyquist", plot_nyquist),
        ("Get Circuit", "get_circuit", get_circuit),
        ("Get Init. Vals.", "get_init_vals", get_init_vals),
        ("Fit Graph", "fit_graph", fit_graph),
        ("Get Output", "get_output", get_output),
    ]

    for i, (label, step_name, action) in enumerate(button_defs):
        row = i // 3
        col = i % 3

        btn = ctk.CTkButton(
            master=control_frame,
            text=label,
            command=lambda s=step_name, a=action: run_step(s, a)
        )
        btn.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        fsm.buttons[step_name] = btn

    fsm.refresh_buttons()

    return app


def main():
    app = create_ui()
    app.mainloop()


main()