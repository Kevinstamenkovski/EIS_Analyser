import re
import customtkinter as ctk
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from tkinter import filedialog
from impedance.visualization import plot_nyquist
from impedance.models.circuits import CustomCircuit


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
            "get_output": ["fit_graph"],
        }

        self.downstream = {
            "select_data": {"plot_nyquist", "get_circuit", "get_init_vals", "fit_graph", "get_output"},
            "get_circuit": {"get_init_vals", "fit_graph", "get_output"},
            "get_init_vals": {"fit_graph", "get_output"},
            "fit_graph": {"get_output"},
            "plot_nyquist": set(),
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

        self.selected_material = None
        self.selected_resistivity = None
        self.selected_data = None

        self.circuit_formula = None
        self.parameter_names = []
        self.initial_guess = []

        self.fit_parameters = None
        self.fit_prediction = None
        self.fit_circuit_string = None
        self.output_df = None


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


def parse_parameter_names(circuit_formula):
    """
    Builds parameter names in left-to-right order matching the circuit text.
    Extend param_map if you use more element types.
    """
    element_tokens = re.findall(r"[A-Za-z]+\d+", circuit_formula)

    param_map = {
        "R":  lambda tok: [tok],
        "C":  lambda tok: [tok],
        "L":  lambda tok: [tok],
        "W":  lambda tok: [tok],
        "CPE": lambda tok: [f"{tok}_T", f"{tok}_P"],
        "Wo":  lambda tok: [f"{tok}_R", f"{tok}_T"],
        "Ws":  lambda tok: [f"{tok}_R", f"{tok}_T"],
        "G":   lambda tok: [f"{tok}_Y", f"{tok}_A"],
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

    app_frame = ctk.CTkFrame(master=app)
    app_frame.grid_rowconfigure(0, weight=1)
    app_frame.grid_columnconfigure(0, weight=1)
    app_frame.grid_columnconfigure(1, weight=3)

    user_input_frame = ctk.CTkScrollableFrame(master=app_frame)
    user_output_frame = ctk.CTkTabview(master=app_frame)
    control_frame = ctk.CTkFrame(master=app)

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

    for tab_name in ["Data", "Circuit", "Initial Values", "Fit", "Output"]:
        user_output_frame.add(tab_name)

    material_var = ctk.StringVar(value="")
    resistivity_var = ctk.StringVar(value="")

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
        elif step_name == "fit_graph":
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

    def reset_selected_block():
        app_data.selected_material = None
        app_data.selected_resistivity = None
        app_data.selected_data = None
        app_data.circuit_formula = None
        app_data.parameter_names = []
        app_data.initial_guess = []
        app_data.fit_parameters = None
        app_data.fit_prediction = None
        app_data.fit_circuit_string = None
        app_data.output_df = None

    def format_data_summary(extra_text=""):
        if app_data.selected_data is None:
            return "No selected data."

        data = app_data.selected_data
        point_count = len(data["Frequency"])
        first_freq = data["Frequency"][0] if data["Frequency"] else "N/A"
        last_freq = data["Frequency"][-1] if data["Frequency"] else "N/A"

        text = (
            f"File: {app_data.filepath}\n"
            f"Material: {app_data.selected_material}\n"
            f"Resistivity: {app_data.selected_resistivity}\n"
            f"Points loaded: {point_count}\n"
            f"Frequency range: {first_freq} -> {last_freq}"
        )

        if extra_text:
            text += f"\n\n{extra_text}"

        return text

    def format_circuit_summary(extra_text=""):
        if app_data.circuit_formula is None:
            text = "No circuit formula selected."
        else:
            params_text = "\n".join(app_data.parameter_names) if app_data.parameter_names else "No parameters detected."
            text = (
                f"Circuit formula:\n{app_data.circuit_formula}\n\n"
                f"Fit parameters in order:\n{params_text}"
            )

        if extra_text:
            text += f"\n\n{extra_text}"

        return text

    def format_init_vals_summary(extra_text=""):
        if not app_data.initial_guess:
            text = "No initial values entered."
        else:
            vals_text = "\n".join(
                f"{name} = {value}" for name, value in zip(app_data.parameter_names, app_data.initial_guess)
            )
            text = f"Initial values:\n{vals_text}"

        if extra_text:
            text += f"\n\n{extra_text}"

        return text

    def format_fit_summary(extra_text=""):
        if app_data.fit_parameters is None:
            text = "Fit not run yet."
        else:
            fit_vals = "\n".join(
                f"{name} = {value}"
                for name, value in zip(app_data.parameter_names, app_data.fit_parameters)
            )
            text = (
                f"Fitted circuit:\n{app_data.fit_circuit_string}\n\n"
                f"Fitted parameters:\n{fit_vals}"
            )

        if extra_text:
            text += f"\n\n{extra_text}"

        return text

    def format_output_summary(extra_text=""):
        if app_data.output_df is None:
            text = "No output generated."
        else:
            text = (
                f"Output ready.\n\n"
                f"Rows: {len(app_data.output_df)}\n"
                f"Columns: {list(app_data.output_df.columns)}"
            )

        if extra_text:
            text += f"\n\n{extra_text}"

        return text

    def invalidate_after_data_change():
        fsm.invalidate_downstream("select_data")
        fsm.completed.discard("select_data")
        fsm.completed.add("select_data")
        fsm.refresh_buttons()
        clear_tabs_from("select_data")

    def show_file_and_selectors():
        clear_frame(user_input_frame)

        if not app_data.result:
            ctk.CTkLabel(
                user_input_frame,
                text="No file loaded.",
                anchor="w"
            ).pack(fill="x", padx=10, pady=10)
            return

        materials = sorted(app_data.result.keys())

        ctk.CTkLabel(
            user_input_frame,
            text=f"Selected file:\n{app_data.filepath}",
            justify="left",
            anchor="w"
        ).pack(fill="x", padx=10, pady=(10, 15))

        ctk.CTkLabel(
            user_input_frame,
            text="Material",
            anchor="w"
        ).pack(fill="x", padx=10, pady=(0, 5))

        material_menu = ctk.CTkOptionMenu(
            master=user_input_frame,
            variable=material_var,
            values=materials,
            command=on_material_change
        )
        material_menu.pack(fill="x", padx=10, pady=(0, 10))

        if materials:
            if app_data.selected_material in materials:
                material_var.set(app_data.selected_material)
            else:
                material_var.set(materials[0])
                on_material_change(materials[0])
                return

        if app_data.selected_material:
            ctk.CTkLabel(
                user_input_frame,
                text="Resistivity",
                anchor="w"
            ).pack(fill="x", padx=10, pady=(10, 5))

            resistivity_values = sorted(
                [str(block["res"]) for block in app_data.result.get(app_data.selected_material, [])],
                key=lambda x: float(x)
            )

            if resistivity_values:
                resistivity_menu = ctk.CTkOptionMenu(
                    master=user_input_frame,
                    variable=resistivity_var,
                    values=resistivity_values,
                    command=on_resistivity_change
                )
                resistivity_menu.pack(fill="x", padx=10, pady=(0, 10))

                if app_data.selected_resistivity is not None:
                    resistivity_var.set(str(app_data.selected_resistivity))
                else:
                    resistivity_var.set(resistivity_values[0])
                    on_resistivity_change(resistivity_values[0])

    def show_circuit_prompt():
        clear_frame(user_input_frame)

        ctk.CTkLabel(
            user_input_frame,
            text="Enter circuit formula",
            anchor="w"
        ).pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            user_input_frame,
            text="Example: L0-R0-p(R1,CPE1)-p(R2,CPE2)",
            anchor="w",
            justify="left"
        ).pack(fill="x", padx=10, pady=(0, 10))

        circuit_entry = ctk.CTkEntry(
            master=user_input_frame,
            placeholder_text="L0-R0-p(R1,CPE1)-p(R2,CPE2)"
        )
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
            app_data.initial_guess = []
            app_data.fit_parameters = None
            app_data.fit_prediction = None
            app_data.fit_circuit_string = None
            app_data.output_df = None

            fsm.invalidate_downstream("get_circuit")
            fsm.completed.discard("get_circuit")
            fsm.completed.add("get_circuit")
            fsm.refresh_buttons()

            clear_tabs_from("get_circuit")
            show_tab_text("Circuit", format_circuit_summary())
            show_init_vals_prompt()

        ctk.CTkButton(
            master=user_input_frame,
            text="Confirm Circuit",
            command=confirm_circuit
        ).pack(fill="x", padx=10, pady=(5, 10))

        ctk.CTkButton(
            master=user_input_frame,
            text="Back to Data Selectors",
            command=show_file_and_selectors
        ).pack(fill="x", padx=10, pady=(0, 10))

    def show_init_vals_prompt():
        clear_frame(user_input_frame)

        ctk.CTkLabel(
            user_input_frame,
            text="Enter initial values",
            anchor="w"
        ).pack(fill="x", padx=10, pady=(10, 10))

        if not app_data.parameter_names:
            ctk.CTkLabel(
                user_input_frame,
                text="No parameters found. Confirm circuit first.",
                anchor="w"
            ).pack(fill="x", padx=10, pady=10)
            return

        entries = {}

        for i, param_name in enumerate(app_data.parameter_names):
            ctk.CTkLabel(
                user_input_frame,
                text=param_name,
                anchor="w"
            ).pack(fill="x", padx=10, pady=(5, 2))

            entry = ctk.CTkEntry(
                master=user_input_frame,
                placeholder_text=f"Initial value for {param_name}"
            )
            entry.pack(fill="x", padx=10, pady=(0, 5))

            if i < len(app_data.initial_guess):
                entry.insert(0, str(app_data.initial_guess[i]))

            entries[param_name] = entry

        def confirm_init_vals():
            initial_guess = []

            try:
                for param_name in app_data.parameter_names:
                    raw_value = entries[param_name].get().strip()
                    if not raw_value:
                        raise ValueError(f"Missing value for {param_name}")
                    initial_guess.append(float(raw_value))
            except ValueError as exc:
                show_tab_text("Initial Values", f"Invalid initial values:\n{exc}")
                return

            app_data.initial_guess = initial_guess
            app_data.fit_parameters = None
            app_data.fit_prediction = None
            app_data.fit_circuit_string = None
            app_data.output_df = None

            fsm.invalidate_downstream("get_init_vals")
            fsm.completed.discard("get_init_vals")
            fsm.completed.add("get_init_vals")
            fsm.refresh_buttons()

            clear_tabs_from("get_init_vals")
            show_tab_text("Initial Values", format_init_vals_summary())

        ctk.CTkButton(
            master=user_input_frame,
            text="Confirm Initial Values",
            command=confirm_init_vals
        ).pack(fill="x", padx=10, pady=(10, 10))

        ctk.CTkButton(
            master=user_input_frame,
            text="Back to Circuit Input",
            command=show_circuit_prompt
        ).pack(fill="x", padx=10, pady=(0, 10))

    def show_export_output_prompt():
        clear_frame(user_input_frame)

        ctk.CTkLabel(
            user_input_frame,
            text="Output options",
            anchor="w"
        ).pack(fill="x", padx=10, pady=(10, 10))

        def export_excel():
            if app_data.output_df is None:
                show_tab_text("Output", "No output available to export.")
                return

            filepath = filedialog.asksaveasfilename(
                title="Save output as Excel",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")]
            )

            if not filepath:
                return

            app_data.output_df.to_excel(filepath, index=False)
            show_tab_text("Output", format_output_summary(f"Saved to:\n{filepath}"))

        ctk.CTkButton(
            master=user_input_frame,
            text="Export Output to Excel",
            command=export_excel
        ).pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkButton(
            master=user_input_frame,
            text="Back to Data Selectors",
            command=show_file_and_selectors
        ).pack(fill="x", padx=10, pady=(0, 10))

    def on_resistivity_change(choice):
        if not app_data.result or not app_data.selected_material:
            return

        try:
            resistivity = float(choice)
        except ValueError:
            show_tab_text("Data", f"Invalid resistivity value: {choice}")
            return

        data = access_data(app_data.result, app_data.selected_material, resistivity)

        if data is None:
            app_data.selected_resistivity = None
            app_data.selected_data = None
            show_tab_text("Data", f"No data found for material='{app_data.selected_material}' and resistivity={resistivity}.")
            return

        app_data.selected_resistivity = resistivity
        app_data.selected_data = data
        app_data.circuit_formula = None
        app_data.parameter_names = []
        app_data.initial_guess = []
        app_data.fit_parameters = None
        app_data.fit_prediction = None
        app_data.fit_circuit_string = None
        app_data.output_df = None

        invalidate_after_data_change()
        show_tab_text("Data", format_data_summary())

    def on_material_change(choice):
        if not app_data.result:
            return

        app_data.selected_material = choice
        app_data.selected_resistivity = None
        app_data.selected_data = None
        app_data.circuit_formula = None
        app_data.parameter_names = []
        app_data.initial_guess = []
        app_data.fit_parameters = None
        app_data.fit_prediction = None
        app_data.fit_circuit_string = None
        app_data.output_df = None

        show_file_and_selectors()

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
            }.get(step_name, "Output")

            show_tab_text(error_tab, f"{step_name} failed:\n{e}")

    def select_data():
        filepath = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )

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
        if app_data.selected_data is None:
            raise ValueError("No selected material/resistivity data.")

        fig, ax = plt.subplots()
        generate_nyquist_graph(ax, app_data.selected_data)
        plt.legend(["Data"])
        plt.tight_layout()
        plt.show()
        plt.close(fig)

        show_tab_text("Data", format_data_summary("Nyquist plot generated."))

    def get_circuit_step():
        if app_data.selected_data is None:
            raise ValueError("No selected material/resistivity data.")

        show_circuit_prompt()

        if app_data.circuit_formula:
            show_tab_text("Circuit", format_circuit_summary())
        else:
            show_tab_text(
                "Circuit",
                "Enter a circuit formula in the input panel.\n\nExample:\nL0-R0-p(R1,CPE1)-p(R2,CPE2)"
            )

    def get_init_vals_step():
        if app_data.circuit_formula is None:
            raise ValueError("Circuit not available.")

        show_init_vals_prompt()

        if app_data.initial_guess:
            show_tab_text("Initial Values", format_init_vals_summary())
        else:
            show_tab_text(
                "Initial Values",
                "Enter initial values in the input panel.\n\n"
                "These values must match the fit-parameter order shown in the Circuit tab."
            )

    def fit_graph_step():
        if app_data.selected_data is None:
            raise ValueError("No selected data available.")
        if app_data.circuit_formula is None:
            raise ValueError("No circuit formula available.")
        if not app_data.initial_guess:
            raise ValueError("No initial guess available.")

        data = app_data.selected_data
        circuit_obj = CustomCircuit(app_data.circuit_formula, initial_guess=app_data.initial_guess)
        circuit_obj.fit(data["Frequency"], data["Z"])

        app_data.fit_parameters = list(circuit_obj.parameters_)
        app_data.fit_circuit_string = str(circuit_obj)
        app_data.fit_prediction = circuit_obj.predict(data["Frequency"])

        fig, ax = plt.subplots()
        generate_nyquist_graph(ax, data)
        plot_nyquist(app_data.fit_prediction, fmt="-", scale=10, ax=ax)
        plt.legend(["Data", "Fit"])
        plt.tight_layout()
        plt.show()
        plt.close(fig)

        show_tab_text("Fit", format_fit_summary("Fit graph generated and displayed."))

    def get_output_step():
        if app_data.fit_parameters is None:
            raise ValueError("Fit results are not available.")

        data = app_data.selected_data
        z_fit = app_data.fit_prediction

        app_data.output_df = pd.DataFrame({
            "Frequency": data["Frequency"],
            "Z_data": data["Z"],
            "Z_fit": z_fit,
            "Z_prime": data["Z'"],
            "minus_Z_double_prime": data["-Z''"],
        })

        params_df = pd.DataFrame({
            "parameter": app_data.parameter_names,
            "value": app_data.fit_parameters
        })

        output_text = format_output_summary(
            "Fitted parameters:\n" +
            "\n".join(f"{row['parameter']} = {row['value']}" for _, row in params_df.iterrows())
        )
        show_tab_text("Output", output_text)
        show_export_output_prompt()

    button_defs = [
        ("Select Data", "select_data", select_data),
        ("Plot Nyquist", "plot_nyquist", plot_nyquist_step),
        ("Get Circuit", "get_circuit", get_circuit_step),
        ("Get Init. Vals.", "get_init_vals", get_init_vals_step),
        ("Fit Graph", "fit_graph", fit_graph_step),
        ("Get Output", "get_output", get_output_step),
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
    show_file_and_selectors()

    return app


def main():
    app = create_ui()
    app.mainloop()


if __name__ == "__main__":
    main()
    