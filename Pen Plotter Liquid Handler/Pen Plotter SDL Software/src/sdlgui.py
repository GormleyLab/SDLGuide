import tkinter as tk
import os
import code
from ctypes import windll
from tkinter import messagebox
from tkinter import ttk

from ttkthemes import ThemedTk

import src.liquidhandler as lh
import src.guifunctions as gf
import src.sdlvariables as var
import src.dataprocessing as dp

# Main application window, if the GUI does not fit, change the root.geometry to a different geometry,
# and the scale factor to change the main home tab entry boxes and button sizes.
root = ThemedTk(theme='clearlooks')
root.title('SDL - AxiDraw')
root.geometry()
windll.shcore.SetProcessDpiAwareness(1)
root.geometry("1920x1080")

scale_factor = 0.8

# Set the window icon
icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources", "gl_logo.ico"))
root.iconbitmap(icon_path)

# Command prompt to inferace with the program for debugging - does not work during the dispensing process
def open_console():
    console = code.InteractiveConsole(locals=globals())
    console.interact("Interactive Console. Type exit() to close.")

# Define styles
style = ttk.Style(root)
style.configure('Custom.TLabel', font=('Franklin', int(24 * scale_factor)))
style.configure('Custom.TButton', font=('Franklin', int(24 * scale_factor)))
style.configure('Custom.TEntry', font=('Franklin', int(24 * scale_factor)))
style.configure('Custom.TCheckbutton', font=('Franklin', int(24 * scale_factor)))
style.configure('TNotebook.Tab', font=('Franklin', int(24 * scale_factor)))
style.configure('Custom.TCombobox', font=('Franklin', int(24 * scale_factor)))

#----------------------------- Initialize Tab Control -------------------------------
# Tab control is used to update the GUI as you move around, and to initialize axidraw manual movement when
# switched to the manual tab.
tabControl = ttk.Notebook(root)

# Home Tab
home_tab = ttk.Frame(tabControl)
tabControl.add(home_tab, text='Home')

# Enzyme Assay Tab
enzyme_assay_tab = ttk.Frame(tabControl)
tabControl.add(enzyme_assay_tab, text='Enzyme Assay')

# Liquid Handler
liquid_handler_tab = ttk.Frame(tabControl)
tabControl.add(liquid_handler_tab, text='Liquid Handler')

# Manual Tab
manual_tab = ttk.Frame(tabControl)
tabControl.add(manual_tab, text='Manual')
tabControl.pack(expand=1, fill="both")


#----------------------------- Home Tab Layout -------------------------------
# This initializes all the entry boxes, dropdowns and buttons that will be most commonly used in the program.
header_frame = ttk.Frame(home_tab)
header_frame.grid(row=0, column=0, columnspan=10, padx=2, pady=2)
for i in range(10):
    ttk.Label(header_frame, text=f'Reagent {i+1}', style='Custom.TLabel').grid(row=0, column=i+1, padx=2, pady=2)

# Initialize input lists
reagents = []
group_vars = []
stock_concentrations = []
ph_inputs = []
# Iterate per reagent, which is set to 10 by default for limited space.
for i in range(10):

    # Entry for reagent
    reagent = ttk.Entry(header_frame, style='Custom.TEntry', width=12)
    reagent.grid(row=1, column=i+1, padx=2, pady=6, ipadx=0, ipady=4)
    reagent.config(font=('Franklin', 18))
    reagents.append(reagent)

    reagent.bind("<FocusOut>", lambda e, idx=i: update_reagents(idx))
    
    # Dropdown for group selection
    group_var = tk.StringVar()
    group_var.set('No Group')  # Set default value
    group_vars.append(group_var)
    
    # Entry for stock concentration
    stock_concentration_entry = ttk.Entry(header_frame, style='Custom.TEntry', width=12)
    stock_concentration_entry.grid(row=2, column=i+1, padx=2, pady=6, ipadx=0, ipady=4)
    stock_concentration_entry.config(font=('Franklin', 18))
    stock_concentrations.append(stock_concentration_entry)

    # Entry for ph
    ph_entry = ttk.Entry(header_frame, style='Custom.TEntry', width=12)
    ph_entry.grid(row=3, column=i+1, padx=2, pady=6, ipadx=0, ipady=4)
    ph_entry.config(font=('Franklin', 18))
    ph_inputs.append(ph_entry)

    # Dropdown for group selection
    group_dropdown = ttk.Combobox(header_frame, textvariable=group_var, values=['No Group', 'Buffer', 'Group 1', 'Group 2', 'Group 3', 'Group 4'], style='Custom.TCombobox', width=10)
    group_dropdown.grid(row=4, column=i+1, padx=2, pady=6, ipadx=4, ipady=8)
    group_dropdown.config(font=('Franklin', 18))

def create_initial_mixtures_header(parent):
    headers = ['Component 1', 'Volume (μL)', 'Component 2', 'Volume (μL)', 
               'Component 3', 'Volume (μL)', 'Component 4', 'Volume (μL)', 
               'Component 5', 'Volume (μL)']
    for i, header in enumerate(headers):
        ttk.Label(parent, text=header, style='Custom.TLabel').grid(row=6, column=i, padx=1, pady=1)

def create_initial_mixtures_entries(parent, num_rows):
    entries = []
    for row in range(num_rows):
        row_entries = []
        for col in range(10):
            entry = ttk.Entry(parent, style='Custom.TEntry', width=10)
            entry.grid(row=row+7, column=col, padx=1, pady=1, ipadx=2, ipady=2)
            entry.config(font=('Franklin', int(16 * scale_factor)))
            row_entries.append(entry)
        entries.append(row_entries)
    return entries
create_initial_mixtures_header(home_tab)
entries = create_initial_mixtures_entries(home_tab, 16)

# Bind dropdown events after dropdowns are created
ttk.Label(home_tab, text='Initial Mixtures', style='Custom.TLabel').grid(row=5, column=0, columnspan=10, padx=2, pady=2)
ttk.Label(header_frame, text='Name', style='Custom.TLabel').grid(row=1, column=0, padx=2, pady=2)
ttk.Label(header_frame, text='Concentration', style='Custom.TLabel').grid(row=2, column=0, padx=2, pady=2)
ttk.Label(header_frame, text='Ph', style='Custom.TLabel').grid(row=3, column=0, padx=2, pady=2)


ttk.Button(home_tab, text='Upload CSV', command=lambda: gf.upload_csv(entries), style='Custom.TButton').grid(row=23, column=0, columnspan=3,padx=10, pady=10)
ttk.Button(home_tab, text='Clear', command=lambda: gf.clear_entries(entries), style='Custom.TButton').grid(row=23, column=8, columnspan=3, padx=10, pady=10)
ttk.Button(home_tab, text='Reset Data', command=lambda: var.set_variables(), style='Custom.TButton').grid(row=23, column=3, columnspan=4, padx=10, pady=10)
ttk.Button(home_tab, text='Command Prompt', command=lambda: open_console(), style='Custom.TButton').grid(row=24, column=0, columnspan=3, padx=10, pady=10)

#----------------------------- Enzyme Assay Tab Layout -------------------------------
# Generate the enzyme assay tab, this is updated whenever the enzyme assay tab is opened. It takes the reagents
# that have been inputted in the home tab, and add's additional options for them for the assay/ml specific expeiment
variable_type_widgets = {}
bounds_widgets = {}
desired_activity_entry = {}
num_samples_entry = {}

def create_enzyme_assay_layout(parent):
    global update_reagents
    ttk.Label(parent, text='Enzyme Assay', style='Custom.TLabel', font=('Franklin', 28)).grid(row=0, column=0, columnspan=6, pady=(16, 16), padx=2)
    types_of_reagents = [""] * 10
    def update_reagents(idx=None):
        global variable_type_widgets
        global bounds_widgets
        global variable_type_values
        global bounds_values
        global num_of_samples
        global num_samples_entry
        global desired_activity_entry
        global get_num_samples_value

        # Save current dropdown values before clearing the UI
        for i in variable_type_widgets:
            if i in variable_type_widgets:
                variable_type_values[i] = variable_type_widgets[i].get()

        # Save current bounds values before clearing the UI
        for i in bounds_widgets:
            if i in bounds_widgets:
                bounds_values[i] = bounds_widgets[i].get()

        if desired_activity_entry:
            var.desired_activity = desired_activity_entry.get()
        
        if num_samples_entry:
            num_of_samples = num_samples_entry.get()

        # Clear existing UI elements
        for widget in parent.winfo_children():
            widget.grid_forget()

        # Recreate the header
        ttk.Label(parent, text='Enzyme Assay', style='Custom.TLabel', font=('Franklin', 28)).grid(row=0, column=0, columnspan=6, pady=(16, 16), padx=2)

        # Ensure the idx is within range before accessing types_of_reagents
        if idx is not None and idx < len(reagents):
            types_of_reagents[idx] = reagents[idx].get()  # Update the specific reagent in the list

        # Initialize variables for storing comboboxes and entries
        variable_type_widgets = {}
        bounds_widgets = {}
        desired_activity_entry = {}
        num_samples_entry = {}

        # Initialize the dictionaries if not already defined
        if 'variable_type_values' not in globals():
            variable_type_values = {}

        if 'bounds_values' not in globals():
            bounds_values = {}  # Initialize a dictionary to store bounds values

        if 'num_of_samples' not in globals():
            num_of_samples = {}  # Initialize a dictionary to store num of samples value

        row_counter = 1  # Keeps track of which row to add the elements to
        for i, reagent_name in enumerate(types_of_reagents):
            # Skip empty reagent names
            if reagent_name == "":
                continue

            # Create UI elements only for non-empty reagent names
            ttk.Label(parent, text=reagent_name, style='Custom.TLabel').grid(row=row_counter, column=0, padx=2, pady=2)

            # Dropdown to select discrete or continuous variable
            variable_type = ttk.Combobox(parent, values=["Discrete", "Continuous"], style='Custom.TCombobox', state="normal")
            variable_type.grid(row=row_counter, column=1, padx=2, pady=2, ipadx=5, ipady=5)

            # Restore previously selected value if it exists
            if i in variable_type_values:
                variable_type.set(variable_type_values[i])
            else:
                variable_type.set("Continuous")  # Set default value for new reagents

            variable_type.config(font=('Franklin', 20), width=15)
            variable_type['state'] = 'readonly'  # Make the combobox read-only

            # Save the combobox to the dictionary
            variable_type_widgets[i] = variable_type

            # Entry for bounds
            bounds_entry = ttk.Entry(parent, style='Custom.TEntry', width=10)
            bounds_entry.grid(row=row_counter, column=2, padx=2, pady=2)

            # Restore previously entered bounds if it exists
            if i in bounds_values:
                bounds_entry.insert(0, bounds_values[i])
            else:
                bounds_entry.insert(0, "(10,50)")  # Set default value for new bounds

            bounds_entry.config(font=('Franklin', 20))

            # Store the bounds entry with its index
            bounds_widgets[i] = bounds_entry

            row_counter += 1  # Increment the row counter for the next reagent

        # Desired Activity
        ttk.Label(parent, text='Desired Activity', style='Custom.TLabel', font=('Franklin', 28)).grid(row=row_counter, column=0, columnspan=2, pady=(16, 16), padx=2)
        desired_activity_entry = ttk.Entry(parent, style='Custom.TEntry', width=10)
        desired_activity_entry.grid(row=row_counter, column=2, padx=2, pady=2)
        desired_activity_entry.config(font=('Franklin', 20))

        if var.desired_activity:
            desired_activity_entry.insert(0,var.desired_activity)
        else:
            desired_activity_entry.insert(0,'0')

        # Entry for num_samples
        ttk.Label(parent, text='Number Samples', style='Custom.TLabel', font=('Franklin', 24)).grid(row=row_counter + 1, column=0, columnspan=2, pady=(16, 16), padx=2)
        num_samples_entry = ttk.Entry(parent, style='Custom.TEntry', width=10)
        num_samples_entry.grid(row=row_counter + 1, column=2, columnspan=2, pady=4, padx=2)
        num_samples_entry.config(font=('Franklin', 20))

        if num_of_samples:
            num_samples_entry.insert(0,num_of_samples)
        else:
            num_samples_entry.insert(0,12)

        def get_num_samples_value():
            # Function to retrieve and print the value from num_samples
            value = num_samples_entry.get()
            return value

        # Run button
        ttk.Button(parent, text='Run', command=lambda: gf.enzyme_assay_run(), style='Custom.TButton').grid(row=row_counter + 2, column=0, columnspan=4, pady=(16, 16), padx=2)

    # Initialize the layout
    update_reagents()
    
create_enzyme_assay_layout(enzyme_assay_tab)

#----------------------------- Liquid Handler Tab Layout -------------------------------
# Generate the manual tab for manual control of the axidraw. Syringe pump control has not been added
# because it can easily be controlled using the interface on the device

run_button = tk.Button(
    liquid_handler_tab,
    text="Run",
    font=('Arial', 32, 'bold'),
    width=25,
    command=lambda: gf.generate_custom_input_formulations()
)
run_button.pack(pady=20, padx=20)

#----------------------------- Manual Tab Layout -------------------------------
# Generate the manual tab for manual control of the axidraw. Syringe pump control has not been added
# because it can easily be controlled using the interface on the device
def create_manual_layout(parent, speed_pendown, speed_penup, penposup, penposdown):
    global x_entry
    global y_entry
    ttk.Button(parent, text='Pen Up', style='Custom.TButton',command=lambda: lh.penup()).grid(row=1, column=0, padx=2, pady=2)
    ttk.Button(parent, text='Pen Down', style='Custom.TButton',command=lambda: lh.pendown()).grid(row=2, column=0, padx=2, pady=2)
    
    ttk.Label(parent, text='X-Position', style='Custom.TLabel').grid(row=3, column=0, padx=2, pady=2)
    ttk.Label(parent, text='Y-Position', style='Custom.TLabel').grid(row=3, column=1, padx=2, pady=2)

    x_entry = ttk.Entry(parent, style='Custom.TEntry', width=5)
    x_entry.insert(1,0)
    x_entry.grid(row=4, column=0, padx=4, pady=4, ipadx=8, ipady=6)
    x_entry.config(font=('Franklin', 24))
    y_entry = ttk.Entry(parent, style='Custom.TEntry', width=5)
    y_entry.insert(1,0)
    y_entry.grid(row=4, column=1, padx=4, pady=4, ipadx=8, ipady=6)
    y_entry.config(font=('Franklin', 24))
    ttk.Button(parent, text='Move', style='Custom.TButton',command=lambda: move_axidraw()).grid(row=4, column=2, padx=2, pady=2)
    
    ttk.Label(parent, text='Pen Lower Speed (0-110)', style='Custom.TLabel').grid(row=5, column=0, columnspan=2, padx=2, pady=2)
    entry = ttk.Entry(parent, style='Custom.TEntry', width=5)
    entry.insert(0, str(speed_pendown))
    entry.grid(row=5, column=2, padx=4, pady=4, ipadx=8, ipady=6)
    entry.config(font=('Franklin', 24))
    ttk.Label(parent, text='Pen Raise Speed (0-110)', style='Custom.TLabel').grid(row=6, column=0, columnspan=2, padx=2, pady=2)
    entry = ttk.Entry(parent, style='Custom.TEntry', width=5)
    entry.insert(0, str(speed_penup))
    entry.grid(row=6, column=2, padx=4, pady=4, ipadx=8, ipady=6)
    entry.config(font=('Franklin', 24))
    ttk.Label(parent, text='Pen Upper Limit (0-100)', style='Custom.TLabel').grid(row=7, column=0, columnspan=2, padx=2, pady=2)
    entry = ttk.Entry(parent, style='Custom.TEntry', width=5)
    entry.insert(0, str(penposup))
    entry.grid(row=7, column=2, padx=4, pady=4, ipadx=8, ipady=6)
    entry.config(font=('Franklin', 24))
    ttk.Label(parent, text='Pen Lower Limit (0-100)', style='Custom.TLabel').grid(row=8, column=0, columnspan=2, padx=2, pady=2)
    entry = ttk.Entry(parent, style='Custom.TEntry', width=5)
    entry.insert(0, str(penposdown))
    entry.grid(row=8, column=2, padx=4, pady=4, ipadx=8, ipady=6)
    entry.config(font=('Franklin', 24))
   
    ttk.Button(parent, text='Update Settings', style='Custom.TButton').grid(row=9, column=0, columnspan=3, padx=2, pady=2)
    
    ttk.Button(parent, text='Open Drawer', style='Custom.TButton',command=lambda: dp.spectramax_opendrawer()).grid(row=10, column=0, padx=2, pady=2)
    ttk.Button(parent, text='Close Drawer', style='Custom.TButton',command=lambda: dp.spectramax_closedrawer()).grid(row=10, column=1, padx=2, pady=2)

def on_key_press(event):
    # Check which key was pressed and handle accordingly
    key = event.keysym
    move_by = 10 if event.state & 0x0001 else 1  # Check if Shift key is held (state bit 0x0001)

    if key == 'Left':
        lh.ad_left(move_by)
        x_entry.delete(0, 'end')
        x_entry.insert(1, lh.x)
    elif key == 'Right':
        lh.ad_right(move_by)
        x_entry.delete(0, 'end')
        x_entry.insert(1, lh.x)
    elif key == 'Up':
        lh.ad_up(move_by)
        y_entry.delete(1, 'end')
        y_entry.insert(1, lh.y)
    elif key == 'Down':
        lh.ad_down(move_by)
        y_entry.delete(1, 'end')
        y_entry.insert(1, lh.y)

def check_focus(event):
    # Check if the "Manual" tab is selected
    current_tab = tabControl.select()
    if tabControl.tab(current_tab, "text") == "Manual":
        # Bind arrow keys to the on_key_press function
        root.bind('<Left>', on_key_press)
        root.bind('<Right>', on_key_press)
        root.bind('<Up>', on_key_press)
        root.bind('<Down>', on_key_press)
    else:
        # Unbind arrow keys
        root.unbind('<Left>')
        root.unbind('<Right>')
        root.unbind('<Up>')
        root.unbind('<Down>')

def move_axidraw():
    # Retrieve the values from the coordinate entry fields
    x_val = int(x_entry.get())
    y_val = int(y_entry.get())

    # Ensure coordinates are within AxiDraw bounds, otherwise it will crash
    if 190 >= x_val >= 0 and 140 >= y_val >= 0:
        print(x_val)
        print(y_val)
        # Call the AxiDraw move function with these coordinates
        lh.ad_move(x_val, y_val)

        # Update x and y values on LH module to keep track of position
        lh.x = x_val
        lh.y = y_val

create_manual_layout(manual_tab, lh.speed_pendown, lh.speed_penup, lh.penposup, lh.penposdown)
tabControl.bind('<<NotebookTabChanged>>', check_focus)

def ask_to_continue_with_dataframe(df):
    global root
    result = {"continue": None}

    def on_continue():
        result["continue"] = True
        popup.destroy()

    def on_cancel():
        result["continue"] = False
        popup.destroy()

    # Filter to show only index and 'Well'
    df_filtered = df.copy()
    df_filtered.insert(0, "Index", df_filtered.index)
    if "Well" in df_filtered.columns:
        df_filtered = df_filtered[["Index", "Well"]]
    else:
        df_filtered = df_filtered[["Index"]]

    popup = tk.Toplevel(root)
    popup.title("Review DataFrame")
    popup.grab_set()  # Modal

    big_font = ("Helvetica", 20)
    btn_font = ("Helvetica", 18)
    row_height = 40  # Increased row height

    # Styling for Treeview
    style = ttk.Style(popup)
    style.configure("Treeview", font=big_font, rowheight=row_height)
    style.configure("Treeview.Heading", font=("Helvetica", 20, "bold"))

    # Label
    tk.Label(
        popup,
        text="Reagent Locations",
        font=("Helvetica", 24),
        wraplength=600,
        justify="center",
        pady=10
    ).pack(pady=20)

    # Frame for Treeview + Scrollbar
    tree_frame = tk.Frame(popup)
    tree_frame.pack(expand=True, fill="both", padx=20, pady=20)

    tree = ttk.Treeview(tree_frame, columns=list(df_filtered.columns), show="headings", height=10)
    for col in df_filtered.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=250)

    for _, row in df_filtered.iterrows():
        tree.insert("", "end", values=list(row))

    tree.grid(row=0, column=0, sticky="nsew")

    # Scrollbar that stays aligned
    scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.grid(row=0, column=1, sticky="ns")

    tree_frame.grid_rowconfigure(0, weight=1)
    tree_frame.grid_columnconfigure(0, weight=1)

    # Buttons
    btn_frame = tk.Frame(popup)
    btn_frame.pack(pady=20)
    tk.Button(btn_frame, text="Continue", width=14, height=2, font=btn_font, command=on_continue).pack(side="left", padx=20)
    tk.Button(btn_frame, text="Cancel", width=14, height=2, font=btn_font, command=on_cancel).pack(side="right", padx=20)

    popup.wait_window()
    return result["continue"]


#----------------------------- Closing GUI -------------------------------
def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        # Perform any cleanup actions here, if needed
        #lh.ad_move(0,0)  
        root.destroy() # Close the window

root.protocol("WM_DELETE_WINDOW", on_closing)
def start_gui():
    global root
    root.mainloop()
