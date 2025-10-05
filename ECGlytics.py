import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
from filter import ecg_low_filter, ecg_high_filter
from xml_parser import mortaraXMLPARSER, GEXMLparser
import scipy.io


class ECGlyticsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECGlytics - ECG Analytics")
        self.root.geometry("1500x1000")

        # Variables to store data
        self.signal_data = None        # Raw signal data
        self.filtered_signal = None    # Filtered signal data
        self.current_index = 0         # Current signal index
        self.current_lead = 0          # Current lead index (for 12-lead ECG)
        self.segmentation_data = {}    # Dictionary to store segmentation points
        self.current_file = None
        self.view_start = 0            # Start sample for 1000-sample window
        self.window_size = 1000        # Number of samples to show at once
        self.leads = 1                 # Number of leads (default: single lead)
        self.sample_rate = 250         # Default sample rate
        self.file_name = ""

        # NEW: Variables for multi-lead visualization
        self.view_mode = "single"      # "single", "overlay", or "grid"
        self.show_other_lead_segmentation = True  # Show segmentation from other leads

        # Variables for undo and point selection
        self.last_action_stack = []    # Stack to store last actions for undo
        # Currently selected point for movement (signal_idx, wave_type, point_idx)
        self.selected_point = None

        # Create main frames
        self.control_frame = ttk.Frame(root, padding="10")
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        self.lead_filter_frame = ttk.Frame(root, padding="5")
        self.lead_filter_frame.pack(side=tk.TOP, fill=tk.X)

        # NEW: View control frame for multi-lead options
        self.view_control_frame = ttk.LabelFrame(
            root, text="View Options", padding="10")
        self.view_control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.wave_selection_frame = ttk.LabelFrame(
            root, text="Wave Selection", padding="10")
        self.wave_selection_frame.pack(
            side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH,
                             expand=True, padx=10, pady=10)

        # Create navigation controls
        self.create_control_buttons()

        # Create lead selection and filter controls
        self.create_lead_filter_controls()

        # NEW: Create view control options
        self.create_view_controls()

        # Create wave selection radio buttons
        self.create_wave_selection()

        # Create plot area
        self.create_plot_area()

        # Bind the keys .
        self.root.bind('<Left>', lambda event: self.move_point_left())
        self.root.bind('<Right>', lambda event: self.move_point_right())
        self.root.focus_set()

    def create_view_controls(self):
        """Create controls for multi-lead visualization options"""
        # View mode selection
        ttk.Label(self.view_control_frame, text="View Mode:").grid(
            row=0, column=0, padx=5, pady=5)

        self.view_mode_var = tk.StringVar(value="single")
        view_modes = [("Single Lead", "single"),
                      ("Overlay All", "overlay"), ("Grid View", "grid")]

        for i, (text, value) in enumerate(view_modes):
            ttk.Radiobutton(self.view_control_frame, text=text, variable=self.view_mode_var,
                            value=value, command=self.view_mode_changed).grid(row=0, column=i+1, padx=5, pady=5)

        # Segmentation visibility option
        self.show_other_segmentation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.view_control_frame, text="Show segmentation on all leads",
                        variable=self.show_other_segmentation_var,
                        command=self.segmentation_visibility_changed).grid(row=0, column=5, padx=20, pady=5)

    def view_mode_changed(self):
        """Called when view mode changes"""
        self.view_mode = self.view_mode_var.get()
        self.create_plot_area()  # Recreate plot area with new layout
        self.update_plot()

    def segmentation_visibility_changed(self):
        """Called when segmentation visibility option changes"""
        self.show_other_lead_segmentation = self.show_other_segmentation_var.get()
        self.update_plot()

    def create_plot_area(self):
        """Create matplotlib figure and canvas with appropriate layout"""
        # Clear existing plot frame
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        if self.view_mode == "grid" and self.leads > 1:
            # Grid layout for multiple leads
            rows = int(np.ceil(np.sqrt(self.leads)))
            cols = int(np.ceil(self.leads / rows))

            self.figure, self.axes = plt.subplots(rows, cols, figsize=(12, 8))
            if rows == 1 and cols == 1:
                self.axes = [self.axes]  # Make it a list for consistency
            elif rows == 1 or cols == 1:
                self.axes = self.axes.flatten()
            else:
                self.axes = self.axes.flatten()

            # Hide unused subplots
            for i in range(self.leads, len(self.axes)):
                self.axes[i].set_visible(False)
        else:
            # Single plot for single lead or overlay mode
            self.figure, self.ax = plt.subplots(figsize=(10, 6))
            self.axes = [self.ax]  # Make it a list for consistency

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add click event handler to the plot
        self.figure.canvas.mpl_connect(
            'button_press_event', self.on_plot_click)

        # Display empty plot initially
        self.update_plot()

    def create_control_buttons(self):
        # File operations buttons
        ttk.Button(self.control_frame, text="Load Signal",
                   command=self.load_signal).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.control_frame, text="Save Segmentation",
                   command=self.save_segmentation).grid(row=0, column=1, padx=5, pady=5)

        # NEW: Undo button
        ttk.Button(self.control_frame, text="Undo", command=self.undo_last_action).grid(
            row=0, column=2, padx=5, pady=5)

        # Navigation buttons for samples window
        ttk.Button(self.control_frame, text="◀ Pan Left",
                   command=self.pan_left).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(self.control_frame, text="Pan Right ▶",
                   command=self.pan_right).grid(row=0, column=4, padx=5, pady=5)
        ttk.Button(self.control_frame, text="END",
                   command=self.move_to_end).grid(row=0, column=5, padx=5, pady=5)
        ttk.Button(self.control_frame, text="START",
                   command=self.move_to_start).grid(row=0, column=6, padx=5, pady=5)
        
        # NEW: Point movement buttons
        ttk.Button(self.control_frame, text="◀ Move Point Left",
                   command=self.move_point_left).grid(row=0, column=7, padx=5, pady=5)
        ttk.Button(self.control_frame, text="Move Point Right ▶",
                   command=self.move_point_right).grid(row=0, column=8, padx=5, pady=5)

        # Window size control
        ttk.Label(self.control_frame, text="Window Size:").grid(
            row=0, column=9, padx=5, pady=5)
        self.window_size_var = tk.StringVar(value="1000")
        ttk.Entry(self.control_frame, textvariable=self.window_size_var,
                  width=5).grid(row=0, column=10, padx=5, pady=5)

        # Window size update button
        ttk.Button(self.control_frame, text="Update Window Size",
                   command=self.update_window_size).grid(row=0, column=11, padx=5, pady=5)

        # Status information
        self.status_var = tk.StringVar(value="No signal loaded")
        ttk.Label(self.control_frame, textvariable=self.status_var).grid(
            row=0, column=12, padx=20, pady=5)

    def create_wave_selection(self):
        self.wave_var = tk.StringVar(value="none")

        # Create frame for wave buttons with title
        ttk.Label(self.wave_selection_frame, text="Select wave type to mark:").grid(
            row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))

        # P wave options
        ttk.Label(self.wave_selection_frame, text="P Wave:").grid(
            row=1, column=0, sticky=tk.W, pady=5)
        ttk.Radiobutton(self.wave_selection_frame, text="P On", variable=self.wave_var,
                        value="p_on").grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(self.wave_selection_frame, text="P Off", variable=self.wave_var,
                        value="p_off").grid(row=1, column=2, sticky=tk.W, pady=5)

        # Q wave
        ttk.Radiobutton(self.wave_selection_frame, text="Q", variable=self.wave_var, value="q").grid(
            row=2, column=0, sticky=tk.W, pady=5)

        # R wave
        ttk.Radiobutton(self.wave_selection_frame, text="R", variable=self.wave_var, value="r").grid(
            row=3, column=0, sticky=tk.W, pady=5)

        # S wave
        ttk.Radiobutton(self.wave_selection_frame, text="S", variable=self.wave_var, value="s").grid(
            row=4, column=0, sticky=tk.W, pady=5)

        # T wave options
        ttk.Label(self.wave_selection_frame, text="T Wave:").grid(
            row=5, column=0, sticky=tk.W, pady=5)
        ttk.Radiobutton(self.wave_selection_frame, text="T On", variable=self.wave_var,
                        value="t_on").grid(row=5, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(self.wave_selection_frame, text="T Off", variable=self.wave_var,
                        value="t_off").grid(row=5, column=2, sticky=tk.W, pady=5)

        # PAC options
        ttk.Label(self.wave_selection_frame, text="PAC:").grid(
            row=6, column=0, sticky=tk.W, pady=5)
        ttk.Radiobutton(self.wave_selection_frame, text="PAC On", variable=self.wave_var,
                        value="pac_on").grid(row=6, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(self.wave_selection_frame, text="PAC Off", variable=self.wave_var,
                        value="pac_off").grid(row=6, column=2, sticky=tk.W, pady=5)

        # PVC options
        ttk.Label(self.wave_selection_frame, text="PVC:").grid(
            row=7, column=0, sticky=tk.W, pady=5)
        ttk.Radiobutton(self.wave_selection_frame, text="PVC On", variable=self.wave_var,
                        value="pvc_on").grid(row=7, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(self.wave_selection_frame, text="PVC Off", variable=self.wave_var,
                        value="pvc_off").grid(row=7, column=2, sticky=tk.W, pady=5)
        
        # ECG start and end options for cropping
        ttk.Label(self.wave_selection_frame, text="ECG Start/End:").grid(
            row=8, column=0, sticky=tk.W, pady=5)
        ttk.Radiobutton(self.wave_selection_frame, text="ECG Start", variable=self.wave_var,
                value="ecg_start").grid(row=8, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(self.wave_selection_frame, text="ECG End", variable=self.wave_var,
                value="ecg_end").grid(row=8, column=2, sticky=tk.W, pady=5)

        # None option to clear selection
        ttk.Radiobutton(self.wave_selection_frame, text="None", variable=self.wave_var, value="none").grid(
            row=9, column=0, columnspan=3, sticky=tk.W, pady=10)

        # Add a separator
        ttk.Separator(self.wave_selection_frame, orient='horizontal').grid(
            row=10, column=0, columnspan=3, sticky=tk.EW, pady=10)

        # Current points list
        ttk.Label(self.wave_selection_frame, text="Current Marked Points:").grid(
            row=11, column=0, columnspan=3, sticky=tk.W, pady=5)

        # Add a listbox to show marked points
        self.points_listbox = tk.Listbox(
            self.wave_selection_frame, width=25, height=10)
        self.points_listbox.grid(
            row=12, column=0, columnspan=3, sticky=tk.NSEW, pady=5)

        # NEW: Bind selection event for point movement
        self.points_listbox.bind('<<ListboxSelect>>', self.on_point_select)

        # Add scrollbar for listbox
        scrollbar = ttk.Scrollbar(
            self.wave_selection_frame, orient="vertical", command=self.points_listbox.yview)
        scrollbar.grid(row=13, column=3, sticky=tk.NS)
        self.points_listbox.configure(yscrollcommand=scrollbar.set)

        # Delete selected point button
        ttk.Button(self.wave_selection_frame, text="Delete Selected Point",
                   command=self.delete_selected_point).grid(row=14, column=0, columnspan=3, sticky=tk.W, pady=10)
        
        # Crop Signal window
        ttk.Button(self.wave_selection_frame, text="Crop Signal",
                     command=self.crop_signal).grid(row=15, column=0, columnspan=3, sticky=tk.W, pady=10)
        # Bind the listbox selection event
        self.points_listbox.bind('<<ListboxSelect>>', self.on_point_select)
        
    def crop_signal(self):

        if self.signal_data is None:
             messagebox.showwarning("Warning", "Start by loading a signal!")
             return
        if self.segmentation_data is None:
             messagebox.showwarning("Warning", "Segment the signal!")
             return
        data = self.segmentation_data
        signal = self.signal_data[self.current_index]
        signal = np.array(signal)

        try:
            startPoints = data[self.current_index]['ecg_start']
        except: 
            messagebox.showwarning("Warning", "No start point data to crop")
            return

        try:
            endPoints = data[self.current_index]['ecg_end']
        except: 
            endPoints = None

        foldername = filedialog.askdirectory(
            title="Select Folder to Save cropped signals into"
        )

        if endPoints is None:
            for point,i in zip(startPoints, range(len(startPoints))):
                croppedSignal = signal[:,point:]
                signalDic = {"ECG": croppedSignal, "fs": self.sample_rate}
                scipy.io.savemat(os.path.join(foldername,self.file_name + f"_cropped_{i}.mat" ), signalDic)

        elif len(endPoints) == len(startPoints):
            for start, end, i in zip(startPoints, endPoints, range(len(startPoints))):
                if start < end:
                    croppedSignal = signal[:,start:end]
                    signalDic = {"ECG": croppedSignal, "fs": self.sample_rate}
                    scipy.io.savemat(os.path.join(foldername,self.file_name + f"_cropped_{i}.mat" ), signalDic)
                else:
                    messagebox.showerror("Error!", f"Signal was not saved, start ({start}) must be before ({end})")
        elif len(endPoints) + 1 == len(startPoints):
            for start, end, i in zip(startPoints, endPoints.append(signal.shape[-1]), range(len(startPoints))):
                if start < end:
                    croppedSignal = signal[:,start:end]
                    signalDic = {"ECG": croppedSignal, "fs": self.sample_rate}
                    scipy.io.savemat(os.path.join(foldername,self.file_name + f"_cropped_{i}.mat" ), signalDic)
                else:
                    messagebox.showerror("Error!", f"A Signal was not saved, start ({start}) must be before ({end})")
        
        else:
            messagebox.showerror("Error!", f"Number of ending points should be 0, equale to starting points or less by one points")
            return
        messagebox.showinfo("Sucess", f"Signal(s) were saved into {foldername}\\")

    def create_lead_filter_controls(self):
        # Lead selection dropdown
        ttk.Label(self.lead_filter_frame, text="Lead:").grid(
            row=0, column=0, padx=5, pady=5)

        self.lead_var = tk.StringVar(value="Lead I")
        self.lead_dropdown = ttk.Combobox(self.lead_filter_frame, textvariable=self.lead_var,
                                          values=["Lead I", "Lead II", "Lead III",
                                                  "aVR", "aVL", "aVF",
                                                  "V1", "V2", "V3", "V4", "V5", "V6"],
                                          state="readonly", width=10)
        self.lead_dropdown.grid(row=0, column=1, padx=5, pady=5)
        self.lead_dropdown.bind("<<ComboboxSelected>>", self.lead_changed)

        # Filter controls
        ttk.Button(self.lead_filter_frame, text="Apply Filters",
                   command=self.apply_filters).grid(row=0, column=2, padx=5, pady=5)

        # Sample rate entry
        ttk.Label(self.lead_filter_frame, text="Sample Rate (Hz):").grid(
            row=0, column=3, padx=5, pady=5)
        self.sample_rate_var = tk.StringVar(value="250")
        ttk.Entry(self.lead_filter_frame, textvariable=self.sample_rate_var,
                  width=5).grid(row=0, column=4, padx=5, pady=5)

        # High-pass filter frequency
        ttk.Label(self.lead_filter_frame,
                  text="High-pass (Hz):").grid(row=0, column=5, padx=5, pady=5)
        self.highpass_var = tk.StringVar(value="0.5")
        ttk.Entry(self.lead_filter_frame, textvariable=self.highpass_var,
                  width=5).grid(row=0, column=6, padx=5, pady=5)

        # Low-pass filter frequency
        ttk.Label(self.lead_filter_frame,
                  text="Low-pass (Hz):").grid(row=0, column=7, padx=5, pady=5)
        self.lowpass_var = tk.StringVar(value="60")
        ttk.Entry(self.lead_filter_frame, textvariable=self.lowpass_var,
                  width=5).grid(row=0, column=8, padx=5, pady=5)

    def update_window_size(self):
        """Update the window size based on user input"""
        try:
            new_size = int(self.window_size_var.get())
            if new_size > 0:
                self.window_size = new_size
                self.view_start = 0  # Reset view position
                self.update_plot()
            else:
                messagebox.showerror(
                    "Invalid Size", "Window size must be a positive integer.")
        except ValueError:
            messagebox.showerror(
                "Invalid Size", "Please enter a valid integer for window size.")

    def update_plot(self):
        """Update plot based on current view mode"""
        # Clear all axes
        if hasattr(self, 'axes'):
            for ax in self.axes:
                if ax.get_visible():
                    ax.clear()
        elif hasattr(self, 'ax'):
            self.ax.clear()

        if self.signal_data is None or len(self.signal_data) == 0:
            # Handle empty signal case
            if hasattr(self, 'ax'):
                self.ax.set_title("No ECG Signal Loaded")
                self.ax.text(0.5, 0.5, "Please load an ECG signal file",
                             horizontalalignment='center', verticalalignment='center',
                             transform=self.ax.transAxes)
            self.status_var.set("No signal loaded")
            self.canvas.draw()
            return

        if self.view_mode == "single":
            self.plot_single_lead()
        elif self.view_mode == "overlay":
            self.plot_overlay_leads()
        elif self.view_mode == "grid":
            self.plot_grid_leads()

        self.canvas.draw()

    def update_points_listbox(self):
        """Update the listbox showing all marked points"""
        self.points_listbox.delete(0, tk.END)  # Clear the listbox

        if self.current_index not in self.segmentation_data:
            return

        segmentation = self.segmentation_data[self.current_index]

        # Add each point to the listbox
        for wave_type in sorted(segmentation.keys()):
            positions = segmentation[wave_type]
            if positions:
                wave_name = wave_type.replace('_', ' ').capitalize()
                for i, position in enumerate(positions):
                    self.points_listbox.insert(
                        tk.END, f"{wave_name} #{i+1}: Sample {position}")

    def plot_single_lead(self):
        """Plot only the currently selected lead"""
        if not hasattr(self, 'ax'):
            return

        # Get the current signal data
        if self.filtered_signal is not None:
            data = self.filtered_signal[self.current_index][self.current_lead]
        else:
            data = self.signal_data[self.current_index][self.current_lead]

        # Only display window_size samples at a time
        end_idx = min(self.view_start + self.window_size, len(data))
        visible_data = data[self.view_start:end_idx]
        x = range(self.view_start, end_idx)

        self.ax.plot(x, visible_data, 'b-', linewidth=1)

        # Plot segmentation points for current lead only or all leads based on setting
        if self.show_other_lead_segmentation:
            self.plot_segmentation_points_all_leads(self.ax)
        else:
            self.plot_segmentation_points_single_lead(
                self.ax, self.current_lead)

        # Set plot properties
        self.ax.set_xlim(self.view_start, self.view_start + self.window_size)
        lead_name = self.lead_var.get()
        self.ax.set_title(
            f"ECG Signal {self.current_index + 1}/{len(self.signal_data)} - {lead_name}")
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.3)

        self.update_status_and_listbox(data, end_idx)

    def plot_overlay_leads(self):
        """Plot all leads overlaid on the same axes"""
        if not hasattr(self, 'ax'):
            return

        colors = plt.cm.tab10(np.linspace(0, 1, self.leads))
        lead_names = self.get_lead_names()

        for lead_idx in range(self.leads):
            # Get the signal data for this lead
            if self.filtered_signal is not None:
                data = self.filtered_signal[self.current_index][lead_idx]
            else:
                data = self.signal_data[self.current_index][lead_idx]

            # Only display window_size samples at a time
            end_idx = min(self.view_start + self.window_size, len(data))
            visible_data = data[self.view_start:end_idx]
            x = range(self.view_start, end_idx)

            # Normalize data for better overlay visualization
            normalized_data = (visible_data - np.mean(visible_data)
                               ) / (np.std(visible_data) + 1e-8)
            offset = lead_idx * 3  # Vertical offset for each lead

            self.ax.plot(x, normalized_data + offset, color=colors[lead_idx],
                         linewidth=0.8, label=lead_names[lead_idx])

        # Plot segmentation points
        if self.show_other_lead_segmentation:
            self.plot_segmentation_points_overlay()
        else:
            self.plot_segmentation_points_single_lead_overlay(
                self.current_lead)

        # Set plot properties
        self.ax.set_xlim(self.view_start, self.view_start + self.window_size)
        self.ax.set_title(
            f"ECG Signal {self.current_index + 1}/{len(self.signal_data)} - All Leads Overlay")
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("Normalized Amplitude (with offset)")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right', fontsize=8)

        # Update status for overlay mode
        if self.filtered_signal is not None:
            sample_data = self.filtered_signal[self.current_index][0]
        else:
            sample_data = self.signal_data[self.current_index][0]
        end_idx = min(self.view_start + self.window_size, len(sample_data))
        self.update_status_and_listbox(sample_data, end_idx)

    def plot_grid_leads(self):
        """Plot leads in a grid layout"""
        if not hasattr(self, 'axes') or len(self.axes) == 0:
            return

        lead_names = self.get_lead_names()

        for lead_idx in range(min(self.leads, len(self.axes))):
            ax = self.axes[lead_idx]

            # Get the signal data for this lead
            if self.filtered_signal is not None:
                data = self.filtered_signal[self.current_index][lead_idx]
            else:
                data = self.signal_data[self.current_index][lead_idx]

            # Only display window_size samples at a time
            end_idx = min(self.view_start + self.window_size, len(data))
            visible_data = data[self.view_start:end_idx]
            x = range(self.view_start, end_idx)

            ax.plot(x, visible_data, 'b-', linewidth=0.8)

            # Plot segmentation points for this lead
            if self.show_other_lead_segmentation:
                self.plot_segmentation_points_single_lead(ax, lead_idx)
            elif lead_idx == self.current_lead:
                self.plot_segmentation_points_single_lead(ax, lead_idx)

            # Set plot properties
            ax.set_xlim(self.view_start, self.view_start + self.window_size)
            ax.set_title(f"{lead_names[lead_idx]}", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Add highlighting for current lead
            if lead_idx == self.current_lead:
                ax.set_facecolor('#f0f0f0')

        # Update status for grid mode
        if self.filtered_signal is not None:
            sample_data = self.filtered_signal[self.current_index][self.current_lead]
        else:
            sample_data = self.signal_data[self.current_index][self.current_lead]
        end_idx = min(self.view_start + self.window_size, len(sample_data))
        self.update_status_and_listbox(sample_data, end_idx)

    def on_plot_click(self, event):
        if event.inaxes != self.ax or self.signal_data is None:
            return

        # Get the x-coordinate of the click (sample number)
        x_pos = int(round(event.xdata))

        # Get the currently selected signal data
        if self.filtered_signal is not None:
            data = self.filtered_signal[self.current_index][self.current_lead]
        else:
            data = self.signal_data[self.current_index][self.current_lead]

        # Ensure the position is within the signal range
        if 0 <= x_pos < len(data):
            # Get the currently selected wave type
            wave_type = self.wave_var.get()

            if wave_type != "none":
                # Initialize segmentation data for current index if not exists
                if self.current_index not in self.segmentation_data:
                    self.segmentation_data[self.current_index] = {}

                # Initialize the wave type as a list if it doesn't exist
                if wave_type not in self.segmentation_data[self.current_index]:
                    self.segmentation_data[self.current_index][wave_type] = []

                # Add the position to the list for the selected wave type
                self.segmentation_data[self.current_index][wave_type].append(
                    x_pos)

                # Store action for undo
                action = {
                    'type': 'add_point',
                    'signal_idx': self.current_index,
                    'wave_type': wave_type,
                    'position': x_pos,
                    'point_idx': len(self.segmentation_data[self.current_index][wave_type]) - 1
                }
                self.last_action_stack.append(action)

                # Limit undo stack size to prevent memory issues
                if len(self.last_action_stack) > 20:
                    self.last_action_stack.pop(0)

                self.selected_point = self.current_index, wave_type, len(
                    self.segmentation_data[self.current_index][wave_type]) - 1

                # Update the plot to show the marked point
                self.update_plot()

    def undo_last_action(self):
        """Undo the last action"""
        if not self.last_action_stack:
            messagebox.showinfo("No Action", "No action to undo.")
            return

        last_action = self.last_action_stack.pop()

        if last_action['type'] == 'add_point':
            # Remove the last added point
            signal_idx = last_action['signal_idx']
            wave_type = last_action['wave_type']

            if (signal_idx in self.segmentation_data and
                wave_type in self.segmentation_data[signal_idx] and
                    self.segmentation_data[signal_idx][wave_type]):

                # Remove the last point
                self.segmentation_data[signal_idx][wave_type].pop()

                # If no points left for this wave type, remove the wave type
                if not self.segmentation_data[signal_idx][wave_type]:
                    del self.segmentation_data[signal_idx][wave_type]

                # Clear selected point if it was the undone point
                if (self.selected_point and
                    self.selected_point[0] == signal_idx and
                        self.selected_point[1] == wave_type):
                    self.selected_point = None

        elif last_action['type'] == 'move_point':
            # Restore the point to its previous position
            signal_idx = last_action['signal_idx']
            wave_type = last_action['wave_type']
            point_idx = last_action['point_idx']
            old_position = last_action['old_position']

            if (signal_idx in self.segmentation_data and
                wave_type in self.segmentation_data[signal_idx] and
                    len(self.segmentation_data[signal_idx][wave_type]) > point_idx):

                self.segmentation_data[signal_idx][wave_type][point_idx] = old_position

        self.update_plot()

    def on_point_select(self, event):
        """Handle point selection from listbox"""
        selection = self.points_listbox.curselection()
        if not selection:
            self.selected_point = None
            return

        # Get the selected index
        idx = selection[0]
        selected_text = self.points_listbox.get(idx)

        # Parse the text to get wave type and point index
        parts = selected_text.split(":")
        if len(parts) < 2:
            self.selected_point = None
            return

        wave_name = parts[0].strip()

        # Handle the wave name format (e.g., "P On #1")
        wave_parts = wave_name.split()
        if len(wave_parts) < 2:
            self.selected_point = None
            return

        # Get wave type (e.g., "p_on" from "P On #1")
        if wave_parts[0].lower() in ["p", "t", "pac"]:
            wave_type = f"{wave_parts[0].lower()}_{wave_parts[1].lower()}"
        else:
            wave_type = wave_parts[0].lower()

        # Get point index (1-based in UI, 0-based in data)
        point_num = int(wave_parts[-1].replace("#", "")) - 1

        # Store selected point info
        self.selected_point = (self.current_index, wave_type, point_num)

    def move_point_left(self):
        """Move the selected point one sample to the left"""
        if not self.selected_point:
            messagebox.showinfo(
                "No Selection", "Please select a point from the list to move.")
            return

        signal_idx, wave_type, point_idx = self.selected_point

        # Check if the point exists
        if (signal_idx not in self.segmentation_data or
            wave_type not in self.segmentation_data[signal_idx] or
                len(self.segmentation_data[signal_idx][wave_type]) <= point_idx):
            messagebox.showinfo("Invalid Selection",
                                "Selected point no longer exists.")
            self.selected_point = None
            return

        current_position = self.segmentation_data[signal_idx][wave_type][point_idx]
        new_position = max(0, current_position - 1)  # Don't go below 0

        if new_position != current_position:
            # Store action for undo
            action = {
                'type': 'move_point',
                'signal_idx': signal_idx,
                'wave_type': wave_type,
                'point_idx': point_idx,
                'old_position': current_position,
                'new_position': new_position
            }
            self.last_action_stack.append(action)

            # Limit undo stack size
            if len(self.last_action_stack) > 50:
                self.last_action_stack.pop(0)

            # Update position
            self.segmentation_data[signal_idx][wave_type][point_idx] = new_position
            self.update_plot()

    def move_point_right(self):
        """Move the selected point one sample to the right"""
        if not self.selected_point:
            messagebox.showinfo(
                "No Selection", "Please select a point from the list to move.")
            return

        signal_idx, wave_type, point_idx = self.selected_point

        # Check if the point exists
        if (signal_idx not in self.segmentation_data or
            wave_type not in self.segmentation_data[signal_idx] or
                len(self.segmentation_data[signal_idx][wave_type]) <= point_idx):
            messagebox.showinfo("Invalid Selection",
                                "Selected point no longer exists.")
            self.selected_point = None
            return

        # Get signal length to prevent going beyond bounds
        if self.filtered_signal is not None:
            data_len = len(self.filtered_signal[signal_idx][self.current_lead])
        else:
            data_len = len(self.signal_data[signal_idx][self.current_lead])

        current_position = self.segmentation_data[signal_idx][wave_type][point_idx]
        # Don't go beyond signal length
        new_position = min(data_len - 1, current_position + 1)

        if new_position != current_position:
            # Store action for undo
            action = {
                'type': 'move_point',
                'signal_idx': signal_idx,
                'wave_type': wave_type,
                'point_idx': point_idx,
                'old_position': current_position,
                'new_position': new_position
            }
            self.last_action_stack.append(action)

            # Limit undo stack size
            if len(self.last_action_stack) > 50:
                self.last_action_stack.pop(0)

            # Update position
            self.segmentation_data[signal_idx][wave_type][point_idx] = new_position
            self.update_plot()

    def delete_selected_point(self):
        """Delete the point selected in the listbox"""
        selection = self.points_listbox.curselection()
        if not selection:
            messagebox.showinfo(
                "No Selection", "Please select a point to delete.")
            return

        # Get the selected index
        idx = selection[0]
        selected_text = self.points_listbox.get(idx)

        # Parse the text to get wave type and point index
        parts = selected_text.split(":")
        if len(parts) < 2:
            return

        wave_name = parts[0].strip()

        # Handle the wave name format (e.g., "P On #1")
        wave_parts = wave_name.split()
        if len(wave_parts) < 2:
            return

        # Get wave type (e.g., "p_on" from "P On #1")
        if wave_parts[0].lower() in ["p", "t", "pac", "pvc", "ecg"]:
            wave_type = f"{wave_parts[0].lower()}_{wave_parts[1].lower()}"
        else:
            wave_type = wave_parts[0].lower()

        # Get point index (1-based in UI, 0-based in data)
        point_num = int(wave_parts[-1].replace("#", "")) - 1

        # Delete the point
        if (self.current_index in self.segmentation_data and
            wave_type in self.segmentation_data[self.current_index] and
                len(self.segmentation_data[self.current_index][wave_type]) > point_num):

            # Store the deleted point for undo (store as add_point action reversed)
            deleted_position = self.segmentation_data[self.current_index][wave_type][point_num]

            del self.segmentation_data[self.current_index][wave_type][point_num]

            # NEW: Store delete action for undo
            action = {
                'type': 'delete_point',
                'signal_idx': self.current_index,
                'wave_type': wave_type,
                'position': deleted_position,
                'point_idx': point_num
            }
            self.last_action_stack.append(action)

            # If that was the last point of this type, remove the wave type entry
            if not self.segmentation_data[self.current_index][wave_type]:
                del self.segmentation_data[self.current_index][wave_type]

            # Clear selected point if it was the deleted point
            if (self.selected_point and
                self.selected_point[0] == self.current_index and
                self.selected_point[1] == wave_type and
                    self.selected_point[2] == point_num):
                self.selected_point = None

            # Update the UI
            self.update_plot()

    def get_lead_names(self):
        """Get list of lead names based on number of leads"""
        if self.leads == 1:
            return ["Lead I"]
        elif self.leads <= 3:
            return ["Lead I", "Lead II", "Lead III"][:self.leads]
        elif self.leads <= 6:
            return ["Lead I", "Lead II", "Lead III", "aVR", "aVL", "aVF"][:self.leads]
        else:
            return ["Lead I", "Lead II", "Lead III", "aVR", "aVL", "aVF",
                    "V1", "V2", "V3", "V4", "V5", "V6"][:self.leads]

    def update_status_and_listbox(self, data, end_idx):
        """Update status bar and points listbox"""
        if self.current_file:
            base_name = os.path.basename(self.current_file)
            self.status_var.set(f"File: {base_name} - Signal {self.current_index + 1}/{len(self.signal_data)} - " +
                                f"Samples {self.view_start}-{end_idx-1} of {len(data)}")

        # Update the points listbox
        self.update_points_listbox()

    def plot_segmentation_points_all_leads(self, ax):
        """Plot segmentation points from all leads on single axis"""
        if self.current_index not in self.segmentation_data:
            return

        segmentation = self.segmentation_data[self.current_index]

        # Define colors for different wave types
        colors = {
            'p_on': 'g', 'p_off': 'r',
            'q': 'c', 'r': 'm', 's': 'y',
            't_on': 'g', 't_off': 'r',
            'pac_on': 'g', 'pac_off': 'r',
            'pvc_on': 'g', 'pvc_off': 'r',
            'ecg_start': 'g', 'ecg_end': 'r'
        }

        # Get current signal data for amplitude reference
        if self.filtered_signal is not None:
            data = self.filtered_signal[self.current_index][self.current_lead]
        else:
            data = self.signal_data[self.current_index][self.current_lead]

        legend_entries = set()

        for wave_type in segmentation:
            positions = segmentation[wave_type]
            if not positions:
                continue

            for position in positions:
                if self.view_start <= position < (self.view_start + self.window_size):
                    if wave_type in colors:
                        ax.axvline(
                            x=position, color=colors[wave_type], linestyle='--', alpha=0.7)
                        ax.plot(position, data[position], 'o',
                                color=colors[wave_type], markersize=6)

                        legend_name = wave_type.replace('_', ' ').capitalize()
                        if legend_name not in legend_entries:
                            ax.plot([], [], 'o', color=colors[wave_type],
                                    label=legend_name)
                            legend_entries.add(legend_name)

        if legend_entries:
            ax.legend(loc='upper right', fontsize=8)

    def plot_segmentation_points_single_lead(self, ax, lead_idx):
        """Plot segmentation points for a specific lead"""
        if self.current_index not in self.segmentation_data:
            return

        segmentation = self.segmentation_data[self.current_index]

        colors = {
            'p_on': 'g', 'p_off': 'r',
            'q': 'c', 'r': 'm', 's': 'y',
            't_on': 'g', 't_off': 'r',
            'pac_on': 'g', 'pac_off': 'r',
            'pvc_on': 'g', 'pvc_off': 'r',
            'ecg_start': 'g', 'ecg_end': 'r'
        }

        # Get signal data for this specific lead
        if self.filtered_signal is not None:
            data = self.filtered_signal[self.current_index][lead_idx]
        else:
            data = self.signal_data[self.current_index][lead_idx]

        legend_entries = set()

        for wave_type in segmentation:
            positions = segmentation[wave_type]
            if not positions:
                continue

            for position in positions:
                if self.view_start <= position < (self.view_start + self.window_size):
                    if wave_type in colors:
                        ax.axvline(
                            x=position, color=colors[wave_type], linestyle='--', alpha=0.7)
                        ax.plot(position, data[position], 'o',
                                color=colors[wave_type], markersize=6)

                        legend_name = wave_type.replace('_', ' ').capitalize()
                        if legend_name not in legend_entries:
                            ax.plot([], [], 'o', color=colors[wave_type],
                                    label=legend_name)
                            legend_entries.add(legend_name)

        if legend_entries and self.view_mode == "grid":
            ax.legend(loc='upper right', fontsize=6)

    def plot_segmentation_points_overlay(self):
        """Plot segmentation points for overlay mode with proper offsets"""
        if self.current_index not in self.segmentation_data:
            return

        segmentation = self.segmentation_data[self.current_index]

        colors = {
            'p_on': 'g', 'p_off': 'r',
            'q': 'c', 'r': 'm', 's': 'y',
            't_on': 'g', 't_off': 'r',
            'pac_on': 'g', 'pac_off': 'r',
            'pvc_on': 'g', 'pvc_off': 'r',
            'ecg_start': 'g', 'ecg_end': 'r'
        }

        for wave_type in segmentation:
            positions = segmentation[wave_type]
            if not positions:
                continue

            for position in positions:
                if self.view_start <= position < (self.view_start + self.window_size):
                    if wave_type in colors:
                        # Draw vertical line across all leads
                        self.ax.axvline(
                            x=position, color=colors[wave_type], linestyle='--', alpha=0.5)

                        # Add points at each lead level
                        for lead_idx in range(self.leads):
                            offset = lead_idx * 3
                            self.ax.plot(position, offset, 'o',
                                         color=colors[wave_type], markersize=4)

    def plot_segmentation_points_single_lead_overlay(self, target_lead):
        """Plot segmentation points for only one lead in overlay mode"""
        if self.current_index not in self.segmentation_data:
            return

        segmentation = self.segmentation_data[self.current_index]

        colors = {
            'p_on': 'g', 'p_off': 'r',
            'q': 'c', 'r': 'm', 's': 'y',
            't_on': 'g', 't_off': 'r',
            'pac_on': 'g', 'pac_off': 'r',
            'pvc_on': 'g', 'pvc_off': 'r',
            'ecg_start': 'g', 'ecg_end': 'r'
        }

        offset = target_lead * 3

        for wave_type in segmentation:
            positions = segmentation[wave_type]
            if not positions:
                continue

            for position in positions:
                if self.view_start <= position < (self.view_start + self.window_size):
                    if wave_type in colors:
                        self.ax.axvline(
                            x=position, color=colors[wave_type], linestyle='--', alpha=0.7)
                        self.ax.plot(position, offset, 'o',
                                     color=colors[wave_type], markersize=6)

    def on_plot_click(self, event):
        """Handle plot click events - modified to work with different view modes"""
        if self.signal_data is None:
            return

        # Determine which axis was clicked
        clicked_ax = event.inaxes
        if clicked_ax is None:
            return
        
        
        target_lead = self.current_lead  # Default to current lead      
        if self.view_mode == "overlay":
            # For overlay mode, determine lead based on y-position
            if event.ydata is not None:
                target_lead = int(round(event.ydata / 3))
                target_lead = max(0, min(target_lead, self.leads - 1))

        # Get the x-coordinate of the click (sample number)
        x_pos = int(round(event.xdata))

        # Get the currently selected signal data
        if self.filtered_signal is not None:
            data = self.filtered_signal[self.current_index][target_lead]
        else:
            data = self.signal_data[self.current_index][target_lead]

        # Ensure the position is within the signal range
        if 0 <= x_pos < len(data):
            # Get the currently selected wave type
            wave_type = self.wave_var.get()

            if wave_type != "none":
                # Initialize segmentation data for current index if not exists
                if self.current_index not in self.segmentation_data:
                    self.segmentation_data[self.current_index] = {}

                # Initialize the wave type as a list if it doesn't exist
                if wave_type not in self.segmentation_data[self.current_index]:
                    self.segmentation_data[self.current_index][wave_type] = []

                # Add the position to the list for the selected wave type
                self.segmentation_data[self.current_index][wave_type].append(
                    x_pos)

                # Store action for undo
                action = {
                    'type': 'add_point',
                    'signal_idx': self.current_index,
                    'wave_type': wave_type,
                    'position': x_pos,
                    'point_idx': len(self.segmentation_data[self.current_index][wave_type]) - 1
                }
                self.last_action_stack.append(action)

                # Limit undo stack size to prevent memory issues
                if len(self.last_action_stack) > 20:
                    self.last_action_stack.pop(0)

                self.selected_point = self.current_index, wave_type, len(
                    self.segmentation_data[self.current_index][wave_type]) - 1

                # Update the plot to show the marked point
                self.update_plot()

    def lead_changed(self, event=None):
        """Called when the lead selection changes - modified to handle view modes"""
        lead_name = self.lead_var.get()

        # Convert lead name to index (0-11 for 12-lead ECG)
        lead_names = self.get_lead_names()
        if lead_name in lead_names:
            self.current_lead = lead_names.index(lead_name)
        else:
            self.current_lead = 0

        # Reset view to start of signal
        self.view_start = 0

        # Update the plot - for grid mode, this will highlight the new current lead
        self.update_plot()

    def apply_filters(self):
        """Apply high-pass and low-pass filters to the ECG signal"""
        if self.signal_data is None:
            messagebox.showinfo("No Data", "Please load a signal first.")
            return

        try:
            # Get filter parameters
            sample_rate = float(self.sample_rate_var.get())
            highpass_freq = float(self.highpass_var.get())
            lowpass_freq = float(self.lowpass_var.get())

            # Apply filters to all signals and leads
            self.filtered_signal = []

            for sig_idx in range(len(self.signal_data)):
                filtered_leads = []
                for lead_idx in range(len(self.signal_data[sig_idx])):
                    # Get original signal
                    signal = self.signal_data[sig_idx][lead_idx]

                    # Apply filters
                    filtered = ecg_high_filter(
                        signal, samplerate=sample_rate, highpass_frequency=highpass_freq)
                    filtered = ecg_low_filter(
                        filtered, samplerate=sample_rate, lowpass_frequency=lowpass_freq)

                    filtered_leads.append(filtered)
                self.filtered_signal.append(filtered_leads)

            messagebox.showinfo("Filters Applied", f"High-pass ({highpass_freq} Hz) and "
                                f"low-pass ({lowpass_freq} Hz) filters applied successfully.")

            # Update the plot
            self.update_plot()

        except Exception as e:
            messagebox.showerror(
                "Filter Error", f"Error applying filters: {str(e)}")

    def pan_left(self):
        """Pan the view window to the left"""
        if self.signal_data is None:
            return

        # Pan left by half window size
        self.view_start = max(0, self.view_start - int(self.window_size/2))
        self.update_plot()

    def pan_right(self):
        """Pan the view window to the right"""
        if self.signal_data is None:
            return

        # Get current signal length
        if self.filtered_signal is not None:
            data_len = len(
                self.filtered_signal[self.current_index][self.current_lead])
        else:
            data_len = len(
                self.signal_data[self.current_index][self.current_lead])

        # Pan right by half window size, but don't go beyond signal length
        new_start = self.view_start + int(self.window_size/2)
        if new_start < data_len - 10:  # keep at least 10 samples visible
            self.view_start = new_start
        self.update_plot()

    def read_ecg_file(self, file_name, start_offset=0, length=-1):
        """
        Read an ISHNE ECG file and return the header and signal data.
        Parameters:
            file_name (str): Path to the ECG file.
            start_offset (int): Offset in seconds from the start of the signal.
            length (int): Length in seconds to read from the signal. If -1, read the entire signal.
        Returns:
            ishne_header (dict): Dictionary containing header information.
            ecg_sig (numpy.ndarray): 2D array of ECG signal data with shape (num_samples, num_leads).
        """
        try:
            ishne_header = {}
            ecg_sig = None
            with open(file_name, 'rb') as fid:
                # Magic number
                magic_number = np.fromfile(fid, dtype=np.uint8, count=8)

                # Checksum
                checksum = np.fromfile(fid, dtype=np.uint16, count=1)

                # Read header
                var_length_block_size = np.fromfile(
                    fid, dtype=np.int32, count=1)[0]
                ishne_header['Sample_Size_ECG'] = np.fromfile(
                    fid, dtype=np.int32, count=1)[0]
                offset_var_length_block = np.fromfile(
                    fid, dtype=np.int32, count=1)[0]
                offset_ecg_block = np.fromfile(fid, dtype=np.int32, count=1)[0]
                file_version = np.fromfile(fid, dtype=np.int16, count=1)[0]
                first_name = np.fromfile(fid, dtype=np.uint8, count=40)
                last_name = np.fromfile(fid, dtype=np.uint8, count=40)
                id = np.fromfile(fid, dtype=np.uint8, count=20)
                sex = np.fromfile(fid, dtype=np.int16, count=1)[0]
                race = np.fromfile(fid, dtype=np.int16, count=1)[0]
                birth_date = np.fromfile(fid, dtype=np.int16, count=3)
                record_date = np.fromfile(fid, dtype=np.int16, count=3)
                file_date = np.fromfile(fid, dtype=np.int16, count=3)
                start_time = np.fromfile(fid, dtype=np.int16, count=3)
                ishne_header['nbLeads'] = np.fromfile(
                    fid, dtype=np.int16, count=1)[0]
                lead_spec = np.fromfile(fid, dtype=np.int16, count=12)
                lead_qual = np.fromfile(fid, dtype=np.int16, count=12)
                ishne_header['Resolution'] = np.fromfile(
                    fid, dtype=np.int16, count=12)
                pacemaker = np.fromfile(fid, dtype=np.int16, count=1)[0]
                recorder = np.fromfile(fid, dtype=np.uint8, count=40)
                ishne_header['Sampling_Rate'] = np.fromfile(
                    fid, dtype=np.int16, count=1)[0]
                proprietary = np.fromfile(fid, dtype=np.uint8, count=80)
                copyright = np.fromfile(fid, dtype=np.uint8, count=80)
                reserved = np.fromfile(fid, dtype=np.uint8, count=88)

                # Read variable_length block
                varblock = np.fromfile(
                    fid, dtype=np.uint8, count=var_length_block_size)

                # Calculate the offset to start reading
                # each data has 2 bytes
                offset = start_offset * \
                    ishne_header['Sampling_Rate'] * ishne_header['nbLeads'] * 2
                fid.seek(offset_ecg_block + offset, 0)

                # Determine the total number of samples in the file
                file_size = os.path.getsize(file_name)
                total_data_bytes = file_size - (offset_ecg_block + offset)
                total_samples = total_data_bytes // (
                    ishne_header['nbLeads'] * 2)

                # Adjust the length if it is set to -1 to read the entire file
                if length == -1:
                    length = total_samples // ishne_header['Sampling_Rate']

                # Read the ecgSig signal
                num_sample = length * ishne_header['Sampling_Rate']

                ecg_sig = np.fromfile(fid, dtype=np.int16, count=ishne_header['nbLeads'] * num_sample).reshape(
                    (num_sample, ishne_header['nbLeads']))

            return ishne_header, ecg_sig
        except Exception as e:
            return None, None

    def load_mfbf(self, filename):
        with open(filename, 'rb') as f:
            # Read dimensions (as int32)
            cols = int(np.fromfile(f, dtype=np.int32, count=1)[0])
            rows = int(np.fromfile(f, dtype=np.int32, count=1)[0])

            # Read the matrix data (as float32), total elements = rows * cols
            data = np.fromfile(f, dtype=np.float32, count=rows * cols)

            # Reshape and transpose back (remember M was transposed before saving)
            matrix = data.reshape((cols, rows)).T

            # Try to read the rest as ASCII (extra text), if any
            try:
                extra = f.read().decode('ascii').splitlines()
            except:
                extra = []

        return matrix, extra

    def load_signal(self):
        # Open file dialog to select a signal file
        filetypes = [
            ('All supported files', '*.xml *.mat *.npy *.txt *.csv *.ecg *.bin')  # combined
        ]

        filename = filedialog.askopenfilename(
            title="Select Signal File",
            filetypes=filetypes
        )

        if filename:
            try:
                # Load the file based on its extension
                ext = os.path.splitext(filename)[1].lower()
                if ext == '.mat':
                    # Load MATLAB file
                    DATA = scipy.io.loadmat(filename)
                    valid_keys = [key for key in DATA.keys() if (
                        (not key.startswith('__')) and (key.lower() != 'fs'))]
                    data = DATA[valid_keys[0]]
                    try:
                        fs = DATA['fs'].item()  # convert from array to scalar
                        self.sample_rate = fs
                        self.sample_rate_var = tk.StringVar(value=str(fs))
                        ttk.Entry(self.lead_filter_frame, textvariable=self.sample_rate_var, width=5).grid(
                            row=0, column=4, padx=5, pady=5)
                    except:
                        pass

                elif ext == '.npy':
                    # Load numpy file
                    data = np.load(filename, allow_pickle=True)
                elif ext == '.txt' or ext == '.csv':
                    # Load text or CSV file
                    data = np.loadtxt(filename, delimiter=',')
                elif ext == '.ecg':  # ISHNE
                    header, data = self.read_ecg_file(filename)
                    if header is not None:
                        fs = header['Sampling_Rate']
                        self.sample_rate = fs
                        self.sample_rate_var = tk.StringVar(value=str(fs))
                        ttk.Entry(self.lead_filter_frame, textvariable=self.sample_rate_var, width=5).grid(
                            row=0, column=4, padx=5, pady=5)
                    if data is None:
                        data, header = self.load_mfbf(filename)
                        data = np.array(data)
                        # Check if header is None
                        if header is not None:
                            try:
                                fs = header['Sampling_Rate']
                                self.sample_rate = fs
                                self.sample_rate_var = tk.StringVar(
                                    value=str(fs))
                                ttk.Entry(self.lead_filter_frame, textvariable=self.sample_rate_var, width=5).grid(
                                    row=0, column=4, padx=5, pady=5)
                            except:
                                messagebox.showwarning(
                                    "Warning", "Failed to read sampling rate from ECG file, using default 200 Hz.")
                                fs = 200  # Default value if not found
                elif ext == '.bin':
                    data, extra = self.load_mfbf(filename)
                    data = np.array(data)
                    # Try to extract sampling rate from extra text
                    fs = None
                    for line in extra:
                        if 'sampling' in line.lower() and '=' in line:
                            try:
                                fs = float(line.split('=')[1].strip())
                                break
                            except:
                                continue
                    if fs is None:
                        messagebox.showwarning(
                                    "Warning", "Failed to read sampling rate from ECG file, using default 200 Hz.")
                        fs = 1000  # Default value if not found
                    else:
                        self.sample_rate = fs
                        self.sample_rate_var = tk.StringVar(
                            value=str(fs))
                        ttk.Entry(self.lead_filter_frame, textvariable=self.sample_rate_var, width=5).grid(
                            row=0, column=4, padx=5, pady=5)                
                elif ext == ".xml": #GE MUSE
                    data, fs = self.load_xml(filename)
                    if fs is None:
                        messagebox.showwarning(
                                    "Warning", "Failed to read sampling rate from ECG file, using default 200 Hz.")
                        fs = 200  # Default value if not found
                    else:
                        self.sample_rate = fs
                        self.sample_rate_var = tk.StringVar(
                            value=str(fs))
                        ttk.Entry(self.lead_filter_frame, textvariable=self.sample_rate_var, width=5).grid(
                            row=0, column=4, padx=5, pady=5)
                else:
                    # Try numpy format first, then text
                    try:
                        data = np.load(filename, allow_pickle=True)
                    except:
                        try: 
                            data = np.loadtxt(filename, delimiter=',')
                        except:
                            messagebox.showerror(
                                    "Error", "Failed to read file, file is corrpeted or not supported")

                # Handle different data shapes for 12-lead ECG
                # Reshape data as needed for different signal formats
                # Determine the number of leads and handle accordingly
                if len(data.shape) == 1:
                    # Single signal, single lead
                    self.signal_data = [[data]]  # [signal_idx][lead_idx]
                    self.leads = 1
                elif len(data.shape) == 2:
                    if data.shape[1] < data.shape[0]:
                        data = data.T  # Transpose to get leads as rows
                    if data.shape[0] <= 12:
                        # Likely multiple leads, single signal
                        # One signal with multiple leads
                        self.signal_data = [data]
                        self.leads = data.shape[0]
                    else:
                        # Likely multiple signals, single lead
                        self.signal_data = [[data[i, :]]
                                            for i in range(data.shape[0])]
                        self.leads = 1
                elif len(data.shape) == 3:
                    # Multiple signals with multiple leads
                    self.signal_data = data
                    self.leads = data.shape[1]

                # Update lead dropdown values based on actual lead count
                if self.leads == 1:
                    self.lead_dropdown.configure(values=["Lead I"])
                elif self.leads <= 3:
                    self.lead_dropdown.configure(
                        values=["Lead I (X)", "Lead II (Y)", "Lead III(Z)"][:self.leads])
                elif self.leads <= 6:
                    self.lead_dropdown.configure(values=["Lead I", "Lead II", "Lead III",
                                                         "aVR", "aVL", "aVF"][:self.leads])
                else:
                    self.lead_dropdown.configure(values=["Lead I", "Lead II", "Lead III",
                                                         "aVR", "aVL", "aVF",
                                                         "V1", "V2", "V3", "V4", "V5", "V6"][:self.leads])

                # Set lead dropdown to first lead
                self.lead_var.set(self.lead_dropdown["values"][0])
                self.current_lead = 0

                # Reset current index, view position, and segmentation data
                self.current_index = 0
                self.view_start = 0
                self.segmentation_data = {}
                self.filtered_signal = None
                self.current_file = filename

                # Update the plot
                self.update_plot()
                self.file_name = os.path.basename(filename).split(".")[0]


            except Exception as e:
                messagebox.showerror(
                    "Error", f"Failed to load signal: {str(e)}")

    def save_segmentation(self):
        if not self.segmentation_data:
            messagebox.showwarning("Warning", "No segmentation data to save")
            return

        # Open file dialog to select save location
        filetypes = [
            ('Numpy files', '*.npy'),
            ('JSON files', '*.json'),
            ('CSV files', '*.csv'),
        ]

        filename = filedialog.asksaveasfilename(
            title="Save Segmentation Data",
            filetypes=filetypes,
            defaultextension=".npy"
        )

        if filename:
            try:
                ext = os.path.splitext(filename)[1].lower()

                if ext == '.npy':
                    # Save as numpy file
                    np.save(filename, self.segmentation_data)
                elif ext == '.json':
                    # Save as JSON
                    import json
                    # Convert keys to strings for JSON serialization
                    # Also convert numpy arrays/types to Python native types
                    json_data = {}
                    for signal_idx, signal_data in self.segmentation_data.items():
                        json_data[str(signal_idx)] = {}
                        for wave_type, positions in signal_data.items():
                            json_data[str(signal_idx)][wave_type] = [
                                int(pos) for pos in positions]

                    with open(filename, 'w') as f:
                        json.dump(json_data, f, indent=2)
                elif ext == '.csv':
                    # Save as CSV with multiple entries per wave type
                    import csv
                    with open(filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        # Write header
                        writer.writerow(
                            ['signal_index', 'lead_index', 'wave_type', 'sample_position'])

                        # Write data
                        for signal_idx in sorted(self.segmentation_data.keys()):
                            for wave_type, positions in self.segmentation_data[signal_idx].items():
                                for position in positions:
                                    writer.writerow(
                                        [signal_idx, self.current_lead, wave_type, position])

                messagebox.showinfo(
                    "Success", f"Segmentation data saved to {filename}")

            except Exception as e:
                messagebox.showerror(
                    "Error", f"Failed to save segmentation data: {str(e)}")

    def load_xml(self, filename):
        try: #GE MUSE
            _, df_rhythm, metadata = GEXMLparser(filename)
            try: 
                fs = metadata["samplingFrequency"] 
            except:
                fs = None
            ecg = df_rhythm.to_numpy()

            return ecg, fs 
        except:
            try:
                sample_freq, df_median, df_rythm = mortaraXMLPARSER(filename)
                ecg = df_rythm.to_numpy()
                return ecg, sample_freq 
            except: 
                messagebox.showerror(
                    "Error", f"Failed to load signal, file corrupted or not supported")

    def move_to_start(self):
        """Pan the view window to the left"""
        if self.signal_data is None:
            return
        self.view_start = 0
        self.update_plot()
    
    def move_to_end(self):
        """Pan the view window to the left"""
        if self.signal_data is None:
            return
        self.view_start = np.shape(self.signal_data)[-1] - self.window_size
        
        self.update_plot()
    
def main():
    root = tk.Tk()
    app = ECGlyticsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
