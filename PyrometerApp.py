import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import os
import glob
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import fsolve, root_scalar

class PyrometerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Color Pyrometer Analysis")
        self.root.geometry("1400x900")
        
        # Physical constants
        self.h = 6.626e-34  # Planck's constant
        self.c = 3e8        # Speed of light
        self.k = 1.381e-23  # Boltzmann constant
        
        # Default settings
        self.num_detectors = tk.IntVar(value=3)
        self.snr_threshold = tk.DoubleVar(value=1.0)
        self.snr_percentage = tk.DoubleVar(value=50.0)
        # self.cal_temperature = tk.DoubleVar(value=2796.0)  # ADD THIS LINE
        self.smoothing_window = tk.IntVar(value=100)
        # Multiple calibration temperatures
        self.calibration_temps = []  # List of (temperature, description) tuples
        self.add_default_calibration()
        
        # Data storage
        self.detector_wavelengths = {}
        self.folder_paths = {}
        self.processed_data = {}
        self.snr_results = {}
        self.good_detectors = []
        
        self.setup_gui()
        self.update_detector_settings()
    
    
    def setup_gui(self):
        """Setup the main GUI layout"""
        
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create all tabs
        self.setup_frame = ttk.Frame(self.notebook)
        self.files_frame = ttk.Frame(self.notebook)
        self.analysis_frame = ttk.Frame(self.notebook)
        self.results_frame = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.setup_frame, text="Setup")
        self.notebook.add(self.files_frame, text="Folder Selection")
        self.notebook.add(self.analysis_frame, text="Analysis")
        self.notebook.add(self.results_frame, text="Results")
        
        # Setup each tab
        self.setup_setup_tab()
        self.setup_files_tab()
        self.setup_analysis_tab()
        self.setup_results_tab()
    
    def setup_setup_tab(self):
        """Setup the configuration tab"""
        
        # Main settings frame
        settings_frame = ttk.LabelFrame(self.setup_frame, text="Pyrometer Configuration")
        settings_frame.pack(fill='x', padx=10, pady=10)
        
        # Number of detectors
        ttk.Label(settings_frame, text="Number of Detectors:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        detector_frame = ttk.Frame(settings_frame)
        detector_frame.grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Radiobutton(detector_frame, text="3-Color", variable=self.num_detectors, 
               value=3, command=self.update_detector_settings).pack(side='left', padx=5)
        ttk.Radiobutton(detector_frame, text="32-Color", variable=self.num_detectors, 
                       value=32, command=self.update_detector_settings).pack(side='left', padx=5)
        
        # SNR Threshold
        # SNR Threshold
        ttk.Label(settings_frame, text="SNR Threshold:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(settings_frame, textvariable=self.snr_threshold, width=10).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(settings_frame, text="(minimum signal-to-noise ratio)").grid(row=1, column=2, sticky='w', padx=5, pady=5)

        # SNR Percentage Threshold - NEW
        ttk.Label(settings_frame, text="SNR Good Time %:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(settings_frame, textvariable=self.snr_percentage, width=10).grid(row=2, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(settings_frame, text="(% of time SNR must be above threshold)").grid(row=2, column=2, sticky='w', padx=5, pady=5)

        ttk.Label(settings_frame, text="Smoothing Window:").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(settings_frame, textvariable=self.smoothing_window, width=10).grid(row=3, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(settings_frame, text="(data points for temperature smoothing)").grid(row=3, column=2, sticky='w', padx=5, pady=5)

        # # Move Calibration Temperature down to row=3
        # ttk.Label(settings_frame, text="Calibration Temperature (K):").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        # ttk.Entry(settings_frame, textvariable=self.cal_temperature, width=10).grid(row=3, column=1, sticky='w', padx=5, pady=5)
        # ttk.Label(settings_frame, text="(grey body source temperature)").grid(row=3, column=2, sticky='w', padx=5, pady=5)
        # Calibration temperatures section
        cal_frame = ttk.LabelFrame(self.setup_frame, text="Calibration Temperatures")
        cal_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Calibration temperature controls
        cal_control_frame = ttk.Frame(cal_frame)
        cal_control_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(cal_control_frame, text="Add Calibration Temperature", 
                  command=self.add_calibration_temp).pack(side='left', padx=5)
        ttk.Button(cal_control_frame, text="Remove Selected", 
                  command=self.remove_calibration_temp).pack(side='left', padx=5)
        
        # Calibration temperature list
        self.cal_listbox_frame = ttk.Frame(cal_frame)
        self.cal_listbox_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.cal_listbox = tk.Listbox(self.cal_listbox_frame, height=4)
        cal_scrollbar = ttk.Scrollbar(self.cal_listbox_frame, orient='vertical', command=self.cal_listbox.yview)
        self.cal_listbox.configure(yscrollcommand=cal_scrollbar.set)
        
        self.cal_listbox.pack(side='left', fill='both', expand=True)
        cal_scrollbar.pack(side='right', fill='y')
        
        self.update_calibration_list()
        
        # Detector wavelengths frame
        self.wavelength_frame = ttk.LabelFrame(self.setup_frame, text="Detector Wavelengths (nm)")
        self.wavelength_frame.pack(fill='x', padx=10, pady=10)
        
        # Instructions
        instructions = ttk.LabelFrame(self.setup_frame, text="Instructions")
        instructions.pack(fill='both', expand=True, padx=10, pady=10)
        
        instruction_text = """
1. Choose 2-Color or 32-Color pyrometry
2. Set SNR threshold (detectors below this will be excluded)
3. Add calibration temperatures for your reference sources
4. Adjust detector wavelengths if needed
5. Go to 'Folder Selection' tab to select data folders
6. Files should be named with C1, C2, C3... prefixes for different detectors
        """
        
        ttk.Label(instructions, text=instruction_text, justify='left').pack(padx=10, pady=10)

        window_frame = ttk.LabelFrame(self.setup_frame, text="Polycarbonate Window Correction")
        window_frame.pack(fill='x', padx=10, pady=10)
        
        # Enable/disable window correction
        self.enable_window_correction = tk.BooleanVar(value=False)
        ttk.Checkbutton(window_frame, text="Apply window transmission correction",
                    variable=self.enable_window_correction).grid(row=0, column=0, columnspan=4, sticky='w', padx=5, pady=5)
        
        # Reference thickness for transmission values
        ttk.Label(window_frame, text="Reference Thickness (mm):").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.reference_thickness = tk.DoubleVar(value=3.0)
        ttk.Entry(window_frame, textvariable=self.reference_thickness, width=8).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(window_frame, text="(thickness at which transmission values were measured)").grid(row=1, column=2, columnspan=2, sticky='w', padx=5, pady=5)
        
        # Actual window thickness
        ttk.Label(window_frame, text="Actual Window Thickness (mm):").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.actual_thickness = tk.DoubleVar(value=3.0)
        ttk.Entry(window_frame, textvariable=self.actual_thickness, width=8).grid(row=2, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(window_frame, text="(your actual window thickness)").grid(row=2, column=2, columnspan=2, sticky='w', padx=5, pady=5)
        
        # Detector transmission values frame
        self.transmission_frame = ttk.LabelFrame(window_frame, text="Detector Transmission Values")
        self.transmission_frame.grid(row=3, column=0, columnspan=4, sticky='ew', padx=5, pady=10)
        
        # Storage for transmission values
        self.detector_transmissions = {}
        
        # Instructions
        instructions_text = """
    Instructions:
    1. Enter the reference thickness (thickness at which your transmission values were measured)
    2. Enter your actual window thickness
    3. Input transmission values (0.0 to 1.0) for each detector wavelength
    4. Transmission will be automatically adjusted for thickness difference using Beer-Lambert law
        """
        
        ttk.Label(window_frame, text=instructions_text, justify='left', font=('Arial', 9)).grid(row=4, column=0, columnspan=4, sticky='w', padx=5, pady=5)
    
    def add_calibration_temp(self):
        """Add a new calibration temperature"""
        dialog = CalibrationTempDialog(self.root)
        if dialog.result:
            temp, description = dialog.result
            self.calibration_temps.append((temp, description))
            self.update_calibration_list()
    
    def remove_calibration_temp(self):
        """Remove selected calibration temperature"""
        selection = self.cal_listbox.curselection()
        if selection and len(self.calibration_temps) > 1:  # Keep at least one
            index = selection[0]
            del self.calibration_temps[index]
            self.update_calibration_list()
        elif len(self.calibration_temps) <= 1:
            messagebox.showwarning("Warning", "Must have at least one calibration temperature!")
    
    def update_calibration_list(self):
        """Update the calibration temperature listbox"""
        self.cal_listbox.delete(0, tk.END)
        for temp, description in self.calibration_temps:
            self.cal_listbox.insert(tk.END, f"{temp:.1f}K - {description}")
    
    def update_detector_settings(self):
        """Update detector wavelength settings and transmission inputs"""
        
        # Clear existing wavelength widgets
        for widget in self.wavelength_frame.winfo_children():
            widget.destroy()
        
        # Clear existing transmission widgets
        if hasattr(self, 'transmission_frame'):
            for widget in self.transmission_frame.winfo_children():
                widget.destroy()
        
        num_det = self.num_detectors.get()
        self.detector_wavelengths = {}
        self.detector_transmissions = {}
        
        if num_det == 3:
            # === 3-COLOR SYSTEM ===
            default_wavelengths = [800, 1100, 1400]
            default_transmissions = [0.85, 0.82, 0.75]  # Typical polycarbonate values
            
            # WAVELENGTH INPUTS
            for i in range(3):
                detector_name = f"detector{i+1}"
                
                ttk.Label(self.wavelength_frame, text=f"Detector {i+1}:").grid(row=i, column=0, sticky='w', padx=5, pady=5)
                
                wavelength_var = tk.DoubleVar(value=default_wavelengths[i])
                self.detector_wavelengths[detector_name] = wavelength_var
                
                ttk.Entry(self.wavelength_frame, textvariable=wavelength_var, width=10).grid(row=i, column=1, sticky='w', padx=5, pady=5)
                ttk.Label(self.wavelength_frame, text="nm").grid(row=i, column=2, sticky='w', padx=5, pady=5)
            
            # TRANSMISSION INPUTS (3 detectors)
            if hasattr(self, 'transmission_frame'):
                ttk.Label(self.transmission_frame, text="Detector", font=('Arial', 9, 'bold')).grid(row=0, column=0, padx=5, pady=2)
                ttk.Label(self.transmission_frame, text="Wavelength", font=('Arial', 9, 'bold')).grid(row=0, column=1, padx=5, pady=2)
                ttk.Label(self.transmission_frame, text="Transmission", font=('Arial', 9, 'bold')).grid(row=0, column=2, padx=5, pady=2)
                ttk.Label(self.transmission_frame, text="(0.0 - 1.0)", font=('Arial', 8)).grid(row=0, column=3, padx=5, pady=2)
                
                for i in range(3):
                    detector_name = f"detector{i+1}"
                    
                    # Detector label
                    ttk.Label(self.transmission_frame, text=f"C{i+1}").grid(row=i+1, column=0, sticky='w', padx=5, pady=2)
                    
                    # Wavelength display (updates automatically)
                    wavelength_display = ttk.Label(self.transmission_frame, text=f"{default_wavelengths[i]:.0f}nm")
                    wavelength_display.grid(row=i+1, column=1, sticky='w', padx=5, pady=2)
                    
                    # Transmission input
                    transmission_var = tk.DoubleVar(value=default_transmissions[i])
                    self.detector_transmissions[detector_name] = transmission_var
                    
                    entry = ttk.Entry(self.transmission_frame, textvariable=transmission_var, width=8)
                    entry.grid(row=i+1, column=2, sticky='w', padx=5, pady=2)
                    
                    # Status indicator
                    status_label = ttk.Label(self.transmission_frame, text="✓", foreground="green")
                    status_label.grid(row=i+1, column=3, sticky='w', padx=5, pady=2)
                    
                    # Update wavelength display when wavelength changes
                    def create_wavelength_updater(detector=detector_name, display=wavelength_display, status=status_label):
                        def update_display(*args):
                            try:
                                wavelength = self.detector_wavelengths[detector].get()
                                display.config(text=f"{wavelength:.0f}nm")
                                
                                # Update status based on transmission value
                                transmission = self.detector_transmissions[detector].get()
                                if 0.0 < transmission <= 1.0:
                                    status.config(text="✓", foreground="green")
                                else:
                                    status.config(text="✗", foreground="red")
                            except:
                                status.config(text="?", foreground="orange")
                        return update_display
                    
                    # Bind both wavelength and transmission changes
                    self.detector_wavelengths[detector_name].trace('w', create_wavelength_updater())
                    self.detector_transmissions[detector_name].trace('w', create_wavelength_updater())
        
        elif num_det == 32:
            # === 32-COLOR SYSTEM ===
            
            # WAVELENGTH INPUTS (scrollable)
            canvas = tk.Canvas(self.wavelength_frame, height=200)
            scrollbar = ttk.Scrollbar(self.wavelength_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Default wavelengths for 32-color
            start_wavelength = 400
            end_wavelength = 1100
            wavelength_step = (end_wavelength - start_wavelength) / 31
            
            # Create wavelength entries in a grid (8 columns: 4 detectors per row)
            for i in range(32):
                detector_name = f"detector{i+1}"
                default_wavelength = start_wavelength + i * wavelength_step
                
                row = i // 4
                col = (i % 4) * 3  # 3 columns per detector (label, entry, unit)
                
                ttk.Label(scrollable_frame, text=f"C{i+1}:").grid(row=row, column=col, sticky='w', padx=2, pady=2)
                
                wavelength_var = tk.DoubleVar(value=round(default_wavelength, 1))
                self.detector_wavelengths[detector_name] = wavelength_var
                
                ttk.Entry(scrollable_frame, textvariable=wavelength_var, width=6).grid(row=row, column=col+1, sticky='w', padx=2, pady=2)
                ttk.Label(scrollable_frame, text="nm").grid(row=row, column=col+2, sticky='w', padx=2, pady=2)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # TRANSMISSION INPUTS (32 detectors - also scrollable)
            if hasattr(self, 'transmission_frame'):
                # Create scrollable transmission frame
                trans_canvas = tk.Canvas(self.transmission_frame, height=300)
                trans_scrollbar = ttk.Scrollbar(self.transmission_frame, orient="vertical", command=trans_canvas.yview)
                trans_scrollable_frame = ttk.Frame(trans_canvas)
                
                trans_scrollable_frame.bind(
                    "<Configure>",
                    lambda e: trans_canvas.configure(scrollregion=trans_canvas.bbox("all"))
                )
                
                trans_canvas.create_window((0, 0), window=trans_scrollable_frame, anchor="nw")
                trans_canvas.configure(yscrollcommand=trans_scrollbar.set)
                
                # Headers
                ttk.Label(trans_scrollable_frame, text="Det", font=('Arial', 8, 'bold')).grid(row=0, column=0, padx=2, pady=2)
                ttk.Label(trans_scrollable_frame, text="λ(nm)", font=('Arial', 8, 'bold')).grid(row=0, column=1, padx=2, pady=2)
                ttk.Label(trans_scrollable_frame, text="Trans", font=('Arial', 8, 'bold')).grid(row=0, column=2, padx=2, pady=2)
                ttk.Label(trans_scrollable_frame, text="✓", font=('Arial', 8, 'bold')).grid(row=0, column=3, padx=2, pady=2)
                
                # Repeat headers every 8 rows for readability
                for header_row in range(8, 40, 8):
                    if header_row < 32:
                        ttk.Label(trans_scrollable_frame, text="Det", font=('Arial', 8, 'bold')).grid(row=header_row+1, column=0, padx=2, pady=2)
                        ttk.Label(trans_scrollable_frame, text="λ(nm)", font=('Arial', 8, 'bold')).grid(row=header_row+1, column=1, padx=2, pady=2)
                        ttk.Label(trans_scrollable_frame, text="Trans", font=('Arial', 8, 'bold')).grid(row=header_row+1, column=2, padx=2, pady=2)
                        ttk.Label(trans_scrollable_frame, text="✓", font=('Arial', 8, 'bold')).grid(row=header_row+1, column=3, padx=2, pady=2)
                
                # Create transmission entries for all 32 detectors
                for i in range(32):
                    detector_name = f"detector{i+1}"
                    default_wavelength = start_wavelength + i * wavelength_step
                    
                    # Estimate transmission based on wavelength
                    if default_wavelength < 600:
                        default_transmission = 0.88
                    elif default_wavelength < 800:
                        default_transmission = 0.86
                    elif default_wavelength < 1000:
                        default_transmission = 0.83
                    else:
                        default_transmission = 0.78
                    
                    # Calculate row (skip header rows)
                    display_row = i + 1 + (i // 8)  # Add extra row for every 8 detectors (headers)
                    
                    # Detector number
                    ttk.Label(trans_scrollable_frame, text=f"C{i+1}").grid(row=display_row, column=0, sticky='w', padx=2, pady=1)
                    
                    # Wavelength display
                    wavelength_display = ttk.Label(trans_scrollable_frame, text=f"{default_wavelength:.0f}")
                    wavelength_display.grid(row=display_row, column=1, sticky='w', padx=2, pady=1)
                    
                    # Transmission input
                    transmission_var = tk.DoubleVar(value=default_transmission)
                    self.detector_transmissions[detector_name] = transmission_var
                    
                    entry = ttk.Entry(trans_scrollable_frame, textvariable=transmission_var, width=6, font=('Arial', 8))
                    entry.grid(row=display_row, column=2, sticky='w', padx=2, pady=1)
                    
                    # Status indicator
                    status_label = ttk.Label(trans_scrollable_frame, text="✓", foreground="green", font=('Arial', 8))
                    status_label.grid(row=display_row, column=3, sticky='w', padx=2, pady=1)
                    
                    # Update functions
                    def create_32_channel_updater(detector=detector_name, display=wavelength_display, status=status_label):
                        def update_display(*args):
                            try:
                                wavelength = self.detector_wavelengths[detector].get()
                                display.config(text=f"{wavelength:.0f}")
                                
                                transmission = self.detector_transmissions[detector].get()
                                if 0.0 < transmission <= 1.0:
                                    status.config(text="✓", foreground="green")
                                else:
                                    status.config(text="✗", foreground="red")
                            except:
                                status.config(text="?", foreground="orange")
                        return update_display
                    
                    self.detector_wavelengths[detector_name].trace('w', create_32_channel_updater())
                    self.detector_transmissions[detector_name].trace('w', create_32_channel_updater())
                
                trans_canvas.pack(side="left", fill="both", expand=True)
                trans_scrollbar.pack(side="right", fill="y")
                
            # Add mouse wheel scrolling for both canvases
            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                if hasattr(self, 'transmission_frame'):
                    trans_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def apply_window_transmission_correction(self, signals_dict):
        """
        Apply window transmission correction using user-input values
        Accounts for thickness difference using Beer-Lambert law
        """
        
        if not self.enable_window_correction.get():
            return signals_dict
        
        corrected_signals = {}
        reference_thickness = self.reference_thickness.get()
        actual_thickness = self.actual_thickness.get()
        
        self.snr_text.insert(tk.END, f"\nAPPLYING WINDOW TRANSMISSION CORRECTION:\n")
        self.snr_text.insert(tk.END, f"Reference thickness: {reference_thickness:.1f}mm\n")
        self.snr_text.insert(tk.END, f"Actual thickness: {actual_thickness:.1f}mm\n")
        self.snr_text.insert(tk.END, "-" * 50 + "\n")
        
        for detector, signal in signals_dict.items():
            if detector in self.detector_transmissions:
                # Get user-input transmission value (at reference thickness)
                reference_transmission = self.detector_transmissions[detector].get()
                
                # Adjust transmission for actual thickness using Beer-Lambert law
                # T_actual = T_reference ^ (actual_thickness / reference_thickness)
                if reference_thickness > 0:
                    thickness_ratio = actual_thickness / reference_thickness
                    actual_transmission = reference_transmission ** thickness_ratio
                else:
                    actual_transmission = reference_transmission
                
                # Correct signal by dividing by transmission
                corrected_signal = signal / actual_transmission
                corrected_signals[detector] = corrected_signal
                
                # Get wavelength for logging
                wavelength = self.detector_wavelengths[detector].get()
                correction_factor = 1 / actual_transmission
                
                self.snr_text.insert(tk.END, 
                    f"{detector} ({wavelength:.0f}nm): "
                    f"T_ref={reference_transmission:.3f}, "
                    f"T_actual={actual_transmission:.3f}, "
                    f"Correction={correction_factor:.2f}x\n")
            else:
                # No transmission value provided, use uncorrected signal
                corrected_signals[detector] = signal
                wavelength = self.detector_wavelengths.get(detector, tk.DoubleVar(value=0)).get()
                self.snr_text.insert(tk.END, 
                    f"{detector} ({wavelength:.0f}nm): No transmission value - using uncorrected signal\n")
        
        return corrected_signals

    def show_transmission_summary(self):
        """Display transmission correction summary"""
        
        if not self.enable_window_correction.get():
            return
        
        self.snr_text.insert(tk.END, f"\nTRANSMISSION CORRECTION SUMMARY:\n")
        self.snr_text.insert(tk.END, "=" * 40 + "\n")
        
        reference_thickness = self.reference_thickness.get()
        actual_thickness = self.actual_thickness.get()
        thickness_ratio = actual_thickness / reference_thickness if reference_thickness > 0 else 1.0
        
        self.snr_text.insert(tk.END, f"Thickness scaling factor: {thickness_ratio:.3f}\n")
        self.snr_text.insert(tk.END, f"{'Detector':<12} {'Wavelength':<12} {'T_input':<10} {'T_actual':<10} {'Correction':<10}\n")
        self.snr_text.insert(tk.END, "-" * 60 + "\n")
        
        for detector in self.good_detectors:
            if detector in self.detector_transmissions:
                wavelength = self.detector_wavelengths[detector].get()
                t_input = self.detector_transmissions[detector].get()
                t_actual = t_input ** thickness_ratio
                correction = 1 / t_actual
                
                self.snr_text.insert(tk.END, 
                    f"{detector:<12} {wavelength:<12.0f} {t_input:<10.3f} {t_actual:<10.3f} {correction:<10.2f}\n")


    def get_matlab_style_corrected_signals(self):
        """Get corrected signals with window transmission correction"""
        
        corrected_signals = {}
        
        # First apply basic dark subtraction
        for detector in self.good_detectors:
            if (detector in self.processed_data['experimental'] and
                detector in self.processed_data['dark']):
                
                exp_signal = self.processed_data['experimental'][detector]['signal']
                dark_signal = self.processed_data['dark'][detector]['signal']
                
                # MATLAB-style: simple mean dark subtraction
                dark_mean = np.mean(dark_signal)
                corrected = exp_signal - dark_mean
                
                # Ensure positive values
                corrected = np.maximum(corrected, 1e-12)
                
                corrected_signals[detector] = corrected
        
        # Then apply window transmission correction
        corrected_signals = self.apply_window_transmission_correction(corrected_signals)
        
        return corrected_signals

    def setup_files_tab(self):
        """Setup folder selection tab"""
        
        # File categories
        self.file_categories = ['experimental', 'calibration', 'background', 'dark']
        self.folder_paths = {category: tk.StringVar() for category in self.file_categories}
        
        # Instructions
        instructions = ttk.Label(self.files_frame, 
                               text="Select folders containing data files. Files should be named with C1, C2, C3... prefixes for different detectors.",
                               font=('Arial', 10, 'bold'))
        instructions.pack(pady=10)
        
        # Folder selection frame
        folder_frame = ttk.Frame(self.files_frame)
        folder_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        for i, category in enumerate(self.file_categories):
            # Category label
            ttk.Label(folder_frame, text=f"{category.title()} Folder:", 
                     font=('Arial', 10, 'bold')).grid(row=i*2, column=0, sticky='w', pady=(10,5))
            
            # Folder path frame
            path_frame = ttk.Frame(folder_frame)
            path_frame.grid(row=i*2+1, column=0, sticky='ew', pady=(0,10))
            folder_frame.grid_columnconfigure(0, weight=1)
            path_frame.grid_columnconfigure(0, weight=1)
            
            # Path entry
            path_entry = ttk.Entry(path_frame, textvariable=self.folder_paths[category], width=80)
            path_entry.grid(row=0, column=0, sticky='ew', padx=(0,10))
            
            # Browse button
            ttk.Button(path_frame, text="Browse", 
                      command=lambda cat=category: self.browse_folder(cat)).grid(row=0, column=1)
        
        # Process button
        ttk.Button(self.files_frame, text="Load Files from Folders", 
                  command=self.load_files_from_folders).pack(pady=20)
        
        # File preview frame
        preview_frame = ttk.LabelFrame(self.files_frame, text="File Preview")
        preview_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.file_preview_text = tk.Text(preview_frame, height=8, width=100, font=('Courier', 9))
        preview_scrollbar = ttk.Scrollbar(preview_frame, orient='vertical', command=self.file_preview_text.yview)
        self.file_preview_text.configure(yscrollcommand=preview_scrollbar.set)
        
        self.file_preview_text.pack(side='left', fill='both', expand=True)
        preview_scrollbar.pack(side='right', fill='y')
    
    def browse_folder(self, category):
        """Browse and select folder for a specific category"""
        
        folder = filedialog.askdirectory(
            title=f"Select {category} folder"
        )
        
        if folder:
            self.folder_paths[category].set(folder)
            self.preview_files_in_folder(category, folder)
    
    def preview_files_in_folder(self, category, folder):
        """Preview files found in the selected folder"""
        
        self.file_preview_text.delete(1.0, tk.END)
        self.file_preview_text.insert(tk.END, f"Files in {category} folder:\n")
        self.file_preview_text.insert(tk.END, "="*50 + "\n")
        
        num_detectors = self.num_detectors.get()
        
        for i in range(num_detectors):
            detector_prefix = f"C{i+1}"
            pattern = os.path.join(folder, f"{detector_prefix}*.txt")
            files = glob.glob(pattern)
            
            self.file_preview_text.insert(tk.END, f"\n{detector_prefix} files ({len(files)} found):\n")
            
            if files:
                for file in sorted(files)[:5]:  # Show first 5 files
                    filename = os.path.basename(file)
                    self.file_preview_text.insert(tk.END, f"  {filename}\n")
                
                if len(files) > 5:
                    self.file_preview_text.insert(tk.END, f"  ... and {len(files)-5} more files\n")
            else:
                self.file_preview_text.insert(tk.END, f"  No files found with pattern {detector_prefix}*.txt\n")
        
        self.file_preview_text.see(tk.END)
    
    def load_files_from_folders(self):
        """Load files from selected folders"""
        
        try:
            self.processed_data = {}
            num_detectors = self.num_detectors.get()
            
            self.file_preview_text.delete(1.0, tk.END)
            self.file_preview_text.insert(tk.END, "LOADING FILES...\n")
            self.file_preview_text.insert(tk.END, "="*50 + "\n")
            
            # Load data from each folder
            for category in self.file_categories:
                folder_path = self.folder_paths[category].get()
                
                if not folder_path:
                    self.file_preview_text.insert(tk.END, f"No folder selected for {category}\n")
                    continue
                
                self.processed_data[category] = {}
                self.file_preview_text.insert(tk.END, f"\nLoading {category} files from: {folder_path}\n")
                
                for i in range(num_detectors):
                    detector_name = f"detector{i+1}"
                    detector_prefix = f"C{i+1}"
                    
                    # Find files matching the detector prefix
                    pattern = os.path.join(folder_path, f"{detector_prefix}*.txt")
                    files = glob.glob(pattern)
                    
                    if files:
                        time_data, signal_data = self.load_and_average_files_from_list(files)
                        
                        if signal_data is not None:
                            self.processed_data[category][detector_name] = {
                                'time': time_data,
                                'signal': signal_data
                            }
                            self.file_preview_text.insert(tk.END, 
                                f"  {detector_name} ({detector_prefix}): {len(files)} files, {len(signal_data)} data points\n")
                        else:
                            self.file_preview_text.insert(tk.END, 
                                f"  {detector_name} ({detector_prefix}): Failed to load data\n")
                    else:
                        self.file_preview_text.insert(tk.END, 
                            f"  {detector_name} ({detector_prefix}): No files found\n")
            
            # Automatically check SNR
            self.file_preview_text.insert(tk.END, "\nFiles loaded! Checking SNR...\n")
            self.file_preview_text.see(tk.END)
            self.root.update()
            
            self.check_snr()
            
            # Switch to analysis tab
            self.notebook.select(2)
            
            messagebox.showinfo("Success", 
                              f"Files loaded successfully!\n"
                              f"Found {len(self.good_detectors)} detectors with good SNR.\n"
                              f"Check Analysis tab for details.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading files: {str(e)}")
    
    # def load_and_average_files_from_list(self, file_list):
    #     """Load and average multiple files from a list"""
        
    #     if not file_list:
    #         return None, None
        
    #     all_data = []
    #     time_data = None
        
    #     for file_path in file_list:
    #         try:
    #             # Try different separators and headers
    #             data_loaded = False
    #             for sep in [',', '\t', ' ', ';']:
    #                 for header in [4, 0, None]:  # Try header=4 first (your format)
    #                     try:
    #                         data = pd.read_csv(file_path, sep=sep, header=header)
    #                         if len(data.columns) >= 2 and len(data) > 0:
    #                             # Check if data is numeric
    #                             if pd.api.types.is_numeric_dtype(data.iloc[:, 0]) and pd.api.types.is_numeric_dtype(data.iloc[:, 1]):
    #                                 if time_data is None:
    #                                     time_data = data.iloc[:, 0].values
    #                                 all_data.append(data.iloc[:, 1].values)
    #                                 data_loaded = True
    #                                 break
    #                     except:
    #                         continue
    #                 if data_loaded:
    #                     break
                        
    #             if not data_loaded:
    #                 print(f"Could not load {file_path}")
                    
    #         except Exception as e:
    #             print(f"Error loading {file_path}: {e}")
        
    #     if all_data and time_data is not None:
    #         # Make sure all data arrays have the same length
    #         min_length = min(len(d) for d in all_data)
    #         time_data = time_data[:min_length]
    #         averaged = np.mean([d[:min_length] for d in all_data], axis=0)
    #         return time_data, averaged
        
    #     return None, None
    
    def load_and_average_files_from_list(self, file_list):
        """Load and average multiple files from a list - using exact same method as standalone code"""
        
        if not file_list:
            return None, None
        
        all_data = []
        time_data = None
        
        for file_path in file_list:
            try:
                # Use exact same loading method as your standalone code
                data = pd.read_csv(file_path, sep=',', header=4)
                if len(data.columns) >= 2 and len(data) > 0:
                    if time_data is None:
                        time_data = data.iloc[:, 0].values
                    all_data.append(data.iloc[:, 1].values)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        if all_data and time_data is not None:
            min_length = min(len(d) for d in all_data)
            time_data = time_data[:min_length]
            averaged = np.mean([d[:min_length] for d in all_data], axis=0)
            return time_data, averaged
        
        return None, None

    def setup_analysis_tab(self):
        """Setup analysis tab"""
        
        # Analysis controls
        control_frame = ttk.LabelFrame(self.analysis_frame, text="Analysis Controls")
        control_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(control_frame, text="Check SNR", 
                  command=self.check_snr).pack(side='left', padx=5, pady=5)
        ttk.Button(control_frame, text="Calculate Temperature", 
                  command=self.calculate_temperature).pack(side='left', padx=5, pady=5)
        ttk.Button(control_frame, text="Clear Results", 
                  command=self.clear_analysis).pack(side='left', padx=5, pady=5)
        
        # SNR Results
        self.snr_frame = ttk.LabelFrame(self.analysis_frame, text="SNR Analysis Results")
        self.snr_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create text widget with scrollbar for SNR results
        text_frame = ttk.Frame(self.snr_frame)
        text_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.snr_text = tk.Text(text_frame, height=20, width=100, font=('Courier', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=self.snr_text.yview)
        self.snr_text.configure(yscrollcommand=scrollbar.set)
        
        self.snr_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
    
    def setup_results_tab(self):
        """Setup results tab with matplotlib plots"""
        
        # Create matplotlib figure with 3 subplots
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 12))
        plt.tight_layout()
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.results_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Control frame
        control_frame = ttk.Frame(self.results_frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="Save Plot", 
                command=self.save_results).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Export All Data", 
                command=self.export_data).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Export Plot Data", 
                command=self.export_individual_plots_data).pack(side='left', padx=5)  # NEW BUTTON
        ttk.Button(control_frame, text="Refresh Plot", 
                command=self.refresh_plot).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Plot SNR vs Time", 
                command=self.plot_snr_vs_time).pack(side='left', padx=5)
        
    def planck_function(self, wavelength_nm, temperature):
        """Calculate Planck function exactly like MATLAB"""
        wavelength_m = wavelength_nm * 1e-9
        
        try:
            # MATLAB: C1/lam^5 * (exp(C2/(lam*T))-1)^-1
            C1 = 2 * self.h * self.c**2
            C2 = self.h * self.c / self.k
            
            exp_term = np.exp(C2 / (wavelength_m * temperature))
            result = (C1 / wavelength_m**5) / (exp_term - 1)
            
            return result
        except:
            return 1e-10
    
    def calculate_snr(self, signal):
        """Calculate comprehensive signal-to-noise ratio metrics with rolling SNR analysis"""
        if signal is None or len(signal) == 0:
            return 0, False, {}
        
        try:
            # Calculate overall SNR metrics (same as before)
            mean_signal = np.mean(signal)
            std_signal = np.std(signal)
            snr = mean_signal / std_signal if std_signal > 0 else 0
            
            # Peak-to-peak noise
            pp_noise = np.max(signal) - np.min(signal)
            
            # RMS noise (better measure)
            rms_noise = np.sqrt(np.mean((signal - mean_signal)**2))
            rms_snr = mean_signal / rms_noise if rms_noise > 0 else 0
            
            # Dynamic range
            dynamic_range = 20 * np.log10(np.max(signal) / np.min(signal)) if np.min(signal) > 0 else 0
            
            # NEW: Rolling SNR analysis
            window_size = max(100, len(signal) // 50)  # Adaptive window size (at least 100 points, or 2% of data)
            rolling_snr = []
            
            for i in range(len(signal) - window_size + 1):
                window_data = signal[i:i + window_size]
                window_mean = np.mean(window_data)
                window_std = np.std(window_data)
                
                if window_std > 0:
                    window_snr = window_mean / window_std
                    rolling_snr.append(window_snr)
                else:
                    rolling_snr.append(0)
            
            rolling_snr = np.array(rolling_snr)
            
            # Calculate percentage of time SNR is above threshold
            threshold = self.snr_threshold.get()
            percentage_threshold = self.snr_percentage.get()
            
            good_snr_points = np.sum(rolling_snr > threshold)
            total_points = len(rolling_snr)
            percentage_good = (good_snr_points / total_points) * 100 if total_points > 0 else 0
            
            # Detector is good if percentage of good SNR time exceeds threshold
            is_good = percentage_good >= percentage_threshold
            
            # Quality assessment using overall RMS SNR
            if rms_snr > 100:
                quality = "Excellent"
            elif rms_snr > 50:
                quality = "Good"
            elif rms_snr > 20:
                quality = "Fair"
            elif rms_snr > 10:
                quality = "Poor"
            else:
                quality = "Very Poor"
            
            # Return comprehensive metrics
            metrics = {
                'mean_signal': mean_signal,
                'std_signal': std_signal,
                'snr': snr,
                'rms_noise': rms_noise,
                'rms_snr': rms_snr,
                'pp_noise': pp_noise,
                'dynamic_range': dynamic_range,
                'quality': quality,
                'min_signal': np.min(signal),
                'max_signal': np.max(signal),
                'rolling_snr': rolling_snr,
                'rolling_snr_mean': np.mean(rolling_snr),
                'rolling_snr_std': np.std(rolling_snr),
                'percentage_good': percentage_good,
                'window_size': window_size,
                'threshold_used': threshold,
                'percentage_threshold': percentage_threshold
            }
            
            return rms_snr, is_good, metrics
            
        except Exception as e:
            print(f"Error calculating SNR: {e}")
            return 0, False, {}
    
    def check_snr(self):
        """Check SNR for fully corrected signals using proper MATLAB pipeline"""
        
        self.snr_text.delete(1.0, tk.END)
        self.snr_results = {}
        
        num_detectors = self.num_detectors.get()
        threshold = self.snr_threshold.get()
        percentage_threshold = self.snr_percentage.get()
        
        self.snr_text.insert(tk.END, "SNR ANALYSIS ON MATLAB-STYLE CORRECTED SIGNALS\n")
        self.snr_text.insert(tk.END, "="*60 + "\n")
        self.snr_text.insert(tk.END, "Pipeline: Dark subtraction → Calibration scaling → Window correction\n\n")
        
        # First determine which detectors have all required data
        available_detectors = []
        for i in range(num_detectors):
            detector_name = f"detector{i+1}"
            if (detector_name in self.processed_data.get('experimental', {}) and
                detector_name in self.processed_data.get('dark', {}) and
                detector_name in self.processed_data.get('calibration', {})):
                available_detectors.append(detector_name)
        
        if not available_detectors:
            self.snr_text.insert(tk.END, "No detectors have all required data (experimental, dark, calibration)\n")
            return
        
        # Temporarily set good_detectors to all available for correction pipeline
        self.good_detectors = available_detectors
        
        # Get fully corrected signals using MATLAB pipeline
        corrected_signals = self.get_matlab_corrected_signals()
        
        # Now evaluate SNR and determine truly good detectors
        final_good_detectors = []
        
        for detector_name in available_detectors:
            if detector_name in corrected_signals:
                fully_corrected = corrected_signals[detector_name]
                
                # Calculate SNR on fully corrected signal
                rms_snr, is_good, metrics = self.calculate_snr(fully_corrected)
                
                # Store results
                self.snr_results['corrected'] = self.snr_results.get('corrected', {})
                self.snr_results['corrected'][detector_name] = {
                    'snr': rms_snr,
                    'is_good': is_good,
                    'metrics': metrics,
                    'time': self.processed_data['experimental'][detector_name]['time'],
                    'signal': fully_corrected
                }
                
                if is_good:
                    final_good_detectors.append(detector_name)
                
                # Display results
                status = "✓ GOOD" if is_good else "✗ POOR"
                wavelength = self.detector_wavelengths.get(detector_name, tk.DoubleVar(value=0)).get()
                
                self.snr_text.insert(tk.END, f"  {detector_name} ({wavelength:.1f}nm):\n")
                self.snr_text.insert(tk.END, f"    Final Corrected Mean: {metrics['mean_signal']:.3e}\n")
                self.snr_text.insert(tk.END, f"    RMS SNR:              {metrics['rms_snr']:.1f} {status}\n")
                self.snr_text.insert(tk.END, f"    Good SNR Time:        {metrics['percentage_good']:.1f}%\n\n")
        
        # Update good_detectors with final results
        self.good_detectors = final_good_detectors
        
        # Summary
        self.snr_text.insert(tk.END, f"FINAL SUMMARY:\n")
        self.snr_text.insert(tk.END, f"Available detectors: {len(available_detectors)}\n")
        self.snr_text.insert(tk.END, f"Good detectors: {', '.join(self.good_detectors)} ({len(self.good_detectors)} total)\n")
        
        if len(self.good_detectors) >= 2:
            self.snr_text.insert(tk.END, f"✓ Ready for {len(self.good_detectors)}-color pyrometry!\n")
        else:
            self.snr_text.insert(tk.END, f"✗ Need at least 2 good detectors for pyrometry\n")
        
        self.snr_text.see(tk.END)
                
        def plot_snr_vs_time(self):
            """Plot SNR vs time for corrected signals"""
            
            if 'corrected' not in self.snr_results:
                messagebox.showwarning("No Data", "No corrected SNR data available. Run SNR check first.")
                return
            
            try:
                # Clear the third subplot
                self.axes[2].clear()
                
                colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
                threshold = self.snr_threshold.get()
                
                for i, (detector, data) in enumerate(self.snr_results['corrected'].items()):
                    if 'rolling_snr' in data['metrics']:
                        rolling_snr = data['metrics']['rolling_snr']
                        time_data = data['time']
                        
                        # Create time axis for rolling SNR (centered in windows)
                        window_size = data['metrics']['window_size']
                        rolling_time = time_data[window_size//2:window_size//2 + len(rolling_snr)]
                        
                        # Get wavelength and correction info for label
                        wavelength = self.detector_wavelengths.get(detector, tk.DoubleVar(value=0)).get()
                        correction_factor = data['transmission_correction_factor']
                        
                        # Plot rolling SNR
                        color = colors[i % len(colors)]
                        line_style = '-' if data['is_good'] else '--'
                        alpha = 1.0 if data['is_good'] else 0.6
                        
                        label = f'{detector} ({wavelength:.0f}nm, {correction_factor:.1f}x) - {data["metrics"]["percentage_good"]:.1f}% good'
                        
                        self.axes[2].plot(rolling_time, rolling_snr, 
                                        color=color, linewidth=2, linestyle=line_style, alpha=alpha,
                                        label=label)
                
                # Add threshold line
                self.axes[2].axhline(y=threshold, color='red', linestyle=':', 
                                    alpha=0.8, linewidth=2, label=f'SNR Threshold ({threshold})')
                
                self.axes[2].set_xlabel('Time (s)')
                self.axes[2].set_ylabel('Rolling SNR (Corrected Signals)')
                self.axes[2].set_title('Rolling SNR vs Time (Dark + Window Corrected)')
                self.axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                self.axes[2].grid(True, alpha=0.3)
                self.axes[2].set_ylim(0, max(10, threshold * 3))
                
                self.fig.tight_layout()
                self.canvas.draw()
                
                print("DEBUG: Corrected SNR vs Time plot completed")
                
            except Exception as e:
                print(f"DEBUG: Error plotting corrected SNR vs time: {e}")
                messagebox.showerror("Error", f"Error plotting SNR vs time: {str(e)}") 


    def calculate_matlab_calibration_factors(self):
        """Calculate calibration factors for temperature calculation (ratios only)"""
        
        all_cal_factors = {}
        
        for temp, description in self.calibration_temps:
            cal_factors = {}
            
            self.snr_text.insert(tk.END, f"\nCalculation ratios for {temp}K ({description}):\n")
            
            for detector in self.good_detectors:
                # For temperature calculation, we just need the ratio correction
                # The absolute scaling was already applied in signal correction
                wavelength = self.detector_wavelengths[detector].get()
                theoretical = self.planck_function(wavelength, temp)
                
                # Simple ratio factor (since signals are already calibration-scaled)
                cal_factors[detector] = theoretical
                
                self.snr_text.insert(tk.END, 
                    f"  {detector}: theoretical_intensity={theoretical:.2e}\n")
            
            all_cal_factors[temp] = cal_factors
            
        return all_cal_factors

    def get_matlab_corrected_signals(self):
        """Get corrected signals using MATLAB methodology with proper calibration scaling"""
        
        corrected_signals = {}
        
        self.snr_text.insert(tk.END, f"MATLAB-style signal correction:\n")
        self.snr_text.insert(tk.END, f"1. Dark subtraction: experimental - dark_mean\n")
        self.snr_text.insert(tk.END, f"2. Window transmission correction (if enabled)\n")
        self.snr_text.insert(tk.END, f"3. Calibration scaling based on reference temperature\n\n")
        
        # Get calibration factors for scaling
        calibration_factors = {}
        reference_temp = self.calibration_temps[0][0]  # Use first calibration temp as reference
        
        for detector in self.good_detectors:
            if (detector in self.processed_data['calibration'] and
                detector in self.processed_data['dark']):
                
                cal_signal = self.processed_data['calibration'][detector]['signal']
                dark_signal = self.processed_data['dark'][detector]['signal']
                
                # MATLAB-style calibration factor calculation
                dark_mean = np.mean(dark_signal)
                cal_corrected = cal_signal - dark_mean
                
                # Use stable middle region
                start_idx = len(cal_corrected) // 4
                end_idx = 3 * len(cal_corrected) // 4
                stable_cal_signal = np.mean(cal_corrected[start_idx:end_idx])
                
                # Calculate theoretical Planck intensity at reference temperature
                wavelength = self.detector_wavelengths[detector].get()
                theoretical_intensity = self.planck_function(wavelength, reference_temp)
                
                # Calibration factor (like MATLAB intensity correction)
                calibration_factors[detector] = stable_cal_signal / theoretical_intensity
                
                self.snr_text.insert(tk.END, 
                    f"Calibration factor for {detector}: {calibration_factors[detector]:.2e}\n")
        
        # Now correct experimental signals
        for detector in self.good_detectors:
            if (detector in self.processed_data['experimental'] and
                detector in self.processed_data['dark']):
                
                exp_signal = self.processed_data['experimental'][detector]['signal']
                dark_signal = self.processed_data['dark'][detector]['signal']
                
                # Step 1: MATLAB-style dark subtraction
                dark_mean = np.mean(dark_signal)
                dark_corrected = exp_signal - dark_mean
                dark_corrected = np.maximum(dark_corrected, 1e-12)
                
                # Step 2: Apply calibration scaling (like MATLAB correction factors)
                if detector in calibration_factors:
                    calibration_scaled = dark_corrected / calibration_factors[detector]
                else:
                    calibration_scaled = dark_corrected
                
                # Step 3: Window transmission correction (if enabled)
                if self.enable_window_correction.get() and detector in self.detector_transmissions:
                    reference_transmission = self.detector_transmissions[detector].get()
                    reference_thickness = self.reference_thickness.get()
                    actual_thickness = self.actual_thickness.get()
                    
                    if reference_thickness > 0:
                        thickness_ratio = actual_thickness / reference_thickness
                        actual_transmission = reference_transmission ** thickness_ratio
                    else:
                        actual_transmission = reference_transmission
                    
                    final_corrected = calibration_scaled / actual_transmission
                    window_factor = 1 / actual_transmission
                else:
                    final_corrected = calibration_scaled
                    window_factor = 1.0
                
                corrected_signals[detector] = final_corrected
                
                wavelength = self.detector_wavelengths[detector].get()
                cal_factor = calibration_factors.get(detector, 1.0)
                
                self.snr_text.insert(tk.END, 
                    f"{detector} ({wavelength:.0f}nm): "
                    f"dark_corr={np.mean(dark_corrected):.2e}, "
                    f"cal_scaled={np.mean(calibration_scaled):.2e}, "
                    f"final={np.mean(final_corrected):.2e}, "
                    f"cal_factor={cal_factor:.2e}, window_factor={window_factor:.2f}\n")
        
        return corrected_signals

    def calculate_temperature(self):
        """Calculate temperature using MATLAB methodology: numerical solver + all possible ratios"""
        
        if len(self.good_detectors) < 2:
            messagebox.showerror("Error", 
                            f"Need at least 2 detectors with good SNR!\n"
                            f"Currently have: {len(self.good_detectors)}")
            return
        
        try:
            self.snr_text.insert(tk.END, f"\nCALCULATING TEMPERATURE (MATLAB-style: numerical solver + all ratios)...\n")
            self.snr_text.insert(tk.END, "="*70 + "\n")
            
            # Get corrected signals (MATLAB-style: experimental - dark only)
            corrected_signals = self.get_matlab_corrected_signals()
            
            if len(corrected_signals) < 2:
                raise ValueError("Could not correct signals for at least 2 detectors")
            
            # Calculate calibration factors for each calibration temperature
            all_cal_factors = self.calculate_matlab_calibration_factors()
            
            # Generate ALL possible detector combinations (like MATLAB's 3 ratios)
            detector_names = list(corrected_signals.keys())
            all_temperature_arrays = []
            combination_descriptions = []
            
            self.snr_text.insert(tk.END, f"Using {len(detector_names)} good detectors: {', '.join(detector_names)}\n")
            
            # Calculate temperature for ALL detector pair combinations
            for i in range(len(detector_names)):
                for j in range(i+1, len(detector_names)):
                    det1, det2 = detector_names[i], detector_names[j]
                    lambda1 = self.detector_wavelengths[det1].get()
                    lambda2 = self.detector_wavelengths[det2].get()
                    
                    self.snr_text.insert(tk.END, f"\nRatio {det1}({lambda1:.0f}nm) / {det2}({lambda2:.0f}nm):\n")
                    
                    # Calculate for each calibration temperature
                    for temp, description in self.calibration_temps:
                        if temp in all_cal_factors:
                            cal_factors = all_cal_factors[temp]
                            
                            if det1 in cal_factors and det2 in cal_factors:
                                # Use numerical Planck solver (like MATLAB fzero)
                                temperature_array = self.solve_planck_numerical(
                                    corrected_signals[det1], 
                                    corrected_signals[det2],
                                    lambda1, lambda2, 
                                    cal_factors[det1], cal_factors[det2],
                                    calibration_temp=temp
                                )
                                
                                all_temperature_arrays.append(temperature_array)
                                combo_desc = f"{det1}/{det2} @ {temp}K {description}"
                                combination_descriptions.append(combo_desc)
                                
                                # Statistics for this combination
                                valid_temps = temperature_array[np.isfinite(temperature_array)]
                                if len(valid_temps) > 0:
                                    self.snr_text.insert(tk.END, 
                                        f"  {temp}K cal: {valid_temps.mean():.0f}K "
                                        f"(range: {valid_temps.min():.0f}-{valid_temps.max():.0f}K, "
                                        f"{len(valid_temps)}/{len(temperature_array)} valid)\n")
                                else:
                                    self.snr_text.insert(tk.END, f"  {temp}K cal: No valid temperatures\n")
            
            if not all_temperature_arrays:
                raise ValueError("No valid temperature calculations from any detector combination")
            
            # Average ALL temperature arrays (like MATLAB T_P1_AVG)
            self.snr_text.insert(tk.END, f"\nAveraging {len(all_temperature_arrays)} temperature arrays...\n")
            
            # Convert to numpy array and calculate mean (ignoring NaN)
            all_temps_array = np.array(all_temperature_arrays)
            mean_temperature = np.nanmean(all_temps_array, axis=0)
            std_temperature = np.nanstd(all_temps_array, axis=0)
            
            # Apply smoothing (like MATLAB)
            smooth_window = self.smoothing_window.get()
            mean_temperature_smooth = gaussian_filter1d(mean_temperature, sigma=smooth_window)
            std_temperature_smooth = gaussian_filter1d(std_temperature, sigma=smooth_window)
            
            # Clip unrealistic temperatures
            mean_temperature_smooth = np.clip(mean_temperature_smooth, 300, 8000)
            
            # Store results
            corrected_time = self.processed_data['experimental'][detector_names[0]]['time'][:len(mean_temperature_smooth)]
            
            self.temperature_results = {
                'time': corrected_time,
                'temperature_mean': mean_temperature_smooth,
                'temperature_std': std_temperature_smooth,
                'temperature_individual': all_temperature_arrays,
                'temperature_descriptions': combination_descriptions,
                'corrected_signals': corrected_signals,
                'detectors_used': detector_names,
                'wavelengths': [self.detector_wavelengths[det].get() for det in detector_names],
                'has_uncertainty': True,
                'method': 'MATLAB-style: Numerical Planck solver with all detector combinations',
                'num_combinations': len(all_temperature_arrays)
            }
            
            # Plot results
            self.plot_results()
            self.notebook.select(3)
            
            # Summary
            valid_final = np.sum(np.isfinite(mean_temperature_smooth))
            total_final = len(mean_temperature_smooth)
            
            self.snr_text.insert(tk.END, f"\n✓ MATLAB-style temperature calculation complete!\n")
            self.snr_text.insert(tk.END, f"Detector combinations used: {len(all_temperature_arrays)}\n")
            self.snr_text.insert(tk.END, f"Valid temperature points: {valid_final}/{total_final}\n")
            self.snr_text.insert(tk.END, f"Final average: {np.nanmean(mean_temperature_smooth):.0f} ± {np.nanmean(std_temperature_smooth):.0f}K\n")
            
            messagebox.showinfo("Success", 
                            f"MATLAB-style temperature calculated!\n"
                            f"Used {len(all_temperature_arrays)} detector combinations\n"
                            f"Final average: {np.nanmean(mean_temperature_smooth):.0f} ± {np.nanmean(std_temperature_smooth):.0f}K")
            
        except Exception as e:
            error_msg = f"Error calculating temperature: {str(e)}"
            self.snr_text.insert(tk.END, f"\n✗ {error_msg}\n")
            messagebox.showerror("Error", error_msg)
    
    def plot_results(self):
        """Plot temperature and signal results with uncertainty bands"""
        
        if not hasattr(self, 'temperature_results'):
            print("DEBUG: No temperature_results attribute")
            return
        
        try:
            results = self.temperature_results
            
            # Clear first two plots (keep SNR plot if it exists)
            self.axes[0].clear()
            self.axes[1].clear()
            
            # Plot 1: Temperature vs Time
            if results['has_uncertainty']:
                # Multiple calibrations - plot with uncertainty bands
                colors_temp = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightpink']
                for i, (temp_data, description) in enumerate(zip(results['temperature_individual'], 
                                                                results['temperature_descriptions'])):
                    color = colors_temp[i % len(colors_temp)]
                    self.axes[0].plot(results['time'], temp_data, 
                                    color=color, alpha=0.5, linewidth=1, 
                                    label=f'{description}')
                
                # Plot mean with uncertainty bands
                mean_temp = results['temperature_mean']
                std_temp = results['temperature_std']
                
                self.axes[0].plot(results['time'], mean_temp, 
                                'red', linewidth=3, label='Mean Temperature')
                
                # Add shaded error bands
                self.axes[0].fill_between(results['time'], 
                                        mean_temp - std_temp, 
                                        mean_temp + std_temp,
                                        alpha=0.3, color='red', 
                                        label='±1σ uncertainty')
                
                self.axes[0].fill_between(results['time'], 
                                        mean_temp - 2*std_temp, 
                                        mean_temp + 2*std_temp,
                                        alpha=0.15, color='red', 
                                        label='±2σ uncertainty')
                
                title = f'Temperature vs Time (Multi-Calibration)'
                y_max = min(6000, mean_temp.max() + 2*std_temp.max())
                
            else:
                # Single calibration result
                self.axes[0].plot(results['time'], results['temperature'], 
                                'red', linewidth=2, label='Temperature')
                title = 'Temperature vs Time (Single Calibration)'
                y_max = min(6000, results['temperature'].max() * 1.1)
            
            self.axes[0].set_xlabel('Time (s)')
            self.axes[0].set_ylabel('Temperature (K)')
            self.axes[0].set_title(title)
            self.axes[0].legend()
            self.axes[0].grid(True, alpha=0.3)
            self.axes[0].set_ylim(0, y_max)
            
            # Plot 2: Detector Signals and Ratios
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
            
            # Plot detector signals
            for i, (detector, signal) in enumerate(results['corrected_signals'].items()):
                wavelength = results['wavelengths'][i]
                self.axes[1].plot(results['time'], signal, 
                                color=colors[i], linewidth=2, 
                                label=f'{detector} ({wavelength:.1f}nm)')
            
            # Plot ratio(s) if multiple detectors
            if len(results['corrected_signals']) >= 2:
                detector_names = list(results['corrected_signals'].keys())
                
                # Plot primary ratio
                ratio = results['corrected_signals'][detector_names[0]] / results['corrected_signals'][detector_names[1]]
                ratio_scaled = ratio * np.mean(list(results['corrected_signals'].values())) / np.mean(ratio)
                self.axes[1].plot(results['time'], ratio_scaled, 'purple', 
                                linewidth=2, alpha=0.7, linestyle='--',
                                label=f'Ratio {detector_names[0]}/{detector_names[1]} (scaled)')
            
            self.axes[1].set_xlabel('Time (s)')
            self.axes[1].set_ylabel('Corrected Signal')
            self.axes[1].set_title('Detector Signals and Ratios (Background & Dark Corrected)')
            self.axes[1].set_yscale('log')
            self.axes[1].legend()
            self.axes[1].grid(True, alpha=0.3)
            
            # Automatically plot SNR vs time if data is available
            if hasattr(self, 'snr_results') and 'experimental' in self.snr_results:
                self.plot_snr_vs_time()
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            print("DEBUG: Plot completed successfully")
            
        except Exception as e:
            print(f"DEBUG: Error in plot_results: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_analysis(self):
        """Clear analysis results"""
        self.snr_text.delete(1.0, tk.END)
        self.snr_results = {}
        self.good_detectors = []
        if hasattr(self, 'temperature_results'):
            delattr(self, 'temperature_results')
        
        # Clear plots
        for ax in self.axes:
            ax.clear()
        self.canvas.draw()
        
        messagebox.showinfo("Cleared", "Analysis results cleared.")
    
    def refresh_plot(self):
        """Refresh the plot display"""
        if hasattr(self, 'temperature_results'):
            self.plot_results()
        else:
            messagebox.showwarning("No Data", "No temperature results to plot. Calculate temperature first.")
    

    def solve_planck_numerical(self, signal1, signal2, lambda1, lambda2, cal_factor1, cal_factor2, calibration_temp):
        """Solve Planck equation numerically like MATLAB fzero function"""
        
        # Calculate signal ratios
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = signal1 / signal2
        
        # Apply calibration correction (like MATLAB correction factors)
        cal_ratio = cal_factor1 / cal_factor2
        corrected_ratio = ratio / cal_ratio
        
        # Apply smoothing to ratio (like MATLAB)
        corrected_ratio = gaussian_filter1d(corrected_ratio, sigma=5)
        
        # Physical constants
        C1 = 2 * self.h * self.c**2
        C2 = self.h * self.c / self.k
        
        # Convert wavelengths to meters
        lam1_m = lambda1 * 1e-9
        lam2_m = lambda2 * 1e-9
        
        def planck_ratio_function(T, target_ratio):
            """Planck ratio equation to solve (exactly like MATLAB func)"""
            try:
                if T <= 0:
                    return 1e10
                
                # Calculate Planck functions (like MATLAB)
                exp1 = np.exp(C2 / (lam1_m * T))
                exp2 = np.exp(C2 / (lam2_m * T))
                
                if exp1 == np.inf or exp2 == np.inf:
                    return 1e10
                
                planck1 = (C1 / lam1_m**5) / (exp1 - 1)
                planck2 = (C1 / lam2_m**5) / (exp2 - 1)
                
                if planck2 == 0:
                    return 1e10
                
                theoretical_ratio = planck1 / planck2
                return target_ratio - theoretical_ratio
                
            except (OverflowError, ZeroDivisionError, RuntimeWarning):
                return 1e10
        
        # Solve for each data point (like MATLAB loop)
        temperature = np.full_like(corrected_ratio, np.nan)
        
        for i, ratio_val in enumerate(corrected_ratio):
            if not np.isfinite(ratio_val) or ratio_val <= 0:
                continue
            
            # Try multiple initial guesses (like MATLAB T0 = [3000 2000 4000])
            solved = False
            for T0 in [3000, 2000, 4000, 1500, 5000]:
                try:
                    # Use fsolve (like MATLAB fzero)
                    T_solution = fsolve(
                        lambda T: planck_ratio_function(T, ratio_val), 
                        T0, 
                        xtol=1.0  # 1K tolerance like MATLAB
                    )[0]
                    
                    # Check bounds (like MATLAB: and(Temp1 >= 100,Temp1 <= 6000))
                    if 100 <= T_solution <= 6000:
                        temperature[i] = T_solution
                        solved = True
                        break
                            
                except:
                    continue
            
            # If no solution, set to NaN (like MATLAB)
            if not solved:
                temperature[i] = np.nan
        
        return temperature


    def save_results(self):
        """Save results plot to file"""
        
        if not hasattr(self, 'temperature_results'):
            messagebox.showerror("Error", "No results to save! Calculate temperature first.")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"), 
                    ("PDF files", "*.pdf"), 
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ],
                title="Save Results Plot"
            )
            
            if filename:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved to:\n{filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error saving plot: {str(e)}")
    
    def export_individual_plots_data(self):
        """Export data for each plot separately as CSV files"""
        
        if not hasattr(self, 'temperature_results'):
            messagebox.showerror("Error", "No data to export! Calculate temperature first.")
            return
        
        try:
            # Ask user to select directory for saving multiple files
            save_directory = filedialog.askdirectory(
                title="Select Directory to Save Plot Data Files"
            )
            
            if not save_directory:
                return
            
            results = self.temperature_results
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Export Temperature Plot Data
            temp_data = {'Time_s': results['time']}
            
            if results['has_uncertainty']:
                temp_data['Temperature_Mean_K'] = results['temperature_mean']
                temp_data['Temperature_Std_K'] = results['temperature_std']
                temp_data['Temperature_Upper_1sigma_K'] = results['temperature_mean'] + results['temperature_std']
                temp_data['Temperature_Lower_1sigma_K'] = results['temperature_mean'] - results['temperature_std']
                temp_data['Temperature_Upper_2sigma_K'] = results['temperature_mean'] + 2*results['temperature_std']
                temp_data['Temperature_Lower_2sigma_K'] = results['temperature_mean'] - 2*results['temperature_std']
                
                # Add individual calibration results
                for i, description in enumerate(results['temperature_descriptions']):
                    clean_desc = description.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')
                    temp_data[f'Temperature_{clean_desc}_K'] = results['temperature_individual'][i]
            else:
                temp_data['Temperature_K'] = results['temperature']
            
            temp_df = pd.DataFrame(temp_data)
            temp_filename = os.path.join(save_directory, f"temperature_plot_data_{timestamp}.csv")
            temp_df.to_csv(temp_filename, index=False)
            
            # 2. Export Detector Signals Plot Data
            signal_data = {'Time_s': results['time']}
            
            for i, (detector, signal) in enumerate(results['corrected_signals'].items()):
                wavelength = results['wavelengths'][i]
                signal_data[f'{detector}_Signal_{wavelength:.1f}nm'] = signal
            
            # Add ratios
            if len(results['corrected_signals']) >= 2:
                detector_names = list(results['corrected_signals'].keys())
                ratio = results['corrected_signals'][detector_names[0]] / results['corrected_signals'][detector_names[1]]
                signal_data[f'Ratio_{detector_names[0]}_{detector_names[1]}'] = ratio
                
                # Scaled ratio for plotting
                ratio_scaled = ratio * np.mean(list(results['corrected_signals'].values())) / np.mean(ratio)
                signal_data[f'Ratio_{detector_names[0]}_{detector_names[1]}_Scaled_for_Plot'] = ratio_scaled
            
            signal_df = pd.DataFrame(signal_data)
            signal_filename = os.path.join(save_directory, f"detector_signals_plot_data_{timestamp}.csv")
            signal_df.to_csv(signal_filename, index=False)
            
            # 3. Export SNR Plot Data
            if hasattr(self, 'snr_results') and 'experimental' in self.snr_results:
                snr_data = {}
                
                for detector, snr_info in self.snr_results['experimental'].items():
                    if 'rolling_snr' in snr_info['metrics']:
                        rolling_snr = snr_info['metrics']['rolling_snr']
                        window_size = snr_info['metrics']['window_size']
                        
                        # Create time axis for rolling SNR (centered in windows)
                        time_data = snr_info['time']
                        rolling_time = time_data[window_size//2:window_size//2 + len(rolling_snr)]
                        
                        # Get wavelength for column name
                        wavelength = self.detector_wavelengths.get(detector, tk.DoubleVar(value=0)).get()
                        
                        # Store data (pad to same length)
                        if 'Time_s' not in snr_data:
                            snr_data['Time_s'] = rolling_time
                            snr_data['SNR_Threshold'] = [self.snr_threshold.get()] * len(rolling_time)
                        
                        snr_data[f'{detector}_Rolling_SNR_{wavelength:.0f}nm'] = rolling_snr
                        snr_data[f'{detector}_Percentage_Good_Time'] = [snr_info['metrics']['percentage_good']] * len(rolling_snr)
                
                if snr_data:
                    snr_df = pd.DataFrame(snr_data)
                    snr_filename = os.path.join(save_directory, f"snr_plot_data_{timestamp}.csv")
                    snr_df.to_csv(snr_filename, index=False)
                
                files_created = [temp_filename, signal_filename, snr_filename]
            else:
                files_created = [temp_filename, signal_filename]
            
            # Create summary file
            summary_filename = os.path.join(save_directory, f"plot_data_summary_{timestamp}.txt")
            with open(summary_filename, 'w') as f:
                f.write("PYROMETER PLOT DATA EXPORT SUMMARY\n")
                f.write("="*50 + "\n\n")
                f.write(f"Export Date: {pd.Timestamp.now()}\n")
                f.write(f"Analysis Parameters:\n")
                f.write(f"  - Detectors Used: {', '.join(results['detectors_used'])}\n")
                f.write(f"  - Wavelengths: {results['wavelengths']} nm\n")
                f.write(f"  - SNR Threshold: {self.snr_threshold.get()}\n")
                f.write(f"  - SNR Percentage Threshold: {self.snr_percentage.get()}%\n")
                f.write(f"  - Smoothing Window: {self.smoothing_window.get()} data points\n")
                f.write(f"  - Calibration Temperatures: {[temp for temp, _ in self.calibration_temps]} K\n\n")
                
                f.write("Files Created:\n")
                for i, filename in enumerate(files_created, 1):
                    f.write(f"  {i}. {os.path.basename(filename)}\n")
                
                f.write(f"\nSmoothing Window Explanation:\n")
                f.write(f"  - Value: {self.smoothing_window.get()} data points\n")
                f.write(f"  - Type: Gaussian filter sigma parameter\n")
                f.write(f"  - Effect: Smooths temperature data over ~{self.smoothing_window.get()} neighboring points\n")
                f.write(f"  - Higher values = more smoothing, lower values = less smoothing\n")
            
            files_created.append(summary_filename)
            
            messagebox.showinfo("Export Complete", 
                            f"Plot data exported successfully!\n\n"
                            f"Files created:\n" + 
                            "\n".join([f"• {os.path.basename(f)}" for f in files_created]) +
                            f"\n\nLocation: {save_directory}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting plot data: {str(e)}")

    def export_data(self):
        """Export temperature and signal data to CSV"""
        
        if not hasattr(self, 'temperature_results'):
            messagebox.showerror("Error", "No data to export! Calculate temperature first.")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Export Data to CSV"
            )
            
            if filename:
                results = self.temperature_results
                
                # Create DataFrame with all data
                data = {
                    'Time_s': results['time']
                }
                
                # Add temperature data
                if results['has_uncertainty']:
                    data['Temperature_Mean_K'] = results['temperature_mean']
                    data['Temperature_Std_K'] = results['temperature_std']
                    
                    # Add individual calibration results
                    for i, description in enumerate(results['temperature_descriptions']):
                        clean_desc = description.replace(' ', '_').replace('(', '').replace(')', '')
                        data[f'Temperature_{clean_desc}_K'] = results['temperature_individual'][i]
                else:
                    data['Temperature_K'] = results['temperature']
                
                # Add detector signals
                for i, (detector, signal) in enumerate(results['corrected_signals'].items()):
                    wavelength = results['wavelengths'][i]
                    data[f'{detector}_{wavelength:.1f}nm_Signal'] = signal
                
                # Add ratios
                if len(results['corrected_signals']) >= 2:
                    detector_names = list(results['corrected_signals'].keys())
                    ratio = results['corrected_signals'][detector_names[0]] / results['corrected_signals'][detector_names[1]]
                    data[f'Ratio_{detector_names[0]}_{detector_names[1]}'] = ratio
                
                # ADD SNR DATA HERE (MOVED FROM METADATA SECTION)
                if hasattr(self, 'snr_results') and 'experimental' in self.snr_results:
                    for detector, snr_data in self.snr_results['experimental'].items():
                        if 'rolling_snr' in snr_data['metrics']:
                            rolling_snr = snr_data['metrics']['rolling_snr']
                            window_size = snr_data['metrics']['window_size']
                            
                            # Pad rolling SNR to match time length
                            padded_snr = np.full(len(results['time']), np.nan)
                            start_idx = window_size // 2
                            end_idx = start_idx + len(rolling_snr)
                            if end_idx <= len(padded_snr):
                                padded_snr[start_idx:end_idx] = rolling_snr
                            
                            data[f'{detector}_Rolling_SNR'] = padded_snr
                            data[f'{detector}_SNR_Percentage_Good'] = [snr_data['metrics']['percentage_good']] * len(results['time'])
                
                # Create and save DataFrame
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
                
                # Also save metadata
                metadata_filename = filename.replace('.csv', '_metadata.txt')
                with open(metadata_filename, 'w') as f:
                    f.write("PYROMETRY ANALYSIS METADATA\n")
                    f.write("="*40 + "\n\n")
                    f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
                    f.write(f"Detectors Used: {', '.join(results['detectors_used'])}\n")
                    f.write(f"Wavelengths: {results['wavelengths']} nm\n")
                    f.write(f"Calibration Temperatures: {[temp for temp, _ in self.calibration_temps]} K\n")
                    f.write(f"SNR Threshold: {self.snr_threshold.get()}\n")
                    f.write(f"SNR Percentage Threshold: {self.snr_percentage.get()}%\n")  # ADD this
                    f.write(f"Smoothing Window: {self.smoothing_window.get()}\n")  # ADD this
                    f.write(f"Number of Data Points: {len(results['time'])}\n")
                    f.write(f"Time Range: {results['time'].min():.6f} to {results['time'].max():.6f} s\n")
                    
                    if results['has_uncertainty']:
                        f.write(f"Temperature Range (Mean): {results['temperature_mean'].min():.1f} to {results['temperature_mean'].max():.1f} K\n")
                        f.write(f"Average Temperature: {results['temperature_mean'].mean():.1f} ± {results['temperature_std'].mean():.1f} K\n")
                        f.write(f"Number of Calibration Points: {len(self.calibration_temps)}\n")
                    else:
                        f.write(f"Temperature Range: {results['temperature'].min():.1f} to {results['temperature'].max():.1f} K\n")
                        f.write(f"Average Temperature: {results['temperature'].mean():.1f} K\n")
                    
                    # ADD SNR SUMMARY TO METADATA
                    if hasattr(self, 'snr_results') and 'experimental' in self.snr_results:
                        f.write(f"\nSNR ANALYSIS SUMMARY:\n")
                        f.write(f"Good Detectors: {', '.join(self.good_detectors)}\n")
                        for detector, snr_data in self.snr_results['experimental'].items():
                            if 'metrics' in snr_data:
                                f.write(f"{detector}: {snr_data['metrics']['percentage_good']:.1f}% good time, RMS SNR = {snr_data['metrics']['rms_snr']:.1f}\n")

                messagebox.showinfo("Success", 
                                f"Data exported to:\n{filename}\n\n"
                                f"Metadata saved to:\n{metadata_filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting data: {str(e)}")


class CalibrationTempDialog:
    def __init__(self, parent):
        self.result = None
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Add Calibration Temperature")
        self.dialog.geometry("400x200")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (200 // 2)
        self.dialog.geometry(f"400x200+{x}+{y}")
        
        # Create widgets
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Temperature entry
        ttk.Label(main_frame, text="Temperature (K):").grid(row=0, column=0, sticky='w', pady=5)
        self.temp_var = tk.DoubleVar(value=2796.0)
        ttk.Entry(main_frame, textvariable=self.temp_var, width=15).grid(row=0, column=1, sticky='ew', pady=5)
        
        # Description entry
        ttk.Label(main_frame, text="Description:").grid(row=1, column=0, sticky='w', pady=5)
        self.desc_var = tk.StringVar(value="Calibration source")
        ttk.Entry(main_frame, textvariable=self.desc_var, width=30).grid(row=1, column=1, sticky='ew', pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side='left', padx=5)
        
        # Configure grid weights
        main_frame.grid_columnconfigure(1, weight=1)
        
        # Focus on temperature entry
        self.dialog.focus_set()
        
        # Wait for dialog to close
        self.dialog.wait_window()
    
    def ok_clicked(self):
        try:
            temp = self.temp_var.get()
            desc = self.desc_var.get().strip()
            
            if temp <= 0:
                messagebox.showerror("Error", "Temperature must be positive!")
                return
            
            if not desc:
                messagebox.showerror("Error", "Description cannot be empty!")
                return
            
            self.result = (temp, desc)
            self.dialog.destroy()
            
        except tk.TclError:
            messagebox.showerror("Error", "Invalid temperature value!")
    
    def cancel_clicked(self):
        self.dialog.destroy()


def main():
    """Main function to run the application"""
    try:
        root = tk.Tk()
        app = PyrometerApp(root)
        
        # Set window icon if available
        try:
            root.iconbitmap('pyrometer_icon.ico')  # Optional: add an icon file
        except:
            pass  # Icon file not found, continue without it
        
        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Start the application
        root.mainloop()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Startup Error", f"Error starting application: {str(e)}")

if __name__ == "__main__":
    main()