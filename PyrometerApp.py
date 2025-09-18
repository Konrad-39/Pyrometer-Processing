import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import os
import glob
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
import json
from itertools import combinations

class AdvancedPyrometerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Multi-Channel Ratio Pyrometer")
        self.root.geometry("1400x900")
        
        self.smoothing_method = tk.StringVar(value="gaussian")

        # Detector sensitivity data
        self.sensitivity_data = {}
        self.sensitivity_loaded = False

        # Physical constants
        self.h = 6.62607015e-34  # Planck's constant (J⋅s)
        self.c = 299792458       # Speed of light (m/s)
        self.k = 1.380649e-23    # Boltzmann constant (J/K)
        self.c1 = 2 * np.pi * self.h * self.c**2  # First radiation constant
        self.c2 = self.h * self.c / self.k        # Second radiation constant
        
        # Settings
        self.num_detectors = tk.IntVar(value=3)
        self.calibration_temp = tk.DoubleVar(value=2796.0)
        self.smoothing_window = tk.IntVar(value=5)
        self.snr_threshold = tk.DoubleVar(value=10.0)
        self.snr_percentage = tk.DoubleVar(value=80.0)
        self.emissivity = tk.DoubleVar(value=0.35)  # Tungsten emissivity
        
        # Window correction settings
        self.window_correction_enabled = tk.BooleanVar(value=False)
        self.window_material = tk.StringVar(value="polycarbonate")
        self.window_correction_experimental = tk.BooleanVar(value=True)
        self.window_correction_background = tk.BooleanVar(value=False)
        self.window_transmission_scale = tk.DoubleVar(value=1.0)

        # Data storage
        self.detector_wavelengths = {}
        self.folder_paths = {}
        self.processed_data = {}
        self.temperature_results = None
        self.detector_pairs = []
        self.window_transmission = {}
        self.good_detectors = []
        # Polycarbonate transmission approximation
        self.polycarbonate_transmission = {
            300: 0.85, 400: 0.88, 500: 0.89, 600: 0.89, 700: 0.89,
            800: 0.88, 900: 0.87, 1000: 0.86, 1100: 0.85, 1200: 0.84,
            1300: 0.83, 1400: 0.82, 1500: 0.80, 1600: 0.78, 1700: 0.75,
            1800: 0.72, 1900: 0.68, 2000: 0.65
        }
        
        self.setup_gui()
        self.update_detector_settings()

    # ...all other methods from setup_gui to the end of the class should be indented here...
    
    def setup_gui(self):
        """Setup comprehensive GUI"""
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Setup tab
        setup_frame = ttk.Frame(notebook)
        notebook.add(setup_frame, text="Setup")
        
        # Files tab
        files_frame = ttk.Frame(notebook)
        notebook.add(files_frame, text="Files")
        
        # Sensitivity tab (NEW)
        sensitivity_frame = ttk.Frame(notebook)
        notebook.add(sensitivity_frame, text="Sensitivity")
        
        # Analysis tab
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="Analysis")
        
        # Results tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results")
        
        self.setup_setup_tab(setup_frame)
        self.setup_files_tab(files_frame)
        self.setup_sensitivity_tab(sensitivity_frame)  # NEW
        self.setup_analysis_tab(analysis_frame)
        self.setup_results_tab(results_frame)
        
        self.notebook = notebook

    def setup_sensitivity_tab(self, frame):
        """Setup sensitivity files tab"""
        # Main container
        main_frame = ttk.Frame(frame)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text="Detector Sensitivity Configuration", 
                            font=('Arial', 12, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Instructions
        instructions = ttk.Label(main_frame, 
                            text="Load sensitivity/responsivity files for each detector channel.\n"
                                    "Files should contain wavelength (nm) and sensitivity data.",
                            font=('Arial', 10), justify='center')
        instructions.pack(pady=(0, 20))
        
        # Sensitivity file selection frame
        sens_frame = ttk.LabelFrame(main_frame, text="Sensitivity Files Location")
        sens_frame.pack(fill='x', pady=(0, 20))
        
        # Path selection
        path_frame = ttk.Frame(sens_frame)
        path_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(path_frame, text="Folder containing sensitivity files:").pack(anchor='w', pady=(0, 5))
        
        self.sensitivity_path = tk.StringVar()
        
        entry_frame = ttk.Frame(path_frame)
        entry_frame.pack(fill='x')
        
        ttk.Entry(entry_frame, textvariable=self.sensitivity_path, width=60).pack(side='left', fill='x', expand=True)
        ttk.Button(entry_frame, text="Browse", 
                command=self.browse_sensitivity_folder).pack(side='left', padx=(5, 0))
        
        # Load button
        ttk.Button(sens_frame, text="Load Sensitivity Files", 
                command=self.load_sensitivity_files,
                style='Accent.TButton').pack(pady=10)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status")
        status_frame.pack(fill='both', expand=True)
        
        # Info label
        self.sensitivity_info = ttk.Label(status_frame, text="No sensitivity data loaded", 
                                        font=('Arial', 10))
        self.sensitivity_info.pack(pady=10)
        
        # Details text
        self.sensitivity_details = tk.Text(status_frame, height=10, width=70, font=('Courier', 9))
        scroll = ttk.Scrollbar(status_frame, orient='vertical', command=self.sensitivity_details.yview)
        self.sensitivity_details.configure(yscrollcommand=scroll.set)
        
        text_frame = ttk.Frame(status_frame)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.sensitivity_details.pack(side='left', fill='both', expand=True, in_=text_frame)
        scroll.pack(side='right', fill='y', in_=text_frame)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(button_frame, text="View Sensitivity Curves", 
                command=self.plot_sensitivity_curves).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Export Sensitivity Data", 
                command=self.export_sensitivity_data).pack(side='left', padx=5)

    
    
    def setup_setup_tab(self, frame):
        """Enhanced setup configuration tab"""
        # Main settings frame
        main_frame = ttk.Frame(frame)
        main_frame.pack(fill='both', expand=True)
        
        # Left column - Basic settings
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        # Basic settings
        settings_frame = ttk.LabelFrame(left_frame, text="Basic Settings")
        settings_frame.pack(fill='x', pady=(0, 10))
        
        # Number of detectors
        ttk.Label(settings_frame, text="Number of Channels:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        detector_frame = ttk.Frame(settings_frame)
        detector_frame.grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Radiobutton(detector_frame, text="3", variable=self.num_detectors, 
                    value=3, command=self.update_detector_settings).pack(side='left', padx=5)
        ttk.Radiobutton(detector_frame, text="32", variable=self.num_detectors, 
                    value=32, command=self.update_detector_settings).pack(side='left', padx=5)
        
        # Calibration temperature
        ttk.Label(settings_frame, text="Calibration Temperature (K):").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(settings_frame, textvariable=self.calibration_temp, width=10).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        # Emissivity
        ttk.Label(settings_frame, text="Emissivity:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(settings_frame, textvariable=self.emissivity, width=10).grid(row=2, column=1, sticky='w', padx=5, pady=5)
        
        # ADD DETECTOR RESISTANCE HERE (if using A/W sensitivity)
        ttk.Label(settings_frame, text="Detector Load (Ω):").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.detector_resistance = tk.DoubleVar(value=50.0)  # Add to __init__ if not there
        ttk.Entry(settings_frame, textvariable=self.detector_resistance, width=10).grid(row=3, column=1, sticky='w', padx=5, pady=5)
        
        # SNR settings
        snr_frame = ttk.LabelFrame(left_frame, text="SNR Settings")
        snr_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(snr_frame, text="SNR Threshold:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(snr_frame, textvariable=self.snr_threshold, width=10).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Label(snr_frame, text="Required % Above Threshold:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(snr_frame, textvariable=self.snr_percentage, width=10).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        # Smoothing settings
        smooth_frame = ttk.LabelFrame(left_frame, text="Smoothing Settings")
        smooth_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(smooth_frame, text="Window Size:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        ttk.Entry(smooth_frame, textvariable=self.smoothing_window, width=10).grid(row=0, column=1, sticky='w', padx=5, pady=5)

        ttk.Label(smooth_frame, text="Method:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        method_combo = ttk.Combobox(smooth_frame, textvariable=self.smoothing_method, 
                                values=["gaussian", "savgol", "moving_average", "median"], 
                                width=15, state='readonly')
        method_combo.grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        # Window correction
        window_frame = ttk.LabelFrame(left_frame, text="Window Correction")
        window_frame.pack(fill='x')
        
        # Enable checkbox
        ttk.Checkbutton(window_frame, text="Enable Window Correction", 
                    variable=self.window_correction_enabled,
                    command=self.update_window_correction_ui).grid(row=0, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        # Window material
        ttk.Label(window_frame, text="Window Material:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        material_combo = ttk.Combobox(window_frame, textvariable=self.window_material, 
                                    values=["polycarbonate", "quartz", "sapphire"], width=15)
        material_combo.grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        # Apply to which measurements
        apply_frame = ttk.LabelFrame(window_frame, text="Apply Correction To:")
        apply_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        self.exp_checkbox = ttk.Checkbutton(apply_frame, text="Experimental Data", 
                                        variable=self.window_correction_experimental)
        self.exp_checkbox.grid(row=0, column=0, sticky='w', padx=5, pady=2)
        
        self.bg_checkbox = ttk.Checkbutton(apply_frame, text="Background Data", 
                                        variable=self.window_correction_background)
        self.bg_checkbox.grid(row=1, column=0, sticky='w', padx=5, pady=2)
        
        # Transmission scale
        ttk.Label(apply_frame, text="Transmission Scale:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        scale_frame = ttk.Frame(apply_frame)
        scale_frame.grid(row=2, column=1, sticky='w', padx=5, pady=5)
        ttk.Entry(scale_frame, textvariable=self.window_transmission_scale, width=8).pack(side='left')
        ttk.Label(scale_frame, text="(1.0 = use material data)", font=('Arial', 8, 'italic')).pack(side='left', padx=5)
        
        # Note
        note_label = ttk.Label(window_frame, 
                            text="Note: Calibration and dark measurements should always be taken without window",
                            font=('Arial', 9, 'italic'), foreground='gray')
        note_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

        # Right column - Wavelengths
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        self.wavelength_frame = ttk.LabelFrame(right_frame, text="Channel Wavelengths (nm)")
        self.wavelength_frame.pack(fill='both', expand=True)
        
    def setup_files_tab(self, frame):
        """Enhanced file selection tab"""
        # File categories
        self.file_categories = ['experimental', 'calibration', 'dark', 'background']
        self.folder_paths = {cat: tk.StringVar() for cat in self.file_categories}
        
        # Instructions
        instructions = ttk.Label(frame, text="Select folders containing detector files (C1_*.txt, C2_*.txt, etc.)",
                            font=('Arial', 10, 'italic'))
        instructions.grid(row=0, column=0, columnspan=2, pady=10)
        
        # File selection
        for i, category in enumerate(self.file_categories):
            row = i + 1
            ttk.Label(frame, text=f"{category.title()} Folder:", 
                    font=('Arial', 10, 'bold')).grid(row=row*2-1, column=0, sticky='w', padx=10, pady=(10,5))
            
            path_frame = ttk.Frame(frame)
            path_frame.grid(row=row*2, column=0, columnspan=2, sticky='ew', padx=10, pady=(0,10))
            frame.grid_columnconfigure(0, weight=1)
            path_frame.grid_columnconfigure(0, weight=1)
            
            entry = ttk.Entry(path_frame, textvariable=self.folder_paths[category], width=70)
            entry.grid(row=0, column=0, sticky='ew', padx=(0,10))
            
            ttk.Button(path_frame, text="Browse", 
                    command=lambda cat=category: self.browse_folder(cat)).grid(row=0, column=1)
            
            # Optional indicator
            if category == 'background':
                ttk.Label(frame, text="(Optional - for additional background correction)",
                        font=('Arial', 9, 'italic')).grid(row=row*2, column=2, sticky='w', padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=10, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Load All Files", 
                command=self.load_files, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(button_frame, text="Clear All", 
                command=self.clear_all_data).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Check File Format", 
                command=self.check_file_format).pack(side='left', padx=5)
        
        # Status display
        self.status_text = tk.Text(frame, height=10, width=80, font=('Courier', 9))
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        text_frame = ttk.Frame(frame)
        text_frame.grid(row=11, column=0, columnspan=2, sticky='ew', padx=10, pady=10)
        frame.grid_rowconfigure(11, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        
        self.status_text.grid(row=0, column=0, sticky='nsew', in_=text_frame)
        scrollbar.grid(row=0, column=1, sticky='ns', in_=text_frame)


    def browse_sensitivity_folder(self):
        """Browse for sensitivity files folder"""
        folder = filedialog.askdirectory(title="Select folder containing sensitivity files")
        if folder:
            self.sensitivity_path.set(folder)
    
    def load_sensitivity_files(self):
        """Load detector sensitivity files"""
        folder = self.sensitivity_path.get()
        if not folder:
            messagebox.showwarning("No Folder", "Please select a sensitivity files folder")
            return
        
        try:
            self.sensitivity_data = {}
            num_detectors = self.num_detectors.get()
            files_loaded = 0
            
            # Progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Loading Sensitivity Files")
            progress_window.geometry("400x200")
            
            ttk.Label(progress_window, text="Loading sensitivity data...", 
                     font=('Arial', 10)).pack(pady=20)
            
            details = tk.Text(progress_window, height=6, width=50, font=('Courier', 9))
            details.pack(padx=20, pady=10)
            
            for i in range(num_detectors):
                detector_name = f"detector{i+1}"
                
                # Try different file patterns - INCLUDING EXCEL
                patterns = [
                    f"C{i+1}_sensitivity*.xlsx",  # ADD EXCEL PATTERNS
                    f"C{i+1}_sens*.xlsx",
                    f"C{i+1}_sensitivity*.txt",
                    f"C{i+1}_sens*.txt",
                    f"Channel{i+1}_sensitivity*.xlsx",
                    f"Channel{i+1}_sensitivity*.txt",
                    f"detector{i+1}_sensitivity*.xlsx",
                    f"detector{i+1}_sensitivity*.txt"
                ]
            
                
                file_found = False
                for pattern in patterns:
                    files = glob.glob(os.path.join(folder, pattern))
                    if files:
                        # Load the first matching file
                        wavelengths, sensitivity = self.load_sensitivity_file(files[0])
                        if wavelengths is not None:
                            self.sensitivity_data[detector_name] = {
                                'wavelengths': wavelengths,
                                'sensitivity': sensitivity,
                                'file': files[0],
                                'interpolator': interp1d(wavelengths, sensitivity, 
                                                       kind='cubic', bounds_error=False, 
                                                       fill_value=0)
                            }
                            files_loaded += 1
                            details.insert(tk.END, f"✓ {detector_name}: {os.path.basename(files[0])}\n")
                            file_found = True
                            break
                
                if not file_found:
                    details.insert(tk.END, f"✗ {detector_name}: No sensitivity file found\n")
                
                details.see(tk.END)
                progress_window.update()
            
            # Update info label
            if files_loaded > 0:
                self.sensitivity_loaded = True
                self.sensitivity_info.config(text=f"✓ {files_loaded}/{num_detectors} sensitivity files loaded")
                details.insert(tk.END, f"\n✓ Successfully loaded {files_loaded} files")
            else:
                self.sensitivity_loaded = False
                self.sensitivity_info.config(text="❌ No sensitivity files loaded")
            
            # Add close button
            ttk.Button(progress_window, text="Close", 
                      command=progress_window.destroy).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading sensitivity files: {str(e)}")
    
    def load_sensitivity_file(self, filepath):
        """Load a single sensitivity file (txt or xlsx)"""
        try:
            if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                # Read Excel file
                try:
                    # Try reading with headers first
                    df = pd.read_excel(filepath)
                    
                    # Check if first row contains non-numeric headers
                    if df.iloc[0].dtype == 'object' or not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
                        # Has headers, use the dataframe as is
                        wavelengths = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
                        sensitivity = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
                    else:
                        # No headers, read again without header
                        df = pd.read_excel(filepath, header=None)
                        wavelengths = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
                        sensitivity = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
                        
                except Exception as e:
                    print(f"Error reading Excel file: {e}")
                    # Try reading without header as fallback
                    df = pd.read_excel(filepath, header=None)
                    wavelengths = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
                    sensitivity = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
            
            else:
                # Read text file
                data = pd.read_csv(filepath, sep=None, engine='python', header=None, 
                                skiprows=0, comment='#')
                
                # Assume first column is wavelength, second is sensitivity
                if len(data.columns) >= 2:
                    wavelengths = pd.to_numeric(data.iloc[:, 0], errors='coerce').values
                    sensitivity = pd.to_numeric(data.iloc[:, 1], errors='coerce').values
                else:
                    print(f"File {filepath} doesn't have at least 2 columns")
                    return None, None
            
            # Remove NaN values
            mask = ~(np.isnan(wavelengths) | np.isnan(sensitivity))
            wavelengths = wavelengths[mask]
            sensitivity = sensitivity[mask]
            
            # Check if we have valid data
            if len(wavelengths) == 0 or len(sensitivity) == 0:
                print(f"No valid data found in {filepath}")
                return None, None
            
            # Convert units if needed (e.g., if wavelengths are in μm)
            if np.max(wavelengths) < 100:  # Likely in μm
                wavelengths = wavelengths * 1000  # Convert to nm
                print(f"Converted wavelengths from μm to nm for {os.path.basename(filepath)}")
            
            # Convert A/W to V/W using resistance
            if hasattr(self, 'detector_resistance'):
                resistance = self.detector_resistance.get()
                sensitivity = sensitivity * resistance  # Convert A/W to V/W
                print(f"Converted A/W to V/W using {resistance}Ω")
            
            # Print some debug info
            print(f"Loaded {os.path.basename(filepath)}:")
            print(f"  Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
            print(f"  Sensitivity range: {sensitivity.min():.3e} - {sensitivity.max():.3e}")
            print(f"  Number of points: {len(wavelengths)}")
            
            return wavelengths, sensitivity
            
        except Exception as e:
            print(f"Error reading sensitivity file {filepath}: {e}")
            import traceback
            traceback.print_exc()
            
        return None, None
    
    def get_detector_sensitivity(self, detector, wavelength):
        """Get interpolated sensitivity at specific wavelength"""
        if not self.sensitivity_loaded or detector not in self.sensitivity_data:
            return 1.0  # Default to 1 if no sensitivity data
        
        # Use interpolator
        sensitivity = self.sensitivity_data[detector]['interpolator'](wavelength)
        
        # Ensure positive value
        return max(sensitivity, 1e-10)

    def manual_analyze_snr(self):
        """Manually run SNR analysis from button click"""
        if not self.processed_data or 'experimental' not in self.processed_data:
            messagebox.showwarning("No Data", "Please load experimental data files first!")
            return
        
        # Clear the analysis text
        self.analysis_text.delete(1.0, tk.END)
        
        # Show that analysis is running
        self.analysis_text.insert(tk.END, "Running SNR analysis...\n\n")
        self.analysis_text.update()
        
        # Run the analysis
        self.analyze_snr()
        
        # Show completion message
        self.analysis_text.insert(tk.END, "\n" + "="*60 + "\n")
        self.analysis_text.insert(tk.END, "Analysis complete!\n")
        self.analysis_text.see(tk.END)
        
        # Show summary in message box
        if hasattr(self, 'good_detectors'):
            num_good = len(self.good_detectors)
            num_total = len(self.processed_data.get('experimental', {}))
            messagebox.showinfo("SNR Analysis Complete", 
                            f"{num_good} out of {num_total} channels pass SNR criteria\n"
                            f"Threshold: {self.snr_threshold.get():.1f}\n"
                            f"Required percentage: {self.snr_percentage.get():.1f}%")



    def setup_analysis_tab(self, frame):
        """Setup analysis tab"""
        # Control panel
        control_frame = ttk.LabelFrame(frame, text="Analysis Controls")
        control_frame.pack(fill='x', padx=10, pady=10)
        
        # First row of buttons
        button_row1 = ttk.Frame(control_frame)
        button_row1.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(button_row1, text="Calculate Temperature", 
                command=self.calculate_temperature, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(button_row1, text="Analyze SNR", 
                command=self.manual_analyze_snr).pack(side='left', padx=5)
        ttk.Button(button_row1, text="Show Detector Pairs", 
                command=self.show_detector_pairs).pack(side='left', padx=5)
        
        # Second row of buttons
        button_row2 = ttk.Frame(control_frame)
        button_row2.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(button_row2, text="View Sensitivity Curves", 
                command=self.plot_sensitivity_curves).pack(side='left', padx=5)
        
        # THIS BUTTON SHOULD BE HERE, INSIDE THE METHOD
        ttk.Button(button_row2, text="Min Temperature Analysis", 
                command=self.calculate_minimum_temperature_from_specs,
                style='Accent.TButton').pack(side='left', padx=5)
        
        ttk.Button(button_row2, text="Quick Min Temp Check", 
                command=self.quick_min_temp_check).pack(side='left', padx=5)

        # Results display
        self.analysis_text = tk.Text(frame, height=25, width=100, font=('Courier', 9))
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=self.analysis_text.yview)
        self.analysis_text.configure(yscrollcommand=scrollbar.set)
        
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.analysis_text.pack(side='left', fill='both', expand=True, in_=text_frame)
        scrollbar.pack(side='right', fill='y', in_=text_frame)
    
    def setup_results_tab(self, frame):
        """Setup results tab with multiple plots"""
        # Create matplotlib figure with subplots
        self.fig = plt.figure(figsize=(12, 10))
        
        # Temperature plot
        self.ax_temp = plt.subplot(3, 1, 1)
        # Signal plot
        self.ax_signals = plt.subplot(3, 1, 2)
        # Ratio plot
        self.ax_ratios = plt.subplot(3, 1, 3)
        
        plt.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Control buttons
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="Save Plots", command=self.save_plots).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Export Data", command=self.export_data).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Export Report", command=self.export_report).pack(side='left', padx=5)
    
    def update_window_correction_ui(self):
        """Update UI when window correction is enabled/disabled"""
        if self.window_correction_enabled.get():
            self.exp_checkbox.configure(state='normal')
            self.bg_checkbox.configure(state='normal')
        else:
            self.exp_checkbox.configure(state='disabled')
            self.bg_checkbox.configure(state='disabled')
    
    
        def apply_config():
            self.window_affected_detectors = {det: var.get() for det, var in checkbox_vars.items()}
            self.window_transmission_scale = scale_var.get()
            
            # Update label
            affected_count = sum(self.window_affected_detectors.values())
            if affected_count > 0:
                self.window_config_label.config(
                    text=f"{affected_count} channel(s) with window correction"
                )
            else:
                self.window_config_label.config(text="No channels configured")
            
            config_window.destroy()
        
        def select_all():
            for var in checkbox_vars.values():
                var.set(True)
        
        def select_none():
            for var in checkbox_vars.values():
                var.set(False)
        
        ttk.Button(button_frame, text="Select All", command=select_all).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Select None", command=select_none).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Apply", command=apply_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=config_window.destroy).pack(side='left', padx=5)



    def update_detector_settings(self):
        """Update detector wavelength inputs"""
        # Clear existing widgets
        for widget in self.wavelength_frame.winfo_children():
            widget.destroy()
        
        num_det = self.num_detectors.get()
        self.detector_wavelengths = {}
        
        if num_det == 3:
            # 3-detector system
            default_wavelengths = [800, 1100, 1400]  # Your specified wavelengths
            
            for i in range(3):
                detector_name = f"detector{i+1}"
                row_frame = ttk.Frame(self.wavelength_frame)
                row_frame.grid(row=i, column=0, sticky='ew', padx=5, pady=5)
                
                ttk.Label(row_frame, text=f"Channel {i+1} (C{i+1}):").pack(side='left', padx=5)
                
                wavelength_var = tk.DoubleVar(value=default_wavelengths[i])
                self.detector_wavelengths[detector_name] = wavelength_var
                
                entry = ttk.Entry(row_frame, textvariable=wavelength_var, width=10)
                entry.pack(side='left', padx=5)
                ttk.Label(row_frame, text="nm").pack(side='left')
        
        else:  # 32 detectors
            # Create scrollable frame for 32 detectors
            canvas = tk.Canvas(self.wavelength_frame, height=300)
            scrollbar = ttk.Scrollbar(self.wavelength_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Distribute wavelengths between 300nm and 2000nm
            wavelengths = np.linspace(300, 2000, 32)
            
            # Create grid layout - 8 rows x 4 columns
            for i in range(32):
                detector_name = f"detector{i+1}"
                row = i // 4
                col = i % 4
                
                frame = ttk.Frame(scrollable_frame)
                frame.grid(row=row, column=col, padx=5, pady=2, sticky='ew')
                
                ttk.Label(frame, text=f"C{i+1}:", width=4).pack(side='left')
                
                wavelength_var = tk.DoubleVar(value=round(wavelengths[i], 1))
                self.detector_wavelengths[detector_name] = wavelength_var
                
                entry = ttk.Entry(frame, textvariable=wavelength_var, width=8)
                entry.pack(side='left', padx=2)
                ttk.Label(frame, text="nm").pack(side='left')
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
    
    def browse_folder(self, category):
        """Browse and select folder"""
        folder = filedialog.askdirectory(title=f"Select {category} folder")
        if folder:
            self.folder_paths[category].set(folder)
    
    def clear_all_data(self):
        """Clear all loaded data"""
        self.processed_data = {}
        self.temperature_results = None
        self.status_text.delete(1.0, tk.END)
        self.analysis_text.delete(1.0, tk.END)
        for ax in [self.ax_temp, self.ax_signals, self.ax_ratios]:
            ax.clear()
        self.canvas.draw()
        messagebox.showinfo("Cleared", "All data has been cleared.")

    def load_files(self):
        """Load and process all files with progress dialog"""
        try:
            # Create progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Loading Files")
            progress_window.geometry("500x400")
            progress_window.transient(self.root)
            
            # Progress bar
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, 
                                         maximum=100, length=400)
            progress_bar.pack(pady=20, padx=50)
            
            # Status label
            status_label = ttk.Label(progress_window, text="Initializing...", 
                                   font=('Arial', 10))
            status_label.pack(pady=10)
            
            # Details text
            details_text = tk.Text(progress_window, height=15, width=60, 
                                 font=('Courier', 9))
            details_scroll = ttk.Scrollbar(progress_window, orient='vertical', 
                                         command=details_text.yview)
            details_text.configure(yscrollcommand=details_scroll.set)
            
            text_frame = ttk.Frame(progress_window)
            text_frame.pack(fill='both', expand=True, padx=20, pady=10)
            
            details_text.pack(side='left', fill='both', expand=True, in_=text_frame)
            details_scroll.pack(side='right', fill='y', in_=text_frame)
            
            # Update window
            progress_window.update()
            
            self.processed_data = {}
            num_detectors = self.num_detectors.get()
            
            # Clear status text
            self.status_text.delete(1.0, tk.END)
            
            # Calculate total steps
            total_steps = len(self.file_categories) * num_detectors
            current_step = 0
            
            # Load each category
            for category in self.file_categories:
                folder_path = self.folder_paths[category].get()
                
                if not folder_path and category != 'background':
                    details_text.insert(tk.END, f"⚠️  No folder selected for {category}\n")
                    details_text.see(tk.END)
                    continue
                elif not folder_path:
                    continue
                
                self.processed_data[category] = {}
                status_label.config(text=f"Loading {category} files...")
                details_text.insert(tk.END, f"\n{'='*50}\n")
                details_text.insert(tk.END, f"Loading {category.upper()} files\n")
                details_text.insert(tk.END, f"Folder: {folder_path}\n")
                details_text.insert(tk.END, f"{'='*50}\n")
                details_text.see(tk.END)
                progress_window.update()
                
                # Load files for each detector
                files_loaded = 0
                for i in range(num_detectors):
                    detector_name = f"detector{i+1}"
                    detector_prefix = f"C{i+1}"
                    
                    # Update progress
                    current_step += 1
                    progress_var.set((current_step / total_steps) * 100)
                    
                    # Find all files matching pattern
                    pattern = os.path.join(folder_path, f"{detector_prefix}*.txt")
                    files = glob.glob(pattern)
                    
                    if files:
                        details_text.insert(tk.END, f"\n{detector_name} ({detector_prefix}): ")
                        details_text.see(tk.END)
                        progress_window.update()
                        
                        time_data, signal_data = self.load_and_average_files(files)
                        
                        if signal_data is not None:
                            self.processed_data[category][detector_name] = {
                                'time': time_data,
                                'signal': signal_data,
                                'files': files
                            }
                            files_loaded += len(files)
                            details_text.insert(tk.END, f"✓ {len(files)} files loaded\n")
                            
                            # Show file names if only a few
                            if len(files) <= 3:
                                for f in files:
                                    details_text.insert(tk.END, f"  - {os.path.basename(f)}\n")
                        else:
                            details_text.insert(tk.END, f"✗ Failed to load\n")
                    else:
                        details_text.insert(tk.END, f"\n{detector_name}: No files found\n")
                    
                    details_text.see(tk.END)
                    progress_window.update()
                
                details_text.insert(tk.END, f"\nTotal: {files_loaded} files loaded\n")
                self.status_text.insert(tk.END, f"{category.upper()}: {files_loaded} files loaded\n")
            
            # Final status
            status_label.config(text="Loading complete!")
            progress_var.set(100)
            details_text.insert(tk.END, f"\n{'='*50}\n")
            details_text.insert(tk.END, "✅ All files loaded successfully!\n")
            details_text.insert(tk.END, f"{'='*50}\n")
            details_text.see(tk.END)
            
            # Add close button
            ttk.Button(progress_window, text="Close", 
                      command=progress_window.destroy).pack(pady=10)
            
            # Perform initial SNR analysis
            self.analyze_snr()
            
            self.status_text.insert(tk.END, "\n✅ All files loaded successfully!\n")
            self.status_text.see(tk.END)
            
        except Exception as e:
            error_msg = f"Error loading files: {str(e)}"
            self.status_text.insert(tk.END, f"\n❌ {error_msg}\n")
            messagebox.showerror("Error", error_msg)
            if 'progress_window' in locals():
                progress_window.destroy()
    
    def load_and_average_files(self, file_list):
        """Load and average multiple files with proper error handling"""
        if not file_list:
            return None, None
        
        all_data = []
        time_data = None
        
        for file_path in file_list:
            try:
                # Try different separators and handle various formats
                # First try comma separator
                try:
                    data = pd.read_csv(file_path, sep=',', header=None, skiprows=4, 
                                    engine='python', encoding='utf-8')
                except:
                    # Try tab separator
                    try:
                        data = pd.read_csv(file_path, sep='\t', header=None, skiprows=4, 
                                        engine='python', encoding='utf-8')
                    except:
                        # Try whitespace separator
                        data = pd.read_csv(file_path, sep='\s+', header=None, skiprows=4, 
                                        engine='python', encoding='utf-8')
                
                # Check if we have at least 2 columns
                if len(data.columns) >= 2 and len(data) > 0:
                    # Convert to numeric, handling any string values
                    time_col = pd.to_numeric(data.iloc[:, 0], errors='coerce')
                    signal_col = pd.to_numeric(data.iloc[:, 1], errors='coerce')
                    
                    # Remove any NaN values
                    mask = ~(time_col.isna() | signal_col.isna())
                    time_col = time_col[mask].values
                    signal_col = signal_col[mask].values
                    
                    # Check if we have valid data
                    if len(time_col) > 0 and len(signal_col) > 0:
                        if time_data is None:
                            time_data = time_col
                        
                        # Interpolate signal to match time_data if needed
                        if len(time_col) == len(time_data) and np.array_equal(time_col, time_data):
                            all_data.append(signal_col)
                        else:
                            # Interpolate to common time base
                            f = interp1d(time_col, signal_col, kind='linear', 
                                    fill_value='extrapolate', bounds_error=False)
                            interpolated_signal = f(time_data)
                            all_data.append(interpolated_signal)
                        
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                # Try to read the file line by line as a fallback
                try:
                    time_list = []
                    signal_list = []
                    
                    with open(file_path, 'r') as f:
                        # Skip header lines
                        for _ in range(4):
                            f.readline()
                        
                        # Read data lines
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                # Try to parse the line
                                parts = line.replace(',', ' ').split()
                                if len(parts) >= 2:
                                    try:
                                        t = float(parts[0])
                                        s = float(parts[1])
                                        time_list.append(t)
                                        signal_list.append(s)
                                    except ValueError:
                                        continue
                    
                    if time_list and signal_list:
                        time_array = np.array(time_list)
                        signal_array = np.array(signal_list)
                        
                        if time_data is None:
                            time_data = time_array
                            all_data.append(signal_array)
                        else:
                            # Interpolate to common time base
                            f = interp1d(time_array, signal_array, kind='linear', 
                                    fill_value='extrapolate', bounds_error=False)
                            interpolated_signal = f(time_data)
                            all_data.append(interpolated_signal)
                            
                except Exception as e2:
                    print(f"Warning: Fallback reading also failed for {file_path}: {e2}")
        
        if all_data and time_data is not None:
            # Find the common length
            min_length = min(len(time_data), min(len(d) for d in all_data))
            
            # Trim all arrays to common length
            time_data = time_data[:min_length]
            trimmed_data = [d[:min_length] for d in all_data]
            
            # Average all signals
            averaged = np.nanmean(trimmed_data, axis=0)
            
            # Final check for valid data
            if np.any(np.isfinite(averaged)):
                return time_data, averaged
        
        return None, None
    
    def diagnose_file_format(self, file_path):
        """Diagnose the format of a data file"""
        try:
            with open(file_path, 'r') as f:
                # Read first 10 lines
                lines = []
                for i in range(10):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)
            
            print(f"\nDiagnosing file: {file_path}")
            print("First 10 lines:")
            for i, line in enumerate(lines):
                print(f"Line {i}: {repr(line)}")
            
            # Try to detect separator
            if len(lines) > 4:
                data_line = lines[4].strip()
                if ',' in data_line:
                    print("Detected separator: comma (,)")
                elif '\t' in data_line:
                    print("Detected separator: tab (\\t)")
                else:
                    print("Detected separator: whitespace")
            
        except Exception as e:
            print(f"Error reading file: {e}")

# Add this method to help debug
    def check_file_format(self):
        """Check the format of a sample file from each category"""
        for category in self.file_categories:
            folder_path = self.folder_paths[category].get()
            if folder_path:
                # Find first C1 file
                pattern = os.path.join(folder_path, "C1*.txt")
                files = glob.glob(pattern)
                if files:
                    self.diagnose_file_format(files[0])

    def get_window_transmission(self, wavelength):
        """Get window transmission for given wavelength"""
        if not self.window_correction_enabled.get():
            return 1.0
        
        base_transmission = 1.0
        
        if self.window_material.get() == "polycarbonate":
            # Interpolate transmission values
            wavelengths = sorted(self.polycarbonate_transmission.keys())
            transmissions = [self.polycarbonate_transmission[w] for w in wavelengths]
            
            if wavelength <= wavelengths[0]:
                base_transmission = transmissions[0]
            elif wavelength >= wavelengths[-1]:
                base_transmission = transmissions[-1]
            else:
                # Linear interpolation
                f = interp1d(wavelengths, transmissions, kind='linear')
                base_transmission = float(f(wavelength))
        
        elif self.window_material.get() == "quartz":
            # Quartz transmission approximation
            if wavelength < 200:
                base_transmission = 0.0
            elif wavelength < 300:
                base_transmission = 0.85
            elif wavelength < 2500:
                base_transmission = 0.92
            else:
                base_transmission = 0.90
        
        elif self.window_material.get() == "sapphire":
            # Sapphire transmission approximation
            if wavelength < 200:
                base_transmission = 0.0
            elif wavelength < 300:
                base_transmission = 0.80
            elif wavelength < 5000:
                base_transmission = 0.85
            else:
                base_transmission = 0.80
        
        # Apply scale factor
        base_transmission *= self.window_transmission_scale.get()
        
        return base_transmission
    
    def analyze_snr(self):
        """Comprehensive SNR analysis with clear pass/fail indication"""
        if 'experimental' not in self.processed_data:
            return
        
        current_text = self.analysis_text.get(1.0, tk.END)
        if "Running SNR analysis..." not in current_text:
            self.analysis_text.delete(1.0, tk.END)
        
        # self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, "SIGNAL-TO-NOISE RATIO ANALYSIS\n")
        self.analysis_text.insert(tk.END, "="*60 + "\n\n")
        
        from datetime import datetime
        self.analysis_text.insert(tk.END, f"Analysis performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        snr_results = {}
        threshold = self.snr_threshold.get()
        required_percentage = self.snr_percentage.get()

        num_detectors = len(self.processed_data.get('experimental', {}))
        self.analysis_text.insert(tk.END, f"Analyzing {num_detectors} channels...\n\n")
        self.analysis_text.update()
        
        # Calculate SNR for each detector
        for i, detector in enumerate(self.processed_data.get('experimental', {})):
        # Update progress
            if i % 5 == 0:  # Update every 5 channels for performance
                self.analysis_text.insert(tk.END, f"Processing channel {i+1}/{num_detectors}...\r")
                self.analysis_text.update()
            
            signal = self.processed_data['experimental'][detector]['signal']
            # Calculate SNR using sliding window
            window_size = min(100, len(signal) // 10)  # Adaptive window size
            snr_values = []
                
            for i in range(0, len(signal) - window_size, 10):
                window = signal[i:i+window_size]
                mean_val = np.mean(window)
                std_val = np.std(window)
                
                if std_val > 0:
                    snr = mean_val / std_val
                    snr_values.append(snr)
            
            if snr_values:
                avg_snr = np.mean(snr_values)
                percentage_above = (np.sum(np.array(snr_values) > threshold) / len(snr_values)) * 100
                
                wavelength = self.detector_wavelengths.get(detector, tk.DoubleVar(value=0)).get()
                
                snr_results[detector] = {
                    'avg_snr': avg_snr,
                    'percentage_above': percentage_above,
                    'wavelength': wavelength,
                    'passes': percentage_above >= required_percentage
                }
        
        self.analysis_text.delete("end-2l", "end-1l")


        # Display results
        self.analysis_text.insert(tk.END, f"SNR Threshold: {threshold:.1f}\n")
        self.analysis_text.insert(tk.END, f"Required Percentage: {required_percentage:.1f}%\n\n")
        
        # Separate passing and failing channels
        passing_channels = []
        failing_channels = []
        
        for detector, results in sorted(snr_results.items()):
            if results['passes']:
                passing_channels.append((detector, results))
            else:
                failing_channels.append((detector, results))
        
        # Display PASSING channels first
        self.analysis_text.insert(tk.END, "✅ CHANNELS WITH ACCEPTABLE SNR:\n")
        self.analysis_text.insert(tk.END, "-"*60 + "\n")
        
        if passing_channels:
            self.analysis_text.insert(tk.END, f"{'Channel':<12} {'λ (nm)':<10} {'Avg SNR':<12} {'% Above':<12}\n")
            self.analysis_text.insert(tk.END, "-"*60 + "\n")
            
            for detector, results in passing_channels:
                # Extract channel number for cleaner display
                channel_num = int(detector.replace('detector', ''))
                self.analysis_text.insert(tk.END, 
                    f"Channel {channel_num:<4} {results['wavelength']:<10.0f} "
                    f"{results['avg_snr']:<12.1f} {results['percentage_above']:<12.1f}\n")
            
            self.analysis_text.insert(tk.END, f"\nTotal: {len(passing_channels)} channels pass SNR criteria\n")
        else:
            self.analysis_text.insert(tk.END, "   None - No channels meet the SNR criteria!\n")
        
        # Display FAILING channels
        self.analysis_text.insert(tk.END, "\n❌ CHANNELS WITH INSUFFICIENT SNR:\n")
        self.analysis_text.insert(tk.END, "-"*60 + "\n")
        
        if failing_channels:
            self.analysis_text.insert(tk.END, f"{'Channel':<12} {'λ (nm)':<10} {'Avg SNR':<12} {'% Above':<12}\n")
            self.analysis_text.insert(tk.END, "-"*60 + "\n")
            
            for detector, results in failing_channels:
                channel_num = int(detector.replace('detector', ''))
                self.analysis_text.insert(tk.END, 
                    f"Channel {channel_num:<4} {results['wavelength']:<10.0f} "
                    f"{results['avg_snr']:<12.1f} {results['percentage_above']:<12.1f}\n")
            
            self.analysis_text.insert(tk.END, f"\nTotal: {len(failing_channels)} channels fail SNR criteria\n")
        else:
            self.analysis_text.insert(tk.END, "   None - All channels meet the SNR criteria!\n")
        
        # Summary for ratio pyrometry
        self.analysis_text.insert(tk.END, "\n" + "="*60 + "\n")
        self.analysis_text.insert(tk.END, "RATIO PYROMETRY SUMMARY:\n")
        self.analysis_text.insert(tk.END, "="*60 + "\n")
        
        if len(passing_channels) >= 2:
            self.detector_pairs = list(combinations([ch[0] for ch in passing_channels], 2))
            self.analysis_text.insert(tk.END, f"✓ {len(passing_channels)} channels available for pyrometry\n")
            self.analysis_text.insert(tk.END, f"✓ {len(self.detector_pairs)} possible channel pairs\n\n")
            
            # List the passing channels with wavelengths
            self.analysis_text.insert(tk.END, "Channels available for temperature measurement:\n")
            for detector, results in passing_channels:
                channel_num = int(detector.replace('detector', ''))
                self.analysis_text.insert(tk.END, f"  • Channel {channel_num} ({results['wavelength']:.0f}nm)\n")
        else:
            self.detector_pairs = []
            self.analysis_text.insert(tk.END, f"⚠️  Only {len(passing_channels)} channel(s) pass SNR criteria\n")
            self.analysis_text.insert(tk.END, "⚠️  Need at least 2 channels with good SNR for ratio pyrometry\n")
            self.analysis_text.insert(tk.END, "\nPossible solutions:\n")
            self.analysis_text.insert(tk.END, "  • Increase integration time\n")
            self.analysis_text.insert(tk.END, "  • Improve optical alignment\n")
            self.analysis_text.insert(tk.END, "  • Check for blocked channels\n")
            self.analysis_text.insert(tk.END, "  • Lower SNR threshold (current: {:.1f})\n".format(threshold))
            self.analysis_text.insert(tk.END, "  • Lower required percentage (current: {:.1f}%)\n".format(required_percentage))
        
        # Store results
        self.snr_results = snr_results
        self.good_detectors = [ch[0] for ch in passing_channels]
        
        # Scroll to top to show passing channels first
        self.analysis_text.see("1.0")
    
    def show_detector_pairs(self):
        """Show all possible detector pairs and their characteristics"""
        if not hasattr(self, 'detector_pairs') or not self.detector_pairs:
            messagebox.showwarning("No Pairs", "No valid detector pairs available. Run SNR analysis first.")
            return
        
        self.analysis_text.insert(tk.END, "\n\nDETECTOR PAIR ANALYSIS\n")
        self.analysis_text.insert(tk.END, "="*60 + "\n\n")
        
        for i, (det1, det2) in enumerate(self.detector_pairs):
            lambda1 = self.detector_wavelengths[det1].get()
            lambda2 = self.detector_wavelengths[det2].get()
            
            # Calculate wavelength separation
            separation = abs(lambda1 - lambda2)
            
            # Estimate temperature sensitivity (larger separation = better)
            sensitivity_score = separation / 100  # Normalized score
            
            self.analysis_text.insert(tk.END, f"Pair {i+1}: {det1} ({lambda1:.0f}nm) - {det2} ({lambda2:.0f}nm)\n")
            self.analysis_text.insert(tk.END, f"  Wavelength separation: {separation:.0f}nm\n")
            self.analysis_text.insert(tk.END, f"  Sensitivity score: {sensitivity_score:.2f}\n\n")
    
    def planck_function(self, wavelength_nm, temperature, emissivity=None):
        """Calculate Planck function value"""
        if emissivity is None:
            emissivity = self.emissivity.get()
        
        wavelength_m = wavelength_nm * 1e-9
        
        try:
            # Planck's law: B(λ,T) = ε * (2hc²/λ⁵) / (exp(hc/λkT) - 1)
            exp_term = np.exp(self.c2 / (wavelength_m * temperature))
            
            if exp_term == np.inf:
                return 1e-100
            
            radiance = emissivity * (self.c1 / wavelength_m**5) / (exp_term - 1)
            return radiance
            
        except (OverflowError, ZeroDivisionError):
            return 1e-100
    
    def calculate_temperature(self):
        """Calculate temperature using all valid detector pairs"""
        if not hasattr(self, 'detector_pairs') or not self.detector_pairs:
            messagebox.showerror("Error", "No valid detector pairs. Run SNR analysis first.")
            return
        
        try:
            self.analysis_text.insert(tk.END, "\n\nTEMPERATURE CALCULATION\n")
            self.analysis_text.insert(tk.END, "="*60 + "\n\n")
            
            # Get corrected signals
            corrected_signals = self.get_corrected_signals()
            
            # Calculate temperature for each detector pair
            all_temperatures = []
            pair_info = []
            
            for det1, det2 in self.detector_pairs:
                if det1 in corrected_signals and det2 in corrected_signals:
                    signal1 = corrected_signals[det1]
                    signal2 = corrected_signals[det2]
                    
                    lambda1 = self.detector_wavelengths[det1].get()
                    lambda2 = self.detector_wavelengths[det2].get()
                    
                    # Calculate calibration factors
                    cal_factor1 = self.calculate_calibration_factor(det1)
                    cal_factor2 = self.calculate_calibration_factor(det2)
                    
                    if cal_factor1 and cal_factor2:
                        # Calculate temperature using ratio method
                        temp = self.solve_temperature_ratio(signal1, signal2, lambda1, lambda2, 
                                                          cal_factor1, cal_factor2)
                        
                        if temp is not None:
                            all_temperatures.append(temp)
                            pair_info.append({
                                'det1': det1, 'det2': det2,
                                'lambda1': lambda1, 'lambda2': lambda2,
                                'temperature': temp
                            })
                            
                            self.analysis_text.insert(tk.END, 
                                f"Pair {det1}-{det2} ({lambda1:.0f}nm/{lambda2:.0f}nm): "
                                f"Avg T = {np.nanmean(temp):.0f}K\n")
            
            if not all_temperatures:
                raise ValueError("No valid temperature calculations")
            
            # Calculate average temperature and standard deviation
            # Align all temperature arrays to same time base
            min_length = min(len(t) for t in all_temperatures)
            aligned_temps = np.array([t[:min_length] for t in all_temperatures])
            
            # Calculate mean and std across all pairs
            mean_temperature = np.nanmean(aligned_temps, axis=0)
            std_temperature = np.nanstd(aligned_temps, axis=0)
            
            # Apply smoothing
            window_size = self.smoothing_window.get()
            if window_size > 1:
                method = self.smoothing_method.get()
                self.analysis_text.insert(tk.END, f"\nApplying {method} smoothing with window size {window_size}\n")
                
                original_length = len(mean_temperature)
                mean_temperature = self.smooth_data(mean_temperature, window_size)
                std_temperature = self.smooth_data(std_temperature, window_size)
                
                # Verify length is preserved
                if len(mean_temperature) != original_length:
                    self.analysis_text.insert(tk.END, 
                        f"WARNING: Smoothing changed array length from {original_length} to {len(mean_temperature)}\n")
                else:
                    self.analysis_text.insert(tk.END, f"✓ Array length preserved: {len(mean_temperature)} points\n")

            # Get time axis
            time_axis = corrected_signals[self.detector_pairs[0][0]]['time'][:min_length]
            
            # Store results
            self.temperature_results = {
                'time': time_axis,
                'mean_temperature': mean_temperature,
                'std_temperature': std_temperature,
                'all_temperatures': aligned_temps,
                'pair_info': pair_info,
                'corrected_signals': corrected_signals,
                'num_pairs': len(all_temperatures)
            }
            
            # Display summary
            avg_temp = np.nanmean(mean_temperature)
            std_temp = np.nanmean(std_temperature)
            
            self.analysis_text.insert(tk.END, f"\n{'='*40}\n")
            self.analysis_text.insert(tk.END, f"RESULTS SUMMARY:\n")
            self.analysis_text.insert(tk.END, f"  Number of detector pairs used: {len(all_temperatures)}\n")
            self.analysis_text.insert(tk.END, f"  Average temperature: {avg_temp:.0f} ± {std_temp:.0f} K\n")
            self.analysis_text.insert(tk.END, f"  Min temperature: {np.nanmin(mean_temperature):.0f} K\n")
            self.analysis_text.insert(tk.END, f"  Max temperature: {np.nanmax(mean_temperature):.0f} K\n")
            
            # Plot results
            self.plot_results()
            
            # Switch to results tab
            self.notebook.select(3)
            
            messagebox.showinfo("Success", 
                f"Temperature calculated successfully!\n"
                f"Average: {avg_temp:.0f} ± {std_temp:.0f} K\n"
                f"Using {len(all_temperatures)} detector pairs")
            
        except Exception as e:
            error_msg = f"Error calculating temperature: {str(e)}"
            self.analysis_text.insert(tk.END, f"\n❌ {error_msg}\n")
            messagebox.showerror("Error", error_msg)
    
    def get_corrected_signals(self):
        """Get corrected signals with dark subtraction and window correction"""
        corrected_signals = {}
        
        # Show window correction status if enabled
        if self.window_correction_enabled.get():
            self.analysis_text.insert(tk.END, "\nWindow Correction Status:\n")
            self.analysis_text.insert(tk.END, f"  Material: {self.window_material.get()}\n")
            self.analysis_text.insert(tk.END, f"  Scale factor: {self.window_transmission_scale.get():.3f}\n")
            if self.window_correction_experimental.get():
                self.analysis_text.insert(tk.END, "  ✓ Applied to experimental data\n")
            if self.window_correction_background.get():
                self.analysis_text.insert(tk.END, "  ✓ Applied to background data\n")
        
        for detector in self.good_detectors:
            try:
                # Get experimental signal
                exp_data = self.processed_data['experimental'][detector]
                exp_signal = exp_data['signal'].copy()
                time_data = exp_data['time'].copy()
                
                # Dark subtraction
                if 'dark' in self.processed_data and detector in self.processed_data['dark']:
                    dark_signal = self.processed_data['dark'][detector]['signal']
                    # Align lengths
                    min_len = min(len(exp_signal), len(dark_signal))
                    exp_signal = exp_signal[:min_len]
                    dark_signal = dark_signal[:min_len]
                    time_data = time_data[:min_len]
                    
                    # Subtract dark current
                    exp_signal = exp_signal - dark_signal
                
                # Window correction for experimental data (if enabled)
                if (self.window_correction_enabled.get() and 
                    self.window_correction_experimental.get()):
                    wavelength = self.detector_wavelengths[detector].get()
                    transmission = self.get_window_transmission(wavelength)
                    exp_signal = exp_signal / transmission
                
                # Background subtraction (if available)
                if 'background' in self.processed_data and detector in self.processed_data['background']:
                    bg_signal = self.processed_data['background'][detector]['signal']
                    min_len = min(len(exp_signal), len(bg_signal))
                    
                    # Apply window correction to background if enabled
                    if (self.window_correction_enabled.get() and 
                        self.window_correction_background.get()):
                        wavelength = self.detector_wavelengths[detector].get()
                        transmission = self.get_window_transmission(wavelength)
                        bg_signal = bg_signal / transmission
                    
                    exp_signal = exp_signal[:min_len] - bg_signal[:min_len]
                    time_data = time_data[:min_len]
                
                # Ensure positive values
                exp_signal = np.maximum(exp_signal, 1e-10)
                
                corrected_signals[detector] = {
                    'signal': exp_signal,
                    'time': time_data
                }
                
            except Exception as e:
                print(f"Error correcting signal for {detector}: {e}")
        
        return corrected_signals
    
    def calculate_calibration_factor(self, detector):
        """Calculate calibration factor for a detector including sensitivity"""
        try:
            # Get calibration signal
            cal_data = self.processed_data['calibration'][detector]
            cal_signal = cal_data['signal'].copy()
            
            # Dark subtraction for calibration
            if 'dark' in self.processed_data and detector in self.processed_data['dark']:
                dark_signal = self.processed_data['dark'][detector]['signal']
                min_len = min(len(cal_signal), len(dark_signal))
                cal_signal = cal_signal[:min_len] - dark_signal[:min_len]
            
            # Use stable portion of calibration signal (middle 50%)
            start_idx = len(cal_signal) // 4
            end_idx = 3 * len(cal_signal) // 4
            stable_signal = np.mean(cal_signal[start_idx:end_idx])
            
            # Get wavelength and sensitivity
            wavelength = self.detector_wavelengths[detector].get()
            sensitivity = self.get_detector_sensitivity(detector, wavelength)
            
            # Calculate theoretical Planck function value
            cal_temp = self.calibration_temp.get()
            theoretical = self.planck_function(wavelength, cal_temp)
            
            # Calibration factor = measured / (theoretical * sensitivity)
            cal_factor = stable_signal / (theoretical * sensitivity)
            
            # Log the values for debugging
            self.analysis_text.insert(tk.END, 
                f"  {detector} calibration: signal={stable_signal:.2e}, "
                f"theoretical={theoretical:.2e}, sensitivity={sensitivity:.3f}, "
                f"cal_factor={cal_factor:.2e}\n")
            
            return cal_factor
            
        except Exception as e:
            self.analysis_text.insert(tk.END, f"Warning: Calibration failed for {detector}: {e}\n")
            return None

    def plot_sensitivity_curves(self):
        """Plot detector sensitivity curves"""
        if not self.sensitivity_loaded:
            messagebox.showwarning("No Data", "Please load sensitivity files first")
            return
        
        # Create new window for sensitivity plot
        sens_window = tk.Toplevel(self.root)
        sens_window.title("Detector Sensitivity Curves")
        sens_window.geometry("800x600")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each detector's sensitivity
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.sensitivity_data)))
        
        for i, (detector, data) in enumerate(self.sensitivity_data.items()):
            wavelengths = data['wavelengths']
            sensitivity = data['sensitivity']
            
            # Normalize if needed
            if np.max(sensitivity) > 10:
                sensitivity = sensitivity / np.max(sensitivity)
            
            ax.plot(wavelengths, sensitivity, color=colors[i], 
                   linewidth=2, label=detector)
            
            # Mark the operating wavelength
            if detector in self.detector_wavelengths:
                op_wavelength = self.detector_wavelengths[detector].get()
                op_sensitivity = data['interpolator'](op_wavelength)
                if np.max(sensitivity) > 10:
                    op_sensitivity = op_sensitivity / np.max(data['sensitivity'])
                ax.plot(op_wavelength, op_sensitivity, 'o', color=colors[i], 
                       markersize=8, markeredgecolor='black')
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Relative Sensitivity')
        ax.set_title('Detector Sensitivity Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add to window
        canvas = FigureCanvasTkAgg(fig, sens_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add close button
        ttk.Button(sens_window, text="Close", 
                  command=sens_window.destroy).pack(pady=10)


    def solve_temperature_ratio(self, signal1, signal2, lambda1, lambda2, cal1, cal2):
        """Solve for temperature using Planck ratio method"""
        # Ensure same length
        min_len = min(len(signal1['signal']), len(signal2['signal']))
        sig1 = signal1['signal'][:min_len]
        sig2 = signal2['signal'][:min_len]
        
        # Calculate ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            measured_ratio = (sig1 / cal1) / (sig2 / cal2)
        
        # Initialize temperature array
        temperature = np.full(min_len, np.nan)
        
        # Convert wavelengths to meters
        lam1_m = lambda1 * 1e-9
        lam2_m = lambda2 * 1e-9
        
        # Pre-calculate constants for Wien approximation
        wien_const = self.c2 * (1/lam1_m - 1/lam2_m)
        planck_const = (lam2_m/lam1_m)**5
        
        for i in range(min_len):
            if np.isfinite(measured_ratio[i]) and measured_ratio[i] > 0:
                # Use Wien approximation for initial guess
                if measured_ratio[i] * planck_const > 0:
                    T_wien = wien_const / np.log(measured_ratio[i] * planck_const)
                    
                    if 500 < T_wien < 6000:  # Reasonable range
                        # Refine with full Planck equation
                        def ratio_error(T):
                            if T <= 0:
                                return 1e10
                            B1 = self.planck_function(lambda1, T, emissivity=1.0)
                            B2 = self.planck_function(lambda2, T, emissivity=1.0)
                            if B2 == 0:
                                return 1e10
                            return abs(B1/B2 - measured_ratio[i])
                        
                        # Minimize error
                        result = minimize_scalar(ratio_error, bounds=(T_wien*0.8, T_wien*1.2), 
                                               method='bounded')
                        
                        if result.success and 500 < result.x < 6000:
                            temperature[i] = result.x
        
        return temperature
    
    def smooth_data(self, data, window_size):
        """Apply smoothing to data while preserving array length"""
        if window_size <= 1 or len(data) < window_size:
            return data
        
        method = self.smoothing_method.get()
        
        # Ensure window size is odd for some methods
        if method in ['savgol', 'median'] and window_size % 2 == 0:
            window_size += 1
        
        if method == "gaussian":
            # Gaussian smoothing
            from scipy.ndimage import gaussian_filter1d
            sigma = window_size / 4.0
            return gaussian_filter1d(data, sigma=sigma, mode='nearest')
        
        elif method == "savgol":
            # Savitzky-Golay filter
            from scipy.signal import savgol_filter
            # Use polyorder=3 or less depending on window size
            polyorder = min(3, window_size - 1)
            return savgol_filter(data, window_size, polyorder, mode='nearest')
        
        elif method == "moving_average":
            # Simple moving average with edge handling
            from scipy.ndimage import uniform_filter1d
            return uniform_filter1d(data, size=window_size, mode='nearest')
        
        elif method == "median":
            # Median filter - good for removing spikes
            from scipy.signal import medfilt
            # medfilt pads with zeros, so we'll use a custom implementation
            smoothed = np.copy(data)
            half_window = window_size // 2
            
            for i in range(len(data)):
                start = max(0, i - half_window)
                end = min(len(data), i + half_window + 1)
                smoothed[i] = np.median(data[start:end])
            
            return smoothed
        
        else:
            # Fallback - return original data
            return data

    # Also update calculate_temperature to show what smoothing is being applied:
    # In the calculate_temperature method, update the smoothing section:
    # Apply smoothing
        window_size = self.smoothing_window.get()
        if window_size > 1:
            method = self.smoothing_method.get()
            self.analysis_text.insert(tk.END, f"Applying {method} smoothing with window size: {window_size}\n")
            
            original_length = len(mean_temperature)
            mean_temperature = self.smooth_data(mean_temperature, window_size)
            std_temperature = self.smooth_data(std_temperature, window_size)
            
            # Verify length is preserved
            if len(mean_temperature) != original_length:
                self.analysis_text.insert(tk.END, 
                    f"WARNING: Smoothing changed array length from {original_length} to {len(mean_temperature)}\n")
            else:
                self.analysis_text.insert(tk.END, f"✓ Array length preserved: {len(mean_temperature)} points\n")
        
    def plot_results(self):
        """Create comprehensive plots"""
        if not self.temperature_results:
            return
        
        results = self.temperature_results
        
        # Clear all plots
        self.ax_temp.clear()
        self.ax_signals.clear()
        self.ax_ratios.clear()
        
        # Plot 1: Temperature with uncertainty band
        time = results['time']
        mean_temp = results['mean_temperature']
        std_temp = results['std_temperature']
        
        self.ax_temp.plot(time, mean_temp, 'r-', linewidth=2, label='Mean Temperature')
        self.ax_temp.fill_between(time, mean_temp - std_temp, mean_temp + std_temp, 
                                alpha=0.3, color='red', label='±1σ')
        
        # Add calibration temperature reference
        self.ax_temp.axhline(y=self.calibration_temp.get(), color='gray', 
                        linestyle='--', label=f'Calibration ({self.calibration_temp.get():.0f}K)')
        
        self.ax_temp.set_xlabel('Time (s)')
        self.ax_temp.set_ylabel('Temperature (K)')
        self.ax_temp.set_title(f'Temperature vs Time ({results["num_pairs"]} detector pairs)')
        self.ax_temp.legend()
        self.ax_temp.grid(True, alpha=0.3)
        
        # Plot 2: Detector signals
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.good_detectors)))
        
        for i, detector in enumerate(self.good_detectors[:6]):  # Show up to 6 detectors
            if detector in results['corrected_signals']:
                sig_data = results['corrected_signals'][detector]
                wavelength = self.detector_wavelengths[detector].get()
                self.ax_signals.plot(sig_data['time'], sig_data['signal'], 
                                color=colors[i], linewidth=1.5,
                                label=f'{detector} ({wavelength:.0f}nm)')
        
        self.ax_signals.set_xlabel('Time (s)')
        self.ax_signals.set_ylabel('Corrected Signal')
        self.ax_signals.set_title('Detector Signals (Corrected)')
        self.ax_signals.set_yscale('log')
        self.ax_signals.legend(loc='best', ncol=2)
        self.ax_signals.grid(True, alpha=0.3)
        
        # Plot 3: Temperature from individual pairs
        if len(results['all_temperatures']) <= 6:  # Show individual pairs if not too many
            for i, temp in enumerate(results['all_temperatures']):
                pair = results['pair_info'][i]
                label = f"{pair['det1']}-{pair['det2']} " \
                    f"({pair['lambda1']:.0f}/{pair['lambda2']:.0f}nm)"
                self.ax_ratios.plot(time, temp, linewidth=1, alpha=0.7, label=label)
        else:
            # Show temperature distribution
            percentiles = np.nanpercentile(results['all_temperatures'], [10, 25, 50, 75, 90], axis=0)
            self.ax_ratios.fill_between(time, percentiles[0], percentiles[4], 
                                    alpha=0.2, color='blue', label='10-90 percentile')
            self.ax_ratios.fill_between(time, percentiles[1], percentiles[3], 
                                    alpha=0.4, color='blue', label='25-75 percentile')
            self.ax_ratios.plot(time, percentiles[2], 'b-', linewidth=2, label='Median')
        
        self.ax_ratios.set_xlabel('Time (s)')
        self.ax_ratios.set_ylabel('Temperature (K)')
        self.ax_ratios.set_title('Temperature from Individual Detector Pairs')
        self.ax_ratios.legend(loc='best')
        self.ax_ratios.grid(True, alpha=0.3)
        
        # Adjust layout and draw
        self.fig.tight_layout()
        self.canvas.draw()
    
    def save_plots(self):
        """Save all plots to file"""
        if not self.temperature_results:
            messagebox.showerror("Error", "No results to save!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), 
                      ("SVG files", "*.svg"), ("All files", "*.*")]
        )
        
        if filename:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Plots saved to: {filename}")
    
    def export_data(self):
        """Export processed data to CSV"""
        if not self.temperature_results:
            messagebox.showerror("Error", "No data to export!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            results = self.temperature_results
            
            # Create comprehensive data dictionary
            data_dict = {
                'Time_s': results['time'],
                'Mean_Temperature_K': results['mean_temperature'],
                'Std_Temperature_K': results['std_temperature']
            }
            
            # Add individual pair temperatures
            for i, temp in enumerate(results['all_temperatures']):
                pair = results['pair_info'][i]
                col_name = f"T_{pair['det1']}_{pair['det2']}_K"
                data_dict[col_name] = temp
            
            # Add corrected signals
            for detector in self.good_detectors:
                if detector in results['corrected_signals']:
                    sig_data = results['corrected_signals'][detector]
                    wavelength = self.detector_wavelengths[detector].get()
                    # Align to same length
                    signal = sig_data['signal'][:len(results['time'])]
                    data_dict[f'{detector}_{wavelength:.0f}nm_Signal'] = signal
            
            # Convert to DataFrame and save
            df = pd.DataFrame(data_dict)
            df.to_csv(filename, index=False)
            
            messagebox.showinfo("Success", f"Data exported to: {filename}")
    
    def calculate_minimum_temperature_from_specs(self):
        """Calculate minimum detectable temperature from detector and digitizer specifications"""
        
        # Create a new window for the analysis
        min_temp_window = tk.Toplevel(self.root)
        min_temp_window.title("Minimum Temperature Analysis")
        min_temp_window.geometry("900x800")
        
        # Create scrollable frame
        canvas = tk.Canvas(min_temp_window)
        scrollbar = ttk.Scrollbar(min_temp_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Input parameters frame
        input_frame = ttk.LabelFrame(scrollable_frame, text="System Specifications")
        input_frame.pack(fill='x', padx=10, pady=10)
        
        # Detector specifications
        det_frame = ttk.LabelFrame(input_frame, text="Detector Specifications")
        det_frame.pack(fill='x', padx=5, pady=5)
        
        # Create input variables
        self.det_dark_current = tk.DoubleVar(value=1e-9)  # A
        self.det_shunt_resistance = tk.DoubleVar(value=1e9)  # Ohms
        self.det_capacitance = tk.DoubleVar(value=100e-12)  # F
        self.det_nep = tk.DoubleVar(value=1e-13)  # W/√Hz
        self.det_area = tk.DoubleVar(value=1e-6)  # m²
        self.det_d_star = tk.DoubleVar(value=1e12)  # cm√Hz/W
        
        # Detector inputs
        ttk.Label(det_frame, text="Dark Current (A):").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(det_frame, textvariable=self.det_dark_current, width=15).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(det_frame, text="Shunt Resistance (Ω):").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(det_frame, textvariable=self.det_shunt_resistance, width=15).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(det_frame, text="Capacitance (F):").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(det_frame, textvariable=self.det_capacitance, width=15).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(det_frame, text="NEP (W/√Hz):").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(det_frame, textvariable=self.det_nep, width=15).grid(row=3, column=1, padx=5, pady=2)
        
        ttk.Label(det_frame, text="Active Area (m²):").grid(row=4, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(det_frame, textvariable=self.det_area, width=15).grid(row=4, column=1, padx=5, pady=2)
        
        ttk.Label(det_frame, text="D* (cm√Hz/W):").grid(row=5, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(det_frame, textvariable=self.det_d_star, width=15).grid(row=5, column=1, padx=5, pady=2)
        
        # Amplifier specifications
        amp_frame = ttk.LabelFrame(input_frame, text="Amplifier/TIA Specifications")
        amp_frame.pack(fill='x', padx=5, pady=5)
        
        self.tia_gain = tk.DoubleVar(value=1e6)  # V/A
        self.tia_bandwidth = tk.DoubleVar(value=1e6)  # Hz
        self.amp_noise_voltage = tk.DoubleVar(value=10e-9)  # V/√Hz
        self.amp_noise_current = tk.DoubleVar(value=1e-12)  # A/√Hz
        self.load_resistance = tk.DoubleVar(value=50)  # Ohms
        
        ttk.Label(amp_frame, text="TIA Gain (V/A) or Load R (Ω):").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(amp_frame, textvariable=self.tia_gain, width=15).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(amp_frame, text="Bandwidth (Hz):").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(amp_frame, textvariable=self.tia_bandwidth, width=15).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(amp_frame, text="Voltage Noise (V/√Hz):").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(amp_frame, textvariable=self.amp_noise_voltage, width=15).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(amp_frame, text="Current Noise (A/√Hz):").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(amp_frame, textvariable=self.amp_noise_current, width=15).grid(row=3, column=1, padx=5, pady=2)
        
        # Digitizer specifications
        dig_frame = ttk.LabelFrame(input_frame, text="Digitizer Specifications")
        dig_frame.pack(fill='x', padx=5, pady=5)
        
        self.adc_bits = tk.IntVar(value=16)
        self.adc_range = tk.DoubleVar(value=10.0)  # V
        self.adc_sampling_rate = tk.DoubleVar(value=1e6)  # Hz
        self.adc_noise_lsb = tk.DoubleVar(value=1.0)  # LSB rms
        
        ttk.Label(dig_frame, text="ADC Bits:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(dig_frame, textvariable=self.adc_bits, width=15).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(dig_frame, text="Input Range (V):").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(dig_frame, textvariable=self.adc_range, width=15).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(dig_frame, text="Sampling Rate (Hz):").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(dig_frame, textvariable=self.adc_sampling_rate, width=15).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(dig_frame, text="Noise (LSB rms):").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(dig_frame, textvariable=self.adc_noise_lsb, width=15).grid(row=3, column=1, padx=5, pady=2)
        
        # Optical system specifications
        opt_frame = ttk.LabelFrame(input_frame, text="Optical System")
        opt_frame.pack(fill='x', padx=5, pady=5)
        
        self.optical_throughput = tk.DoubleVar(value=0.5)
        self.solid_angle = tk.DoubleVar(value=0.01)  # sr
        self.optical_bandwidth = tk.DoubleVar(value=50)  # nm
        self.target_emissivity = tk.DoubleVar(value=0.35)
        
        ttk.Label(opt_frame, text="Optical Throughput:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(opt_frame, textvariable=self.optical_throughput, width=15).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(opt_frame, text="Collection Solid Angle (sr):").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(opt_frame, textvariable=self.solid_angle, width=15).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(opt_frame, text="Optical Bandwidth (nm):").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(opt_frame, textvariable=self.optical_bandwidth, width=15).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(opt_frame, text="Target Emissivity:").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(opt_frame, textvariable=self.target_emissivity, width=15).grid(row=3, column=1, padx=5, pady=2)
        
        # Calculate button
        ttk.Button(input_frame, text="Calculate Minimum Temperature", 
                command=lambda: self.perform_min_temp_calculation(results_text),
                style='Accent.TButton').pack(pady=10)
        
        # Results display
        results_frame = ttk.LabelFrame(scrollable_frame, text="Analysis Results")
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        results_text = tk.Text(results_frame, height=20, width=100, font=('Courier', 9))
        results_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add close button
        ttk.Button(min_temp_window, text="Close", command=min_temp_window.destroy).pack(pady=5)

    def perform_min_temp_calculation(self, results_text):
        """Perform the minimum temperature calculation"""
        results_text.delete(1.0, tk.END)
        results_text.insert(tk.END, "MINIMUM DETECTABLE TEMPERATURE ANALYSIS\n")
        results_text.insert(tk.END, "="*70 + "\n\n")
        
        # Get all parameters
        dark_current = self.det_dark_current.get()
        shunt_R = self.det_shunt_resistance.get()
        capacitance = self.det_capacitance.get()
        nep = self.det_nep.get()
        det_area = self.det_area.get()
        d_star = self.det_d_star.get()
        
        tia_gain = self.tia_gain.get()
        bandwidth = self.tia_bandwidth.get()
        v_noise = self.amp_noise_voltage.get()
        i_noise = self.amp_noise_current.get()
        
        adc_bits = self.adc_bits.get()
        adc_range = self.adc_range.get()
        fs = self.adc_sampling_rate.get()
        adc_noise_lsb = self.adc_noise_lsb.get()
        
        throughput = self.optical_throughput.get()
        solid_angle = self.solid_angle.get()
        optical_bw = self.optical_bandwidth.get()
        emissivity = self.target_emissivity.get()
        
        # Calculate noise sources
        results_text.insert(tk.END, "1. NOISE ANALYSIS\n")
        results_text.insert(tk.END, "-"*50 + "\n")
        
        # 1. Shot noise from dark current
        q = 1.602e-19  # electron charge
        i_shot = np.sqrt(2 * q * dark_current * bandwidth)
        results_text.insert(tk.END, f"Shot noise current: {i_shot:.3e} A (rms)\n")
        
        # 2. Johnson noise from shunt resistance
        kb = 1.381e-23  # Boltzmann constant
        T_det = 300  # detector temperature (K)
        i_johnson = np.sqrt(4 * kb * T_det * bandwidth / shunt_R)
        results_text.insert(tk.END, f"Johnson noise current: {i_johnson:.3e} A (rms)\n")
        
        # 3. Amplifier noise contributions
        v_amp_noise = v_noise * np.sqrt(bandwidth)
        i_amp_noise = i_noise * np.sqrt(bandwidth)
        results_text.insert(tk.END, f"Amplifier voltage noise: {v_amp_noise:.3e} V (rms)\n")
        results_text.insert(tk.END, f"Amplifier current noise: {i_amp_noise:.3e} A (rms)\n")
        
        # 4. Total current noise
        i_noise_total = np.sqrt(i_shot**2 + i_johnson**2 + i_amp_noise**2)
        results_text.insert(tk.END, f"Total current noise: {i_noise_total:.3e} A (rms)\n")
        
        # 5. Total voltage noise at amplifier output
        v_noise_total = np.sqrt((i_noise_total * tia_gain)**2 + v_amp_noise**2)
        results_text.insert(tk.END, f"Total voltage noise: {v_noise_total:.3e} V (rms)\n")
        
        # 6. ADC noise
        lsb = adc_range / (2**adc_bits)
        adc_noise_v = adc_noise_lsb * lsb
        results_text.insert(tk.END, f"ADC LSB: {lsb:.3e} V\n")
        results_text.insert(tk.END, f"ADC noise: {adc_noise_v:.3e} V (rms)\n")
        
        # 7. Total system noise
        v_system_noise = np.sqrt(v_noise_total**2 + adc_noise_v**2)
        i_system_noise = v_system_noise / tia_gain
        results_text.insert(tk.END, f"\nTotal system voltage noise: {v_system_noise:.3e} V (rms)\n")
        results_text.insert(tk.END, f"Equivalent current noise: {i_system_noise:.3e} A (rms)\n")
        
        # Calculate NEP from noise
        if self.sensitivity_loaded and len(self.good_detectors) > 0:
            # Use actual sensitivity data
            detector = self.good_detectors[0]
            wavelength = self.detector_wavelengths[detector].get()
            sensitivity_V_W = self.get_detector_sensitivity(detector, wavelength)
            sensitivity_A_W = sensitivity_V_W / tia_gain
            nep_calculated = i_system_noise / sensitivity_A_W
        else:
            # Use D* to estimate NEP
            nep_calculated = np.sqrt(det_area * 1e4) / (d_star * 1e-2)  # Convert units
        
        results_text.insert(tk.END, f"\nNEP (from noise): {nep_calculated:.3e} W/√Hz\n")
        results_text.insert(tk.END, f"NEP (specified): {nep:.3e} W/√Hz\n")
        
        # Use worse of the two
        nep_effective = max(nep_calculated, nep)
        results_text.insert(tk.END, f"Effective NEP: {nep_effective:.3e} W/√Hz\n")
        
        # 2. MINIMUM DETECTABLE POWER
        results_text.insert(tk.END, "\n2. MINIMUM DETECTABLE POWER\n")
        results_text.insert(tk.END, "-"*50 + "\n")
        
        # For SNR = 1, minimum power = NEP * sqrt(bandwidth)
        # For practical SNR = 3, multiply by 3
        snr_required = 3.0
        min_power = nep_effective * np.sqrt(bandwidth) * snr_required
        results_text.insert(tk.END, f"Minimum detectable power (SNR={snr_required}): {min_power:.3e} W\n")
        
        # 3. OPTICAL COLLECTION EFFICIENCY
        results_text.insert(tk.END, "\n3. OPTICAL COLLECTION\n")
        results_text.insert(tk.END, "-"*50 + "\n")
        
        # Étendue (throughput)
        etendue = det_area * solid_angle
        results_text.insert(tk.END, f"Étendue (AΩ): {etendue:.3e} m²·sr\n")
        results_text.insert(tk.END, f"Optical throughput: {throughput:.3f}\n")
        results_text.insert(tk.END, f"Optical bandwidth: {optical_bw:.1f} nm\n")
        
        # 4. CALCULATE MINIMUM TEMPERATURE FOR EACH WAVELENGTH
        results_text.insert(tk.END, "\n4. MINIMUM TEMPERATURE BY WAVELENGTH\n")
        results_text.insert(tk.END, "-"*50 + "\n")
        
        # Test wavelengths
        if hasattr(self, 'detector_wavelengths') and self.detector_wavelengths:
            wavelengths = [wl_var.get() for wl_var in self.detector_wavelengths.values()]
        else:
            wavelengths = [800, 1100, 1400]  # Default wavelengths
        
        min_temps = []
        
        for wavelength in wavelengths:
            results_text.insert(tk.END, f"\nWavelength: {wavelength:.0f} nm\n")
            
            # Binary search for minimum temperature
            T_min, T_max = 300, 5000
            target_power = min_power / (throughput * etendue * emissivity)
            
            while T_max - T_min > 1:
                T_mid = (T_min + T_max) / 2
                
                # Calculate spectral radiance at this temperature
                # L = ε * (2hc²/λ⁵) / (exp(hc/λkT) - 1)
                lambda_m = wavelength * 1e-9
                try:
                    radiance = self.planck_function(wavelength, T_mid, emissivity)
                    
                    # Integrate over bandwidth
                    # Approximate as radiance * bandwidth
                    bandwidth_m = optical_bw * 1e-9
                    integrated_radiance = radiance * bandwidth_m
                    
                    if integrated_radiance < target_power:
                        T_min = T_mid
                    else:
                        T_max = T_mid
                        
                except:
                    T_min = T_mid
            
            min_temps.append(T_max)
            results_text.insert(tk.END, f"  Minimum temperature: {T_max:.0f} K\n")
            
            # Also calculate what signal this produces
            signal_power = self.planck_function(wavelength, T_max, emissivity) * bandwidth_m * throughput * etendue
            signal_current = signal_power * sensitivity_A_W if 'sensitivity_A_W' in locals() else signal_power / wavelength * 0.8
            signal_voltage = signal_current * tia_gain
            
            results_text.insert(tk.END, f"  Signal power: {signal_power:.3e} W\n")
            results_text.insert(tk.END, f"  Signal voltage: {signal_voltage:.3e} V\n")
            results_text.insert(tk.END, f"  SNR: {signal_voltage/v_system_noise:.1f}\n")
        
        # 5. RATIO PYROMETRY MINIMUM TEMPERATURE
        results_text.insert(tk.END, "\n5. RATIO PYROMETRY ANALYSIS\n")
        results_text.insert(tk.END, "-"*50 + "\n")
        
        if len(wavelengths) >= 2:
            # For ratio pyrometry, both channels need adequate SNR
            ratio_min_temp = max(min_temps)
            results_text.insert(tk.END, f"Minimum temperature for ratio pyrometry: {ratio_min_temp:.0f} K\n")
            
            # Calculate ratio sensitivity at this temperature
            T_test = ratio_min_temp + 100  # Test slightly above minimum
            for i in range(len(wavelengths)-1):
                for j in range(i+1, len(wavelengths)):
                    wl1, wl2 = wavelengths[i], wavelengths[j]
                    
                    # Calculate ratio change per degree
                    R1 = self.planck_function(wl1, T_test, 1.0) / self.planck_function(wl2, T_test, 1.0)
                    R2 = self.planck_function(wl1, T_test+1, 1.0) / self.planck_function(wl2, T_test+1, 1.0)
                    
                    sensitivity = abs(R2 - R1) / R1 * 100  # %/K
                    
                    results_text.insert(tk.END, f"\nPair {wl1:.0f}/{wl2:.0f} nm:\n")
                    results_text.insert(tk.END, f"  Ratio sensitivity at {T_test:.0f}K: {sensitivity:.3f} %/K\n")
        
        # 6. PRACTICAL CONSIDERATIONS
        results_text.insert(tk.END, "\n6. PRACTICAL CONSIDERATIONS\n")
        results_text.insert(tk.END, "-"*50 + "\n")
        
        # Integration time effects
        integration_time = 1.0 / bandwidth  # Effective integration time
        results_text.insert(tk.END, f"Effective integration time: {integration_time*1e6:.1f} μs\n")
        
        # Averaging improvement
        if fs > bandwidth:
            averaging_factor = np.sqrt(fs / bandwidth)
            improved_min_temp = ratio_min_temp * (1 / averaging_factor)**(1/4)  # Approximate
            results_text.insert(tk.END, f"Averaging improvement factor: {averaging_factor:.1f}\n")
            results_text.insert(tk.END, f"Potential minimum with averaging: {improved_min_temp:.0f} K\n")
        
        # Dynamic range
        # Maximum temperature limited by ADC range
        max_signal_v = adc_range * 0.9  # 90% of range
        max_signal_current = max_signal_v / tia_gain
        
        # Find maximum temperature for shortest wavelength
        T_min, T_max = ratio_min_temp, 10000
        wavelength_min = min(wavelengths)
        
        while T_max - T_min > 1:
            T_mid = (T_min + T_max) / 2
            
            signal_power = self.planck_function(wavelength_min, T_mid, emissivity) * bandwidth_m * throughput * etendue
            signal_current = signal_power * sensitivity_A_W if 'sensitivity_A_W' in locals() else signal_power / wavelength_min * 0.8
            signal_voltage = signal_current * tia_gain
            
            if signal_voltage < max_signal_v:
                T_min = T_mid
            else:
                T_max = T_mid
        
        results_text.insert(tk.END, f"\nDynamic Range:\n")
        results_text.insert(tk.END, f"  Minimum temperature: {ratio_min_temp:.0f} K\n")
        results_text.insert(tk.END, f"  Maximum temperature: {T_min:.0f} K\n")
        results_text.insert(tk.END, f"  Temperature range: {T_min/ratio_min_temp:.1f}:1\n")
        results_text.insert(tk.END, f"  Dynamic range: {20*np.log10(T_min/ratio_min_temp):.1f} dB\n")
        
        # 7. RECOMMENDATIONS
        results_text.insert(tk.END, "\n7. RECOMMENDATIONS\n")
        results_text.insert(tk.END, "-"*50 + "\n")
        
        if ratio_min_temp > 2000:
            results_text.insert(tk.END, "⚠️ Minimum temperature is quite high. Consider:\n")
            results_text.insert(tk.END, "  • Using more sensitive detectors (higher D*)\n")
            results_text.insert(tk.END, "  • Increasing optical collection (larger lens/aperture)\n")
            results_text.insert(tk.END, "  • Using shorter wavelengths if possible\n")
            results_text.insert(tk.END, "  • Increasing TIA gain\n")
            results_text.insert(tk.END, "  • Cooling detectors to reduce noise\n")
        else:
            results_text.insert(tk.END, "✓ System should work well for your temperature range\n")
        
        # Save results for later use
        self.min_temp_analysis = {
            'min_temperatures': dict(zip(wavelengths, min_temps)),
            'ratio_min_temp': ratio_min_temp if 'ratio_min_temp' in locals() else max(min_temps),
            'system_noise': v_system_noise,
            'nep': nep_effective,
            'dynamic_range': (ratio_min_temp, T_min) if 'T_min' in locals() else (ratio_min_temp, 5000)
        }
        
        results_text.insert(tk.END, f"\n{'='*70}\n")
        results_text.insert(tk.END, "Analysis complete!\n")
        results_text.see(tk.END)

    # The button for Min Temperature Analysis should be created inside setup_analysis_tab, not at the class level.
    # Remove this misplaced code. The correct code is already present in setup_analysis_tab.

    def quick_min_temp_check(self):
        """Quick minimum temperature estimate based on current data"""
        if not hasattr(self, 'good_detectors') or not self.good_detectors:
            messagebox.showinfo("No Data", "Please load data and run SNR analysis first")
            return
        
        # Get noise statistics from actual data
        noise_estimates = {}
        
        for detector in self.good_detectors:
            if 'dark' in self.processed_data and detector in self.processed_data['dark']:
                dark_signal = self.processed_data['dark'][detector]['signal']
                noise_estimates[detector] = {
                    'mean': np.mean(dark_signal),
                    'std': np.std(dark_signal),
                    'peak_to_peak': np.max(dark_signal) - np.min(dark_signal)
                }
        
        # Create simple report
        report = "QUICK MINIMUM TEMPERATURE CHECK\n"
        report += "="*50 + "\n\n"
        
        for detector in self.good_detectors:
            wavelength = self.detector_wavelengths[detector].get()
            
            if detector in noise_estimates:
                noise = noise_estimates[detector]['std'] * 3  # 3-sigma
                
                # Find temperature where signal = noise
                T_min, T_max = 500, 5000
                
                while T_max - T_min > 10:
                    T_mid = (T_min + T_max) / 2
                    
                    # Estimate signal (need calibration for absolute)
                    # This is a rough estimate
                    signal_est = self.planck_function(wavelength, T_mid, 1.0) * 1e-15  # Rough scaling
                    
                    if signal_est < noise:
                        T_min = T_mid
                    else:
                        T_max = T_mid
                
                report += f"{detector} ({wavelength:.0f}nm):\n"
                report += f"  Noise level (3σ): {noise:.3e} V\n"
                report += f"  Estimated min temp: ~{T_max:.0f} K\n\n"
        
        # Show in message box
        messagebox.showinfo("Quick Check", report)

    def export_sensitivity_data(self):
        """Export sensitivity data to file"""
        if not self.sensitivity_loaded:
            messagebox.showwarning("No Data", "No sensitivity data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            # Combine all sensitivity data
            all_data = {}
            max_length = 0
            
            # Find maximum length and prepare data
            for detector, data in self.sensitivity_data.items():
                all_data[f"{detector}_wavelength"] = data['wavelengths']
                all_data[f"{detector}_sensitivity"] = data['sensitivity']
                max_length = max(max_length, len(data['wavelengths']))
            
            # Pad shorter arrays with NaN
            for key in all_data:
                if len(all_data[key]) < max_length:
                    all_data[key] = np.pad(all_data[key], (0, max_length - len(all_data[key])), 
                                        constant_values=np.nan)
            
            # Create DataFrame and save
            df = pd.DataFrame(all_data)
            df.to_csv(filename, index=False)
            
            messagebox.showinfo("Success", f"Sensitivity data exported to: {filename}")

    def export_report(self):
        """Export comprehensive analysis report"""
        if not self.temperature_results:
            messagebox.showerror("Error", "No results to export!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write("MULTI-CHANNEL PYROMETRY ANALYSIS REPORT\n")
                f.write("="*60 + "\n\n")
                
                # Settings
                f.write("MEASUREMENT SETTINGS:\n")
                f.write(f"  Number of channels: {self.num_detectors.get()}\n")
                f.write(f"  Calibration temperature: {self.calibration_temp.get():.1f} K\n")
                f.write(f"  Emissivity: {self.emissivity.get():.3f}\n")
                f.write(f"  SNR threshold: {self.snr_threshold.get():.1f}\n")
                f.write(f"  Required SNR percentage: {self.snr_percentage.get():.1f}%\n")
                f.write(f"  Smoothing window: {self.smoothing_window.get()}\n")
                
                if self.window_correction_enabled.get():
                    f.write(f"  Window correction: Enabled\n")
                    f.write(f"    Material: {self.window_material.get()}\n")
                    f.write(f"    Transmission scale: {self.window_transmission_scale.get():.3f}\n")
                    f.write(f"    Applied to experimental: {'Yes' if self.window_correction_experimental.get() else 'No'}\n")
                    f.write(f"    Applied to background: {'Yes' if self.window_correction_background.get() else 'No'}\n")
                    
                    # Show transmission values for each wavelength
                    f.write("    Transmission values:\n")
                    for detector, wavelength_var in sorted(self.detector_wavelengths.items()):
                        if detector in self.good_detectors:
                            wavelength = wavelength_var.get()
                            transmission = self.get_window_transmission(wavelength)
                            f.write(f"      {wavelength:.0f}nm: {transmission:.3f}\n")
                else:
                    f.write(f"  Window correction: Disabled\n")
                
                # Continue with rest of report...
                f.write("\n")
                
                # Channel wavelengths
                f.write("CHANNEL WAVELENGTHS:\n")
                for detector, wavelength_var in sorted(self.detector_wavelengths.items()):
                    f.write(f"  {detector}: {wavelength_var.get():.1f} nm\n")
                
                f.write("\n")
                
                # SNR Results
                if hasattr(self, 'snr_results'):
                    f.write("SIGNAL-TO-NOISE ANALYSIS:\n")
                    f.write(f"  Channels passing SNR criteria: {len(self.good_detectors)}/{len(self.snr_results)}\n")
                    f.write(f"  Valid detector pairs: {len(self.detector_pairs)}\n\n")
                    
                    f.write("  Channel Details:\n")
                    for detector, results in sorted(self.snr_results.items()):
                        status = "PASS" if results['passes'] else "FAIL"
                        f.write(f"    {detector} ({results['wavelength']:.0f}nm): "
                               f"SNR={results['avg_snr']:.1f}, "
                               f"{results['percentage_above']:.1f}% above threshold - {status}\n")
                
                f.write("\n")
                
                # Temperature results
                results = self.temperature_results
                avg_temp = np.nanmean(results['mean_temperature'])
                std_temp = np.nanmean(results['std_temperature'])
                min_temp = np.nanmin(results['mean_temperature'])
                max_temp = np.nanmax(results['mean_temperature'])
                
                f.write("TEMPERATURE RESULTS:\n")
                f.write(f"  Number of detector pairs used: {results['num_pairs']}\n")
                f.write(f"  Average temperature: {avg_temp:.1f} ± {std_temp:.1f} K\n")
                f.write(f"  Temperature range: {min_temp:.1f} - {max_temp:.1f} K\n")
                f.write(f"  Total measurement time: {results['time'][-1]:.2f} s\n")
                
                f.write("\n")
                
                # Detector pair details
                f.write("DETECTOR PAIR DETAILS:\n")
                for i, pair in enumerate(results['pair_info']):
                    pair_temp = results['all_temperatures'][i]
                    avg_pair_temp = np.nanmean(pair_temp)
                    f.write(f"  Pair {i+1}: {pair['det1']}-{pair['det2']} "
                           f"({pair['lambda1']:.0f}nm/{pair['lambda2']:.0f}nm) - "
                           f"Avg T = {avg_pair_temp:.1f} K\n")
                
                f.write("\n")
                
                # File information
                f.write("FILE INFORMATION:\n")
                for category in self.file_categories:
                    folder = self.folder_paths[category].get()
                    if folder:
                        f.write(f"  {category.title()}: {folder}\n")
                
                f.write("\n")
                f.write("Report generated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            
            messagebox.showinfo("Success", f"Report exported to: {filename}")


def main():
    """Main function to run the application"""
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    style.theme_use('clam')  # Modern looking theme
    
    # Configure colors
    style.configure('Accent.TButton', foreground='white', background='#0078D4')
    style.map('Accent.TButton',
              background=[('active', '#106EBE'), ('pressed', '#005A9E')])
    
    # Create and run application
    app = AdvancedPyrometerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()