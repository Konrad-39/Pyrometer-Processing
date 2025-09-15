import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import os
import glob
from scipy.ndimage import gaussian_filter1d

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
    
    def add_default_calibration(self):
        """Add default calibration temperature"""
        self.calibration_temps = [(2796.0, "Grey body source")]
    
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
        """Update detector wavelength settings based on number of detectors"""
        
        # Clear existing wavelength widgets
        for widget in self.wavelength_frame.winfo_children():
            widget.destroy()
        
        num_det = self.num_detectors.get()
        self.detector_wavelengths = {}
        
        if num_det == 3:
            # Default wavelengths for 2-color
            default_wavelengths = [800, 1100, 1400]
            
            for i in range(3):
                detector_name = f"detector{i+1}"
                
                ttk.Label(self.wavelength_frame, text=f"Detector {i+1}:").grid(row=i, column=0, sticky='w', padx=5, pady=5)
                
                wavelength_var = tk.DoubleVar(value=default_wavelengths[i])
                self.detector_wavelengths[detector_name] = wavelength_var
                
                ttk.Entry(self.wavelength_frame, textvariable=wavelength_var, width=10).grid(row=i, column=1, sticky='w', padx=5, pady=5)
                ttk.Label(self.wavelength_frame, text="nm").grid(row=i, column=2, sticky='w', padx=5, pady=5)
        
        elif num_det == 32:
            # Create scrollable frame for 32 detectors
            canvas = tk.Canvas(self.wavelength_frame, height=200)
            scrollbar = ttk.Scrollbar(self.wavelength_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Default wavelengths for 32-color (visible to near-IR spectrum)
            start_wavelength = 400  # nm
            end_wavelength = 1100   # nm
            wavelength_step = (end_wavelength - start_wavelength) / 31
            
            # Create wavelength entries in a grid
            for i in range(32):
                detector_name = f"detector{i+1}"
                default_wavelength = start_wavelength + i * wavelength_step
                
                row = i // 4
                col = (i % 4) * 3
                
                ttk.Label(scrollable_frame, text=f"C{i+1}:").grid(row=row, column=col, sticky='w', padx=2, pady=2)
                
                wavelength_var = tk.DoubleVar(value=round(default_wavelength, 1))
                self.detector_wavelengths[detector_name] = wavelength_var
                
                entry = ttk.Entry(scrollable_frame, textvariable=wavelength_var, width=8)
                entry.grid(row=row, column=col+1, sticky='w', padx=2, pady=2)
                ttk.Label(scrollable_frame, text="nm").grid(row=row, column=col+2, sticky='w', padx=2, pady=2)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Add mouse wheel scrolling
            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
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
        """Calculate Planck function for given wavelength (nm) and temperature (K)"""
        wavelength_m = wavelength_nm * 1e-9  # Convert nm to m
        try:
            result = (2 * self.h * self.c**2 / wavelength_m**5) / \
                    (np.exp(self.h * self.c / (wavelength_m * self.k * temperature)) - 1)
            return result
        except:
            return 1e-10  # Return small value if calculation fails
    
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
        """Check SNR for all loaded files using comprehensive analysis"""
        
        self.snr_text.delete(1.0, tk.END)
        self.snr_results = {}
        
        num_detectors = self.num_detectors.get()
        threshold = self.snr_threshold.get()
        percentage_threshold = self.snr_percentage.get()  # ADD this line
        
        self.snr_text.insert(tk.END, "COMPREHENSIVE SNR ANALYSIS RESULTS\n")
        self.snr_text.insert(tk.END, "="*60 + "\n")
        self.snr_text.insert(tk.END, f"Number of detectors: {num_detectors}\n")
        self.snr_text.insert(tk.END, f"SNR Threshold: {threshold}\n")
        self.snr_text.insert(tk.END, f"Required Good Time: {percentage_threshold}%\n\n")
        
        for category in self.file_categories:
            self.snr_text.insert(tk.END, f"{category.upper()} FILES:\n")
            self.snr_text.insert(tk.END, "-"*50 + "\n")
            self.snr_results[category] = {}
            
            detector_data = {}  # For detector comparisons
            
            for i in range(num_detectors):
                detector_name = f"detector{i+1}"
                
                if (category in self.processed_data and 
                    detector_name in self.processed_data[category]):
                    
                    signal_data = self.processed_data[category][detector_name]['signal']
                    rms_snr, is_good, metrics = self.calculate_snr(signal_data)
                    
                    self.snr_results[category][detector_name] = {
                        'snr': rms_snr,
                        'is_good': is_good,
                        'metrics': metrics,
                        'time': self.processed_data[category][detector_name]['time'],
                        'signal': signal_data
                    }
                    
                    detector_data[detector_name] = signal_data
                    
                    status = "✓ GOOD" if is_good else "✗ POOR"
                    wavelength = self.detector_wavelengths.get(detector_name, tk.DoubleVar(value=0)).get()
                    
                    self.snr_text.insert(tk.END, f"  C{i+1} ({wavelength:.1f}nm):\n")
                    self.snr_text.insert(tk.END, f"    Mean Signal:     {metrics['mean_signal']:.3e}\n")
                    self.snr_text.insert(tk.END, f"    Std Deviation:   {metrics['std_signal']:.3e}\n")
                    self.snr_text.insert(tk.END, f"    SNR (mean/std):  {metrics['snr']:.1f}\n")
                    self.snr_text.insert(tk.END, f"    RMS Noise:       {metrics['rms_noise']:.3e}\n")
                    self.snr_text.insert(tk.END, f"    RMS SNR:         {metrics['rms_snr']:.1f}\n")
                    self.snr_text.insert(tk.END, f"    Rolling SNR:     {metrics['rolling_snr_mean']:.1f} ± {metrics['rolling_snr_std']:.1f}\n")
                    self.snr_text.insert(tk.END, f"    Good SNR Time:   {metrics['percentage_good']:.1f}% {status}\n")
                    self.snr_text.insert(tk.END, f"    Window Size:     {metrics['window_size']} points\n")
                    self.snr_text.insert(tk.END, f"    Peak-Peak:       {metrics['pp_noise']:.3e}\n")
                    self.snr_text.insert(tk.END, f"    Dynamic Range:   {metrics['dynamic_range']:.1f} dB\n")
                    self.snr_text.insert(tk.END, f"    Quality:         {metrics['quality']}\n")
                    self.snr_text.insert(tk.END, "\n")
                    
                else:
                    self.snr_text.insert(tk.END, f"  C{i+1}: No data found\n\n")
            
            # Add detector comparisons
            if len(detector_data) >= 2:
                self.snr_text.insert(tk.END, "  Detector Comparison:\n")
                detector_names = list(detector_data.keys())
                for i in range(len(detector_names)):
                    for j in range(i+1, len(detector_names)):
                        det1, det2 = detector_names[i], detector_names[j]
                        ratio = np.mean(detector_data[det1]) / np.mean(detector_data[det2])
                        ratio_std = np.std(detector_data[det1] / detector_data[det2])
                        self.snr_text.insert(tk.END, f"    {det1}/{det2} ratio: {ratio:.3f} ± {ratio_std:.3f}\n")
                self.snr_text.insert(tk.END, "\n")
            
            self.snr_text.insert(tk.END, "\n")
        
        # Determine which detectors to use - only check experimental data SNR
        self.good_detectors = []
        
        for i in range(num_detectors):
            detector_name = f"detector{i+1}"
            
            # Only check experimental data SNR
            if ('experimental' in self.snr_results and 
                detector_name in self.snr_results['experimental']):
                
                if self.snr_results['experimental'][detector_name]['is_good']:
                    self.good_detectors.append(detector_name)
                    print(f"DEBUG: {detector_name} added - experimental good time = {self.snr_results['experimental'][detector_name]['metrics']['percentage_good']:.1f}%")
                else:
                    print(f"DEBUG: {detector_name} excluded - experimental good time = {self.snr_results['experimental'][detector_name]['metrics']['percentage_good']:.1f}% < {percentage_threshold}%")
            else:
                print(f"DEBUG: {detector_name} excluded - no experimental data found")
        
        self.snr_text.insert(tk.END, "SUMMARY:\n")
        self.snr_text.insert(tk.END, "="*30 + "\n")
        self.snr_text.insert(tk.END, f"SNR Threshold: {threshold}\n")
        self.snr_text.insert(tk.END, f"Required Good Time: {percentage_threshold}%\n")
        self.snr_text.insert(tk.END, f"Good detectors: {', '.join(self.good_detectors)}\n")
        self.snr_text.insert(tk.END, f"Poor detectors excluded: {num_detectors - len(self.good_detectors)}\n")

        # Show which detectors were excluded and why
        for i in range(num_detectors):
            detector_name = f"detector{i+1}"
            if detector_name not in self.good_detectors:
                if ('experimental' in self.snr_results and 
                    detector_name in self.snr_results['experimental']):
                    percentage_good = self.snr_results['experimental'][detector_name]['metrics']['percentage_good']
                    self.snr_text.insert(tk.END, f"  {detector_name} excluded: good SNR time = {percentage_good:.1f}% < {percentage_threshold}%\n")
                else:
                    self.snr_text.insert(tk.END, f"  {detector_name} excluded: no experimental data found\n")

        if len(self.good_detectors) >= 2:
            self.snr_text.insert(tk.END, f"\n✓ Ready for {len(self.good_detectors)}-color pyrometry!\n")
        elif len(self.good_detectors) == 1:
            self.snr_text.insert(tk.END, "\n⚠ Only 1 good detector - need at least 2 for ratio pyrometry\n")
        else:
            self.snr_text.insert(tk.END, "\n✗ No detectors meet SNR threshold!\n")
        
        # Auto-scroll to bottom
        self.snr_text.see(tk.END)
            
    def plot_snr_vs_time(self):
        """Plot SNR vs time for all detectors"""
        
        if 'experimental' not in self.snr_results:
            messagebox.showwarning("No Data", "No SNR data available. Run SNR check first.")
            return
        
        try:
            # Clear the third subplot
            self.axes[2].clear()
            
            colors = ['red', 'green', 'blue', 'orange', 'purple']
            threshold = self.snr_threshold.get()
            
            for i, (detector, data) in enumerate(self.snr_results['experimental'].items()):
                if 'rolling_snr' in data['metrics']:
                    rolling_snr = data['metrics']['rolling_snr']
                    time_data = data['time']
                    
                    # Create time axis for rolling SNR (centered in windows)
                    window_size = data['metrics']['window_size']
                    rolling_time = time_data[window_size//2:window_size//2 + len(rolling_snr)]
                    
                    # Get wavelength for label
                    wavelength = self.detector_wavelengths.get(detector, tk.DoubleVar(value=0)).get()
                    
                    # Plot rolling SNR
                    color = colors[i % len(colors)]
                    self.axes[2].plot(rolling_time, rolling_snr, 
                                    color=color, linewidth=2, 
                                    label=f'{detector} ({wavelength:.0f}nm) - {data["metrics"]["percentage_good"]:.1f}% good')
            
            # Add threshold line
            self.axes[2].axhline(y=threshold, color='red', linestyle='--', 
                                alpha=0.7, linewidth=2, label=f'SNR Threshold ({threshold})')
            
            self.axes[2].set_xlabel('Time (s)')
            self.axes[2].set_ylabel('Rolling SNR')
            self.axes[2].set_title('Rolling SNR vs Time (Experimental Data)')
            self.axes[2].legend()
            self.axes[2].grid(True, alpha=0.3)
            self.axes[2].set_ylim(0, max(10, threshold * 3))  # Dynamic y-limit
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            print("DEBUG: SNR vs Time plot completed")
            
        except Exception as e:
            print(f"DEBUG: Error plotting SNR vs time: {e}")
            messagebox.showerror("Error", f"Error plotting SNR vs time: {str(e)}")        

    def plot_rolling_snr(self):
        """Plot rolling SNR analysis for experimental data"""
        
        if 'experimental' not in self.snr_results:
            return
        
        fig, axes = plt.subplots(len(self.snr_results['experimental']), 1, figsize=(12, 8))
        if len(self.snr_results['experimental']) == 1:
            axes = [axes]
        
        colors = ['red', 'green', 'blue']
        threshold = self.snr_threshold.get()
        
        for i, (detector, data) in enumerate(self.snr_results['experimental'].items()):
            if 'rolling_snr' in data['metrics']:
                rolling_snr = data['metrics']['rolling_snr']
                time_data = data['time']
                
                # Create time axis for rolling SNR (centered in windows)
                window_size = data['metrics']['window_size']
                rolling_time = time_data[window_size//2:window_size//2 + len(rolling_snr)]
                
                axes[i].plot(rolling_time, rolling_snr, color=colors[i], linewidth=1, label=f'{detector} Rolling SNR')
                axes[i].axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
                axes[i].set_ylabel('Rolling SNR')
                axes[i].set_title(f'{detector} - {data["metrics"]["percentage_good"]:.1f}% above threshold')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()


    def calculate_temperature(self):
        """Calculate temperature using ratio pyrometry with multiple calibration points"""
        
        if len(self.good_detectors) < 2:
            messagebox.showerror("Error", 
                               f"Need at least 2 detectors with good SNR!\n"
                               f"Currently have: {len(self.good_detectors)}")
            return
        
        try:
            # Use first two good detectors for now
            det1, det2 = self.good_detectors[0], self.good_detectors[1]
            
            self.snr_text.insert(tk.END, f"\nCALCULATING TEMPERATURE using {det1} and {det2}...\n")
            self.snr_text.insert(tk.END, "="*50 + "\n")
            
            # Get wavelengths
            lambda1 = self.detector_wavelengths[det1].get()  # nm
            lambda2 = self.detector_wavelengths[det2].get()  # nm
            
            self.snr_text.insert(tk.END, f"Wavelengths: {lambda1:.1f}nm and {lambda2:.1f}nm\n")
            self.snr_text.insert(tk.END, f"Calibration temperatures: {[temp for temp, _ in self.calibration_temps]}\n")
            
            # Get corrected signals
            corrected_signals = {}
            corrected_time = None

            for detector in [det1, det2]:
                if (detector in self.processed_data['experimental'] and
                    detector in self.processed_data['background'] and
                    detector in self.processed_data['dark']):
                    exp_signal = self.processed_data['experimental'][detector]['signal']
                    bg_signal = self.processed_data['background'][detector]['signal']
                    dark_signal = self.processed_data['dark'][detector]['signal']
                
                    min_length = min(len(exp_signal), len(bg_signal), len(dark_signal))
                    
                    # Correct signals: (Experimental - Dark) - (Background - Dark)
                    corrected = (exp_signal[:min_length] - dark_signal[:min_length]) - \
                               (bg_signal[:min_length] - dark_signal[:min_length])
                    corrected = np.maximum(corrected, 1e-10)  # Avoid negative values
                    
                    corrected_signals[detector] = corrected
                    
                    if corrected_time is None:
                        corrected_time = self.processed_data['experimental'][detector]['time'][:min_length]
                    
                    self.snr_text.insert(tk.END, 
                        f"{detector} corrected signal range: {corrected.min():.2e} to {corrected.max():.2e}\n")
            
            if len(corrected_signals) < 2:
                raise ValueError("Could not correct signals for both detectors")
            
            # Calculate calibration factors for each calibration temperature
            all_cal_factors = {}
            
            for temp, description in self.calibration_temps:
                cal_factors = {}
                
                self.snr_text.insert(tk.END, f"\nCalculating calibration factors for {temp}K ({description}):\n")
                
                for detector in [det1, det2]:
                    if (detector in self.processed_data['calibration'] and
                        detector in self.processed_data['dark']):
                        
                        cal_signal = self.processed_data['calibration'][detector]['signal']
                        dark_signal = self.processed_data['dark'][detector]['signal']
                        
                        min_length = min(len(cal_signal), len(dark_signal))
                        corrected_cal = cal_signal[:min_length] - dark_signal[:min_length]
                        corrected_cal = np.maximum(corrected_cal, 1e-10)
                        
                        # Get wavelength and calculate theoretical Planck function
                        wavelength = self.detector_wavelengths[detector].get()
                        theoretical = self.planck_function(wavelength, temp)
                        
                        # Use stable region (middle 50% of data)
                        start_idx = len(corrected_cal) // 4
                        end_idx = 3 * len(corrected_cal) // 4
                        stable_signal = np.mean(corrected_cal[start_idx:end_idx])
                        
                        cal_factors[detector] = stable_signal / theoretical
                        
                        self.snr_text.insert(tk.END, 
                            f"  {detector}: {cal_factors[detector]:.2e}\n")
                
                all_cal_factors[temp] = cal_factors
            
            # Calculate temperature using each calibration point
            temperature_results = []
            temperature_descriptions = []
            
            for temp, description in self.calibration_temps:
                if temp in all_cal_factors and len(all_cal_factors[temp]) >= 2:
                    cal_factors = all_cal_factors[temp]
                    
                    # Calculate temperature using ratio pyrometry
                    ratio = corrected_signals[det1] / corrected_signals[det2]
                    
                    # Apply smoothing to ratio
                    ratio_smooth = gaussian_filter1d(ratio, sigma=5)
                    
                    # Apply calibration correction
                    cal_ratio = cal_factors[det1] / cal_factors[det2]
                    corrected_ratio = ratio_smooth / cal_ratio
                    corrected_ratio = np.maximum(corrected_ratio, 1e-10)
                    
                    # Wien's law temperature calculation
                    # Wien's law temperature calculation - CORRECTED VERSION
                    C2 = self.h * self.c / self.k  # Second radiation constant
                    lambda1_m = lambda1 * 1e-9  # Convert to meters
                    lambda2_m = lambda2 * 1e-9

                    # Calculate theoretical ratio at calibration temperature (for display)
                    theoretical_ratio = self.planck_function(lambda1, temp) / self.planck_function(lambda2, temp)
                    self.snr_text.insert(tk.END, f"Theoretical ratio at {temp}K: {theoretical_ratio:.4f}\n")
                    self.snr_text.insert(tk.END, f"Mean measured ratio: {np.mean(corrected_ratio):.4f}\n")

                    # Calculate the wavelength-dependent terms (use meters!)
                    wavelength_factor = (1/lambda2_m - 1/lambda1_m)  # This will be negative
                    wavelength_ratio_term = 5 * np.log(lambda2_m/lambda1_m)

                    self.snr_text.insert(tk.END, f"Wavelength factor (1/λ2 - 1/λ1): {wavelength_factor:.2e}\n")
                    self.snr_text.insert(tk.END, f"Wavelength ratio term 5*ln(λ2/λ1): {wavelength_ratio_term:.4f}\n")

                    # Initialize temperature list
                    temperature = []

                    # Calculate temperature for each point
                    for i, ratio_val in enumerate(corrected_ratio):
                        try:
                            # Proper Wien's law formula
                            log_ratio = np.log(ratio_val)
                            denominator = log_ratio - wavelength_ratio_term
                            
                            if abs(denominator) > 1e-10:  # Avoid division by zero
                                T_point = C2 * wavelength_factor / denominator
                                
                                # The temperature should be positive
                                if T_point > 0:
                                    temperature.append(T_point)
                                else:
                                    # If negative, try the reciprocal
                                    T_point = -T_point
                                    temperature.append(T_point)
                            else:
                                temperature.append(temp)  # Default to calibration temperature
                                
                        except (ValueError, RuntimeWarning, ZeroDivisionError):
                            temperature.append(temp)

                    # Convert to numpy array
                    temperature = np.array(temperature)

                    # Filter unrealistic temperatures BEFORE smoothing
                    temperature = np.clip(temperature, 300, 8000)

                    # Apply smoothing to temperature
                    smooth_sig = self.smoothing_window.get()
                    temperature_smooth = gaussian_filter1d(temperature, sigma=smooth_sig)

                    # Final clipping after smoothing
                    temperature_smooth = np.clip(temperature_smooth, 300, 8000)
                                        
                    temperature_results.append(temperature_smooth)
                    temperature_descriptions.append(f"{temp}K {description}")
                    
                    self.snr_text.insert(tk.END, 
                        f"\nUsing {temp}K calibration:\n")
                    self.snr_text.insert(tk.END, 
                        f"  Temperature range: {temperature_smooth.min():.0f}K to {temperature_smooth.max():.0f}K\n")
                    self.snr_text.insert(tk.END, 
                        f"  Average temperature: {temperature_smooth.mean():.0f}K\n")
            
            if not temperature_results:
                raise ValueError("No valid temperature calculations from calibration data")
            
            # Calculate statistics if multiple calibrations
            if len(temperature_results) > 1:
                # Calculate mean and standard deviation across calibrations
                temp_array = np.array(temperature_results)
                mean_temperature = np.mean(temp_array, axis=0)
                std_temperature = np.std(temp_array, axis=0)
                
                self.snr_text.insert(tk.END, f"\nMULTI-CALIBRATION STATISTICS:\n")
                self.snr_text.insert(tk.END, f"Mean temperature: {mean_temperature.mean():.0f} ± {std_temperature.mean():.0f}K\n")
                self.snr_text.insert(tk.END, f"Temperature range: {mean_temperature.min():.0f}K to {mean_temperature.max():.0f}K\n")
                
                # Store results with uncertainty
                self.temperature_results = {
                    'time': corrected_time,
                    'temperature_individual': temperature_results,
                    'temperature_descriptions': temperature_descriptions,
                    'temperature_mean': mean_temperature,
                    'temperature_std': std_temperature,
                    'corrected_signals': corrected_signals,
                    'detectors_used': [det1, det2],
                    'wavelengths': [lambda1, lambda2],
                    'has_uncertainty': True
                }
            else:
                # Single calibration result
                self.temperature_results = {
                    'time': corrected_time,
                    'temperature': temperature_results[0],
                    'temperature_smooth': temperature_results[0],
                    'corrected_signals': corrected_signals,
                    'detectors_used': [det1, det2],
                    'wavelengths': [lambda1, lambda2],
                    'has_uncertainty': False
                }
            
            # Plot results
            self.plot_results()
            
            # Switch to results tab
            self.notebook.select(3)
            
            self.snr_text.insert(tk.END, f"\n✓ Temperature calculation complete!\n")
            self.snr_text.see(tk.END)
            
            if len(temperature_results) > 1:
                messagebox.showinfo("Success", 
                                  f"Temperature calculated using {len(temperature_results)} calibration points!\n"
                                  f"Using {det1} ({lambda1:.1f}nm) and {det2} ({lambda2:.1f}nm)\n"
                                  f"Mean temperature: {mean_temperature.mean():.0f} ± {std_temperature.mean():.0f}K")
            else:
                messagebox.showinfo("Success", 
                                  f"Temperature calculated successfully!\n"
                                  f"Using {det1} ({lambda1:.1f}nm) and {det2} ({lambda2:.1f}nm)\n"
                                  f"Average temperature: {temperature_results[0].mean():.0f}K")
            
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