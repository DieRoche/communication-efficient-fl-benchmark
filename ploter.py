import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, EngFormatter, FixedLocator
import pandas as pd
from pathlib import Path
import os
from typing import Dict, Tuple, List

# Configuration
EXCEL_FILES_FOLDER = r"C:\Users\drroc\OneDrive\Documents\Repos_copy\compression_FL_final_version"  # Change this to your folder path
TARGET_ROUND_CIFAR10 = 100  # Target round for CIFAR-10
TARGET_ROUND_CIFAR100 = 100   # Target round for CIFAR-100

def load_excel_files(folder_path: str, target_round_cifar10: int, target_round_cifar100: int) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict]:
    """
    Load Excel files from folder and extract data.
    Returns: (data_1, data_2, flops_1, flops_2, cd_server_1, cd_server_2, cd_clients_1, cd_clients_2, total_cd_1, total_cd_2)
    
    Expected filename format: Model_dataset_resnet_nncl.xlsx
    where Model: fedavg, FedQClip, flocora, Sparsyfed, Sparsyfed+CSR
          dataset: cifar10, cifar100
          nn: 10, 50, 100
          cl: resnet (fixed)
    """
    data_cifar10 = {}
    data_cifar100 = {}
    flops_cifar10 = {}
    flops_cifar100 = {}
    cd_server_cifar10 = {}
    cd_server_cifar100 = {}
    cd_clients_cifar10 = {}
    cd_clients_cifar100 = {}
    total_cd_cifar10 = {}
    total_cd_cifar100 = {}
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist. Using empty data.")
        return data_cifar10, data_cifar100, flops_cifar10, flops_cifar100, cd_server_cifar10, cd_server_cifar100, cd_clients_cifar10, cd_clients_cifar100, total_cd_cifar10, total_cd_cifar100
    
    # Find all Excel files
    excel_files = list(Path(folder_path).glob("*.xlsx"))
    
    for excel_file in excel_files:
        filename = excel_file.stem  # Remove .xlsx extension
        
        # Parse filename: Model_dataset_resnet_nncl
        parts = filename.split("_")
        if len(parts) < 4:
            print(f"Skipping {filename}: does not match expected format")
            continue
        
        model = parts[0]  # fedavg, FedQClip, flocora, Sparsyfed
        dataset = parts[1]  # cifar10 or cifar100
        # parts[2] is "resnet" (fixed)
        nncl_str = parts[3]  # e.g., "10resnet" or "100resnet"
        
        # Extract number of clients (nn)
        nn_clients = int(''.join(filter(str.isdigit, nncl_str)))
        label_key = f"{model}_{nn_clients}cl"
        
        # Determine target round based on dataset
        target_round = target_round_cifar10 if dataset.lower() == "cifar10" else target_round_cifar100
        
        try:
            # Read accuracy data from acc_servers_highest sheet
            acc_df = pd.read_excel(excel_file, sheet_name="acc_servers_highest")
            # Read traffic data from overall_traffic sheet
            traffic_df = pd.read_excel(excel_file, sheet_name="overall_traffic")
            
            # Read FLOPs sheets
            round_flops_df = pd.read_excel(excel_file, sheet_name="round_flops")
            comp_server_df = pd.read_excel(excel_file, sheet_name="compression_flops_server")
            decomp_server_df = pd.read_excel(excel_file, sheet_name="decompression_flops_server")
            comp_clients_df = pd.read_excel(excel_file, sheet_name="compression_flops_clients")
            decomp_clients_df = pd.read_excel(excel_file, sheet_name="decompression_flops_clients")
            
            # Extract data for the target round
            acc_data = extract_round_data(acc_df, target_round, cumulative=False)
            traffic_data = extract_round_data(traffic_df, target_round, cumulative=True)
            
            # Calculate FLOPs metrics
            round_flops_data = calculate_avg_flops_across_rounds(round_flops_df, exclude_round_0=False)
            cd_server_data = calculate_sum_flops_across_rounds(comp_server_df, decomp_server_df, exclude_round_0=True)
            cd_clients_data = calculate_sum_flops_across_rounds(comp_clients_df, decomp_clients_df, exclude_round_0=True)
            
            # Calculate total C/D (server + clients) - need to combine rep values
            if cd_server_data and cd_clients_data:
                total_cd_reps = [s + c for s, c in zip(cd_server_data["reps"], cd_clients_data["reps"])]
                total_cd_data = {
                    "reps": total_cd_reps,
                    "mean": np.median(total_cd_reps)
                }
            else:
                total_cd_data = None
            
            if acc_data is not None and traffic_data is not None:
                # Multiply accuracy values by 100 (convert from decimal to percentage)
                acc_reps_scaled = [val * 100 for val in acc_data["reps"]]
                acc_mean_scaled = acc_data["mean"] * 100
                
                data_entry = {
                    "semi_transparent": list(zip(traffic_data["reps"], acc_reps_scaled)),
                    "full": (traffic_data["mean"], acc_mean_scaled)
                }
                
                # Create entries for FLOPs plots
                if round_flops_data:
                    flops_entry = {
                        "semi_transparent": list(zip(round_flops_data["reps"], acc_reps_scaled)),
                        "full": (round_flops_data["mean"], acc_mean_scaled)
                    }
                else:
                    flops_entry = None
                
                if cd_server_data:
                    cd_server_entry = {
                        "semi_transparent": list(zip(cd_server_data["reps"], acc_reps_scaled)),
                        "full": (cd_server_data["mean"], acc_mean_scaled)
                    }
                else:
                    cd_server_entry = None
                
                if cd_clients_data:
                    cd_clients_entry = {
                        "semi_transparent": list(zip(cd_clients_data["reps"], acc_reps_scaled)),
                        "full": (cd_clients_data["mean"], acc_mean_scaled)
                    }
                else:
                    cd_clients_entry = None
                
                if total_cd_data:
                    total_cd_entry = {
                        "semi_transparent": list(zip(total_cd_data["reps"], acc_reps_scaled)),
                        "full": (total_cd_data["mean"], acc_mean_scaled)
                    }
                else:
                    total_cd_entry = None
                
                if dataset.lower() == "cifar10":
                    data_cifar10[label_key] = data_entry
                    if flops_entry:
                        flops_cifar10[label_key] = flops_entry
                    if cd_server_entry:
                        cd_server_cifar10[label_key] = cd_server_entry
                    if cd_clients_entry:
                        cd_clients_cifar10[label_key] = cd_clients_entry
                    if total_cd_entry:
                        total_cd_cifar10[label_key] = total_cd_entry
                elif dataset.lower() == "cifar100":
                    data_cifar100[label_key] = data_entry
                    if flops_entry:
                        flops_cifar100[label_key] = flops_entry
                    if cd_server_entry:
                        cd_server_cifar100[label_key] = cd_server_entry
                    if cd_clients_entry:
                        cd_clients_cifar100[label_key] = cd_clients_entry
                    if total_cd_entry:
                        total_cd_cifar100[label_key] = total_cd_entry
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return data_cifar10, data_cifar100, flops_cifar10, flops_cifar100, cd_server_cifar10, cd_server_cifar100, cd_clients_cifar10, cd_clients_cifar100, total_cd_cifar10, total_cd_cifar100

def extract_round_data(df: pd.DataFrame, target_round: int, cumulative: bool = False) -> Dict:
    """
    Extract data for a specific round from a DataFrame.
    First column is 'round' or similar identifier.
    Other columns are individual replicates (rep 1, rep 2, etc.).
    
    If cumulative=True: returns sum of all rows up to target_round (for traffic data)
    If cumulative=False: returns data at target_round only (for accuracy data)
    
    Returns dict with 'reps' (list of values) and 'mean' (mean of all reps).
    """
    # Get first column name (round column)
    round_col = df.columns[0]
    
    # Filter columns to exclude metadata columns (Mean, Total, Std, etc.)
    exclude_keywords = ["mean", "total", "std", "sum", "min", "max", "average"]
    rep_cols = [col for col in df.columns[1:] 
                if not any(keyword in str(col).lower() for keyword in exclude_keywords)]
    
    if not rep_cols:
        print(f"Warning: No replicate columns found (all columns seem to be metadata)")
        rep_cols = df.columns[1:]
    
    if cumulative:
        # Sum all rows up to and including target_round
        rows = df[df[round_col] <= target_round]
        
        if rows.empty:
            print(f"No data up to round {target_round}")
            return None
        
        # Sum all values for each replicate column
        rep_values = []
        for col in rep_cols:
            col_sum = rows[col].sum()
            if pd.notna(col_sum):
                rep_values.append(float(col_sum))
    else:
        # Find the row with target_round
        row = df[df[round_col] == target_round]
        
        if row.empty:
            print(f"Round {target_round} not found in sheet")
            return None
        
        row = row.iloc[0]
        
        # Get all rep values for replicate columns
        rep_values = []
        for col in rep_cols:
            val = row[col]
            if pd.notna(val):
                rep_values.append(float(val))
    
    if not rep_values:
        return None
    
    return {
        "reps": rep_values,
        "mean": np.median(rep_values)
    }

def calculate_avg_flops_across_rounds(df: pd.DataFrame, exclude_round_0: bool = False) -> Dict:
    """
    Calculate average FLOPs across all rounds for each rep.
    Returns dict with 'reps' (list of average values for each rep) and 'mean' (median of all reps).
    If exclude_round_0=True, excludes round 0 from the calculation.
    """
    round_col = df.columns[0]
    
    # Filter columns to exclude metadata columns
    exclude_keywords = ["mean", "total", "std", "sum", "min", "max", "average"]
    rep_cols = [col for col in df.columns[1:] 
                if not any(keyword in str(col).lower() for keyword in exclude_keywords)]
    
    if not rep_cols:
        return None
    
    # Filter out round 0 if requested
    data_df = df.copy()
    if exclude_round_0:
        data_df = data_df[data_df[round_col] != 0]
    
    if data_df.empty:
        return None
    
    rep_values = []
    for col in rep_cols:
        col_mean = float(pd.to_numeric(data_df[col], errors="coerce").mean())
        if pd.notna(col_mean):
            rep_values.append(col_mean)
    
    if not rep_values:
        return None
    
    return {
        "reps": rep_values,
        "mean": np.median(rep_values)
    }

def calculate_sum_flops_across_rounds(df1: pd.DataFrame, df2: pd.DataFrame, exclude_round_0: bool = False) -> Dict:
    """
    Calculate average of (df1 + df2) across all rounds for each rep.
    Used for compression + decompression flops.
    Returns dict with 'reps' (list of average values for each rep) and 'mean' (median of all reps).
    """
    round_col = df1.columns[0]
    
    # Filter columns to exclude metadata columns
    exclude_keywords = ["mean", "total", "std", "sum", "min", "max", "average"]
    rep_cols = [col for col in df1.columns[1:] 
                if not any(keyword in str(col).lower() for keyword in exclude_keywords)]
    
    if not rep_cols:
        return None
    
    # Filter out round 0 if requested
    data_df1 = df1.copy()
    data_df2 = df2.copy()
    if exclude_round_0:
        data_df1 = data_df1[data_df1[round_col] != 0]
        data_df2 = data_df2[data_df2[round_col] != 0]
    
    if data_df1.empty or data_df2.empty:
        return None
    
    rep_values = []
    for col in rep_cols:
        col1_mean = float(pd.to_numeric(data_df1[col], errors="coerce").mean())
        col2_mean = float(pd.to_numeric(data_df2[col], errors="coerce").mean())
        if pd.notna(col1_mean) and pd.notna(col2_mean):
            rep_values.append(col1_mean + col2_mean)
    
    if not rep_values:
        return None
    
    return {
        "reps": rep_values,
        "mean": np.median(rep_values)
    }

# Load data from Excel files with separate target rounds
data_1, data_2, flops_1, flops_2, cd_server_1, cd_server_2, cd_clients_1, cd_clients_2, total_cd_1, total_cd_2 = load_excel_files(EXCEL_FILES_FOLDER, TARGET_ROUND_CIFAR10, TARGET_ROUND_CIFAR100)

def get_axis_limits(data_dict):
    """
    Calculate appropriate axis limits from data.
    Returns (x_min, x_max, y_min, y_max) with some padding.
    """
    x_values = []
    y_values = []
    
    for vals in data_dict.values():
        if not vals:
            continue
        
        # Extract semi-transparent values
        for x, y in vals.get("semi_transparent", []):
            if (x, y) != (0, 0) and np.isfinite(x) and np.isfinite(y) and x > 0 and y >= 0:
                x_values.append(x)
                y_values.append(y)
        
        # Extract full (mean) values
        f = vals.get("full", (0, 0))
        if f != (0, 0) and np.isfinite(f[0]) and np.isfinite(f[1]) and f[0] > 0 and f[1] >= 0:
            x_values.append(f[0])
            y_values.append(f[1])
    
    if not x_values or not y_values:
        return (1, 1000, 0, 100)
    
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    
    # Ensure valid ranges
    if x_min <= 0 or not np.isfinite(x_min):
        x_min = 1
    if x_max <= 0 or not np.isfinite(x_max):
        x_max = 1000
    if x_min >= x_max:
        x_min, x_max = 1, 1000
    
    # Add 10% padding
    x_padding = (x_max - x_min) * 0.1 if x_max > x_min else 1
    y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 1
    
    return (x_min * 0.9, x_max * 1.1, y_min - y_padding, y_max + y_padding)

def get_global_axis_limits_for_communication_cost_plots(data_1, data_2):
    """
    Calculate global axis limits for communication cost plots (plots 1 and 6).
    X-axis: Global across all datasets
    Y-axis: Separate for CIFAR-10 and CIFAR-100
    Returns ((x_min, x_max), (y_min_1, y_max_1), (y_min_2, y_max_2))
    """
    x_values = []
    y_values_1 = []  # CIFAR-10
    y_values_2 = []  # CIFAR-100
    
    # Collect X values (communication cost, global) and Y values (accuracy, dataset-specific)
    for vals in data_1.values():
        if not vals:
            continue
        f = vals.get("full", (0, 0))
        if f != (0, 0) and np.isfinite(f[0]) and np.isfinite(f[1]) and f[0] > 0 and f[1] >= 0:
            x_values.append(f[0])
            y_values_1.append(f[1])
    
    for vals in data_2.values():
        if not vals:
            continue
        f = vals.get("full", (0, 0))
        if f != (0, 0) and np.isfinite(f[0]) and np.isfinite(f[1]) and f[0] > 0 and f[1] >= 0:
            x_values.append(f[0])
            y_values_2.append(f[1])
    
    # Calculate global X limits
    if not x_values:
        x_min, x_max = 1, 1000
    else:
        x_min, x_max = min(x_values), max(x_values)
        if x_min <= 0 or not np.isfinite(x_min):
            x_min = 1
        if x_max <= 0 or not np.isfinite(x_max):
            x_max = 1000
        if x_min >= x_max:
            x_min, x_max = 1, 1000
    
    # Calculate Y limits for CIFAR-10
    if not y_values_1:
        y_min_1, y_max_1 = 0, 100
    else:
        y_min_1, y_max_1 = min(y_values_1), max(y_values_1)
        y_padding_1 = (y_max_1 - y_min_1) * 0.12 if y_max_1 > y_min_1 else 1
        y_min_1 = y_min_1 - y_padding_1
        y_max_1 = y_max_1 + y_padding_1
    
    # Calculate Y limits for CIFAR-100
    if not y_values_2:
        y_min_2, y_max_2 = 0, 100
    else:
        y_min_2, y_max_2 = min(y_values_2), max(y_values_2)
        y_padding_2 = (y_max_2 - y_min_2) * 0.12 if y_max_2 > y_min_2 else 1
        y_min_2 = y_min_2 - y_padding_2
        y_max_2 = y_max_2 + y_padding_2
    
    # Add 10% more room for bubble plot markers
    x_min = x_min * 0.8
    x_max = x_max * 1.2
    
    return ((x_min, x_max), (y_min_1, y_max_1), (y_min_2, y_max_2))

def get_global_axis_limits_for_cd_plots(cd_server_1, cd_server_2, cd_clients_1, cd_clients_2, total_cd_1, total_cd_2):
    """
    Calculate global axis limits for all C/D FLOPs plots (plots 3, 4, 5).
    X-axis: Global across all datasets
    Y-axis: Separate for CIFAR-10 and CIFAR-100
    Includes extra padding so large markers are not clipped.
    Returns ((x_min, x_max), (y_min_1, y_max_1), (y_min_2, y_max_2))
    """
    x_values = []
    y_values_1 = []  # CIFAR-10
    y_values_2 = []  # CIFAR-100
    
    # Collect X values (global) and Y values (dataset-specific)
    for dataset, y_list in [(cd_server_1, y_values_1), (cd_clients_1, y_values_1), (total_cd_1, y_values_1)]:
        for vals in dataset.values():
            if not vals:
                continue
            f = vals.get("full", (0, 0))
            if f != (0, 0) and np.isfinite(f[0]) and np.isfinite(f[1]) and f[0] > 0 and f[1] >= 0:
                x_values.append(f[0])
                y_list.append(f[1])
    
    for dataset, y_list in [(cd_server_2, y_values_2), (cd_clients_2, y_values_2), (total_cd_2, y_values_2)]:
        for vals in dataset.values():
            if not vals:
                continue
            f = vals.get("full", (0, 0))
            if f != (0, 0) and np.isfinite(f[0]) and np.isfinite(f[1]) and f[0] > 0 and f[1] >= 0:
                x_values.append(f[0])
                y_list.append(f[1])
    
    # Calculate global X limits
    if not x_values:
        x_min, x_max = 1, 1000
    else:
        x_min, x_max = min(x_values), max(x_values)
        if x_min <= 0 or not np.isfinite(x_min):
            x_min = 1
        if x_max <= 0 or not np.isfinite(x_max):
            x_max = 1000
        if x_min >= x_max:
            x_min, x_max = 1, 1000
    
    # Calculate Y limits for CIFAR-10
    if not y_values_1:
        y_min_1, y_max_1 = 0, 100
    else:
        y_min_1, y_max_1 = min(y_values_1), max(y_values_1)
        y_padding_1 = (y_max_1 - y_min_1) * 0.12 if y_max_1 > y_min_1 else 1
        y_min_1 = y_min_1 - y_padding_1
        y_max_1 = y_max_1 + y_padding_1
    
    # Calculate Y limits for CIFAR-100
    if not y_values_2:
        y_min_2, y_max_2 = 0, 100
    else:
        y_min_2, y_max_2 = min(y_values_2), max(y_values_2)
        y_padding_2 = (y_max_2 - y_min_2) * 0.12 if y_max_2 > y_min_2 else 1
        y_min_2 = y_min_2 - y_padding_2
        y_max_2 = y_max_2 + y_padding_2
    
    # Add 10% more room than the previous C/D limits so large bubbles fit.
    x_min = x_min * 0.8
    x_max = x_max * 1.2
    
    return ((x_min, x_max), (y_min_1, y_max_1), (y_min_2, y_max_2))

# --- Helpers ---
def has_point(v):
    f = v.get("full", (0, 0))
    st = v.get("semi_transparent", [])
    return (f != (0, 0)) or (len(st) > 0)

def split_label(label):
    # Example: Fedavg_10cl -> ("Fedavg", "10cl")
    model, size = label.rsplit("_", 1)
    return model, size

def size_to_number(size_label):
    return int(size_label.replace("cl", ""))

# Optional: cleaner display for legends
MODEL_DISPLAY_NAMES = {
    "fedavg": "FedAvg",
    "fedqclip": "FedQClip",
    "sparsyfed": "SparsyFed",
    "sparsyfed+csr": "SparsyFed+CSR",
    "flocora": "FLoCoRA",
}

def display_model(model):
    return MODEL_DISPLAY_NAMES.get(model.lower(), model)

def display_size(size_label):
    return size_label.replace("cl", " clients")

def get_equidistant_linear_ticks(vmin, vmax, num_ticks=5):
    """
    Generate equidistant ticks on a linear scale that include min and max values.
    """
    return np.linspace(vmin, vmax, num_ticks)

def get_equidistant_log_ticks(vmin, vmax, num_ticks=5):
    """
    Generate equidistant ticks on a log scale that include min and max values.
    """
    return np.logspace(np.log10(vmin), np.log10(vmax), num_ticks)

def get_minor_log_ticks_between_major_ticks(major_ticks, minor_ticks_per_interval=4):
    """
    Generate minor x-axis ticks that follow the same log spacing as the major ticks.
    """
    minor_ticks = []
    for start, end in zip(major_ticks[:-1], major_ticks[1:]):
        if start <= 0 or end <= 0:
            continue
        interval_ticks = np.logspace(
            np.log10(start),
            np.log10(end),
            minor_ticks_per_interval + 2
        )
        minor_ticks.extend(interval_ticks[1:-1])
    return minor_ticks

def exponent_formatter(x, pos):
    """
    Custom formatter for x-axis tick labels showing mantissa and exponent notation.
    Example: 1e12 -> "1e12", 3.16e12 -> "3.2e12", 1e13 -> "1e13"
    """
    if x <= 0 or np.isnan(x):
        return ""
    
    # Get the exponent (floor of log10)
    exp = int(np.floor(np.log10(x)))
    
    # Calculate mantissa (should be between 1 and 10)
    mantissa = x / (10 ** exp)
    
    # Format mantissa - show as integer if close to one, otherwise with 1 decimal place
    if abs(mantissa - round(mantissa)) < 0.05:
        # Close to integer
        return f"{int(round(mantissa))}e{exp}"
    else:
        # Show with 1 decimal place, removing trailing zeros
        mantissa_str = f"{mantissa:.1f}".rstrip('0').rstrip('.')
        return f"{mantissa_str}e{exp}"

# Define marker by model
marker_map = {
    "fedavg": "o",      # circle
    "fedqclip": "s",    # square
    "sparsyfed": "^",   # triangle
    "sparsyfed+csr": "D", # diamond
    "flocora": "X"      # cross
}

model_order = ["fedavg", "fedqclip", "sparsyfed", "sparsyfed+csr", "flocora"]

def marker_for_model(model):
    return marker_map.get(model.lower(), "o")

# Define color by client size
sizes = ["10cl", "50cl", "100cl"]

# Use 3 fixed positions from viridis
cmap = plt.get_cmap("viridis")
size_colors = cmap([0.15, 0.50, 0.85])  # purple-ish, green-ish, yellow-ish
color_map = {size: size_colors[i] for i, size in enumerate(sizes)}

# Define color by model using viridis (0.15-0.85 range)
model_cmap = plt.get_cmap("viridis")
model_positions = np.linspace(0.15, 0.85, len(model_order))
model_color_map = {model: model_cmap(pos) for model, pos in zip(model_order, model_positions)}

# Define plot configurations
plot_configs = [
    ("Communication Cost", data_1, data_2, "Communication Cost vs Accuracy"),
    ("Round FLOPs", flops_1, flops_2, "Round FLOPs vs Accuracy"),
    ("C/D FLOPs Server", cd_server_1, cd_server_2, "C/D FLOPs Server vs Accuracy"),
    ("C/D FLOPs Clients", cd_clients_1, cd_clients_2, "C/D FLOPs Clients vs Accuracy"),
    ("Total C/D FLOPs", total_cd_1, total_cd_2, "Total C/D FLOPs vs Accuracy"),
]

plot_titles = [
    ("CIFAR-10", "CIFAR-100"),
    ("CIFAR-10 - Round FLOPs", "CIFAR-100 - Round FLOPs"),
    ("CIFAR-10 - C/D FLOPs Server", "CIFAR-100 - C/D FLOPs Server"),
    ("CIFAR-10 - C/D FLOPs Clients", "CIFAR-100 - C/D FLOPs Clients"),
    ("CIFAR-10 - Total C/D FLOPs", "CIFAR-100 - Total C/D FLOPs"),
]

def plot_metric_row(axes_row, plot_idx, x_label, dataset_1, dataset_2, global_limits=None):
    # Left subplot (data_1 / CIFAR-10)
    ax1 = axes_row[0]
    for lab, vals in dataset_1.items():
        if not has_point(vals):
            continue

        model, size = split_label(lab)
        color = color_map[size]
        marker = marker_for_model(model)

        f = vals.get("full", (0, 0))
        st = vals.get("semi_transparent", [])

        # Plot semi-transparent individual replicates
        for x, y in st:
            if (x, y) != (0, 0):
                ax1.scatter(
                    x, y,
                    s=80,
                    color=color,
                    alpha=0.3,  # Semi-transparent
                    marker=marker
                )

        # Plot full (median) value - completely solid, no transparency
        if f != (0, 0):
            ax1.scatter(
                f[0], f[1],
                s=80,
                color=color,
                alpha=1.0,  # Completely solid
                marker=marker,
                edgecolors='black',
                linewidths=1
            )
    
    ax1.set_xscale('log')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(plot_titles[plot_idx][0])
    ax1.grid(False)

    # Calculate and set dynamic limits for data_1
    if global_limits:
        x_limits, y_limits_1, y_limits_2 = global_limits
        x_min1, x_max1 = x_limits
        y_min1, y_max1 = y_limits_1
    else:
        x_min1, x_max1, y_min1, y_max1 = get_axis_limits(dataset_1)
    
    ax1.set_xlim(left=x_min1, right=x_max1)
    ax1.set_ylim(bottom=y_min1, top=y_max1)

    # Set equidistant ticks that explicitly show min and max
    x_ticks_1 = get_equidistant_log_ticks(x_min1, x_max1, num_ticks=5)
    y_ticks_1 = get_equidistant_linear_ticks(y_min1, y_max1, num_ticks=5)
    ax1.set_xticks(x_ticks_1)
    ax1.set_yticks(y_ticks_1)

    # Format tick labels: exponent notation for x-axis, integers for y-axis
    ax1.xaxis.set_major_formatter(FuncFormatter(exponent_formatter))
    ax1.set_yticklabels([f'{int(y)}' for y in y_ticks_1])

    # Enable minor ticks
    ax1.minorticks_on()
    ax1.xaxis.set_minor_locator(FixedLocator(get_minor_log_ticks_between_major_ticks(x_ticks_1)))

    # Configure major ticks
    ax1.tick_params(axis='x', which='major', labeltop=False, labelbottom=True, top=True, bottom=True, length=6, direction='in')
    ax1.tick_params(axis='y', which='major', labelleft=True, labelright=False, left=True, right=True, length=6, direction='in')

    # Configure minor ticks
    ax1.tick_params(axis='x', which='minor', labeltop=False, labelbottom=False, top=True, bottom=True, length=3, direction='in')
    ax1.tick_params(axis='y', which='minor', labelleft=False, labelright=False, left=True, right=True, length=3, direction='in')

    # Right subplot (data_2 / CIFAR-100)
    ax2 = axes_row[1]
    for lab, vals in dataset_2.items():
        if not has_point(vals):
            continue

        model, size = split_label(lab)
        color = color_map[size]
        marker = marker_for_model(model)

        f = vals.get("full", (0, 0))
        st = vals.get("semi_transparent", [])

        # Plot semi-transparent individual replicates
        for x, y in st:
            if (x, y) != (0, 0):
                ax2.scatter(
                    x, y,
                    s=80,
                    color=color,
                    alpha=0.3,  # Semi-transparent
                    marker=marker
                )

        # Plot full (median) value - completely solid, no transparency
        if f != (0, 0):
            ax2.scatter(
                f[0], f[1],
                s=80,
                color=color,
                alpha=1.0,  # Completely solid
                marker=marker,
                edgecolors='black',
                linewidths=1
            )
    
    ax2.set_xscale('log')
    ax2.set_xlabel(x_label)
    ax2.set_title(plot_titles[plot_idx][1])
    ax2.grid(False)

    # Calculate and set dynamic limits for data_2
    if global_limits:
        x_limits, y_limits_1, y_limits_2 = global_limits
        x_min2, x_max2 = x_limits
        y_min2, y_max2 = y_limits_2
    else:
        x_min2, x_max2, y_min2, y_max2 = get_axis_limits(dataset_2)
    
    ax2.set_xlim(left=x_min2, right=x_max2)
    ax2.set_ylim(bottom=y_min2, top=y_max2)

    # Set equidistant ticks that explicitly show min and max
    x_ticks_2 = get_equidistant_log_ticks(x_min2, x_max2, num_ticks=5)
    y_ticks_2 = get_equidistant_linear_ticks(y_min2, y_max2, num_ticks=5)
    ax2.set_xticks(x_ticks_2)
    ax2.set_yticks(y_ticks_2)

    # Format tick labels: exponent notation for x-axis, integers for y-axis
    ax2.xaxis.set_major_formatter(FuncFormatter(exponent_formatter))
    ax2.set_yticklabels([f'{int(y)}' for y in y_ticks_2])

    # Enable minor ticks
    ax2.minorticks_on()
    ax2.xaxis.set_minor_locator(FixedLocator(get_minor_log_ticks_between_major_ticks(x_ticks_2)))

    # Configure major ticks
    ax2.tick_params(axis='x', which='major', labeltop=False, labelbottom=True, top=True, bottom=True, length=6, direction='in')
    ax2.tick_params(axis='y', which='major', labelleft=True, labelright=False, left=True, right=True, length=6, direction='in')

    # Configure minor ticks
    ax2.tick_params(axis='x', which='minor', labeltop=False, labelbottom=False, top=True, bottom=True, length=3, direction='in')
    ax2.tick_params(axis='y', which='minor', labelleft=False, labelright=False, left=True, right=True, length=3, direction='in')


def plot_bubble_chart_row(axes_grid, plot_idx, accuracy_data_1, accuracy_data_2, cd_flops_data_1, cd_flops_data_2, comm_cost_data_1, comm_cost_data_2, global_limits=None):
    """
    Plot bubble charts with 3x2 grid (3 rows for client subsets, 2 columns for datasets):
    - X-axis: Communication Cost
    - Y-axis: Accuracy
    - Bubble size: 1 / Total C/D FLOPs
    - All bubbles use circles (no differentiated markers)
    - Reference window for FedAvg in top-left corner of each subplot
    - Uses axis limits from global_limits (same as plot 1)
    """
    def plot_bubble_by_client_and_dataset(ax, accuracy_data, cd_flops_data, comm_cost_data, client_size, dataset_name, global_limits=None):
        """Plot bubbles for a specific client subset and dataset."""
        
        # Collect all bubbles for this client size and dataset (excluding FedAvg)
        bubbles = []  # List of bubble dictionaries
        fedavg_cd_value = None  # Track FedAvg total C/D FLOPs for reference
        fedavg_comm_value = None  # Track FedAvg communication cost for reference
        fedavg_acc_value = None  # Track FedAvg accuracy for reference

        # FedAvg may not have a positive Communication Cost value, so get its
        # total C/D FLOPs directly from the cd_flops_data.
        for lab, cd_vals in cd_flops_data.items():
            model, size = split_label(lab)
            if model.lower() != 'fedavg' or size != client_size:
                continue

            cd_full = cd_vals.get("full", (0, 0))
            if cd_full != (0, 0) and cd_full[0] > 0:
                fedavg_cd_value = cd_full[0]
                break

        # Also try to get FedAvg communication cost from comm_cost_data
        for lab, comm_vals in comm_cost_data.items():
            model, size = split_label(lab)
            if model.lower() != 'fedavg' or size != client_size:
                continue

            comm_full = comm_vals.get("full", (0, 0))
            if comm_full != (0, 0) and comm_full[0] > 0:
                fedavg_comm_value = comm_full[0]
                break

        # And get FedAvg accuracy from accuracy_data (if present)
        for lab, acc_vals in accuracy_data.items():
            model, size = split_label(lab)
            if model.lower() == 'fedavg' and size == client_size:
                f = acc_vals.get("full", (0, 0))
                if f != (0, 0) and f[1] > 0:
                    fedavg_acc_value = f[1]
                break
        
        for lab, acc_vals in accuracy_data.items():
            if not has_point(acc_vals):
                continue
            
            model, size = split_label(lab)
            if size != client_size:
                continue
            
            cd_vals = cd_flops_data.get(lab)
            comm_vals = comm_cost_data.get(lab)
            
            if not cd_vals or not comm_vals:
                continue
            
            acc_full = acc_vals.get("full", (0, 0))
            cd_full = cd_vals.get("full", (0, 0))
            comm_full = comm_vals.get("full", (0, 0))
            
            acc_value = acc_full[1] if acc_full != (0, 0) else 0
            cd_value = cd_full[0] if cd_full != (0, 0) else 0
            comm_value = comm_full[0] if comm_full != (0, 0) else 0
            
            if acc_value > 0 and cd_value > 0 and comm_value > 0:
                # Skip FedAvg - it is shown as the reference inset only.
                if model.lower() != 'fedavg':
                    bubbles.append({
                        'cd': cd_value,
                        'acc': acc_value,
                        'comm': comm_value,
                        'model': model,
                    })
        
        if not bubbles and fedavg_cd_value is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=10)
            return
        
        # Calculate bubble size scaling based on all C/D FLOPs values (including fedavg for reference)
        all_cd_values = [b['cd'] for b in bubbles]
        if fedavg_cd_value is not None:
            all_cd_values.append(fedavg_cd_value)
        
        inverse_cd_values = [1 / value for value in all_cd_values]
        max_inverse_cd = max(inverse_cd_values)
        min_inverse_cd = min(inverse_cd_values)
        inverse_cd_range = (
            max_inverse_cd - min_inverse_cd
            if max_inverse_cd > min_inverse_cd
            else 1
        )

        def scale_bubble_size(cd_value):
            inverse_cd_value = 1 / cd_value
            return (
                50
                + (inverse_cd_value - min_inverse_cd)
                / inverse_cd_range
                * 150
            ) * 1.5  # Scale to 75-300, 1.5x larger
        
        # Plot bubbles with circles and no in-plot model labels.
        for bubble in bubbles:
            bubble_size = scale_bubble_size(bubble['cd'])
            
            # Get color from model color map using viridis
            model = bubble['model']
            color = model_color_map.get(model.lower(), model_cmap(0.5))
            
            # Plot as circle (no edgecolors)
            ax.scatter(
                bubble['comm'], bubble['acc'],
                s=bubble_size,
                color=color,
                alpha=0.8,
                marker='o'
            )
        
        # Add FedAvg reference bubble in top-left corner inset window.
        if fedavg_cd_value is not None:
            ref_bubble_size = scale_bubble_size(fedavg_cd_value)
            
            # Create an inset axes for the reference bubble
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            inset_ax = inset_axes(ax, width="16%", height="16%", loc='upper left', borderpad=0.5)
            
            # Get FedAvg color
            ref_color = model_color_map.get('fedavg', model_cmap(0.15))
            
            # Plot the reference bubble with the same sizing and color rules.
            inset_ax.scatter(
                [0.5],
                [0.5],
                s=ref_bubble_size,
                color=ref_color,
                alpha=0.8,
                marker='o',
                transform=inset_ax.transAxes
            )

            # Keep the inset visible as a small reference window.
            inset_ax.set_xlim(0, 1)
            inset_ax.set_ylim(0, 1)
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_facecolor("white")
            inset_ax.patch.set_alpha(0.85)
            for spine in inset_ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_edgecolor("0.4")
        
        # Draw dotted reference lines for FedAvg: vertical (comm cost) and horizontal (accuracy)
        if fedavg_comm_value is not None and fedavg_acc_value is not None:
            # Use same reference color as inset
            ref_color = model_color_map.get('fedavg', model_cmap(0.15))
            # Vertical line at FedAvg communication cost
            ax.axvline(x=fedavg_comm_value, color=ref_color, linestyle=':', linewidth=1)
            # Horizontal line at FedAvg accuracy
            ax.axhline(y=fedavg_acc_value, color=ref_color, linestyle=':', linewidth=1)
            # Label the intersection point
            try:
                ax.annotate('FedAvg', xy=(fedavg_comm_value, fedavg_acc_value), xytext=(5, 5), textcoords='offset points', color=ref_color, fontsize=9)
            except Exception:
                pass
        
        ax.set_xlabel("Communication Cost")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xscale('log')
        ax.set_title(f"{dataset_name} - {client_size}", pad=5)
        ax.grid(False)
        
        # Apply axis limits from global_limits if provided
        if global_limits:
            x_limits, y_limits_1, y_limits_2 = global_limits
            ax.set_xlim(x_limits[0], x_limits[1])
            # Determine which y limits to use based on dataset
            if dataset_name == "CIFAR-10":
                ax.set_ylim(y_limits_1[0], y_limits_1[1])
            else:
                ax.set_ylim(y_limits_2[0], y_limits_2[1])
            
            # Set matching ticks as in plot 1
            x_ticks = get_equidistant_log_ticks(x_limits[0], x_limits[1], num_ticks=5)
            y_min, y_max = (y_limits_1 if dataset_name == "CIFAR-10" else y_limits_2)
            y_ticks = get_equidistant_linear_ticks(y_min, y_max, num_ticks=5)
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            
            # Format tick labels
            ax.xaxis.set_major_formatter(FuncFormatter(exponent_formatter))
            ax.set_yticklabels([f'{int(y)}' for y in y_ticks])
            
            # Enable minor ticks
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(FixedLocator(get_minor_log_ticks_between_major_ticks(x_ticks)))
        else:
            # Fallback: calculate limits from data
            all_comm_vals = [b['comm'] for b in bubbles]
            all_acc_vals = [b['acc'] for b in bubbles]
            
            if all_comm_vals:
                comm_min, comm_max = min(all_comm_vals), max(all_comm_vals)
                ax.set_xlim(comm_min * 0.8, comm_max * 1.2)
            
            if all_acc_vals:
                acc_min, acc_max = min(all_acc_vals), max(all_acc_vals)
                acc_padding = (acc_max - acc_min) * 0.1
                ax.set_ylim(acc_min - acc_padding, acc_max + acc_padding)
            
            ax.minorticks_on()
        
        # Configure major ticks
        ax.tick_params(axis='x', which='major', labeltop=False, labelbottom=True, top=True, bottom=True, length=6, direction='in')
        ax.tick_params(axis='y', which='major', labelleft=True, labelright=False, left=True, right=True, length=6, direction='in')
        
        # Configure minor ticks
        ax.tick_params(axis='x', which='minor', labeltop=False, labelbottom=False, top=True, bottom=True, length=3, direction='in')
        ax.tick_params(axis='y', which='minor', labelleft=False, labelright=False, left=True, right=True, length=3, direction='in')
    
    # Plot 3x2 grid: 3 rows for client sizes, 2 columns for datasets
    for row_idx, client_size in enumerate(sizes):
        # CIFAR-10 (left column)
        plot_bubble_by_client_and_dataset(
            axes_grid[row_idx, 0],
            accuracy_data_1, cd_flops_data_1, comm_cost_data_1,
            client_size, "CIFAR-10",
            global_limits
        )
        
        # CIFAR-100 (right column)
        plot_bubble_by_client_and_dataset(
            axes_grid[row_idx, 1],
            accuracy_data_2, cd_flops_data_2, comm_cost_data_2,
            client_size, "CIFAR-100",
            global_limits
        )

def make_model_handles(models=None):
    selected_models = models or model_order
    return [
        Line2D(
            [0], [0],
            marker=marker_for_model(model),
            linestyle="None",
            color="black",
            markerfacecolor="black",
            markersize=8,
            label=display_model(model)
        )
        for model in selected_models
        if model.lower() in marker_map
    ]

def add_legends(fig, models=None):
    current_model_handles = make_model_handles(models)
    fig.legend(
        handles=current_model_handles,
        loc="upper center",
        bbox_to_anchor=(0.3, 0.99),
        ncol=len(current_model_handles),
        frameon=True,
        title="Model",
        fontsize=10
    )

    fig.legend(
        handles=size_handles,
        loc="upper center",
        bbox_to_anchor=(0.7, 0.99),
        ncol=len(size_handles),
        frameon=True,
        title="Clients",
        fontsize=10
    )

# --- Legends ---
# Legend 2: sizes -> colors
size_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        color=color_map[size],
        markerfacecolor=color_map[size],
        markersize=8,
        label=display_size(size)
    )
    for size in sizes
]

# --- Plot 1: communication cost only ---
# Calculate global axis limits for communication cost plots (plots 1 and 6)
global_comm_limits = get_global_axis_limits_for_communication_cost_plots(data_1, data_2)

fig_comm, axes_comm = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
plot_metric_row(axes_comm, 0, plot_configs[0][0], plot_configs[0][1], plot_configs[0][2], global_limits=global_comm_limits)
fig_comm.subplots_adjust(top=0.82, hspace=0.3)
add_legends(fig_comm)
fig_comm.savefig("accuracy_vs_communication_cost_stop.pdf", format="pdf", bbox_inches="tight")
plt.show()

# --- Plot 2: round FLOPs shown after closing the communication-cost window ---
fig_round_flops, axes_round_flops = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
plot_metric_row(
    axes_round_flops,
    1,
    plot_configs[1][0],
    plot_configs[1][1],
    plot_configs[1][2]
)
fig_round_flops.subplots_adjust(top=0.82, hspace=0.3)
add_legends(fig_round_flops)
fig_round_flops.savefig("accuracy_vs_round_flops_stop.pdf", format="pdf", bbox_inches="tight")
plt.show()

# --- Plots 3-5: remaining C/D FLOPs figures shown after closing round FLOPs ---
# Calculate global axis limits for all C/D plots (3, 4, 5, 6)
global_cd_limits = get_global_axis_limits_for_cd_plots(
    cd_server_1, cd_server_2, cd_clients_1, cd_clients_2, total_cd_1, total_cd_2
)

#Current axis limits 1.1e8 to 2.4e11 for x-axis, 44 to 76 for y-axis in CIFAR-10, 17 to 37 for y-axis in CIFAR-100

fig_cd_flops, axes_cd_flops = plt.subplots(3, 2, figsize=(16, 16), sharey=False)
for row_idx, (x_label, dataset_1, dataset_2, _) in enumerate(plot_configs[2:]):
    plot_metric_row(axes_cd_flops[row_idx], row_idx + 2, x_label, dataset_1, dataset_2, global_limits=global_cd_limits)

fig_cd_flops.subplots_adjust(top=0.94, hspace=0.3)
add_legends(fig_cd_flops, models=["fedqclip", "sparsyfed", "sparsyfed+csr", "flocora"])
fig_cd_flops.savefig("accuracy_vs_cd_flops_stop.pdf", format="pdf", bbox_inches="tight")
plt.show()



# --- Plot 6: Bubble chart - Accuracy vs Communication Cost (3x2 grid: 3 rows for client subsets, 2 columns for datasets) ---
# Bubble size: 1 / Total C/D FLOPs
fig_bubble, axes_bubble = plt.subplots(3, 2, figsize=(16, 14), sharey=False)
plot_bubble_chart_row(
    axes_bubble,
    5,
    total_cd_1,
    total_cd_2,
    total_cd_1,
    total_cd_2,
    data_1,
    data_2,
    global_limits=global_comm_limits
)
fig_bubble.subplots_adjust(top=0.90, hspace=0.35, wspace=0.25)

# Create a custom legend for model colors using viridis
model_color_handles = []
for model, pos in zip(model_order, model_positions):
    color = model_cmap(pos)
    if model.lower() == 'fedavg':
        # FedAvg uses a dotted line instead of a marker
        model_color_handles.append(
            Line2D([0], [0], color=color, linestyle=':', linewidth=2, label=display_model(model))
        )
    else:
        # Other models use a marker dot
        model_color_handles.append(
            Line2D([0], [0], marker='o', linestyle='None', color=color, markerfacecolor=color, markersize=8, label=display_model(model))
        )

fig_bubble.legend(
    handles=model_color_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.99),
    ncol=len(model_order),
    frameon=True,
    title="Models",
    fontsize=10
)

fig_bubble.savefig("accuracy_vs_communication_cost_bubble_stop.pdf", format="pdf", bbox_inches="tight")
plt.show()
