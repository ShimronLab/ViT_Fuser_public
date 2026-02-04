import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

## usage example:
# python3 Plot_comparisons_BiasVariance.py --r 3 --snr 5 --slice_idx 20 --mask 1D --subject 1

# -----------------------------------------------------------------------------
# ARGUMENT PARSING
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Plot Bias-Variance Concatenation")
parser.add_argument('--r', type=int, default=3, help='Acceleration factor')
parser.add_argument('--snr', type=int, default=10, help='SNR dB')
parser.add_argument('--slice_idx', type=int, default=15, help='Slice index')
parser.add_argument('--mask', type=str, default='1D', help='Mask type')
parser.add_argument('--subject', type=str, default='1', help='Subject number')

args = parser.parse_args()
R = args.r
SNR = args.snr
SLICE = args.slice_idx
MASK = args.mask
subject_num  = args.subject

print(f"Generating Plot with Mean Bias: R={R}, SNR={SNR}dB, Subject={subject_num}, Slice={SLICE}")

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
base_path = '../bias_variance_analysis'

models_to_compare = [
    ('MoDL',    'modl'),
    ('DPS',     'DPS'),
    ('DPS-PI',  'DPS_PI'),
    ('ViT',     'ViT'),
    ('ViTFuser', 'ViTFuser'),
]

# Amplification Config (Optional - kept from previous step)
models_to_amplify = ['MoDL', 'ViT', 'ViTFuser', 'DPS']
amp_factor = [1, 1] #10e4

bias_list = []
std_list = []
labels = []
s_vals_list = []

global_bias_max = 0
global_std_max = 0
ground_truth_img = None 

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
print("Loading data...")
for label, model_type in models_to_compare:
    folder_name = f"{model_type}_SNR{SNR}_R{R}_{MASK}"
    full_folder = f"{folder_name}_slice{SLICE}"
    file_name = f"results_{model_type}_R{R}_SNR{SNR}_sub{subject_num}_slice{SLICE}.mat"
    
    full_path = os.path.join(base_path, full_folder, file_name)
    
    if not os.path.exists(full_path):
        print(f"❌ Missing: {full_path}")
        continue
    
    print(f"✅ Loaded: {label}")
    data = sio.loadmat(full_path)
    
    # Handle GT
    if ground_truth_img is None:
        gt_raw = data['ground_truth']
        ground_truth_img = np.abs(np.squeeze(gt_raw))
            
    # Load Maps
    bias = np.abs(data['bias_map']) 
    std = data['std_map'] 

    # <--- NEW: Calculate Mean Bias
    mean_bias_val = np.mean(bias)

    # Apply Variance Amplification (if needed)
    if label in models_to_amplify:
        if label == 'DPS':
            amp = amp_factor[1]
        elif label == 'DPS-PI':
            amp = 1
        else:
            amp = amp_factor[0]
        std = std * amp
        std_label = f"(x{int(amp)})" 
    else:
        std_label = ""

    # <--- NEW: Update Label to include Mean Bias
    # Format: "ModelName\nMean: 0.0123"
    final_label = f"{label} {std_label}\nMean Bias: {mean_bias_val:.4f}"

    bias_list.append(bias)
    std_list.append(std)
    labels.append(final_label) 
    
    if 'singular_values' in data:
        s_vals_list.append((label, data['singular_values']))
    else:
        s_vals_list.append((label, np.zeros(1)))

    # Track max
    global_bias_max = max(global_bias_max, np.max(bias))
    global_std_max = max(global_std_max, np.max(std))

if not bias_list or ground_truth_img is None:
    print("No data found.")
    exit()

# -----------------------------------------------------------------------------
# CALCULATE SCALES & CONCATENATE
# -----------------------------------------------------------------------------
max_gt_val = np.max(ground_truth_img)
# GT is NO LONGER in the list
bias_strip = np.concatenate(bias_list, axis=1) 
std_strip = np.concatenate(std_list, axis=1)

# Ensure the bias plot covers the max of both GT and Bias for fair comparison
#unified_bias_row_max = max(max_gt_val, global_bias_max)

# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------
# Use GridSpec: Left column (GT), Right column (Strips)
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 2, width_ratios=[1, len(models_to_compare)], wspace=0.05, hspace=0.1)

# --- Left Column: Ground Truth (Spanning both rows) ---
ax_gt = fig.add_subplot(gs[:, 0])
im_gt = ax_gt.imshow(ground_truth_img, cmap='gray', vmin=0, vmax=max_gt_val)
ax_gt.set_title("Ground Truth", fontsize=14, fontweight='bold', pad=10)
ax_gt.axis('off')
# Colorbar for GT (bottom of GT panel)
cbar_gt = plt.colorbar(im_gt, ax=ax_gt, orientation='horizontal', fraction=0.05, pad=0.05)
cbar_gt.set_label('Magnitude', fontsize=10)


# --- Right Column, Top Row: Bias Maps ---
ax_bias = fig.add_subplot(gs[0, 1])
im_bias = ax_bias.imshow(bias_strip, cmap='gray', vmin=0, vmax=global_bias_max)
ax_bias.set_ylabel("Bias Maps", fontsize=14, fontweight='bold')
ax_bias.set_xticks([]); ax_bias.set_yticks([])

# Add Model Labels to Bias Strip
img_width = bias_list[0].shape[1]
for idx, label in enumerate(labels):
    # Fix specific label names if needed
    if "DPS " in label: label = label.replace("DPS ", "DPS (DC) ")
    
    # Calculate center position (GT is no longer index 0, so logic simplifies)
    center_pos = idx * img_width + (img_width / 2)
    ax_bias.text(center_pos, -5, label, ha='center', va='bottom', fontsize=10, fontweight='bold')

# Colorbar for Bias
cbar_bias = plt.colorbar(im_bias, ax=ax_bias, fraction=0.015, pad=0.01)
cbar_bias.set_label('Bias Mag', rotation=270, labelpad=15)


# --- Right Column, Bottom Row: Variance Maps ---
ax_std = fig.add_subplot(gs[1, 1])
im_std = ax_std.imshow(std_strip, cmap='gray', vmin=0, vmax=global_std_max)
ax_std.set_ylabel("Std Dev", fontsize=14, fontweight='bold')
ax_std.set_xticks([]); ax_std.set_yticks([])

# Colorbar for Variance
cbar_std = plt.colorbar(im_std, ax=ax_std, fraction=0.015, pad=0.01)
cbar_std.set_label('Std Dev', rotation=270, labelpad=15)

plt.suptitle(f"Bias (with Mean) vs Variance (Amplified) R={R}, SNR={SNR}dB", fontsize=16, y=0.95)

save_name = f'Comparison_MeanBias_R{R}_SNR{SNR}_sub{subject_num}_slice{args.slice_idx}.png'
save_path = os.path.join(base_path, save_name)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {save_path}")
plt.show()
# -----------------------------------------------------------------------------
# PLOTTING 2: EIGENVALUE DECAY
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
markers = ['o', 's', '^', 'x', '*', 'D']
linestyles = ['-', '--', '-.', ':', '-', '--']

for i, (label, s_vals) in enumerate(s_vals_list):
    if label == "DPS":
        label = "DPS (DC. Atn)"
    if label == "DPS-PI":
        label = "Prior-DPS (DC. Atn)"
    s_vals = s_vals.flatten()
    num_to_plot = min(3, len(s_vals))
    x_axis = np.arange(1, num_to_plot + 1)
    
    if num_to_plot > 0:
        eigenvalues = (s_vals[:num_to_plot] ** 2)
        plt.plot(x_axis, eigenvalues, 
                 marker=markers[i % len(markers)], 
                 linestyle=linestyles[i % len(linestyles)], 
                 linewidth=2, 
                 label=label)

#plt.yscale('log')
plt.xlabel('Principal Component Index', fontsize=12, fontweight='bold')
plt.ylabel('Eigenvalue', fontsize=12, fontweight='bold')
plt.title(f'Scree Plot: Variance Decay (R={R}, SNR={SNR}dB)', fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()

save_name_pca = f'Comparison_Eigenvalues_R{R}_SNR{SNR}_sub{args.subject}_slice{args.slice_idx}.png'
plt.savefig(os.path.join(base_path, save_name_pca), dpi=300)
print(f"✅ Eigenvalue plot saved to: {os.path.join(base_path, save_name_pca)}")

plt.show()