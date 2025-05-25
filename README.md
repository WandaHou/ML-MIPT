# Figure Generation Documentation

This directory contains the code and data for generating figures for both 1D and 2D cases. The figures are organized into two subdirectories: `1d_figure/` and `2d_figure/`.

## 1D Case

The 1D case analyzes three key quantum measures:
1. Negativity
2. QC Entropy
3. Coherent Information

For each measure, we compare four different methods:
- MPS (Trainable Matrix Product State, aka Born Machine)
- TFM (Transformer)
- Post-selection
- Decoding with different depolarization parameters (ε)

### Decoding Parameters
The decoding method is tested with different depolarization parameters:
- ε = 0.1, 0.2, 0.3, 0.4, 0.5
- Note: ε = 0.0 (undepolarized case) is excluded from QC Entropy and Coherent Information plots due to significantly higher entropy values

### Plotting Style
- Each method uses distinct markers and colors
- Decode results are shown with dashed lines
- Error bars are included for all measurements
- Legend is placed outside the plot for better visibility
- A text box explains the exclusion of ε = 0.0 where applicable

## 2D Case

The 2D case analyzes the same three quantum measures but in two different contexts:
1. Variation with theta (angle parameter)
2. Variation with distance (d = 3,4,5,6)

### Methods Compared
- TFM (Transformer)
- Decoding with different depolarization parameters (ε)

### Decoding Parameters
Same as 1D case:
- ε = 0.1, 0.2, 0.3, 0.4, 0.5
- ε = 0.0 excluded from QC Entropy and Coherent Information plots

### Plotting Style
- Consistent with 1D case
- Different markers and colors for each epsilon value
- Dashed lines for decode results
- Error bars included
- Legend outside plot
- Text box explaining ε = 0.0 exclusion where applicable

## Data Structure

### 1D Data Files
- `mps_N_values.pt`, `gpt_N_values.pt`, `ps_N_values.pt`, `decode_N_values.pt` (Negativity)
- `mps_S_values.pt`, `gpt_S_values.pt`, `ps_S_values.pt`, `decode_S_values.pt` (QC Entropy)
- `mps_I_values.pt`, `gpt_I_values.pt`, `ps_I_values.pt`, `decode_I_values.pt` (Coherent Information)

### 2D Data Files
- `decode_N_values_d={3,4,5,6}.pt`, `gpt_N_values_d={3,4,5,6}.pt` (Negativity)
- `decode_S_values_d={3,4,5,6}.pt`, `gpt_S_values_d={3,4,5,6}.pt` (QC Entropy)
- `decode_I_values_d={3,4,5,6}.pt`, `gpt_I_values_d={3,4,5,6}.pt` (Coherent Information)

## Plotting Code

The plotting code is organized in Jupyter notebooks:
- `1d_figure/1dfigure.ipynb` for 1D case
- `2d_figure/2dfigure.ipynb` for 2D case

Each notebook contains code for:
1. Loading and processing data
2. Calculating means and standard deviations
3. Generating plots with consistent styling
4. Adding appropriate labels and legends

## Data Availability

Due to the large size of the data files, they are not included in this repository. The data files can be accessed through the following Google Drive link:

[Google Drive Link to Data Files](https://drive.google.com/drive/folders/your-folder-id)

The data is organized in two zip files:
1. `1d_data`: Contains all 1D case data files
2. `2d_data`: Contains all 2D case data files

<mark>Please contact the authors for access to the data files.</mark>
