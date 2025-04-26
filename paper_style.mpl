# Matplotlib style for general scientific plots

# Set color cycle
axes.prop_cycle : cycler('color', ["0173B2", "DE8F05", "029E73", "D55E00", "CC78BC", "CA9161", "FBAFE4", "949494", "ECE133", "56B4E9"])

# Set default figure size: Determined by the journal width
figure.figsize   : 3.5, 2.625 #Golden ratio
figure.dpi       : 500

# Use serif fonts
font.serif  : Source Han Sans TW
font.family : sans-serif

# Font sizes
axes.labelsize: 7
xtick.labelsize: 7
ytick.labelsize: 7
legend.fontsize: 7
font.size: 7

# Set x axis
xtick.direction : out
xtick.major.size : 3.5
xtick.major.width : 0.55
xtick.minor.size : 2.0
xtick.minor.width : 0.5
xtick.minor.visible :   True
xtick.top : False

# Set y axis
ytick.direction : out 
ytick.major.size : 3.5
ytick.major.width : 0.55
ytick.minor.size : 2.0
ytick.minor.width : 0.5
ytick.minor.visible :   True
ytick.right : False

# Set line widths
axes.linewidth : 0.5
grid.linewidth : 0.5
lines.linewidth : 1.5

# Remove legend frame
legend.frameon : False

# Always save as 'tight'
savefig.bbox : tight
savefig.pad_inches : 0.05

# Use LaTeX for math formatting text.usetex : True
text.latex.preamble : \usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath}