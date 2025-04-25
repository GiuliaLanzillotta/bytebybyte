import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
sns.set(style="ticks")
sns.set_context("paper",
                rc={"lines.linewidth": 2.5,
                'xtick.labelsize':8,
                'ytick.labelsize':8,
                'lines.markersize' : 8,
                'legend.fontsize': 8,
                'axes.labelsize': 8,
                'legend.handlelength': 1,
                'legend.handleheight':1,})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def viz_heatmap(matrix, title, xaxis, yaxis, savepath):
    increase_factor = len(matrix) // 10
    # Set annotation font size and color
    annot_kws = {"size": 5}

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(5*increase_factor, 4*increase_factor), dpi=200)
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap='plasma', linewidths=.5, annot_kws=annot_kws, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    fig.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()

def viz_lineplots(matrix, title, xaxis, yaxis, savepath, lbl_names=None, ylim=None):
    increase_factor = len(matrix) // 10
    # Set annotation font size and color
    annot_kws = {"size": 5}
    num_samples, seq_len = matrix.shape
    palette = sns.color_palette("plasma_r", num_samples+1)


    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(6*increase_factor, 3*increase_factor), dpi=200)
    for i in range(num_samples):
        if lbl_names == "no": 
            label=None
        else: label = label=f"{lbl_names} {i+1}"
        sns.lineplot(x=range(seq_len), y=matrix[i,:], color=palette[i], label=label, marker='o', linewidth=1, markersize=6)
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)

    if ylim is not None: ax.set_ylim(ylim)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=5, fontsize='small')

    fig.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()