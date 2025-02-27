import os
import pickle as pkl
import matplotlib.pyplot as plt

data_dir = "./"
print("[LOG] Simulation started under:", data_dir)


def trial_exists(trial_id):
    return os.path.exists(os.path.join(data_dir, f"{trial_id}.pkl"))


def load_trial(trial_id):
    with open(os.path.join(data_dir, f"{trial_id}.pkl"), "rb") as in_file:
        return pkl.load(in_file)


def save_trial(trial_id, data):
    with open(os.path.join(data_dir, f"{trial_id}.pkl"), "wb") as out_file:
        pkl.dump(data, out_file)


def mpl_line_plot(df, x, y, title, x_label, y_label, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    x_data = df[x]
    for col in y:
        y_data = df[col]
        ax.plot(x_data, y_data, label=col)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if len(y) > 1:
        ax.legend()
    return fig
