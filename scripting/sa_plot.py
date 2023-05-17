import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
plt.style.use(Path(__file__).parents[1] / "calibration_tool/data_config/matplotlib.rc")
data_dir = "/home/bookboi/Nextcloud/1_Docs/4_Master/20_Studienarbeit_paper/results/27_2_2023_SA_sobol/"
data_path = Path(data_dir)
param_keys = "speedFactor,minGap,accel,decel,startupDelay,tau,delta,tpreview,tPersDrive,tPersEstimate,treaction,ccoolness,jerkmax,epsilonacc,taccmax,Mflatness,Mbegin"
param_keys = param_keys.split(",")
type1_files = [data_path / "sensitivity_resultsS1.csv",
               data_path / "sensitivity_resultsST.csv",
               data_path / "sensitivity_resultsS2.csv"]
figs = []
def plot_bar(results):
    names = results["parameterName"].values
    sens_index = results.iloc[:, 0].values
    sens_index_error = results.iloc[:, 1].values
    fig, ax = plt.subplots()
    x_range = np.arange(len(results))
    ax.set_title(list(results.columns)[0])
    ax.bar(x_range, sens_index, yerr=sens_index_error, align="center")
    ax.set_xticks(x_range)
    ax.set_xticklabels(names, rotation=45, ha="right")
    fig.tight_layout()
    return fig

for results_file in type1_files:
    results = pd.read_csv(results_file)
    fig = plot_bar(results)
    figs.append(fig)
    plt.show()
    