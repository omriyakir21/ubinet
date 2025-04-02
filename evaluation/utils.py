from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d


def multi_exp_data_to_fig(multi_experiments_data: dict, 
                          show_std_in_legend: bool = True, log_scale: bool = True, 
                          resolution: int = 1_000) -> go.Figure:

    # Simulation settings
    num_experiments = len(multi_experiments_data)  # Change this number dynamically

    # Define fixed FPR values for interpolation (log scale)
    if log_scale:
        fpr_interp_values = np.logspace(-3, 0, resolution)  # Avoiding zero
    else:
        fpr_interp_values = np.linspace(0, 1, resolution)

    # Dynamically choose colors from Plotly's palette
    color_palette = px.colors.qualitative.Set1  # Can also use Set2, Plotly, Dark24, etc.
    colors = color_palette * (num_experiments // len(color_palette) + 1)  # Repeat if needed

    fig = go.Figure()

    print('computing rocs')
    exp_idx = 0
    exp_to_auc = dict()
    for exp_name, exp_data in tqdm(multi_experiments_data.items()):
        tpr_interpolated = []
        auc_values = []

        # Compute ROC curves per experiment
        for i in range(len(exp_data)):
            fpr, tpr, _ = roc_curve(exp_data[i][0], exp_data[i][1])
            auc_val = auc(fpr, tpr)
            auc_values.append(auc_val)

            # Interpolate TPR at fixed FPR points
            interp_tpr = interp1d(fpr, tpr, bounds_error=False, fill_value=(0, 1))(fpr_interp_values)
            tpr_interpolated.append(interp_tpr)

        # Compute mean and std of interpolated TPR across curves in this experiment
        tpr_mean = np.mean(tpr_interpolated, axis=0)
        tpr_std = np.std(tpr_interpolated, axis=0)

        # Upper and lower bounds for shaded region
        tpr_upper = np.minimum(tpr_mean + tpr_std, 1)
        tpr_lower = np.maximum(tpr_mean - tpr_std, 0)
        
        # Add shaded std region per experiment
        fig.add_trace(go.Scatter(
            x=np.concatenate([fpr_interp_values, fpr_interp_values[::-1]]),
            y=np.concatenate([tpr_upper, tpr_lower[::-1]]),
            fill='toself',
            # fillcolor=f'rgba{tuple(int(c) for c in px.colors.hex_to_rgb(colors[exp_idx])) + (0.3,)}',
            fillcolor='rgba(' + colors[exp_idx].removeprefix('rgb(')[:-1] + ',0.4)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{exp_name} ±1 Std Dev'
        ))

        # Add mean ROC curve per experiment (bold line)
        mean_auc = np.mean(auc_values)
        exp_to_auc[exp_name] = mean_auc
        fig.add_trace(go.Scatter(x=fpr_interp_values, y=tpr_mean, mode='lines', 
                                line=dict(color=colors[exp_idx], width=3),
                                name=f'{exp_name} Mean (AUC = {mean_auc:.4})'))
        exp_idx += 1

    # Adapted random guess line (log space)
    print('adding random roc')
    if log_scale:
        random_guess_fpr = np.logspace(-3, 0, resolution)  # Avoiding zero
    else:
        random_guess_fpr = np.linspace(0, 1, resolution)
    random_guess_tpr = random_guess_fpr  # y = x, but now correctly sampled

    fig.add_trace(go.Scatter(x=random_guess_fpr, y=random_guess_tpr, mode='lines',
                            line=dict(dash='dash', color='black'), name='Random Guess'))

    # Layout
    print('fig update')
    if log_scale:
        xaxis_dict = dict(title='False Positive Rate (Log Scale)', type='log')
    else:
        xaxis_dict = dict(title='False Positive Rate')
    fig.update_layout(
        title='ROC Curves Across Multiple Experiments',
        xaxis=xaxis_dict,
        yaxis=dict(title='True Positive Rate'),
        template='plotly_white'
    )

    if not show_std_in_legend:
        for trace in fig['data']: 
            if ('Std Dev' in trace['name']):
                trace['showlegend'] = False

    return fig
