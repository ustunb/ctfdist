import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category = FutureWarning)

#paper embelisshments
script_metric_labels = {
    'da_out': 'DA',
    }


def plot_conditional_distribution(cf, name):

    plot_data = cf.split_values(variable_name = name)

    for value, samples in plot_data.items():
        label = '%s = %1.0f' % (name, value)
        singular = len(np.unique(samples)) > 1
        sns.distplot(samples, hist = True, kde = not singular, kde_kws = {'shade': True, 'linewidth': 1}, label = label)

        plt.title('Values of %s' % cf.pretty_type)
        plt.xlabel(name)
        plt.ylabel('Density')
        plt.legend()
        plt.show()


def plot_descent_profile(results, max_iterations = None):

    max_iterations = None
    # Plot Discrimination Gap Over Time
    sns.set(font_scale = 2, font = 'sans-serif')
    sns.set_style("whitegrid")

    f = plt.figure(figsize = (10, 8))
    ax = plt.gca()

    plot_df = pd.DataFrame(results)
    plot_test_df = plot_df[plot_df['type'] == 'descent'].get(['iteration', 'gap'])

    # determine best iteration
    best_iteration = plot_df.query('is_best_iteration')['iteration'].values[0]
    best_gap = plot_test_df[plot_test_df['iteration'] == best_iteration]['gap'].mean()

    # progress
    plot_test_df = plot_test_df.groupby(['iteration'], as_index = False).agg(['mean', 'min', 'max'])
    plot_test_df.columns = ['.'.join(col).strip() for col in plot_test_df.columns.values]
    plot_test_df = plot_test_df.reset_index()
    plot_test_df.index = plot_test_df['iteration']
    plot_test_df = plot_test_df.drop(columns = ['iteration'])

    #from ctfdist.debug import ipsh
    #ipsh()



    plt.fill_between(x = plot_test_df.index.tolist(),
                     y1 = plot_test_df['gap.min'].tolist(),
                     y2 = plot_test_df['gap.max'].tolist(),
                     facecolor = 'green', alpha = 0.5, zorder = 1)

    plt.plot(plot_test_df.get(['gap.mean']), lw = 2, zorder = 2)

    plt.scatter(x = [best_iteration], y = [best_gap], s = [300], marker = 'o', label = 'final iteration', color = 'black', zorder = 3)
    plt.legend(frameon = True, loc = 'upper right', framealpha = 1.0)
    plt.show()

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Disparity Metric')
    if max_iterations is not None:
        ax.set_xlim(0, min(max_iterations, max(results['iteration'])))

    return f, ax


