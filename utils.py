import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from shapleycomposition import ilr_inv
from pandas.plotting import parallel_coordinates

plt.rcParams['axes.grid'] = True

cm = plt.cm.tab10
colors = cm.colors


def rotate(vector, angle):
    radians = np.radians(angle)
    return np.inner(np.array([[np.cos(radians), -np.sin(radians)],
                              [np.sin(radians), np.cos(radians)]]),
                    np.array(vector).T)


def plot_ilr_coordinate_system(ax):

    class_coordinates = np.array([
        rotate(np.array([1, 0]), 30),
        rotate(np.array([1, 0]), 30+120),
        rotate(np.array([1, 0]), 30+240)])

    for i, coord in enumerate(class_coordinates):
        ax.arrow(0, 0, coord[0], coord[1], shape='full', head_width=0.1,
                 color=colors[i], linewidth=4)
        ax.text(coord[0]/2, coord[1]/2, f"$C_{i+1}$")

    decision_boundaries = np.array([rotate(np.array([0, 3]), 0),
                                    rotate(np.array([0, 3]), 120),
                                    rotate(np.array([0, 3]), 240)])

    for i, coord in enumerate(decision_boundaries):
        ax.arrow(0, 0, coord[0], coord[1], linestyle='--')

    ax.set_box_aspect(1)


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_instances(base, inst_ilr_tip_list, ax, head_width=0.1, linestyle='-',
                  feature_names=None):
    instance_colors = colors[3:]

    for i, instance in enumerate(inst_ilr_tip_list):
        last = base.copy()
        color = instance_colors[i]
        for j in range(len(instance)):
            d_next = instance[j] - last
            ax.arrow(x=last[0], y=last[1], dx=d_next[0], dy=d_next[1],
                     shape='full', head_width=head_width,
                     length_includes_head=True,
                     color=adjust_lightness(color,
                                            amount=1+0.5*(j/len(instance))),
                     linestyle=linestyle, linewidth=2,
                     label=f"Inst. {i} $X_{j+1}$")
            last += d_next


def plot_composite_shapley_feature_contributions(composite_shapley_values,
                                                 base=None, feature_names=None,
                                                 target_names=None,
                                                 cummulative=False, title=None,
                                                 sort_by_norm=False,
                                                 basis=None, fig=None, ax=None,
                                                 parallel=False):
    """
    Parameters
    ==========
    composite_shapley_values: numpy.ndarray (n_features, n_classes-1)
    base: numpy.ndarray (n_classes-1)
        If a base is passed, the base is plot as the first step
    composite_base: numpy.ndarray (n_classes-1)
    title: string
    basis: numpy.ndarray (n_classes-1, n_classes)
        If a basis is passed, the inverse of the ILR is calculated and plot
    parallel: boolean
        If False it plots histograms, if True it plots parallel coordinates
    """

    n_features = len(composite_shapley_values)

    composite_shapley_values = np.array(composite_shapley_values).T

    if feature_names is None:
        feature_names = np.array([f"$X_{j+1}$" for j in range(n_features)])
    else:
        feature_names = np.array(feature_names)

    if sort_by_norm:
        order = np.argsort([-norm(v) for v in composite_shapley_values.T])
        print(order)
        composite_shapley_values = composite_shapley_values.T[order].T
        feature_names = feature_names[order]

    if base is not None:
        composite_shapley_values = np.hstack([base.reshape(-1, 1),
                                              composite_shapley_values])
        feature_names = np.concatenate([['Base', ], feature_names])

    if cummulative:
        composite_shapley_values = np.cumsum(composite_shapley_values, axis=1)

    if basis is not None:
        composite_shapley_values = np.array([ilr_inv(x, basis=basis) for x in
                                             composite_shapley_values.T]).T

    n_classes = composite_shapley_values.shape[0]
    print(f"Number of classes = {n_classes}")

    if basis is not None:
        if target_names is None:
            target_names = [r"$C_" + str(j+1) + "$" for j in range(n_classes)]
    else:
        target_names = [r"$\tilde{p}_" + str(j+1) + "$" for j in
                        range(n_classes)]

    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=(5, 2))
        ax = fig.add_subplot()

    if parallel:
        return plot_parallel_coordinates(composite_shapley_values,
                                         feature_names, target_names, fig, ax,
                                         title=title)

    return plot_bars(composite_shapley_values, feature_names, target_names,
                     fig, ax, title=title)


def plot_parallel_coordinates(matrix, x_labels, legend, fig, ax, title):
    print(x_labels)
    print(legend)
    print(matrix)
    df_matrix = pd.DataFrame(matrix, index=legend, columns=x_labels)
    df_matrix['class'] = df_matrix.index
    parallel_coordinates(df_matrix, class_column='class', color=colors, ax=ax)
    ax.grid(True)
    ax.set_ylabel('Probability')
    plt.xticks(rotation=45, ha='right')

    if title is not None:
        ax.set_title(title)

    plt.show()
    return fig, ax


def plot_bars(matrix, x_labels, legend, fig, ax, title):
    width = 1/(matrix.shape[1]+1)
    x = np.arange(matrix.shape[1])
    print(f"legend = {legend}")
    print(matrix)
    for j, dimension in enumerate(matrix):
        offset = width * j
        ax.bar(x + offset, dimension, width, label=legend[j])

    ax.set_xticks(x + width, x_labels)
    plt.xticks(rotation=45, ha='right')
    ax.legend()

    if title is not None:
        ax.set_title(title)

    plt.show()
    return fig, ax
