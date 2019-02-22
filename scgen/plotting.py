import numpy
from matplotlib import pyplot
from scipy import stats, sparse
# TODO: Writing comments for each fucntions


def reg_mean_plot(adata, condition_key, axis_keys,  path_to_save="./reg_mean.pdf", gene_list=None):
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    stim = adata[adata.obs[condition_key] == axis_keys["y"]]
    ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
    if "y1"in axis_keys.keys():
        real_stim = adata[adata.obs[condition_key] == axis_keys["y1"]]
    x = numpy.average(ctrl.X, axis=0)
    y = numpy.average(stim.X, axis=0)
    m, b, r_value, p_value, std_err = stats.linregress(x, y)
    _p1 = pyplot.scatter(x, y, marker=".", c="grey", alpha=.5, label=f"x-y")
    pyplot.plot(x, m * x + b, "-", color="green")
    pyplot.xlabel(axis_keys["x"], fontsize=12)
    pyplot.ylabel(axis_keys["y"], fontsize=12)
    if "y1"in axis_keys.keys():
        y1 = numpy.average(real_stim.X, axis=0)
        _p2 = pyplot.scatter(x, y1, marker="*", c="blue", alpha=.5, label="x-y1")
    if gene_list is not None:
        texts = []
        for i in gene_list:
            j = adata.var_names.tolist().index(i)
            x_bar = x[j]
            y_bar = y[j]
            texts.append(pyplot.text(x_bar, y_bar, i, fontsize=11, color="grey"))
            if "y1" in axis_keys.keys():
                y1_bar = y1[j]
                pyplot.plot(x_bar, y1_bar, '*', color="blue", alpha=.5)
    pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pyplot.title(f"", fontsize=12)
    pyplot.savefig(f"{path_to_save}", bbox_inches='tight', dpi=100)
    pyplot.show()


def reg_var_plot(adata, condition_key, axis_keys,  path_to_save="./reg_var.pdf", gene_list=None):
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    stim = adata[adata.obs[condition_key] == axis_keys["y"]]
    ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
    if "y1"in axis_keys.keys():
        real_stim = adata[adata.obs[condition_key] == axis_keys["y1"]]
    x = numpy.var(ctrl.X, axis=0)
    y = numpy.var(stim.X, axis=0)
    m, b, r_value, p_value, std_err = stats.linregress(x, y)
    _p1 = pyplot.scatter(x, y, marker=".", c="grey", alpha=.5, label=f"x-y")
    pyplot.plot(x, m * x + b, "-", color="green")
    pyplot.xlabel(axis_keys["x"], fontsize=12)
    pyplot.ylabel(axis_keys["y"], fontsize=12)
    if "y1"in axis_keys.keys():
        y1 = numpy.var(real_stim.X, axis=0)
        _p2 = pyplot.scatter(x, y1, marker="*", c="blue", alpha=.5, label="x-y1")
    if gene_list is not None:
        texts = []
        for i in gene_list:
            j = adata.var_names.tolist().index(i)
            x_bar = x[j]
            y_bar = y[j]
            texts.append(pyplot.text(x_bar, y_bar, i, fontsize=11, color="grey"))
            if "y1" in axis_keys.keys():
                y1_bar = y1[j]
                pyplot.plot(x_bar, y1_bar, '*', color="blue", alpha=.5)
    pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pyplot.title(f"", fontsize=12)
    pyplot.savefig(f"{path_to_save}", bbox_inches='tight', dpi=100)
    pyplot.show()


def binary_classifier(scg_object, adata, delta, condtion_key, conditions,  path_to_save):
    if sparse.issparse(adata.X):
        adata.X = adata.X.A
    cd = adata[adata.obs[condtion_key] == conditions["ctrl"], :]
    stim = adata[adata.obs[condtion_key] == conditions["stim"], :]
    all_latent_cd = scg_object.to_latent(cd.X)
    all_latent_stim = scg_object.to_latent(stim.X)
    dot_cd = numpy.zeros((len(all_latent_cd)))
    dot_sal = numpy.zeros((len(all_latent_stim)))
    for ind, vec in enumerate(all_latent_cd):
        dot_cd[ind] = numpy.dot(delta, vec)
    for ind, vec in enumerate(all_latent_stim):
        dot_sal[ind] = numpy.dot(delta, vec)
    pyplot.hist(dot_cd, label=conditions["ctrl"], bins=50, )
    pyplot.hist(dot_sal, label=conditions["stim"], bins=50)
    pyplot.legend(loc=1, prop={'size': 7})
    pyplot.axvline(0, color='k', linestyle='dashed', linewidth=1)
    pyplot.title("  ", fontsize=10)
    pyplot.xlabel("  ", fontsize=10)
    pyplot.ylabel("  ", fontsize=10)
    pyplot.xticks(fontsize=8)
    pyplot.yticks(fontsize=8)
    ax = pyplot.gca()
    ax.grid(False)
    pyplot.savefig(f"{path_to_save}", bbox_inches='tight', dpi=100)
