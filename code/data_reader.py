import numpy as np
import scanpy.api as sc
from random import shuffle
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
sc.settings.verbosity = 1  # show logging output
sns.set_style("white")

class data_reader():

    def __init__(self,train_data, valid_data, conditions, tr_ct_list =None,ho_ct_list=None):


        self.conditions = conditions
        if(tr_ct_list and ho_ct_list):
            self.t_in = tr_ct_list
            self.t_out = ho_ct_list
            self.train_real = self.data_remover(train_data)
            self.train_real_adata = self.train_real
            ind_list = [i for i in range(self.train_real.shape[0])]
            shuffle(ind_list)
            self.train_real = self.train_real[ind_list, :].X
            self.valid_real_adata = self.data_remover(valid_data)
            self.valid_real =  self.valid_real_adata.X

        else:
            self.train_real = train_data
            self.train_real_adata = self.train_real
            ind_list = [i for i in range(self.train_real.shape[0])]
            shuffle(ind_list)
            self.train_real = self.train_real[ind_list, :].X
            self.valid_real_adata = valid_data
            self.valid_real = valid_data.X


    def data_remover(self,adata):
        source_data = []
        for i in self.t_in:
            source_data.append(self.extractor(adata, i)[3])
        target_data = []
        for i in self.t_out:
            target_data.append(self.extractor(adata, i)[1])
        mearged_data = self.training_data_provider(source_data,target_data)
        mearged_data.var_names = adata.var_names
        return mearged_data

    def extractor(self, data, cell_type):
        cell_with_both_condition = data[data.obs["cell_type"] == cell_type]
        condtion_1 = data[(data.obs["cell_type"] == cell_type) & (data.obs["condition"] == self.conditions["ctrl"])]
        condtion_2 = data[(data.obs["cell_type"] == cell_type) & (data.obs["condition"] == self.conditions["stim"])]
        training = data[~((data.obs["cell_type"] ==cell_type) & (data.obs["condition"] ==self.conditions["stim"]))]
        return [training, condtion_1, condtion_2, cell_with_both_condition]

    def training_data_provider(self,train_s, train_t):
        train_s_X = []
        train_s_diet = []
        train_s_groups = []
        for i in train_s:
            train_s_X.append(i.X.A)
            train_s_diet.append(i.obs["condition"].tolist())
            train_s_groups.append(i.obs["cell_type"].tolist())
        train_s_X = np.concatenate(train_s_X)
        temp = []
        for i in train_s_diet:
            temp = temp + i
        train_s_diet = temp
        temp = []
        for i in train_s_groups:
            temp = temp + i
        train_s_groups = temp
        train_t_X = []
        train_t_diet = []
        train_t_groups = []
        for i in train_t:
            train_t_X.append(i.X.A)
            train_t_diet.append(i.obs["condition"].tolist())
            train_t_groups.append(i.obs["cell_type"].tolist())
        temp = []
        for i in train_t_diet:
            temp = temp + i
        train_t_diet = temp
        temp = []
        for i in train_t_groups:
            temp = temp + i
        train_t_groups = temp
        train_t_X = np.concatenate(train_t_X)
        # concat_all
        train_real = np.concatenate([train_s_X, train_t_X])
        train_real = sc.AnnData(train_real)
        train_real.obs["condition"] = train_s_diet + train_t_diet
        train_real.obs["cell_type"] = train_s_groups + train_t_groups
        return train_real

    def balancer(self,data):
        class_names = np.unique(data.obs["cell_type"])
        class_pop = {}
        for cls in class_names:
            class_pop[cls] = len(data[data.obs["cell_type"] == cls])

        max_number = np.max(list(class_pop.values()))

        all_data_x = []
        all_data_label = []
        all_data_condition = []

        for cls in class_names:
            temp = data[data.obs["cell_type"] == cls]
            index = np.random.choice(range(len(temp)), max_number)
            temp_x = temp.X[index]
            all_data_x.append(temp_x)
            temp_ct = np.repeat(cls, max_number)
            all_data_label.append(temp_ct)
            temp_cc = np.repeat(np.unique(temp.obs["condition"]), max_number)
            all_data_condition.append(temp_cc)

        balanced_data = sc.AnnData(np.concatenate(all_data_x))
        balanced_data.obs["cell_type"] = np.concatenate(all_data_label)
        balanced_data.obs["condition"] = np.concatenate(all_data_label)

        class_names = np.unique(balanced_data.obs["cell_type"])
        class_pop = {}
        for cls in class_names:
            class_pop[cls] = len(balanced_data[balanced_data.obs["cell_type"] == cls])
        #print(class_pop)
        return balanced_data

    def reg_overlap_plot(self, adata, gene_list, path_to_save, file_name):
        stim = adata[adata.obs["condition"] == "pred_stim"]
        ctrl = adata[adata.obs["condition"] == "ctrl"]
        real_stim = adata[adata.obs["condition"] == "real_stim"]
        x = np.average(ctrl.X, axis=0)
        y = np.average(real_stim.X, axis=0)
        y_pred = np.average(stim.X, axis=0)
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        p1 = plt.scatter(x, y, marker=".", c="grey", alpha=.5, label="ground truth")
        p2 = plt.scatter(x, y_pred, marker="*", c="blue", alpha=.5, label="prediction")
        plt.plot(x, m * x + b, "-", color="green")
        plt.xlabel("ctrl", fontsize=12)
        plt.ylabel("stim", fontsize=12)
        texts = []
        for i in gene_list:
            j = adata.var_names.tolist().index(i)
            x_bar = x[j]
            y_bar = y[j]
            y_pred_bar = y_pred[j]
            texts.append(plt.text(x_bar, y_bar, i, fontsize=11, color="grey"))
            plt.plot(x_bar, y_pred_bar, '*', color="blue", alpha=.5)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f"", fontsize=12)
        plt.savefig(f"{path_to_save}/mean_overlap_{file_name}.pdf", bbox_inches='tight', dpi=100)
        plt.show()

    def reg_overlap_plot_cross(self, adata, gene_list, path_to_save, file_name):
        stim = adata[adata.obs["condition"] == "pred_stim"]
        ctrl = adata[adata.obs["condition"] == "ctrl"]
        real_stim = adata[adata.obs["condition"] == "real_stim"]
        x = np.average(ctrl.X, axis=0)
        y = np.average(real_stim.X, axis=0)
        y_pred = np.average(stim.X, axis=0)
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        p1 = plt.scatter(x, y, marker=".", c="grey", alpha=.5, label="ground truth")
        p2 = plt.scatter(x, y_pred, marker="*", c="blue", alpha=.5, label="prediction")
        plt.plot(x, m * x + b, "-", color="green")
        plt.xlabel("ctrl", fontsize=12)
        plt.ylabel("stim", fontsize=12)
        texts = []
        texts2 = []
        for i in gene_list:
            j = adata.var_names.tolist().index(i)
            x_bar = x[j]
            y_bar = y[j]
            y_pred_bar = y_pred[j]
            texts.append(plt.text(x_bar, y_bar, i, fontsize=11, color="grey"))
            texts2.append(plt.text(x_bar, y_pred_bar, i, fontsize=11, color="blue"))
            plt.plot(x_bar, y_pred_bar, '*', color="blue", alpha=.5, )
            plt.plot(x_bar, y_bar, '.', color="gray", alpha=.5)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f"", fontsize=12)
        plt.savefig(f"{path_to_save}/mean_overlap_{file_name}.pdf", bbox_inches='tight', dpi=100)
        plt.show()


    def reg_mean_plot(self, adata, path_to_save, file_name):
        stim = adata[adata.obs["condition"] == "pred_stim"]
        real_stim = adata[adata.obs["condition"] == "real_stim"]
        x = np.average(stim.X, axis=0)
        y = np.average(real_stim.X, axis=0)
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        plt.plot(x, y, '.')
        plt.plot(x, m * x + b, 'g-')
        plt.xlabel("pred mean", fontsize=12)
        plt.ylabel("real mean", fontsize=12)
        plt.title(f"mean matching", fontsize=12)
        plt.text(max(x) - max(x) * .25, .1, r'$R^2$=' + f"{r_value**2:.2f}")
        plt.savefig(f"{path_to_save}/{file_name}_mean.pdf", bbox_inches='tight', dpi=100)
        plt.show()

    def reg_var_plot(self, adata, path_to_save, file_name):
        stim = adata[adata.obs["condition"] == "pred_stim"]
        real_stim = adata[adata.obs["condition"] == "real_stim"]
        x = np.var(stim.X, axis=0)
        y = np.var(real_stim.X, axis=0)
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        plt.plot(x, y, '.')
        plt.plot(x, m * x + b, 'g-')
        plt.xlabel("pred variance", fontsize=12)
        plt.ylabel("real variance", fontsize=12)
        plt.title(f"variance matching", fontsize=12)
        plt.text(max(x) - .2*max(x), .1, r'$R^2$=' + f"{r_value**2:.2f}")
        plt.savefig(f"{path_to_save}/{file_name}_variance.pdf", bbox_inches='tight', dpi=100)
        plt.show()

    def reg_mean_plot_cross(self, adata, path_to_save, file_name, gene_list, **ploting_keys):
        x_adata = adata[adata.obs["condition"] == ploting_keys["x"]]
        y_adata = adata[adata.obs["condition"] == ploting_keys["y"]]
        x = np.average(x_adata.X, axis=0)
        y = np.average(y_adata.X, axis=0)
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        plt.plot(x, y, ploting_keys["shape"])
        plt.plot(x, m * x + b, ploting_keys["line"])
        texts = []
        for i in gene_list:
            j = adata.var_names.tolist().index(i)
            x_bar = x[j]
            y_bar = y[j]
            texts.append(plt.text(x_bar, y_bar, i, fontsize=11))
            plt.plot(x_bar, y_bar, ploting_keys["shape"], color="red", markersize=10)
        plt.xlabel(ploting_keys["x"], fontsize=12)
        plt.ylabel(ploting_keys["y"], fontsize=12)
        plt.title(f" ", fontsize=12)
        plt.text(max(x) - max(x) * .25, .1, r'$R^2$=' + f"{r_value**2:.2f}")
        plt.savefig(f"{path_to_save}/{file_name}_mean.pdf", bbox_inches='tight', dpi=100)
        plt.show()

    def dpclassifier_hist(self, scg_obj, adata, delta, path_to_save):

        cd = adata[adata.obs["condition"] == self.conditions["ctrl"], :]
        stim = adata[adata.obs["condition"] == self.conditions["stim"], :]
        all_latent_cd = scg_obj._to_latent(cd.X)
        all_latent_stim = scg_obj._to_latent(stim.X)
        dot_cd = np.zeros((len(all_latent_cd)))
        dot_sal = np.zeros((len(all_latent_stim)))
        for ind, vec in enumerate(all_latent_cd):
            dot_cd[ind] = np.dot(delta, vec)
        for ind, vec in enumerate(all_latent_stim):
            dot_sal[ind] = np.dot(delta, vec)
        plt.hist(dot_cd, label=self.conditions["ctrl"], bins=50, )
        plt.hist(dot_sal, label=self.conditions["stim"], bins=50)
        plt.legend(loc=1, prop={'size': 7})
        plt.axvline(0, color='k', linestyle='dashed', linewidth=1)
        plt.title("  ", fontsize=10)
        plt.xlabel("  ", fontsize=10)
        plt.ylabel("  ", fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        ax = plt.gca()
        ax.grid(False)
        plt.savefig(f"{path_to_save}"
                    , bbox_inches='tight', dpi=100)















