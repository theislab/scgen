import anndata
import numpy as np
from scipy import sparse


def extractor(
    data,
    cell_type,
    condition_key,
    cell_type_key,
    ctrl_key,
    stim_key,
):
    """
    Returns a list of `data` files while filtering for a specific `cell_type`.

    Parameters
    ----------
    data: `~anndata.AnnData`
        Annotated data matrix
    cell_type: basestring
        specific cell type to be extracted from `data`.
    condition_key: basestring
        key for `.obs` of `data` where conditions can be found.
    cell_type_key: basestring
        key for `.obs` of `data` where cell types can be found.
    ctrl_key: basestring
        key for `control` part of the `data` found in `condition_key`.
    stim_key: basestring
        key for `stimulated` part of the `data` found in `condition_key`.
    Returns
    ----------
    data_list: list
        list of `data` files while filtering for a specific `cell_type`.
    Example
    ----------
    ```python
    import scgen
    import anndata
    train_data = anndata.read("./data/train.h5ad")
    test_data = anndata.read("./data/test.h5ad")
    train_data_extracted_list = extractor(train_data, "CD4T", "conditions", "cell_type", "control", "stimulated")
    ```
    """
    cell_with_both_condition = data[data.obs[cell_type_key] == cell_type]
    condition_1 = data[
        (data.obs[cell_type_key] == cell_type)
        & (data.obs[condition_key] == ctrl_key)
    ]
    condition_2 = data[
        (data.obs[cell_type_key] == cell_type)
        & (data.obs[condition_key] == stim_key)
    ]
    training = data[
        ~(
            (data.obs[cell_type_key] == cell_type)
            & (data.obs[condition_key] == stim_key)
        )
    ]
    return [training, condition_1, condition_2, cell_with_both_condition]


def balancer(
    adata,
    condition_key,
    cell_type_key,
):

    """
    Makes cell type population equal.

    Parameters
    ----------
    adata: `~anndata.AnnData`
        Annotated data matrix.
    condition_key: basestring
        key for `.obs` of `data` where conditions can be found.
    cell_type_key: basestring
        key for `.obs` of `data` where cell types can be found.
    Returns
    ----------
    balanced_data: `~anndata.AnnData`
        Equal cell type population Annotated data matrix.
    Example
    ----------
    ```python
    import scgen
    import anndata
    train_data = anndata.read("./train_kang.h5ad")
    train_ctrl = train_data[train_data.obs["condition"] == "control", :]
    train_ctrl = balancer(train_ctrl, "conditions", "cell_type")
    ```
    """
    class_names = np.unique(adata.obs[cell_type_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = adata.copy()[adata.obs[cell_type_key] == cls].shape[0]
    max_number = np.max(list(class_pop.values()))
    all_data_x = []
    all_data_label = []
    all_data_condition = []
    for cls in class_names:
        temp = adata.copy()[adata.obs[cell_type_key] == cls]
        index = np.random.choice(range(len(temp)), max_number)
        if sparse.issparse(temp.X):
            temp_x = temp.X.A[index]
        else:
            temp_x = temp.X[index]
        all_data_x.append(temp_x)
        temp_ct = np.repeat(cls, max_number)
        all_data_label.append(temp_ct)
        temp_cc = np.repeat(np.unique(temp.obs[condition_key]), max_number)
        all_data_condition.append(temp_cc)
    balanced_data = anndata.AnnData(np.concatenate(all_data_x))
    balanced_data.obs[cell_type_key] = np.concatenate(all_data_label)
    balanced_data.obs[condition_key] = np.concatenate(all_data_condition)
    class_names = np.unique(balanced_data.obs[cell_type_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = len(balanced_data[balanced_data.obs[cell_type_key] == cls])
    return balanced_data
