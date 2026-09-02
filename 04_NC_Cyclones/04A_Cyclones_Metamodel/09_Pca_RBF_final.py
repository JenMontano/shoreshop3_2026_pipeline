import numpy as np
import pandas as pd
import xarray as xr
import os
import pickle
from bluemath_tk.interpolation.rbf import RBF

os.environ["OMP_NUM_THREADS"] = "1"

n_folds = [10]
n_cases = 3000



xds_post = xr.open_dataset("outputs/Vars_postprocessed_proj.nc")

variables = xds_post.data_vars.keys()

df_parametros = pd.read_csv("outputs/syntetic_tracks_7r_params.csv")

df_parametros_clean = df_parametros.iloc[xds_post.case_num.values].drop(columns=["date_furthest"])

def shuffle_cases(cases, n_fold):
    shuffled_cases = cases.copy()
    np.random.seed(2)
    np.random.shuffle(shuffled_cases)
    return np.array_split(shuffled_cases, n_fold)

folds = n_folds[0]
rmse_df_list = []

for var in variables:
    with open(f"outputs/pca_model_{var}.pkl", "rb") as f:
        pca_output_all = pickle.load(f)

    cases = n_cases
    i = 0

    cases_to_process = df_parametros_clean.index.values[:cases]
    k_cases_list = shuffle_cases(cases_to_process, folds)
    rmse_fold_values = []

    for j, fold in enumerate(range(folds)):
        print(
            f"Processing k-fold for {var} with {cases} cases and {j+1}/{folds} folds, "
        )
        train_cases = np.concatenate(
            [k_cases_list[j] for j in range(folds) if j != fold]
        )
        test_cases = k_cases_list[fold]

        rbf_fitted = RBF()
        
        rbf_fitted.fit(
            subset_data=df_parametros_clean.loc[train_cases],
            target_data=pca_output_all.pcs_df.loc[train_cases],
            num_workers=40,
        )
        try:
            reconstructed_ds = rbf_fitted.predict(
                dataset=df_parametros_clean.loc[test_cases]
            )
            ds_target_test = xr.Dataset(
                {
                    "PCs": (
                        ("case_num", "n_component"),
                        reconstructed_ds.values,
                    ),
                },
                coords={
                    "case_num": test_cases,
                    "n_component": np.arange(reconstructed_ds.shape[1]),
                },
            )
            pca_reconstructed = pca_output_all.inverse_transform(ds_target_test);

            target_test = xds_post.sel(case_num=test_cases)[var].values
            pred_test = pca_reconstructed[var].values
            rmse_all = np.sqrt(np.mean((pred_test - target_test) ** 2))
            rmse_fold_values.append(rmse_all)
        except Exception as e:
            print(f"Error processing fold {j+1}/{folds} for variable {var}: {e}")
            rmse_fold_values.append(np.nan)

    combined_df = pd.DataFrame({
        "fold": range(folds),
        "rmse": rmse_fold_values,
        "var": var,
    })

combined_df.set_index(["var", "fold"], inplace=True)
results_kfold = combined_df.to_xarray()
results_kfold.to_netcdf("outputs/OK_k_fold_metamodel_gausian_proj.nc")