import numpy as np
import pandas as pd
import xarray as xr
import os
from bluemath_tk.datamining.pca import PCA
from bluemath_tk.interpolation.rbf import RBF

os.environ["OMP_NUM_THREADS"] = "1"

n_folds = [10]
#n_cases = [3000, 2750, 2500, 2250, 2000, 1750, 1500, 1250,1125,1050, 1000, 950, 825, 750, 500, 250]

n_cases = np.arange(3000, 250, -250)

xds_post = xr.open_dataset("outputs/Vars_postprocessed_new_OK.nc")

variables = list(xds_post.data_vars.keys())

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
#var = variables[0]
    for i, cases in enumerate(n_cases):

        cases_to_process = df_parametros_clean.index.values[:cases]
        k_cases_list = shuffle_cases(cases_to_process, folds)
        rmse_fold_values = []

        for j, fold in enumerate(range(folds)):

            train_cases = np.concatenate(
                [k_cases_list[j] for j in range(folds) if j != fold]
            )

            pca_output_all = PCA(n_components=0.998, # % of variance to retain
                            is_incremental=False, 
                            debug=True, )
            pca_output_all.fit_transform( 
                data=xds_post.sel(case_num=train_cases), 
                vars_to_stack=[var], 
                coords_to_stack=["time", "point"],
                pca_dim_for_rows="case_num")

            print(
                f"Processing k-fold for {var} with {cases} cases and {j+1}/{folds} folds, "
                f"iteration {i+1}/{len(n_cases)}"
            )

            test_cases = k_cases_list[fold]

            rbf_fitted = RBF(kernel = 'linear')
            
            rbf_fitted.fit(
                subset_data=df_parametros_clean.loc[train_cases],
                target_data=pca_output_all.pcs_df.loc[train_cases],
                num_workers=25,
            )

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

        rmse_df = pd.DataFrame({
            "fold": range(folds),
            "rmse": rmse_fold_values,
            "n_cases": cases,
            "var": var,
        })
        rmse_df_list.append(rmse_df)

combined_df = pd.concat(rmse_df_list, ignore_index=True)
combined_df.set_index(["var", "n_cases", "fold"], inplace=True)
results_kfold = combined_df.to_xarray()
results_kfold.to_netcdf("outputs/OK_k_fold_metamodel_full_linear_OK.nc")

# combined_df = pd.concat(rmse_df_list, ignore_index=True)
# combined_df.set_index(["var", "n_cases", "fold"], inplace=True)
# results_kfold = combined_df.to_xarray()
# results_kfold.to_netcdf("outputs/OK_k_fold_metamodel_linear_proj_test.nc")