import numpy as np
from bluemath_tk.core.operations import spatial_gradient


def process_ds_mslp_mask(ds, mask):
    
    ds['longitude'] = np.where(ds['longitude'] > 180, ds['longitude'] - 360, ds['longitude'])
    ds = ds.sortby('longitude')

    ds = ds.sel(longitude = slice(mask.longitude.values.min()- 5, mask.longitude.values.max() + 5),
                latitude = slice(mask.latitude.values.min()- 5, mask.latitude.values.max() + 5)).load()

    #interp to 1 degree to then coarsen every 2
    ds = ds.interp(longitude = np.arange(ds.longitude.min()-2, ds.longitude.max()+2, 1),
                    latitude = np.arange(ds.latitude.min()+2, ds.latitude.max()+2, 1), method = 'linear').load()
    ds = ds.coarsen(longitude = 2, latitude = 2, boundary = 'trim').mean()

    #ds["mslp_gradient"] = spatial_gradient(ds["mslp"])
    
    # regridder = xe.Regridder(ds, dwt_fit.data.drop('mask'), method="conservative", reuse_weights=False)
    # ds = regridder(ds)
    
    # ds = ds.interp(longitude = dwt_fit.data.longitude.values, latitude = dwt_fit.data.latitude.values, method = 'linear').load()

    # ds['mslp'] = ds.mslp.fillna(np.nanmean(ds.mslp.values))
    # ds['mslp_gradient'] = ds.mslp_gradient.fillna(np.nanmean(ds.mslp_gradient.values))
    ds = ds.interp(longitude = mask.longitude.values, latitude = mask.latitude.values, method = 'nearest').load()
    ds = ds.where(mask.where(mask == 0) == 0)

    return ds

def process_ds_sst_mask(ds, mask):
    
    ds['longitude'] = np.where(ds['longitude'] > 180, ds['longitude'] - 360, ds['longitude'])
    ds = ds.sortby('longitude')

    ds = ds.sel(longitude = slice(mask.longitude.values.min()- 5, mask.longitude.values.max() + 5),
                latitude = slice(mask.latitude.values.min()- 5, mask.latitude.values.max() + 5)).load()

    #interp to 1 degree to then coarsen every 2
    ds = ds.interp(longitude = np.arange(ds.longitude.min()-2, ds.longitude.max()+2, 1),
                    latitude = np.arange(ds.latitude.min()+2, ds.latitude.max()+2, 1), method = 'linear').load()
    ds = ds.coarsen(longitude = 2, latitude = 2, boundary = 'trim').mean()

    #ds["mslp_gradient"] = spatial_gradient(ds["mslp"])
    
    # regridder = xe.Regridder(ds, dwt_fit.data.drop('mask'), method="conservative", reuse_weights=False)
    # ds = regridder(ds)
    
    # ds = ds.interp(longitude = dwt_fit.data.longitude.values, latitude = dwt_fit.data.latitude.values, method = 'linear').load()

    # ds['mslp'] = ds.mslp.fillna(np.nanmean(ds.mslp.values))
    # ds['mslp_gradient'] = ds.mslp_gradient.fillna(np.nanmean(ds.mslp_gradient.values))
    ds = ds.interp(longitude = mask.longitude.values, latitude = mask.latitude.values, method = 'nearest').load()
    ds = ds.where(mask.where(mask == 0) == 0)

    return ds