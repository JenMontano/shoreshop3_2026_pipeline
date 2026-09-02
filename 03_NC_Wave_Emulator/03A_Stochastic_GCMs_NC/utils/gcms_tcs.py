def is_NA(lat, lon):
    return (10 <= lat <= 30) and (280 <= lon <= 340)

def Check_NA_formation(lat,lon): 
    """
    Check if formation is in North Atlantic (this should be inhibited if basin==EP)
    Parameters
    ----------
    lat : latitude coordinate of genesis
    lon : longitude coordinate of genesis

    Returns
    -------
    l : 1=yes (formation in NA) 0=no (no formation in NA).

    """
    if lat<=60. and lat>17.5 and lon>260.:
        l=1
    elif lat<=17.5 and lat>15. and lon>270.:
        l=1
    elif lat<=15. and lat>10 and lon>275.:
        l=1
    elif lat<=10. and lon>276.:
        l=1
    else:
        l=0
    return l

def is_TC_season(time):
    month = time.month
    return (month >= 6) & (month <= 11)

def passes_by_north_carolina(lat_list, lon_list):
    center_lat = 35.5
    center_lon = 77.5
    radius = 7.0

    for lat, lon in zip(lat_list, lon_list):
        distance = ((lat - center_lat) ** 2 + (lon - center_lon) ** 2) ** 0.5
        if distance <= radius:
            return True  # at least one point inside

    return False  # no points inside