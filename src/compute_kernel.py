import numpy as np

def cyclical_time_features(unix_secs):
    ts = np.asarray(unix_secs, dtype=float)
    hour = (ts / 3600) % 24
    day = (ts / 86400) % 365.25
    h = 2 * np.pi * hour / 24
    d = 2 * np.pi * day / 365.25
    return np.sin(h), np.cos(h), np.sin(d), np.cos(d)
