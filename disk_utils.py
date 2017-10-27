# import hashlib
import os
import pickle


def disk_cache(f):
    def wrapper(*args, **kwargs):
        if not os.path.exists("cache/"):
            print("[DISK_CACHE] creating cache dir")
            os.makedirs("cache/")

        fid = f.__name__
        if args:
            fid += "::".join(str(arg) for arg in args)
        cache_file = "cache/{}.pkl".format(fid)
        try:
            with open(cache_file, "rb") as fin:
                retr = pickle.load(fin)
        except FileNotFoundError:
            retr = f(*args, **kwargs)
            with open(cache_file, "wb") as fout:
                pickle.dump(retr, fout)
        return retr
    return wrapper
