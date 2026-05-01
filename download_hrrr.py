import gcsfs

fs = gcsfs.GCSFileSystem()

base = "gs://high-resolution-rapid-refresh/hrrr.20140730/conus/"

for i in range(16):
    f = f"{base}hrrr.t18z.wrfnatf{str(i).zfill(2)}.grib2"
    fs.get(f, f.split("/")[-1])
    print("Downloaded:", f)