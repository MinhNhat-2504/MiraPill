import zipfile, glob, os, time
zips=sorted(glob.glob("C:/Users/Minh Nhat/Downloads/Pill_Images-*.zip"))
dest=r"D:\CURE"
os.makedirs(dest, exist_ok=True)
t0=time.time(); n=0
for z in zips:
    with zipfile.ZipFile(z) as zf:
        zf.extractall(dest)
    n+=1
    print(f"[{n}/6] extracted {os.path.basename(z)}  ({time.time()-t0:.0f}s)", flush=True)
print("DONE extract in", round(time.time()-t0), "s")
