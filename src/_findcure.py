import glob, os
print("=== Pill_Images*.zip anywhere ===")
found=[]
for base in ["C:/","D:/"]:
    found += glob.glob(base+"**/Pill_Images*.zip", recursive=True)
for f in found[:20]:
    print("  ", f, round(os.path.getsize(f)/1e6), "MB")
print("total zips:", len(found))
print()
print("=== folders named Pill_Images ===")
for base in ["C:/","D:/"]:
    for d in glob.glob(base+"**/Pill_Images", recursive=True)[:10]:
        n=len([x for x in glob.glob(os.path.join(d,"**","*.*"),recursive=True) if x.lower().endswith((".jpg",".png",".bmp"))])
        print("  ", d, "->", n, "imgs")
