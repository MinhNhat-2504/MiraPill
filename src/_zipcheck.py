import zipfile, glob, os
zips=sorted(glob.glob("C:/Users/Minh Nhat/Downloads/Pill_Images-*.zip"))
total=set(); refs=set(); custs=set(); classes=set()
for z in zips:
    with zipfile.ZipFile(z) as zf:
        imgs=[n for n in zf.namelist() if n.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        print(os.path.basename(z), "->", len(imgs), "images")
        for n in imgs:
            total.add(n)
            parts=n.replace("\\","/").split("/")
            # find Pill_Images index
            if "Pill_Images" in parts:
                i=parts.index("Pill_Images")
                if len(parts)>i+1: classes.add(parts[i+1])
            if "Reference" in n: refs.add(n)
            if "Customer" in n: custs.add(n)
print("---")
print("UNIQUE images across 6 zips:", len(total))
print("classes:", len(classes), "| Reference imgs:", len(refs), "| Customer imgs:", len(custs))
