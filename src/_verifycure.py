import glob, os
from collections import Counter
root=r"D:\CURE\Pill_Images"
imgs=[f for f in glob.glob(os.path.join(root,"**","*.*"),recursive=True) if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
def rel(f): return f.split("Pill_Images"+os.sep)[1].split(os.sep)
classes=set(rel(f)[0] for f in imgs)
dom=Counter(rel(f)[2] for f in imgs if len(rel(f))>=4)
refc=set(rel(f)[0] for f in imgs if len(rel(f))>=3 and rel(f)[2]=="Reference")
print("classes:",len(classes),"| total imgs:",len(imgs),"| domains:",dict(dom))
print("classes with Reference:",len(refc))
