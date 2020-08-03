import argparse
import glob
import os
import zipfile
from os import path

BLACKLIST = ['__pycache__', '.pyc']
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('homework')
    parser.add_argument('utid')
    args = parser.parse_args()

    this_path = path.dirname(path.realpath(__file__))
    files = []
    for f in glob.glob(path.join(args.homework, '**')) + glob.glob(path.join('writeup', '**')):
        if all(b not in f for b in BLACKLIST):
            files.append(f)

    zf = zipfile.ZipFile(args.utid + '.zip', 'w', compression=zipfile.ZIP_DEFLATED)
    for f in files:
        zf.write(f, f.replace(os.path.join(this_path, 'homework'), args.utid))
    zf.close()
    size = os.path.getsize(args.utid + '.zip')
    if size > 10 * 1024 * 1024:
        print("Warning: The created zip file is larger than expected!",
              "Please make sure you did not accidentally include the training data in the submission directory.")
    print("submission created for %s.zip. Please remember to submit" % args.utid)
