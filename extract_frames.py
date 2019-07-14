import os
import glob
from subprocess import call


def extract_frames(dataset_path):
    """Extract frames of .mov files under dataset_path

    Ouput frames will be stored in a folder named {filename}_all.
    If {filename}_all exists, this file will not be processed.

    Parameters
    ----------
        dataset_path: the folder containing .mov files to be extracted
    """

    files = glob.glob(os.path.join(dataset_path,
                                   "data",
                                   "*",
                                   "*.mov"))  # this line should be updated according to the full path
    files += glob.glob(os.path.join(dataset_path,
                                    "data",
                                    "*",  "*",
                                    "*.mov"))  # this line should be updated according to the full path
    for file in files:
        print("Extracting frames for ", file)
        directory = file.split(".")[0] + "_all"
        if not os.path.exists(directory):
            os.makedirs(directory)
            call(["ffmpeg", "-i",  file,
                  os.path.join(directory, "%05d.jpg"), "-hide_banner"])
