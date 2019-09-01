import os
import sys
import json
import logging as log
import numpy as np
import pandas as pd
import werkzeug.datastructures

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import encoder

# generate attribute.names 
def prepare_names():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attribute.names")
    log.info("creating %s" % path)
    with open(path, 'w+t') as fp:
        for name in encoder.attribute_names():
            fp.write("%s\n" % name)

# generate classes
def prepare_classes():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classes.json")
    log.info("creating %s" % path)
    with open(path, 'w+t') as fp:
        json.dump(encoder.classes, fp)

# trainning dataset
def prepare_dataset(filename):
    prepare_names()
    prepare_classes()
    data = pd.read_csv(filename, sep = ',', header = None)
    return data.replace(encoder.classes_replace)

# templatenya ergo encode <path> <folder>
def prepare_input(x, is_encoding = False):
    # file upload
    if isinstance(x, werkzeug.datastructures.FileStorage):
        return encoder.encode_pe(x)
    # file path
    elif os.path.isfile(x) :
        return encoder.encode_pe(x)
    # raw vector
    else:
        return x.split(',')


