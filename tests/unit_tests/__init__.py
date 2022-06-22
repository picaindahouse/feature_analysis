import sys
from os.path import dirname, join, normpath

THIS_DIR = dirname(__file__)
BIN_PROJ_DIR = normpath(join(THIS_DIR, "..", "..", "src", "feature_analysis", "binary_class"))
REG_PROJ_DIR = normpath(join(THIS_DIR, "..", "..", "src", "feature_analysis", "regression"))
sys.path.append(BIN_PROJ_DIR)
sys.path.append(REG_PROJ_DIR)
