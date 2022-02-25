import os
import sys

GRADUATION_SCRIPTS_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

if GRADUATION_SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, GRADUATION_SCRIPTS_PATH)