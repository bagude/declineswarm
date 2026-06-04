"""Make the repo root importable so `import segmented_decline` works from anywhere."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
