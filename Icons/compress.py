"""
This script converts icon.ico into base64 and compresses it.
This allows the icon to be embedded directly into Plot3D's code.
"""
# Import modules
from zlib import compress
from base64 import b64encode
from io import StringIO

# Load image data as bytes
with open('icon.ico', mode='rb') as rf:
    data = rf.read()

# Compress to bytes data and print result
with open('compressed-icon.txt', mode='w') as wf:
    #wf.write(str(b64encode(compress(data))))
    wf.write(str(b64encode(compress(data))))
