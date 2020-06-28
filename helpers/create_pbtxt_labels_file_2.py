import os

# initialize these parameters
labels      = ['fox', 'badger']
LABELS_PATH = os.getcwd()
LABELS_NAME = 'ssd_fox_bager.pbtxt'

# Label 0 is reserved for background [not used]
LABELS_FILE = os.path.join(LABELS_PATH, LABELS_NAME)
# generate *.pbtxt
with open(LABELS_FILE, 'w') as f:
	for i, label in enumerate(labels):
         f.write('item {\n')
         f.write(f'    id:{i + 1}\n')
         f.write(f'    name:'{label}'\n')
         f.write('}\n\n')