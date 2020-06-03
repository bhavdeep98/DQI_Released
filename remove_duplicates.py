import sys
from tqdm import tqdm
lines_seen = set() # holds lines already seen
outfile = open(sys.argv[2], "w")
for line in tqdm(open(sys.argv[1], "r")):
    if line not in lines_seen: # not a duplicate
        outfile.write(line)
        lines_seen.add(line)
outfile.close()
