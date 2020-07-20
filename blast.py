from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.Blast import NCBIXML, Record
from glob import glob
from Bio import SeqIO
from io import StringIO
import os
from Bio.Blast import NCBIWWW
import time
import datetime

while True:
    print("start",datetime.datetime.now())
    start = time.time()
    g = NCBIWWW.qblast("blastn", "nt", "8332116")
    end = time.time()
    print("end",datetime.datetime.now())
    print("total",end - start)
    print("###########################################")


# start 2020-07-19 23:50:09.877776
# end 2020-07-20 00:13:15.340926
# total 1385.4631497859955
# ###########################################
# start 2020-07-20 00:13:15.340926
# end 2020-07-20 00:44:35.577495
# total 1880.2365696430206
# ###########################################
# start 2020-07-20 00:44:35.577495
# end 2020-07-20 01:10:55.522578
# total 1579.9450829029083
# ###########################################
# start 2020-07-20 01:10:55.522578
# end 2020-07-20 01:28:15.991557
# total 1040.468978881836
# ###########################################
# start 2020-07-20 01:28:15.992554
