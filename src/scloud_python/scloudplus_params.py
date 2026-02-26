import numpy as np


class SCloudPlusParams:
 def __init__(self, l:int):
  assert l==128 or l==192 or l==256, "Wrong Security Level (128/192/256)"
  if l==128:
   self.l = l
   self.ss = 16
   self.m = 600
   self.n = 600
   self.mbar = 8
   self.nbar = 8
   self.logq = 12
   self.logq1 = 9
   self.logq2 = 7
   self.h1 = 150
   self.h2 = 150
   self.eta1 = 7
   self.eta2 = 7
   self.mu = 64
   self.tau = 3
   self.subm = 2
   self.block_number = 75
   self.block_size = 4
   self.block_rowlen = 300
   self.c1 = 5400 # MBAR*N*LOGQ1/sizeof(uint8_t)
   self.c2 = 56
   self.ctx = 5456
   self.pk = 7216
   self.pke_sk = 1200
   self.kem_sk = 8480
   self.m2 = 360000
   self.m3 = 216000000
   self.n2 = 360000
   self.n3 = 216000000
   self.mnin = 679
   self.mnout = 816 #582
   self.rejblocks = 7
   self.bm_size = 10
  if l==192:
   self.l = l
   self.ss = 24
   self.m = 928
   self.n = 896
   self.mbar = 8
   self.nbar = 8
   self.logq = 12
   self.logq1 = 12
   self.logq2 = 10
   self.h1 = 224
   self.h2 = 232
   self.eta1 = 2
   self.eta2 = 1
   self.mu = 96
   self.tau = 4
   self.subm = 2
   self.block_number = 112
   self.block_size = 4
   self.block_rowlen = 448
   self.c1 = 10752
   self.c2 = 80
   self.ctx = 10832
   self.pk = 11152
   self.pke_sk = 1792
   self.kem_sk = 13008
   self.mnin = 671
   self.mnout = 544 #488
   self.rejblocks = 5
   self.bm_size = 15
  if l==256:
   self.l = l
   self.ss = 32
   self.m = 1136
   self.n = 1120
   self.mbar = 12
   self.nbar = 11
   self.logq = 12
   self.logq1 = 10
   self.logq2 = 7
   self.h1 = 280
   self.h2 = 284
   self.eta1 = 3
   self.eta2 = 2
   self.mu = 64
   self.tau = 3
   self.subm = 4
   self.block_number = 140
   self.block_size = 4
   self.block_rowlen = 560
   self.c1 = 16800
   self.c2 = 116
   self.ctx = 16916
   self.pk = 18760
   self.pke_sk = 3080
   self.kem_sk = 21904
   self.n2 = 1254400
   self.n3 = 1404928000
   self.n4 = 1573519360000
   self.n5 = 1762341683200000
   self.m2 = 1290496
   self.m3 = 1466003456
   self.m4 = 1665379926016
   self.m5 = 1891871595954176
   self.mnin = 680
   self.mnout = 320 #530
   self.rejblocks = 3
   self.bm_size = 18

scloudplus128 = SCloudPlusParams(128)
scloudplus192 = SCloudPlusParams(192)
scloudplus256 = SCloudPlusParams(256)


ScpParams = { 128: SCloudPlusParams(128)
            , 192: SCloudPlusParams(192)
            , 256: SCloudPlusParams(256)
            }