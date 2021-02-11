#! /usr/bin/env python3

#SBATCH --mem 4G
#SBATCH --time 1:00:00

import pickle
import numpy as np
import sys
import math
import json

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib

import argparse
parser = argparse.ArgumentParser('Combines and visualizes textual'+
                                 ' and visual video segmentation results')
parser.add_argument('--data', type=str,
                    choices=['ina', 'urheiluruutu'],
                    help='dataset to use')
parser.add_argument('--visual-segment-matrix', type=str,
                    help='pickle result from text-based segmentation')
parser.add_argument('--visual-shot-boundaries', type=str,
                    help='pickle result from text-based segmentation')
parser.add_argument('--text-segment-result', type=str,
                    help='pickle result from text-based segmentation')
parser.add_argument('video', type=str,
                    help='label of the video, such as 5266518001 or 202000823730')
                    
args = parser.parse_args()

#print(matplotlib.__version__)

if args.data:
    xset = args.data+'_'

def equalize(m):
    v = []
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v.append(m[i, j])
    v.sort()
    x = {}
    j = 0
    for i in v:
        if not i in x:
            x[i] = j/(len(v)-1)
        j += 1

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            m[i, j] = x[m[i, j]]
            
    return m

v = args.video

if args.visual_segment_matrix:
    z = np.array(json.load(open(args.visual_segment_matrix, 'rb')))
else:
    z = np.load('visual-input/'+v+'-eq.npy')
    
if args.visual_shot_boundaries:
    t = json.load(open(args.visual_shot_boundaries, 'rb'))
else:
    t = np.load('visual-input/'+v+'-time.npy')
#print(t)

d = math.ceil(t[-1])
m = np.zeros((d, d))

x = []
for i in range(d):
    for a, b in enumerate(t):
        if b>i:
            break
    x.append(a)
        
for i in range(d):
    for j in range(d):
        m[i, j] = z[x[i], x[j]]

#plt.matshow(z)
#plt.show()
#plt.matshow(m)
#plt.show()

mm = [ m ] 

vx = v
if args.data=='ina':
    vx = vx[:7]+'_'+vx[-3:]
elif args.data=='urheiluruutu':
    vx = 'PROG_'+vx[:4]+'_'+vx[4:]

if args.text_segment_result:
    text_segment_results = pickle.load(open(args.text_segment_result, 'rb'))
    #pt = text_segment_results['similarities']
    pt = {}
    pt['start'] = []
    pt['end'] = []
else:
    pt = pickle.load(open('textual-input/'+xset+'parts_timestamps.pickle', 'rb'))
    pt = pt[vx]

#print(pt)
ptset = set()
for i in pt['start']:
    ptset.add(i)
for i in pt['end']:
    ptset.add(i)
ptset = list(ptset)
ptset.sort()
#print(ptset)

if args.text_segment_result:
    y = text_segment_results['similarity']
else:    
    y = pickle.load(open('textual-input/'+xset+'subtitle_neighborhood_similarity.pickle',
                     'rb'))
    y = y[vx]
#print(y.shape)

if args.text_segment_result:
    s = text_segment_results['start']
    e = text_segment_results['end']
else:
    w = pickle.load(open('textual-input/'+xset+'subtitles_timestamps.pickle', 'rb'))
    w = w[vx]
    #print(w)
    s = w['start']
    e = w['end']
    #print(len(s))

x = []
for i in range(d):
    j = -1
    for a in range(len(s)):
        if i>=s[a] and i<e[a]:
            j = a
            break
    x.append(j)
#print(x)

m = np.zeros((d, d))

for i in range(d):
    for j in range(d):
        if x[i]>=0 and x[j]>=0:
            m[i, j] = (1-y[x[i], x[j]])/2
        else:
            m[i, j] = 1

m = equalize(m)
mm.append(m)

#plt.matshow(mm[0])
#plt.show()
#plt.matshow(mm[1])
#plt.show()
#plt.matshow(mm[0]*mm[1])
#plt.show()
#plt.matshow(mm[0]+mm[1])
#plt.show()

#plt.figure(dpi=2400)

mm.append(equalize(mm[0]+mm[1]))
mm.append(equalize(mm[0]*mm[1]))

def seg_value(m, real):
    l = 30
    #return np.random.randint(0, 101, m.shape[0])
    h = m.shape[0]//5
    r = np.zeros((h, m.shape[0]))
    if not real:
        return r
    vv = []
    for t in range(1, r.shape[1]):
        v = 0
        for i in range(-l, l):
            if t+i>=0 and t+i<m.shape[1]:
                if i<0:
                    v += m[t, t+i] - m[t-1, t+i]
                else:
                    v += m[t-1, t+i] - m[t, t+i]
        vv.append(v)

    vmax = np.max(vv)
        
    for t in range(1, r.shape[1]):
        v = int(vv[t-1]/vmax*h)
        for y in range(v):
            r[h-1-y, t] = 1
            
    return r, np.concatenate((np.array([0]), vv/vmax))
    
def gt_value(m, real):
    h = m.shape[0]//15
    r = np.zeros((h, m.shape[0]))
    if not real:
        return r
    for t in range(1, r.shape[1]):
        found = False
        for i in ptset:
            #print(t, i)
            if i-t>-0.5 and i-t<0.5:
                found = True
                break
        if found:
            for y in range(h):
                r[y, t] = 1
    return r

mmseg = []
for i in range(len(mm)):
    seg, vvv = seg_value(mm[i], True)
    #print(mm[i].shape, vvv.shape, vvv)
    gt = gt_value(mm[i], True)
    mm[i] = np.concatenate((mm[i], seg, gt))

    segs = []
    l = 10
    while True:
        mi = np.argmax(vvv)
        if vvv[mi]<=0:
            break
        segs.append((mi, vvv[mi]))
        for j in range(-l, l+1):
            if mi+j>=0 and mi+j<vvv.shape[0]:
                vvv[mi+j] = 0
    mmseg.append(segs)
    
#fig = plt.figure(constrained_layout=True)
#heights = [3, 1, 3, 1]
#widths = [3, 3, 3]
# spec = fig.add_gridspec(nrows=4, ncols=3,
#                         height_ratios=heights, width_ratios=widths)

# print(type(fig.add_subplot(spec[0, 0])))
# axs = np.empty(shape=(len(heights), len(widths)), dtype=type(fig.add_subplot()))
# print(axs.shape)
# for i in range(axs.shape[0]):
#     for j in range(axs.shape[1]):
#         axs[i, j] = fig.add_subplot(spec[i, j])

yx = np.concatenate((y, seg_value(y, False), gt_value(y, False)))
zx = np.concatenate((z, seg_value(z, False), gt_value(z, False)))

fig, axs = plt.subplots(2, 3, constrained_layout=True)

fig.suptitle(vx)
ip = 'bilinear'

axs[0, 0].matshow(yx)
axs[0, 1].matshow(mm[1], interpolation=ip)
axs[0, 2].matshow(mm[3], interpolation=ip)
axs[1, 0].matshow(zx)
axs[1, 1].matshow(mm[0], interpolation=ip)
axs[1, 2].matshow(mm[2], interpolation=ip)
#plt.show()
plt.savefig(vx+'.png', dpi=200)

midxx = [ (1, 'txt') , (0, 'vis'), (2, 'sum') , (3, 'mul') ]

out = open(vx+'.txt', 'w')
for midx, mnam in midxx:
    #print(midx, mnam, mmseg[midx])
    print(mnam, end=' ', file=out)
    for i, j in mmseg[midx]:
        print('({} {:.4f})'.format(i, j), end=' ', file=out)
    print(file=out)
