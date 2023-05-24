import numpy as np

stop_pt = 200

def histmax(hist,stop_pt):
    # Convert histogram to simple list
    hist = [val[0] for val in hist]
    # Find maximum 
    return np.argmax(hist[0:stop_pt])

def histoverlap(hist1, hist2):
    tot1 = sum(hist1[:stop_pt, 0])
    tot2 = sum(hist1[:stop_pt, 0])
    tot = tot1 + tot2
    mins = [0] * (stop_pt)
    for i in range(stop_pt):
        min = np.minimum(int(hist1[i,0]), int(hist2[i,0]))
        mins[i] = min
    return sum(mins) / tot

def histoverlap(hist1, hist2):
    tot1 = sum(hist1[:stop_pt, 0])
    tot2 = sum(hist1[:stop_pt, 0])
    tot = tot1 + tot2
    mins = [0] * (stop_pt)
    for i in range(stop_pt):
        min = np.minimum(int(hist1[i,0]), int(hist2[i,0]))
        mins[i] = min
    return sum(mins) / tot
