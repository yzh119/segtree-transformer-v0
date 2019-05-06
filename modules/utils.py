def clear_feature(g):
    for k in list(g.ndata.keys()):
        g.ndata.pop(k)
    for k in list(g.edata.keys()):
        g.edata.pop(k)

