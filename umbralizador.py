def otsu(eeg):
    h = np.histogram(eeg, bins=100)
    bins = h[1]
    p = h[0] / len(h[0])

    def w1(h,t):
        s = h[0:t].sum()
        return s

    def w2(h,t):
        s = h[t:].sum()
        return s

    def mu1(h,t): 
        val = 0 
        for i in range(0,t): 
            val = val + (i * h[i])/(w1(h,t))   
        return val

    def mu2(h,t): 
        val = 0 
        for i in range(t,len(h)): 
            val = val + (i * h[i])/(w2(h,t))  
        return val

    def sigma1(h,t):
        val = 0
        for i in range(0,t):
            val = val + ((i - mu1(h,t))**2) * h[i] / w1(h,t)
        return val

    def sigma2(h,t):
        val = 0
        for i in range(0,t):
            val = val + ((i - mu2(h,t))**2) * h[i] / w2(h,t)
        return val

    maxval = 0
    maxt = 0
    for t in range(1,len(p)-1):
        print(t)
        val = w1(p,t)*sigma1(p,t) + w2(p,t)*sigma2(p,t)
        print(val)
        if (val > maxval):
            maxval = val
            maxt = t

    print('Trheshold value:' + str(maxt))
    print('Otsu threshold:' + str(bins[maxt]))

