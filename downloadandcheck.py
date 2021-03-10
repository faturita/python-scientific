"""
===============================
Download an online JSON dataset
===============================

How to download json files and process them.

"""
print(__doc__)

import os
import urllib.request, json
import math

def downloadChunks(url):
    """Helper to download large files
        the only arg is a url
       this file will go to a temp directory
       the file will also be downloaded
       in chunks and print out how much remains
    """

    baseFile = os.path.basename(url)

    #move the file to a more uniq path
    #os.umask(0002)
    temp_path = "./"
    try:
        file = os.path.join(temp_path,baseFile)

        req = urllib.request.urlopen(url)
        for key,value in req.getheaders():
            if (key == 'Content-Length'):
                total_size = int(value.strip())
        downloaded = 0
        CHUNK = 256 * 10240
        with open(file, 'wb') as fp:
            while True:
                chunk = req.read(CHUNK)
                downloaded += len(chunk)
                print(math.floor( (downloaded / total_size) * 100 ))
                if not chunk: break
                fp.write(chunk)
    except Exception as e:
        print ("HTTP Error:",e , url)
        return False
    except Exception as e:
        print ("URL Error:",e , url)
        return False

    return file



import urllib.request, json
url = 'http://cdn.buenosaires.gob.ar/datosabiertos/datasets/actividades-estaticas/actividad-estatica-filas-mtb-cts.geojson'
response = urllib.request.urlopen(url)
rawcheckout = json.loads(response.read())
print(json.dumps(rawcheckout, indent=4, sort_keys=True))

# Let's analyze the json.  First check the type

print(type(rawcheckout))

# It is a dictionary, lets see the keys

for i in rawcheckout.keys(): 
    print(i) 

# 'Features' seems interesting, let's see it
print(type(rawcheckout['features']) )

sx = []
sy = []
for feat in rawcheckout['features']:
    loc = feat['geometry']['coordinates']
    sx.append(loc[0])
    sy.append(loc[1])

# Each element is a dictionary, where data is finally located.


import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(sx,sy,  c='r', marker=".", label='Metrobus stops')
plt.legend(loc='upper left');
plt.show()