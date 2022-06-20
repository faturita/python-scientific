<p align="center">
    <img alt="node-express-starter" width="45%" height="auto" src="./images/signal.png" />
    <img alt="node-express-starter" width="45%" height="auto" src="./images/smoothed.png" />
</p>


<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/f/fc/Blue_Python_3.7_Shield_Badge.svg"/>
</p>



# 📟  Python Scientific 

Nowadays, python is ruling the world.  This very humble language that was once the realm of enlighted developers that loved to create the most beautiful scripting language is now becoming the programming language of the world, and particularly of the data deluge.

This repository allows you to install all the necesary stuff to start working with python on scientific applications, and particularly, to work with sensor data and neuroscience time-series.  It allows you to install everything you need and start coding with NumPy, SciPy, OpenAI and MNE.

If you don't know anything at all about python, and you want to know something that will help you to do what you need, this is one place to be.

## Contents

* 🔪 Python Introduction
* 📥 Workspace Setup
* 📈 Signal Processing Overview
* 🔎 Signal Analysis and Feature Extraction
* 🛠 Python ML/DL Tools and Techniques
* 🧠 Neurodata Processing


## Procedure

* Install Git Bash
  https://git-scm.com/downloads
  
* Install Anaconda

  Download Anaconda 3.7 for your platform and install it.
  https://www.anaconda.com/distribution/
 
 * Install Visual Studio Code
 
  Download and install it from https://code.visualstudio.com/download
  
 * Clone this repository
 
  From the Git Bash console, run
  
 ```bash
 git clone https://github.com/faturita/python-scientific.git
 ```
 
 * Run an Anaconda Prompt
 * Move with "cd" to the directory that you just downloaded ("cd python-scientific")
 * Create the environment with:
 
 ```bash 
 conda env update --name mne3 --file config/environmentw.yml
 ```

 (or this one if any error occurs
 
 ```
 conda env update --prefix ./env --file config/environmentw.yml 
 ```

 )
 
 * Activate the newly created environment
 ```bash
 conda activate mne3
 ```

NOTE: if you receive an error about 'umap package missing' or similar, just ''edit'' your local file [environmentw.yaml](config/environmentw.yml) and erase the umap line from the file.  You can later install this package directly from an Anaconda prompt by doing:

 ```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
 ```

 * Install umap (or any package that you want)
 ```bash
 conda install -n mne3 umap
 ```

ó

 ```bash
 conda install -n mne3 umap-learn
 ```

 Conda Cheat sheet: https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

<details>
<summary>🗄 What's on each file</summary>

* [holamundo.py](holamundo.py): Basic Python Data Science sample.  Grab a dataset and visualize it.
* [holamundonotebook.ipynb](holamundonotebook.ipynb): The same, as jupyter notebook.
* [onepasspython.py](onepasspython.py): Basic python 3.x structures, sentences and control keywords.
* [diveintopython3.py](diveintopython3.py): Some python stuff particular to python 3.
* [numpies.py](tensors.py): Basic cookbook on how to deal with numpy tensors.
* [darawrangling.py](datawrangling.py): Basic cookbook for data wrangling in Pandas.
* [downloadandcheck.py](downloadandcheck.py): Basic sample to get an online dataset and plot it.
* [scientificnotation.py](scientificnotation.py): Some numerical stuff for python.
* [filter_temporal.py](filtrostemporales.py): Convolution and numpy slicing.
* [filter_spectral.py](filtrosespectrales.py): Basic sample to extract spectral characteristics from time series.
* [filter_spatial.py](filtrosespaciales.py): Blind source separation sample.
* [onepassfeatureextraction.py](onepassfeatureextraction.py): Program to extract features from an image using opencv.
* [onepassclassifier.py](onepassclassifier.py): Program to classify those features using different classifiers.  This contains all the elements to implement a classification model.
* [InteractiveNotebook.ipynb](InteractiveNotebook.ipynb): Jupyter notebook sample for creating visualizing tools with Altair.
* [baseline.py](baseline.py): Signal baseline removal sample.
* [signalfeatures.py](signalfeatures.py): Basic set of feature extraction procedures for signals (time series).
* [umbralizador.py](umbralizador.py): Otsu method to umbralize a one dimensional time series.
* [eventcounter.py](eventcounter.py): Signal peak counting sample.
* [clustering.py](clustering.py): Clustering sample containing kmeans and dbscan.
* [tsneclustering.py](ejemploclusteringtsne.py): tSNE dimensionality reduction sample on MNIST.
* [umapclustering.py](ejemploclusteringumap.py): UMAP dimensionality reduction sample on MNIST.
* [signalfeatureclassification.py](signalfeatureclassification.py): Runs on environmentann3w.yml environment. Process an EEG signal and detects alpha suppression with eyes closed.  Classify them using Keras.
* [qlearning.py](qlearning.py): Basic OpenAI Gym sample for QLearning.
* [taxilearning.py](taxilearning.py): This is a basic Gym scenario to verify RL optimization.
* [choppersample.py](choppersample.py): A basic game that can be used to test different AI strategies to solve.
* [easygame.py](easygame.py): Pygame engine that can be used as a template.


</details>

# How to update your own repository with new changes from the server repository ?
  
  Run gitbash (windows) or start a new Console on Mac or Linux.

  First you need to upload your own changes to your OWN repository.
  
  ```bash
  git commit -m"Just write down whatever comes to your mind here" .
  ```
  
  After that you need to PULL fresh changes from the server repository at github.
  
  ```bash
  git pull origin master
  ```
  
  If you have happened to modify a file, the previous command will trigger an automatic merge.  If the merge is successful it will open a VI console (just press ':' and 'x') to add a merging comment.
  If there is any conflict, check the modified files looking for any sign of conflict (you will clearly notice it).  After you fix the merging, mark it as resolved with 'git add filename' and finish the operation with 'git commit -m"Merge fixed"'
  
# Do you want to learn python from scratch ?

https://online-learning.harvard.edu/course/using-python-research

# Something else about Git

* learngitbranching.js.org
* atlassian.com/git

# Documentation

* Roadmap for going deeper into [ML](docs/MachineLearning.md)
* [Learning Python](docs/Python.md)
* [Signal Processing](docs/Signals.md)
* [Aprendizaje Modo Maquina](http://monostuff.logdown.com/posts/7835544-aprendizaje-modo-mquina)
  
# References and sources

* Visualizations: https://github.com/rougier/scientific-visualization-book
* Bias on DataScience: https://github.com/koaning/scikit-lego, https://github.com/fairlearn/fairlearn
* https://github.com/ml-tooling/best-of-ml-python Best of Machine Learning resources for Python.
* [MacBook configuration to use GPUs](https://towardsdatascience.com/deep-learning-using-gpu-on-your-macbook-c9becba7c43)
* https://code.visualstudio.com/docs/python/python-tutorial Visual Studio Code tips (debugging line by line included)
* https://github.com/spyder-ide/spyder/wiki/Working-with-packages-and-environments-in-Spyder
* https://jalammar.github.io/visual-numpy/
* [Guía Guit](https://rogerdudler.github.io/git-guide/index.es.html)
* [Tom O'Haver Pragmatic Introduction to Signal Processing](https://terpconnect.umd.edu/~toh/spectrum/): Awesome guide on tools to tackle signal processing.
* Imbalanced Datasets: https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/ and https://link.springer.com/article/10.1186/s40537-018-0151-6
* [Gaussian Processes](https://yugeten.github.io/posts/2019/09/GP/): A tutorial on Gaussian Processes.
* https://towardsdatascience.com/t-sne-python-example-1ded9953f26
* https://github.com/lmcinnes/umap
* https://towardsdatascience.com/t-sne-python-example-1ded9953f26
* https://github.com/aaizemberg/: Everything you would like to know about Visualizations, GIS, databases, and big data science.
* https://github.com/ezequielma20/data-science: Excellent resource of this great Data Science Dev from Baufest.
* https://github.com/jbruestle/plaidvision PlaidML Great Sample
* https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
* https://medium.com/odscjournal/data-scientists-versus-statisticians-8ea146b7a47f
* https://github.com/healthDataScience/deep-learning-HAR
* https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9
* https://www.freecodecamp.org/news/how-to-build-your-first-neural-network-to-predict-house-prices-with-keras-f8db83049159/
* https://towardsdatascience.com/dcgans-deep-convolutional-generative-adversarial-networks-c7f392c2c8f8
* https://medium.com/@ODSC/logistic-regression-with-python-ede39f8573c7
* [Python Data Structures](https://github.com/bhavinjawade/)Advanced-Data-Structures-with-Python
* [Cross Entropy](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)
* [Visual Information Theory](http://colah.github.io/posts/2015-09-Visual-Information/)


