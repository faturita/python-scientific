# Python Scientific

Nowadays, python is ruling the world.  Hence, lets rule python....
This repository allows you to install all the necesary stuff to start working with python on scientific applications, and particularly, to work with neuroscience time-series.  It allows you to install everything you need and start coding with NumPy, SciPy, OpenAI and MNE.

If you don't know anything at all about python, this is the place to be.


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
 * Create the environment with:
 
 ```bash 
 conda env update --name mne3 --file environmentw.yml
 ```

 (or this one if any error occurs
 
 ```
 conda env update --prefix ./env --file environmentw.yml 
 ```

 )
 
  * Activate the newly created environment
  ```bash
  conda activate mne3
  ```

# What's on each file

* [holamundo.py](holamundo.py): Basic Python Data Science sample.  Grab a dataset and visualize it.
* [holamundonotebook.ipynb](holamundonotebook.ipynb): The same, as jupyter notebook.
* [onepasspython.py](onepasspython.py): Basic python 3.x structures, sentences and control keywords.
* [diveintopython3.py](diveintopython3.py): Some python stuff particular to python 3.
* [tensors.py](tensors.py): Basic cookbook on how to deal with numpy tensors.
* [downloadandcheck.py](downloadandcheck.py): Basic sample to get an online dataset and plot it.
* [scientificnotation.py](scientificnotation.py): Some numerical stuff for python.
* [filtrostemporales.py](filtrostemporales.py): Convolution and numpy slicing.
* [filtrosespectrales.py](filtrosespectrales.py): Basic sample to extract spectral characteristics from time series.
* [filtrosespaciales.py](filtrosespaciales.py): Blind source separation sample.
* [onepassfeatureextraction.py](onepassfeatureextraction.py): Program to extract features from an image using opencv.
* [onepassclassifier.py](onepassclassifier.py): Program to classify those features using different classifiers.  This contains all the elements to implement a classification model.
* [InteractiveNotebook.ipynb](InteractiveNotebook.ipynb): Jupyter notebook sample for creating visualizing tools with Altair.
* [baseline.py](baseline.py): Signal baseline removal sample.
* [signalfeatures.py](signalfeatures.py): Basic set of feature extraction procedures for signals (time series).
* [umbralizador.py](umbralizador.py): Otsu method to umbralize a one dimensional time series.
* [contadoreventos.py](contadoreventos.py): Signal peak counting sample.
* [clustering.py](clustering.py): Clustering sample containing kmeans and dbscan.
* [ejemploclusteringtsne.py](ejemploclusteringtsne.py): tSNE dimensionality reduction sample on MNIST.
* [ejemploclusteringumap.py](ejemploclusteringumap.py): UMAP dimensionality reduction sample on MNIST.
* [signalfeatureclassification.py](signalfeatureclassification.py): Runs on environmentann3w.yml environment. Process an EEG signal and detects alpha suppression with eyes closed.  Classify them using Keras.
* [qlearning.py](qlearning.py): Works with environmentaiw.yml. Basic OpenAI Gym sample for QLearning.


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
  
  If you have modified some file, this will trigger an automatic merge.  If the merge is successful it will open a VI console (just press ':' and 'x') to add a merging comment.
  If there is any conflict, check the modified files looking for any sign of conflict (you will clearly notice it).  After you fix the merging, mark it as resolved with 'git add filename' and finish the operation with 'git commit -m"Merge fixed"'
  
# Do you want to learn python from scratch ?

https://online-learning.harvard.edu/course/using-python-research

# Something else about Git

* learngitbranching.js.org
* atlassian.com/git
  
# References and sources

* Visual Studio Code tips (debugging line by line included): https://code.visualstudio.com/docs/python/python-tutorial
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
* https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
* https://medium.com/odscjournal/data-scientists-versus-statisticians-8ea146b7a47f
* https://github.com/healthDataScience/deep-learning-HAR
* https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9
* https://www.freecodecamp.org/news/how-to-build-your-first-neural-network-to-predict-house-prices-with-keras-f8db83049159/
* https://towardsdatascience.com/dcgans-deep-convolutional-generative-adversarial-networks-c7f392c2c8f8
* https://medium.com/@ODSC/logistic-regression-with-python-ede39f8573c7
* https://github.com/bhavinjawade/Advanced-Data-Structures-with-Python
* More info an references [here](http://monostuff.logdown.com/posts/7835544-aprendizaje-modo-mquina)

