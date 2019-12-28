# python-scientific

Nowadays, python is ruling the world.  Hence, lets rule python....

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
 
  * Activate the newly created environment
  ```bash
  conda activate mne3
  ```
  
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
  
# References and sources

* https://towardsdatascience.com/t-sne-python-example-1ded9953f26
* https://github.com/lmcinnes/umap
* More info an references [here](http://monostuff.logdown.com/posts/7835544-aprendizaje-modo-mquina)

