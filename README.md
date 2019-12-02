# python-scientific
Quick guide and tutorial to scientific data python programming


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
  
  # Como actualizar el repo con nuevos cambios?
  
  Desde gitbash (windows) o desde una consola en Mac o Linux
  
  Primero hay que subir los cambios que cada uno haya hecho
  
  ```bash
  git commit -m"ESCRIBAN ACA EL DETALLE DE LO QUE MODIFICARON PARA QUE QUEDE REGISTRADO" .
  ```
  
  Después traigan los cambios desde el servidor a su repo local.
  
  ```bash
  git pull origin master
  ```
  
  Si tienen cambios este comando va a forzar un merge automático.  Si el merge esta ok les abre una consola de VI u otro editor (para salir de vi apreten ':' y después 'x').  Si hay conflictos, revisen los archivos que están con conflicto (se van a dar cuenta) y después marcan el conflicto resuelto con 'git add nombrearchivo' y luego 'git commit-m"Conflicto resuelto"'
  
 
 
 
