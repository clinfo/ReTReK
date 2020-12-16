# Installation
## Python environment
**1. Download pyenv**  
```bash
git clone https://github.com/yyuu/pyenv.git $HOME/.pyenv
```

**2. Setup environment for pyenv**  
* Add the following lines to `$HOME/.bash_profile`
```bash
# Setting for pyenv and pyenv-virtualenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv 1>/dev/null 2>&1; then
    eval "$(pyenv init -)"
fi
```

**3. Update your environment (pyenv will be installed)**
```bash
source $HOME/.bash_profile
```

**4. Install pyenv-virtualenv**
```bash
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> $HOME/.bash_profile
source $HOME/.bash_profile
```
### Create RealRetro implementation environment in pyenv
* Install miniconda3.
```bash
pyenv install miniconda3-4.3.27
```

* Create virtual environment using conda
```bash
conda create -n home python=3.7
pyenv global miniconda3-4.2.12/envs/home
# (Only for Ubuntu18.04 users)
# If you create the environment using environment.yml, you don't need to install the dependency libraries.
conda env create -f environment.yml
# (For NOT Ubuntu18.04 users)
conda create -n realretro python=3.7
pyenv shell miniconda3-4.2.12/envs/realretro
```

* Install the dependency libraries.
```
conda install py4j=10.8.1 tqdm oddt mendeleev
conda install -c rdkit rdkit=2020.03.2.0
conda install keras-gpu=2.3.1 tensorflow-gpu=1.13.1 cudatoolkit=10.0
pip install --upgrade git+https://github.com/clinfo/kGCN.git
```

## Java environment (for Ubuntu18.04)
**1. Download jenv**
```bash
git clone https://github.com/jenv/jenv.git ~/.jenv
```

**2. Setup environment for jenv**
* Add the following lines to `$HOME/.bash_profile`
```bash
# Jenv
export PATH=${PATH}:$HOME/.jenv/bin
eval "$(jenv init -)"
```

**3. Update your environment (pyenv will be installed)**
```bash
source $HOME/.bash_profile
```

**4. Install Java8.0**
```bash
sudo apt install openjdk-8-jdk
jenv add /usr/lib/jvm/java-8-openjdk-amd64/
```

**5. Switch the java version to 1.8**
```bash
jenv global 1.8
```

**6. Download the dependency libraries**
- [jChem](https://chemaxon.com/download?dl=%2Fdata%2Fdownload%2Fjchem%2F20.13.0%2Fjchem_unix_20.13.sh)
- [Commons Collections](https://commons.apache.org/proper/commons-collections/download_collections.cgi)
- [args4j](https://search.maven.org/search?q=g:args4j%20AND%20a:args4j)

**7. Install**
```bash
mkdir -p $HOME/opt/jar
sh jchem_unix_20.13.sh  # install in $HOME/opt
# put args4j-2.33.jar and commons-collections4-4.3 in $HOME/opt/jar
```

**8. Add the path for the libralies to CLASSPATH environment variable**
```bash
# ChemAxon
export PATH=$PATH:$HOME/opt/chemaxon/jchemsuite/bin
export CLASSPATH=$CLASSPATH:$HOME/opt/chemaxon/jchemsuite/lib/jchem.jar
# Jar
export CLASSPATH=$CLASSPATH:$HOME/opt/jar/commons-collections4-4.3/commons-collections4-4.3.jar:$HOME/.pyenv/versions/miniconda3-4.3.27/envs/retro/share/py4j/py4j0.10.8.1.jar:$HOME/opt/jar/args4j-2.33.jar
```

**9. Get the ChemAxon license and its installation**

For the license and instllation, please refer to the following instructions.
- [About ChemAxon Licensing](https://docs.chemaxon.com/display/docs/About+ChemAxon+Licensing)
- [License Server Configuration](https://docs.chemaxon.com/display/docs/License_Server_Configuration.html)