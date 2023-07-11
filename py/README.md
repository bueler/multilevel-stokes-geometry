# py/

The codes here need FASCD.  First do one of the following in the current directory:

        git clone git@bitbucket.org:pefarrell/fascd.git   # clone private repo

or

        ln -s ~/repos/fascd/                              # sym link existing repo

Then do:

        source ~/firedrake/bin/activate                   # activate Firedrake venv
        python3 moveice.py
