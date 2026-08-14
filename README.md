# synreal-mujoco
synreal-mujoco coupling solver

# install guide
the main idea of installation is to install style3dsim/forked mujoco/mujoco_sytle3d into one python venv. here is a recommanded steps:
1. clone style3d forked mujoco repo(https://github.com/SynReal/mujoco) and switch to branch style3d
2. create python virtual env and activate it 
3. pip install style3dsim*.whl
4. run install_py_package.py in mujoco/python folder to generate mujoco python wheel (will output to folder dist)
4.1 install the forked mujoco: cd dist && pip install mujoco*.whl
5. clone mujoco_style3d, and cd mujoco_style3d ,and pip install -e . (within the same venv created in forked mujoco folder)


# examples
There are several examples in mujoco_style3d/examples folder. 
See mj_py_cloth.py first.
mujoco_style3d is just a wrapper of style3dsim for coupling with mujoco, so users can use style3dsim py api directly to set physical properties instead of setting in mujoco xml.
The c style style3dsim plugin in mujoco is deprecated.

# F&Q
1. run install_py_package.py on win with error
a: Turn on uft-8 support on win first. If use python<=3.11, fix VIRTUAL_ENV in .venv/Scripts/activate manually, for example, change VIRTUAL_ENV from "F:\mujoco\python\.venv" to "/f/mujoco/python/.venv"
