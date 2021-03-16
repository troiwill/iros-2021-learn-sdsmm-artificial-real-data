# Learning State-Dependent Sensor Measurement Models with Limited Sensor Measurements

----
## System Requirements

This repository was developed and tested on systems with the following specifications:
- Ubuntu 20.04 LTS and MacOS Catalina (version 10.15.7)
- Python3 (version 3.7.2 up to 3.8.5)

## Download and Environment Setup

__Note:__ Python3 was used for this project.

Run the following commands to download and set up an environment for this project. It is recommended that you run the commands one-by-one.
```
# Create a repo directory for this project, and then clone the repo.
mkdir -p $HOME/repos && cd $HOME/repos
git clone https://github.com/troiwill/iros-2021-learn-sdsmm-artificial-real-data.git iros-2021-sdsmm

# Setup a virtual environment for this project.
cd iros-2021-sdsmm/scripts/env
source setup.sh
bash install_packages.sh
```
You only need to perform this setup once. Each time you open a new terminal environment, run ```source setup.sh``` to start the Python virtual environment.

The rest of this document assumes that the Python virutal environment is active.

------------
## Quick run

First, unpack the data for this project. (Use the terminal commands in the section titled "Data"). Then run the following commands to test the localizer script for the Extended Kalman Particle Filter.

```
# Change to the localizer directory.
cd $IROS21_SDSMM/scripts/localize

# Run the localizer using Course 1, Robot 1, and the A Priori sensor measurement model.
python run_localizer.py 1 1 apriori

# Compute the localization results and graph robot trajectory.
python calc_localize_result.py $IROS21_SDSMM/exps/localize/apriori/robot-1/apriori/ekpf/data-1/results.h5
```

The output graph (from calc_localize_result.py) shows the predicted robot positions from the EKPF and the true robot positions.

![Predicted trajectory for Robot 1 on Course 1](/docs/figures/position_graph_crs1_robot1_apriori_ekpf.svg)

-------
## Data

The **data** directory contains the learning and localization data used for this project. The data was derived from the MR.CLAM dataset from the University of Toronton. Run the terminal command below to unpack all the ZIP files. The ZIP files will unpack into directories called *clean* and *learn*.

```
cd $IROS21_SDSMM/data/mrclam
for p in $(ls *.zip); do unzip $p; done
```

### Learning Data
The following ZIP files contain training data for the Mixture Density Network (MDN). Please refer to the paper to determine the MDNs were trained.

* **mrclam_learn_indiv_data.zip** - Used to train RDO models.
* **artificial_learn_data.zip** - Artificial data used to pre-train an MDN. MDN is later fine-tuned.
* __bootstrapped_mrclam_learn_data_robot*.zip__ - Bootstrapped training data for robot*. This dataset has two purposes.
    1. Used to fine-tune the pre-trained MDN (FLRD models).
    2. Used to train an MDN *without* pre-training (LRDO models).


### Localization Data
The **courses_data_assoc.zip** file contains data for localization. Compared to the original MR.CLAM dataset, some measurement data was removed due to possible landmark mismatches.

----------------------
## Performing Training

**Relevant files and data:**
- [mrclam_learn_indiv_data.zip](/data/mrclam/mrclam_learn_indiv_data.zip)
- [artificial_learn_data.zip](/data/mrclam/artificial_learn_data.zip)
- [bootstrapped_mrclam_learn_data_robot{1-5}.zip](/data/mrclam/)
- [train.py](/scripts/training/train.py)

To determine how to run the training script, run ```python train.py -h``` in the terminal.

If you have questions regarding the training parameters for the networks, please send me an email.

----------------------------
## Performining Localization

**Relevant files and data:**
- [courses_data_assoc.zip](/data/mrclam/courses_data_assoc.zip)
- [run_localizer.py](/scripts/localize/run_localizer.py)
- [calc_localize_result.py](/scripts/localize/calc_localize_result.py)

To determine how to run the localizer, run ```python run_localizer.py -h``` in the terminal.

Once you ran the localization script, you can check yuor results using calc_localize_result.py. To determine how to run this script, run ```python calc_localize_result.py -h``` in the terminal.
