## Download and Environment Setup

__Note:__ It is assumed that you have Python3 installed.

Run the following commands to download and set up an environment for this project.
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

Run the following commands to test the localizer script for the Extended Kalman Particle Filter.
```
# Change to the localizer directory.
cd $IROS21_SDSMM/scripts/localize

# Run the localizer using Course 1, Robot 1, and the A Priori sensor measurement model.
python run_localizer.py 1 1 apriori

# Compute the localization results and graph robot trajectory.
python calc_localize_result.py $IROS21_SDSMM/exps/localize/robot-1/apriori/ekpf/data-1/results.h5
```

The output graph (from calc_localize_result.py) shows the true robot poses and the predicted robot poses from the EKPF.

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
* **articial_learn_data.zip** - Artificial data used to pre-train an MDN. MDN is later fine-tuned.
* __bootstrapped_mrclam_learn_data_robot*.zip__ - Bootstrapped training data for robot*. This dataset has two purposes.
    1. Used to fine-tune the pre-trained MDN (FLRD models).
    2. Used to train an MDN *without* pre-training (LRDO models).


### Localization Data
The **courses_data_assoc.zip** file contains data for localization. Compared to the original MR.CLAM dataset, some measurement data was removed due to possible landmark mismatches.