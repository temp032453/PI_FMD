#  Property Inference Attacks on FedMD via Synthetic Data Generation
Code for the paper "PI-FMD : Property Inference Attacks on FedMD via Synthetic Data Generation."

## file structure
```markdown
├── run_attack.py      # Main script for executing the attack
├── dataset/           # Folder containing datasets
├── models/            # Pre-trained model files
├── profinf/           # Attackunitl or data preprocessing files
└── README.md          # This documentation
```

## Each of the Arguments
```shell
  -dat DATASET, --dataset DATASET
                        dataset name : [adult, bank, census]
  -tp TARGETPROPERTIES, --targetproperties TARGETPROPERTIES
                        list of categories and target attributes. e.g. [(sex, Female), (occupation, Sales)]
  -t0 T0FRAC, --t0frac T0FRAC
                        t0 fraction of target property
  -pant PARTICIPANTS, --participants PARTICIPANTS
                        the number of participants
  -t1 T1FRAC, --t1frac T1FRAC
                        t1 fraction of target property
  -tpub TPUBFRAC, --tpubfrac TPUBFRAC
                        tpub fraction of target property
  -sm SHADOWMODELS, --shadowmodels SHADOWMODELS
                        number of shadow models
  -pw PUB_ENT_W, --pub_ent_w PUB_ENT_W
                        public entropy loss weight
  -ps POISONING_START, --poisoning_start POISONING_START
                        poisoning start round
  -pp PUBLICPOISON, --publicpoison PUBLICPOISON
                        list of poison percent
  -d DEVICE, --device DEVICE
                        PyTorch device
  -nt NTRIALS, --ntrials NTRIALS
                        number of trials
  -pst POISONSTART, --poisonstart POISONSTART
                        number of poisonstart
  -e EPOCHS, --epochs EPOCHS
                        number of epochs
  -ag AGGR, --aggr AGGR
                        logit aggregation method possible values: [fedmd, fedgems, dsfl]
  -piwp PIWP_MODE, --piwp_mode PIWP_MODE
                        make inference with piwp
```

## Experiment Command Examples
```shell 
## adult (baseline)
python run_attack.py -tp="[(workclass, Private)]" -t0=0.2 -t1=0.4 -e 50 -d 0 -m 0 -tpub 1.5 
python run_attack.py  -tp="[(race, White),(sex, Male)]" -t0=0.10 -t1=0.3 -d 0 -m 0 -e 50 -tpub 2
python run_attack.py -tp="[(occupation, Craft-repair)]" -t0=0.02 -t1=0.08 -d 0 -m 0 -e 50 -m 0 -tpub 3 
python run_attack.py -tp="[(marital-status, Divorced),(sex, Male)]" -t0=0.01 -t1=0.05 -d 1 -e 50 -tpub 3 

```

```shell
## adult (ours)
python run_attack.py -tp="[(workclass, Private)]" -t0=0.2 -t1=0.4 -e 50 -d 0 -m 0 -tpub 1.5 -pw 2 -pds
python run_attack.py  -tp="[(race, White),(sex, Male)]" -t0=0.10 -t1=0.3 -d 0 -m 0 -e 50 -tpub 2 -pw 2 -pds
python run_attack.py -tp="[(occupation, Craft-repair)]" -t0=0.02 -t1=0.08 -d 0 -m 0 -e 50 -m 0 -tpub 3 -pw 2 -pds
python run_attack.py -tp="[(marital-status, Divorced),(sex, Male)]" -t0=0.01 -t1=0.05 -d 1 -e 50 -tpub 3 -pw 2 -pds

```
