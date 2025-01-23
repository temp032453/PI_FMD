#  Property Inference Attacks on FedMD via Synthetic Data Generation
Code for the paper PI-FMD : Property Inference Attacks on FedMD via Synthetic Data Generation.

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

```shell 
## adult
python run_attack.py -tp="[(workclass, Private)]" -pp=1.0 -t0=0.2 -t1=0.5 -e 50 -d 0 -m 0 -tpub 1.0
python run_attack.py  -tp="[(race, White),(sex, Male)]" -p="[0.15]" -t0=0.15 -t1=0.40 -pp=1.0 -d 1 -m 0 -e 50 -tpub 1.0
python run_attack.py -tp="[(sex, Female),(occupation, Sales)]" -p="[0.01]" -t0=0.01 -t1=0.035 -d 2 -m 0 -pp 1.0 -e 50 -m 0 -tpub 1.0
python run_attack.py -tp="[(marital-status, Divorced),(sex, Male)]" -t0=0.01 -t1=0.05 -d 2 -e 50 -pp=1.0 -m 0 -tpub 1.0
```
