import gc
import multiprocessing
import math
import time
import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from functorch import make_functional, vmap
from ctgan.data_transformer import DataTransformer
import numpy as np
from tqdm import tqdm
import propinf.data.ModifiedDatasets as data
from propinf.training import training_utils, models
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import csv
import pickle

## tabsyn
from utils import get_column_name_mapping
from torch.utils.data import DataLoader
from utils_train import make_dataset, concat_y_to_X
from utils_train import preprocess, TabularDataset
import src
from tabsyn.vae.model import Model_VAE, Encoder_model, Decoder_model
from utils_train import preprocess
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_train
from tabsyn.diffusion_utils import sample as tabsyn_sample
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target
import pandas as pd
import json




matplotlib.use('Agg')
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim,label_dim,prop_dim):
        super(Generator, self).__init__()
        #self.label_embedding = nn.Embedding(label_dim, label_dim)
        #self.prop_embedding = nn.Embedding(prop_dim, prop_dim)

        self.model = nn.Sequential(  
            nn.Linear(input_dim+2, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, z):
        return self.model(z)
    def forward(self, z, labels,props):
        label_input = labels
        prop_input = props
        
        x = torch.cat([z, label_input.unsqueeze(1),prop_input.unsqueeze(1)], dim=1)
        return self.model(x)
# Discriminator 모델
class Discriminator(nn.Module):
    def __init__(self, data_dim, label_dim, prop_dim):
        super(Discriminator, self).__init__()
        #self.label_embedding = nn.Embedding(label_dim, label_dim)
        #self.prop_embedding = nn.Embedding(prop_dim, prop_dim)
        self.model = nn.Sequential(
            nn.Linear(data_dim +2, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels, props):
        label_input = labels
        prop_input = props
        x = torch.cat([x, label_input.unsqueeze(1), prop_input.unsqueeze(1)], dim=1)
        return self.model(x)
    
class AttackUtil:
    def __init__(self, target_model_layers, df_train, df_test, cat_columns=None,cont_columns=None, verbose=True, dataset='adult'):
        if verbose:
            message = """Before attempting to run the property inference attack, set hyperparameters using
            1. set_attack_hyperparameters()
            2. set_model_hyperparameters()"""
            print(message)
        self.target_model_layers = target_model_layers
        self.df_train = df_train
        self.df_test = df_test
        self.cat_columns = cat_columns
        self.cont_columns = cont_columns
        self.dataset = dataset
        print("dataset is ", self.dataset)

        # Attack Hyperparams
        
        self._categories = None
        self._target_attributes = None
        self._sub_categories = None
        self._sub_attributes = None
        self._poison_class = None
        self._poison_percent = None
        self._k = None
        self._t0 = None
        self._t1 = None
        self._middle = None
        self._variance_adjustment = None
        self._num_queries = None
        self._nsub_samples = None
        self._ntarget_samples = None
        self._subproperty_sampling = False
        self._allow_subsampling = False
        self._allow_target_subsampling = False
        self._restrict_sampling = False
        self._pois_rs = None
        self._model_metrics = None

        # Model + Data Hyperparams
        self._layer_sizes = None
        self._num_classes = None
        self._epochs = None
        self._optim_init = None
        self._optim_kwargs = None
        self._criterion = None
        self._device = None
        self._tol = None
        self._verbose = None
        self._early_stopping = None
        self._dropout = None
        self._shuffle = None
        self._using_ce_loss = None
        self._batch_size = None
        self._num_workers = None
        self._persistent_workers = None
        self._party_nums = 2
        self._local_epoch = 1
        
        
        self.median_stacks = []
        self.mean_stacks = []
        
        self.vic_stacks = []
        
        self.pub_dataset = []
        self.vic_dataset = []
        self.shw_dataset = []
        
        self.pub_label_dist = 0
        self.vic_label_dist = []
        
        self.query_loader = []
        self.nonprop_query_loader = []
        self.public_history = []
        self._no_filter = False
    def set_attack_hyperparameters(
        self,
        categories=["race"],
        target_attributes=[" Black"],
        sub_categories=["occupation"],
        sub_attributes=[" Sales"],
        subproperty_sampling=False,
        restrict_sampling=False,
        poison_class=1,
        poison_percent=0.03,
        k=None,
        t0=0.1,
        t1=0.25,
        tpub=0.6,
        middle="median",
        variance_adjustment=1,
        nsub_samples=5000,
        allow_subsampling=True,
        ntarget_samples=5000,
        num_target_models=25,
        allow_target_subsampling=True,
        pois_random_seed=21,
        num_queries=5000,
        public_poison = 0.5,
        adaptive_step=20,
        poison_unit=0.05,
        poisoning_start = 0,
    ):

        self._categories = categories
        self._target_attributes = target_attributes
        self._sub_categories = sub_categories
        self._sub_attributes = sub_attributes
        self._subproperty_sampling = subproperty_sampling
        self._restrict_sampling = restrict_sampling
        self._poison_class = poison_class
        self._poison_percent = poison_percent
        self._k = k
        self._t0 = t0
        self._t1 = t1
        self._middle = middle
        self._variance_adjustment = variance_adjustment
        self._num_queries = num_queries
        self._nsub_samples = nsub_samples
        self._allow_subsampling = allow_subsampling
        self._num_target_models = num_target_models
        self._ntarget_samples = ntarget_samples
        self._allow_target_subsampling = allow_target_subsampling
        self._pois_rs = pois_random_seed
        self._public_poison = public_poison
        
        print(f"0 : {self._allow_subsampling}")
        self._tpub = self._t0*tpub
        self._taux = self._t1
        self._adaptive_step = adaptive_step
        self._poison_unit = poison_unit
        self.poisoning_start = poisoning_start
        
        self.random_seed = 1111
    def set_shadow_model_hyperparameters(
        self,
        layer_sizes=[64],
        num_classes=2,
        epochs=10,
        optim_init=optim.Adam,
        optim_kwargs={"lr": 0.0003, "weight_decay": 0.0001},
        optim_kd_kwargs={"lr": 0.0003, "weight_decay": 0.0001},
        optim_poison_kwargs={"lr": 1},# "weight_decay": 0.0001},
        #optim_kwargs={"lr": 0.0003, "weight_decay": 0.0001},
        #optim_kd_kwargs={"lr": 0.0003, "weight_decay": 0.0001},
        #optim_kwargs={"lr": 0.0005, "weight_decay": 0.001},
        #optim_kd_kwargs={"lr": 0.0005, "weight_decay": 0.001},
        #optim_kwargs={"lr": 0.03, "weight_decay": 0.0001},
        #optim_kd_kwargs={"lr": 0.03, "weight_decay": 0.0001},
        criterion=nn.CrossEntropyLoss(),
        device="cpu",
        tol=10e-7,
        verbose=False,
        mini_verbose=False,
        early_stopping=True,
        dropout=False,
        shuffle=False,
        using_ce_loss=True,
        batch_size=1000,
        num_workers=8,
        persistent_workers=True,
        num_shadow_models = 4,
    ):
        print(optim_kwargs,optim_kd_kwargs)
        self._num_shadow_models = num_shadow_models
        self._layer_sizes = layer_sizes
        self._num_classes = num_classes
        self._epochs = epochs
        self._optim_init = optim_init
        self._optim_kwargs = optim_kwargs
        self._criterion = criterion
        self._device = device
        self.device = device
        self._tol = tol
        self._verbose = verbose
        self._mini_verbose = mini_verbose
        self._early_stopping = early_stopping
        self._dropout = dropout
        self._shuffle = shuffle
        self._using_ce_loss = using_ce_loss
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._persistent_workers = persistent_workers
        self._activation_val = {}
        self._optim_kd_kwargs = optim_kd_kwargs
        self._optim_poison_kwargs = optim_poison_kwargs
        print(f"1 : {self._allow_subsampling}")
    def lr_scheduler(self):
        self._optim_kwargs ={"lr": self._optim_kwargs["lr"]*0.1, "weight_decay": 0.001}
        self._optim_kd_kwargs = {"lr": self._optim_kd_kwargs["lr"]*0.1, "weight_decay": 0.001}    

    def generate_datasets(self):
        """Generate all datasets required for the property inference attack"""

        self._target_worlds = np.append(
            np.zeros(self._num_target_models), np.ones(self._num_target_models)
        )

        # print(self._target_worlds)

        # Generate Datasets
        if self._mini_verbose:
            print("Generating all datasets...")

        (
            self._D,
            self._D_priv,
            self._D0_priv,
            self._D1_priv,
            self._Dp,
            self._Dtest,
            self._Dacc,
            self._D_public_all,
            self._D0_pub,
            self._D1_pub,
            self._D_aux,
        ) = data.generate_all_datasets(
            self.df_train,
            self.df_test,
            t0=self._t0,
            t1=self._t1,
            tpub=self._tpub,
            taux=self._taux,
            categories=self._categories,
            target_attributes=self._target_attributes,
            sub_categories=self._sub_categories,
            sub_attributes=self._sub_attributes,
            poison_class=self._poison_class,
            poison_percent=self._poison_percent,
            subproperty_sampling=self._subproperty_sampling,
            restrict_sampling=self._restrict_sampling,
            verbose=self._verbose,
        )
        self._D_priv.to_csv(f'./csv_arxiv/table_priv.csv', index = False)
        print()

        if self._t0 == 0:
            self._Dtest = pd.concat([self._Dp, self._Dtest])

        #Changes-Harsh
        # Converting to poisoned class
        # self._Dtest["class"] = self._poison_class

        if self._k is None:
            self._k = int(self._poison_percent * len(self._D0_priv))
        else:
            self._poison_percent = self._k / len(self._D0_priv)

        if len(self._Dp) == 0:
            (
                self._D_OH,
                self._D0_mo_OH,
                self._D1_mo_OH,
                self._D0_OH,
                self._D1_OH,
                self._Dtest_OH,
                self._test_set,
            ) = data.all_dfs_to_one_hot(
                [
                    self.df_train,
                    self._D0_mo,
                    self._D1_mo,
                    self._D0,
                    self._D1,
                    self._Dtest,
                    self.df_test,
                ],
                cat_columns=self.cat_columns,
                class_label="class",
            )

        else:
            ###legacy
            (
                self._D_train_OH_all,
                self._D_OH_all,
                self._D0_OH_all,
                self._D1_OH_all,
                self._Dp_OH,
                self._Dtest_OH,
                self._Dacc_OH,
                self._test_set,
                self._D_pub_OH,
                self._D0_pub_OH,
                self._D1_pub_OH,
                self._D_aux_OH,
            ),self.original_max_values,self.all_columns ,self.original_columns_order= data.all_dfs_to_one_hot(
                [
                    self.df_train,
                    self._D_priv,
                    self._D0_priv,
                    self._D1_priv,
                    self._Dp,
                    self._Dtest,
                    self._Dacc,
                    self.df_test,
                    self._D_public_all,
                    self._D0_pub,
                    self._D1_pub,
                    self._D_aux,
                ],
                cat_columns=self.cat_columns,
                class_label="class",
            )

            
            
            self.transformer_file_path = f"./data_transformer/{self._categories}_{self._target_attributes}.pickle"
            
            if os.path.exists(self.transformer_file_path):
                with open(self.transformer_file_path, 'rb') as file:
                    self.data_transformer = pickle.load(file)
            else:
                
                self.data_transformer = DataTransformer()
                self.data_transformer.fit(self.df_train,self.cat_columns+["class"])
                
            self._D_train_OH_all = self.data_transformer.get_one_hot_encoded_dataframe(self.df_train)
            self._D_OH_all = self.data_transformer.get_one_hot_encoded_dataframe(self._D_priv)
            self._D0_OH_all = self.data_transformer.get_one_hot_encoded_dataframe(self._D0_priv)
            self._D1_OH_all = self.data_transformer.get_one_hot_encoded_dataframe(self._D1_priv)
            self._Dp_OH = self.data_transformer.get_one_hot_encoded_dataframe(self._Dp)
            self._Dtest_OH = self.data_transformer.get_one_hot_encoded_dataframe(self._Dtest)
            self._Dacc_OH = self.data_transformer.get_one_hot_encoded_dataframe(self._Dacc)
            self._test_set = self.data_transformer.get_one_hot_encoded_dataframe(self.df_test)
            self._D_pub_OH = self.data_transformer.get_one_hot_encoded_dataframe(self._D_public_all)
            self._D0_pub_OH = self.data_transformer.get_one_hot_encoded_dataframe(self._D0_pub)
            self._D1_pub_OH = self.data_transformer.get_one_hot_encoded_dataframe(self._D1_pub)
            self._D_aux_OH = self.data_transformer.get_one_hot_encoded_dataframe(self._D_aux)
            self._all_OH_column_names = self._D_train_OH_all.columns.tolist()

            print(len(self._D_train_OH_all.columns.tolist()))
            self._prop_index = [self._D0_OH_all.columns.get_loc(f"{self._categories[i]}_{self._target_attributes[i]}") for i in range(len(self._categories))]
            self._category_index = []
            
            for category in self._categories:
                category_indices = [
                    self._D0_OH_all.columns.get_loc(col) 
                    for col in self._D0_OH_all.columns 
                    if col.startswith(f"{category}_")
                ]
                self._category_index.extend(category_indices)

            #print(self._prop_index)
            #print(self._target_attributes)
            #print(self._categories)
            self._D_OH = []
            self._D0_OH = []
            self._D1_OH = []

            print(f"total orig set size {len(self._D)}, {len(self._D_priv)}, {len(self._D_public_all)}, {len(self._D_priv) + len(self._D_public_all)}")
            
            set_size = len(self._D_OH_all) // (self._party_nums)
            print(f"total D set size is {len(self._D_OH_all)} {len(self._D_pub_OH)}, {set_size}")
            for i in range(self._party_nums):
                start_idx = i* set_size
                end_idx = (i + 1) * set_size if i < self._party_nums else None
                self._D_OH.append(self._D_OH_all.iloc[start_idx:end_idx]) 

            set_size = len(self._D0_OH_all) // (self._party_nums)
            print(f"total D0 set size is {len(self._D0_OH_all)} {len(self._D0_pub_OH)}, {set_size}")
            for i in range(self._party_nums):
                start_idx = i* set_size
                end_idx = (i + 1) * set_size if i < self._party_nums else None
                self._D0_OH.append(data.v2_fix_imbalance_OH(
                    self._D0_OH_all.iloc[start_idx:end_idx],
                    target_split=self._t0,
                    categories=self._categories,
                    target_attributes=self._target_attributes,
                    random_seed=1234,
                )) 
            
            set_size = len(self._D1_OH_all) // (self._party_nums)
            print(f"total D1 set size is {len(self._D1_OH_all)} {len(self._D1_pub_OH)}, {set_size}")
            for i in range(self._party_nums):
                start_idx = i* set_size
                end_idx = (i + 1) * set_size if i < self._party_nums else None
                
                self._D1_OH.append(data.v2_fix_imbalance_OH(
                    self._D1_OH_all.iloc[start_idx:end_idx],
                    target_split=self._t1,
                    categories=self._categories,
                    target_attributes=self._target_attributes,
                    random_seed=1234,
                )
                                        ) 
                #self._D1_OH.append(self._D1_OH_all.iloc[start_idx:end_idx]) 

        for i in range(self._party_nums):
            label_split_d0 = sum(self._D0_OH[i][self._D0_OH[i].columns[-1]]) / len(self._D0_OH[i])
            label_split_d1 = sum(self._D1_OH[i][self._D1_OH[i].columns[-1]]) / len(self._D1_OH[i])
            D0_subpop = data.generate_subpopulation_OH(
            self._D0_OH[i], categories=self._categories, target_attributes=self._target_attributes
         )
            D1_subpop = data.generate_subpopulation_OH(
                self._D1_OH[i], categories=self._categories, target_attributes=self._target_attributes
            )
            print(
            f"D0 has {len(self._D0_OH[i])} points with {len(D0_subpop)} members from the target subpopulation and {label_split_d0*100:.4}% class 1"
            )
            print(
                f"D1 has {len(self._D1_OH[i])} points with {len(D1_subpop)} members from the target subpopulation and {label_split_d1*100:.4}% class 1"
            )
        label_split_d_pub = sum(self._D_pub_OH[self._D_pub_OH.columns[-1]]) / len(self._D_pub_OH)
  
        Dpub_subpop = data.generate_subpopulation_OH(
                self._D_pub_OH, categories=self._categories, target_attributes=self._target_attributes
            )

        print(
                f"D_public has {len(self._D_pub_OH)} points with {len(Dpub_subpop)} members from the target subpopulation and {label_split_d_pub*100:.4}% class 1"
            )
            
            
        self._test_dataloader = training_utils.dataframe_to_dataloader(
                self._Dacc_OH,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
            )
    def intra_domain_alignment2(self,world_split=False,pub_ent_w=None,priv=False,n_trial=0):
        if pub_ent_w is not None:

                
            if False:
                gen_train_0 = pd.read_csv(f'./syn_csv/tabsyn_table_priv_{priv}_{self.dataset}_{self._categories}_{self._target_attributes}_{pub_ent_w}_0.csv')
                if world_split:
                    gen_train_1 = pd.read_csv(f'./syn_csv/tabsyn_table_priv_{priv}_{self.dataset}_{self._categories}_{self._target_attributes}_{pub_ent_w}_1.csv')
                else:
                    gen_train_1 = pd.read_csv(f'./syn_csv/tabsyn_table_priv_{priv}_{self.dataset}_{self._categories}_{self._target_attributes}_{pub_ent_w}_0.csv')
            else:
                gen_train_0 = pd.read_csv(f'./syn_csv/tabsyn_table_priv_{priv}_{self.dataset}_{self._categories}_{self._target_attributes}_{pub_ent_w}_{n_trial}_0.csv')
                if world_split:
                    gen_train_1 = pd.read_csv(f'./syn_csv/tabsyn_table_priv_{priv}_{self.dataset}_{self._categories}_{self._target_attributes}_{pub_ent_w}_{n_trial}_1.csv')
                else:
                    gen_train_1 = pd.read_csv(f'./syn_csv/tabsyn_table_priv_{priv}_{self.dataset}_{self._categories}_{self._target_attributes}_{pub_ent_w}_{n_trial}_0.csv')
        else:
            gen_train = pd.read_csv(f'./syn_csv/tabsyn_table_priv_{priv}_{self.dataset}_{self._categories}_{self._target_attributes}.csv')
        gen_train_0 = gen_train_0.reset_index(drop=True)
        gen_train_1 = gen_train_1.reset_index(drop=True)

        if False:
            _, gen_train_0 = data.split_data_ages(gen_train_0,40)
            _, gen_train_1 = data.split_data_ages(gen_train_1,40)
        gen_train = [gen_train_0, gen_train_1]



        self._shadow_dataloaders = []
        self._shadow_datasets = []
        m=2
        for t_i in range(2):
            # 데이터로더 생성
            self._tabsyn_pub_OH = self.data_transformer.get_one_hot_encoded_dataframe(gen_train[t_i])
            D_align_dataloader = training_utils.dataframe_to_dataloader(
                self._tabsyn_pub_OH,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
                pin_memory=False,
                shuffle=self._shuffle,
            )
            
            D_public_X = torch.Tensor([])
            D_public_y = torch.Tensor([])
            D_public_confi = torch.Tensor([])
            if t_i == 0:
                priv_model = self.temp_models[0].to(self._device)
            else:
                priv_model = self.temp_models[-1].to(self._device)
            pub_model = self.temp_models[1].to(self._device)
            for (inputs, labels) in D_align_dataloader:
                with torch.no_grad():
                    D_public_X = torch.concat([D_public_X, inputs.detach().cpu()])
                    D_public_y = torch.concat([D_public_y, labels.long().detach().clone().cpu()])

                    priv_logits = F.softmax(priv_model(inputs.to(self._device)), dim=1)
                    pub_logits = F.softmax(pub_model(inputs.to(self._device)), dim=1)
                    
                    priv_score = (torch.max(priv_logits, dim=1).values - torch.min(priv_logits, dim=1).values)
                    pub_score = (torch.max(pub_logits, dim=1).values - torch.min(pub_logits, dim=1).values)
                    
                    
                    priv_entropy = -torch.sum(priv_logits * torch.log(priv_logits + 1e-10), dim=1)
                    pub_entropy = -torch.sum(pub_logits * torch.log(pub_logits + 1e-10), dim=1)

                    score =  1.0* pub_entropy +priv_score
                    #score = pub_entropy
                    #score = 1-pub_score
                    #score = 1-priv_score
                    #score = priv_score + pub_score +  0.5 * pub_entropy
                    D_public_confi = torch.concat([D_public_confi, score.detach().clone().cpu()])

            # 타겟 속성을 가진 데이터와 그렇지 않은 데이터의 인덱스 분리
            mask = torch.ones(D_public_X.shape[0], dtype=torch.bool)
            for prop_i in self._prop_index:
                mask &= (D_public_X[:, prop_i] == 1)

            prop_indices = torch.where(mask)[0]
            non_prop_indices = torch.where(~mask)[0]

            print(D_public_X.shape[0], len(prop_indices), len(non_prop_indices))

            # 타겟 속성을 가진 데이터의 레이블 분포 계산
            prop_labels = D_public_y[prop_indices]
            unique_prop_labels, prop_counts = torch.unique(prop_labels, return_counts=True)
            prop_label_distribution = {int(label.item()): prop_counts[i].item() / prop_counts.sum().item() for i, label in enumerate(unique_prop_labels)}

            # 타겟 속성을 가지지 않은 데이터의 레이블 분포 계산
            non_prop_labels = D_public_y[non_prop_indices]
            unique_non_prop_labels, non_prop_counts = torch.unique(non_prop_labels, return_counts=True)
            non_prop_label_distribution = {int(label.item()): non_prop_counts[i].item() / non_prop_counts.sum().item() for i, label in enumerate(unique_non_prop_labels)}

            # 인덱스를 레이블별로 분리
            prop_indices_by_label = {}
            for label in unique_prop_labels:
                label_int = int(label.item())
                prop_indices_by_label[label_int] = prop_indices[prop_labels == label_int]

            non_prop_indices_by_label = {}
            for label in unique_non_prop_labels:
                label_int = int(label.item())
                non_prop_indices_by_label[label_int] = non_prop_indices[non_prop_labels == label_int]

            # D0_public 데이터셋 생성
            D0_public_X_list = []
            D0_public_y_list = []

            num_prop_samples = round(m*self._nsub_samples * self._t0)
            num_non_prop_samples = m*self._nsub_samples - num_prop_samples  # 남은 샘플 수 보정

            # 타겟 속성을 가진 데이터에서 레이블별로 샘플링
            prop_label_list = list(prop_label_distribution.keys())
            prop_num_samples_list = []
            total_prop_assigned = 0

            for label_int in prop_label_list[:-1]:
                label_prop_indices = prop_indices_by_label[label_int]
                num_samples = round(num_prop_samples * prop_label_distribution[label_int])
                num_samples = min(num_samples, len(label_prop_indices))
                prop_num_samples_list.append(num_samples)
                total_prop_assigned += num_samples
                if num_samples > 0:
                    sorted_indices = label_prop_indices[
                        torch.argsort(D_public_confi[label_prop_indices], descending=True)
                    ]
                    selected_indices = sorted_indices[:num_samples]
                    D0_public_X_list.append(D_public_X[selected_indices])
                    D0_public_y_list.append(D_public_y[selected_indices])

            # 남은 샘플 수를 마지막 레이블에 추가
            remaining_prop_samples = num_prop_samples - total_prop_assigned
            last_label_int = prop_label_list[-1]
            label_prop_indices = prop_indices_by_label[last_label_int]
            num_samples = min(remaining_prop_samples, len(label_prop_indices))
            prop_num_samples_list.append(num_samples)
            if num_samples > 0:
                sorted_indices = label_prop_indices[
                    torch.argsort(D_public_confi[label_prop_indices], descending=True)
                ]
                selected_indices = sorted_indices[:num_samples]
                D0_public_X_list.append(D_public_X[selected_indices])
                D0_public_y_list.append(D_public_y[selected_indices])

            # 타겟 속성을 가지지 않은 데이터에서 레이블별로 샘플링
            non_prop_label_list = list(non_prop_label_distribution.keys())
            non_prop_num_samples_list = []
            total_non_prop_assigned = 0

            for label_int in non_prop_label_list[:-1]:
                label_non_prop_indices = non_prop_indices_by_label[label_int]
                num_samples = round(num_non_prop_samples * non_prop_label_distribution[label_int])
                num_samples = min(num_samples, len(label_non_prop_indices))
                non_prop_num_samples_list.append(num_samples)
                total_non_prop_assigned += num_samples
                if num_samples > 0:
                    sorted_indices = label_non_prop_indices[
                        torch.argsort(D_public_confi[label_non_prop_indices], descending=True)
                    ]
                    selected_indices = sorted_indices[:num_samples]
                    D0_public_X_list.append(D_public_X[selected_indices])
                    D0_public_y_list.append(D_public_y[selected_indices])

            # 남은 샘플 수를 마지막 레이블에 추가
            remaining_non_prop_samples = num_non_prop_samples - total_non_prop_assigned
            last_label_int = non_prop_label_list[-1]
            label_non_prop_indices = non_prop_indices_by_label[last_label_int]
            num_samples = min(remaining_non_prop_samples, len(label_non_prop_indices))
            non_prop_num_samples_list.append(num_samples)
            if num_samples > 0:
                sorted_indices = label_non_prop_indices[
                    torch.argsort(D_public_confi[label_non_prop_indices], descending=True)
                ]
                selected_indices = sorted_indices[:num_samples]
                D0_public_X_list.append(D_public_X[selected_indices])
                D0_public_y_list.append(D_public_y[selected_indices])

            # 선택된 데이터를 결합
            D0_public_X = torch.concat(D0_public_X_list)
            D0_public_y = torch.concat(D0_public_y_list).long()

            # 총 샘플 수 확인
            #assert len(D0_public_X) == m*self._nsub_samples, f"D0_public 데이터셋의 샘플 수가 {len(D0_public_X)}개로 예상된 {m*self._nsub_samples}개와 일치하지 않습니다."

            # 데이터를 섞어줌
            perm = torch.randperm(len(D0_public_X))
            D0_public_X = D0_public_X[perm]
            D0_public_y = D0_public_y[perm]

            # D0_public 데이터셋 생성
            D0_public = torch.utils.data.TensorDataset(D0_public_X, D0_public_y)
            df_D0_public = training_utils.torch_dataset_to_dataframe(D0_public, self.dataframe_column_OH)

            # D1_public 데이터셋 생성 (D0_public과 동일한 방식으로 처리)
            D1_public_X_list = []
            D1_public_y_list = []

            num_prop_samples_D1 = round(m*self._nsub_samples * self._t1)
            num_non_prop_samples_D1 = m*self._nsub_samples - num_prop_samples_D1  # 남은 샘플 수 보정

            # 타겟 속성을 가진 데이터에서 레이블별로 샘플링
            prop_label_list = list(prop_label_distribution.keys())
            prop_num_samples_list_D1 = []
            total_prop_assigned_D1 = 0

            for label_int in prop_label_list[:-1]:
                label_prop_indices = prop_indices_by_label[label_int]
                num_samples = round(num_prop_samples_D1 * prop_label_distribution[label_int])
                num_samples = min(num_samples, len(label_prop_indices))
                prop_num_samples_list_D1.append(num_samples)
                total_prop_assigned_D1 += num_samples
                if num_samples > 0:
                    sorted_indices = label_prop_indices[
                        torch.argsort(D_public_confi[label_prop_indices], descending=True)
                    ]
                    selected_indices = sorted_indices[:num_samples]
                    D1_public_X_list.append(D_public_X[selected_indices])
                    D1_public_y_list.append(D_public_y[selected_indices])

            # 남은 샘플 수를 마지막 레이블에 추가
            remaining_prop_samples_D1 = num_prop_samples_D1 - total_prop_assigned_D1
            last_label_int = prop_label_list[-1]
            label_prop_indices = prop_indices_by_label[last_label_int]
            num_samples = min(remaining_prop_samples_D1, len(label_prop_indices))
            prop_num_samples_list_D1.append(num_samples)
            if num_samples > 0:
                sorted_indices = label_prop_indices[
                    torch.argsort(D_public_confi[label_prop_indices], descending=True)
                ]
                selected_indices = sorted_indices[:num_samples]
                D1_public_X_list.append(D_public_X[selected_indices])
                D1_public_y_list.append(D_public_y[selected_indices])

            # 타겟 속성을 가지지 않은 데이터에서 레이블별로 샘플링
            non_prop_label_list = list(non_prop_label_distribution.keys())
            non_prop_num_samples_list_D1 = []
            total_non_prop_assigned_D1 = 0

            for label_int in non_prop_label_list[:-1]:
                label_non_prop_indices = non_prop_indices_by_label[label_int]
                num_samples = round(num_non_prop_samples_D1 * non_prop_label_distribution[label_int])
                num_samples = min(num_samples, len(label_non_prop_indices))
                non_prop_num_samples_list_D1.append(num_samples)
                total_non_prop_assigned_D1 += num_samples
                if num_samples > 0:
                    sorted_indices = label_non_prop_indices[
                        torch.argsort(D_public_confi[label_non_prop_indices], descending=True)
                    ]
                    selected_indices = sorted_indices[:num_samples]
                    D1_public_X_list.append(D_public_X[selected_indices])
                    D1_public_y_list.append(D_public_y[selected_indices])

            # 남은 샘플 수를 마지막 레이블에 추가
            remaining_non_prop_samples_D1 = num_non_prop_samples_D1 - total_non_prop_assigned_D1
            last_label_int = non_prop_label_list[-1]
            label_non_prop_indices = non_prop_indices_by_label[last_label_int]
            num_samples = min(remaining_non_prop_samples_D1, len(label_non_prop_indices))
            non_prop_num_samples_list_D1.append(num_samples)
            if num_samples > 0:
                sorted_indices = label_non_prop_indices[
                    torch.argsort(D_public_confi[label_non_prop_indices], descending=True)
                ]
                selected_indices = sorted_indices[:num_samples]
                D1_public_X_list.append(D_public_X[selected_indices])
                D1_public_y_list.append(D_public_y[selected_indices])

            # 선택된 데이터를 결합
            D1_public_X = torch.concat(D1_public_X_list)
            D1_public_y = torch.concat(D1_public_y_list).long()

            # 총 샘플 수 확인
            #assert len(D1_public_X) == m*self._nsub_samples, f"D1_public 데이터셋의 샘플 수가 {len(D1_public_X)}개로 예상된 {m*self._nsub_samples}개와 일치하지 않습니다."

            # 데이터를 섞어줌
            perm = torch.randperm(len(D1_public_X))
            D1_public_X = D1_public_X[perm]
            D1_public_y = D1_public_y[perm]

            # D1_public 데이터셋 생성
            D1_public = torch.utils.data.TensorDataset(D1_public_X, D1_public_y)
            df_D1_public = training_utils.torch_dataset_to_dataframe(D1_public, self.dataframe_column_OH)

            # 서브인구 생성 및 분석
            sampled_public_D0_subpop = data.generate_subpopulation_OH(
                df_D0_public, categories=self._categories, target_attributes=self._target_attributes
            )

            self.label_split_public_0 = sum(df_D0_public[df_D0_public.columns[-1]]) / len(df_D0_public)
            label_split_public_D0_subpop = sum(sampled_public_D0_subpop[sampled_public_D0_subpop.columns[-1]]) / len(sampled_public_D0_subpop)

            print(f"target D_shadow 0 has {len(df_D0_public)} points with {len(sampled_public_D0_subpop)}, {len(sampled_public_D0_subpop)/len(df_D0_public):.2f} members from the target subpopulation and {self.label_split_public_0*100:.4}%, {label_split_public_D0_subpop*100:.4}% class 1")

            sampled_public_D1_subpop = data.generate_subpopulation_OH(
                df_D1_public, categories=self._categories, target_attributes=self._target_attributes
            )

            self.label_split_public_1 = sum(df_D1_public[df_D1_public.columns[-1]]) / len(df_D1_public)
            label_split_public_D1_subpop = sum(sampled_public_D1_subpop[sampled_public_D1_subpop.columns[-1]]) / len(sampled_public_D1_subpop)

            print(f"target D_shadow 1 has {len(df_D1_public)} points with {len(sampled_public_D1_subpop)}, {len(sampled_public_D1_subpop)/len(df_D1_public):.2f} members from the target subpopulation and {self.label_split_public_1*100:.4}%, {label_split_public_D1_subpop*100:.4}% class 1")
            print("####")
            # 섀도우 데이터로더 생성
            temp_shadow_dataloaders = []
            temp_shadow_datasets = []
            for s_i in range(self._num_shadow_models):
                n_samples = min(self._ntarget_samples, len(df_D0_public))
                sampled_D0_public = data.v2_fix_imbalance_OH(
                    df_D0_public,
                    target_split=self._t0,
                    categories=self._categories,
                    target_attributes=self._target_attributes,
                    random_seed=self.random_seed+s_i,total_samples=n_samples,
                )
                temp_shadow_dataloaders.append(training_utils.dataframe_to_dataloader(
                    sampled_D0_public,
                    batch_size=self._batch_size,
                    using_ce_loss=self._using_ce_loss,
                    num_workers=self._num_workers,
                    persistent_workers=self._persistent_workers,
                    pin_memory=True,
                    shuffle=self._shuffle,
                ))
                temp_shadow_datasets.append(sampled_D0_public)
            for s_i in range(self._num_shadow_models):
                n_samples = min(self._ntarget_samples, len(df_D1_public))

                sampled_D1_public = data.v2_fix_imbalance_OH(
                    df_D1_public,
                    target_split=self._t1,
                    categories=self._categories,
                    target_attributes=self._target_attributes,
                    random_seed=self.random_seed+s_i,total_samples=n_samples,
                )

                temp_shadow_dataloaders.append(training_utils.dataframe_to_dataloader(
                    sampled_D1_public,
                    batch_size=self._batch_size,
                    using_ce_loss=self._using_ce_loss,
                    num_workers=self._num_workers,
                    persistent_workers=self._persistent_workers,
                    pin_memory=True,
                    shuffle=self._shuffle,
                ))
                temp_shadow_datasets.append(sampled_D1_public)
            self._shadow_dataloaders.append(temp_shadow_dataloaders)
            self._shadow_datasets.append(temp_shadow_datasets)
    def intra_domain_alignment(self):
        
        self._D0_pub_OH
        self._D1_pub_OH
        
        self._D_pub_dataloader= training_utils.dataframe_to_dataloader(
                        self._D_pub_OH,
                        batch_size=self._batch_size,
                        using_ce_loss=self._using_ce_loss,
                        num_workers=self._num_workers,
                        persistent_workers=self._persistent_workers,
                        pin_memory=False,
                        shuffle=self._shuffle,
                    )
        self._D0_pub_dataloader = training_utils.dataframe_to_dataloader(
                        self._D0_pub_OH,
                        batch_size=self._batch_size,
                        using_ce_loss=self._using_ce_loss,
                        num_workers=self._num_workers,
                        persistent_workers=self._persistent_workers,
                        pin_memory=False,
                        shuffle=self._shuffle,
                    )
        self._D1_pub_dataloader = training_utils.dataframe_to_dataloader(
                        self._D1_pub_OH,
                        batch_size=self._batch_size,
                        using_ce_loss=self._using_ce_loss,
                        num_workers=self._num_workers,
                        persistent_workers=self._persistent_workers,
                        pin_memory=False,
                        shuffle=self._shuffle,
                    )
        
        D_public_X = torch.Tensor([])
        D_public_y = torch.Tensor([])
        D_public_confi = torch.Tensor([])

        #confi_model = self._local_models[0].to(self._device)
        priv_model = self.temp_models[0].to(self._device)
        pub_model = self.temp_models[1].to(self._device)
        for (inputs,labels) in self._D_pub_dataloader:
            with torch.no_grad():
                D_public_X  =torch.concat([D_public_X , inputs.detach().cpu()])
                D_public_y = torch.concat([D_public_y , labels.long().detach().clone().cpu()])
        
                priv_logits = F.softmax(priv_model(inputs.to(self._device)),dim=1)
                pub_logits = F.softmax(pub_model(inputs.to(self._device)),dim=1)
                priv_score = (torch.max(priv_logits,dim=1).values-torch.min(priv_logits,dim=1).values)
                pub_score =  (torch.max(pub_logits,dim=1).values-torch.min(pub_logits,dim=1).values)
                
                priv_entropy = -torch.sum(priv_logits * torch.log(priv_logits + 1e-10), dim=1) 
                pub_entropy = -torch.sum(pub_logits * torch.log(pub_logits + 1e-10), dim=1) 
                
                score = priv_score + 0.3*pub_entropy
                #score = pub_entropy
                D_public_confi = torch.concat([D_public_confi, score.detach().clone().cpu()])
        
        mask = torch.ones(D_public_X.shape[0],dtype=torch.bool)
        for prop_i in self._prop_index:
            mask &= (D_public_X[:,prop_i]==1)
            
        prop_indices = torch.where(mask)[0]
        non_prop_indices = torch.where(~mask)[0]
        print(D_public_X.shape[0],len(prop_indices),len(non_prop_indices))
        
        #D0_public_X = D0_public_X[sort_index[:2000]]
        #D0_public_y = D0_public_y[sort_index[:2000]]

        D_public_X_prop = D_public_X[prop_indices]
        D_public_y_prop = D_public_y[prop_indices]
        D_public_confi_prop= D_public_confi[prop_indices]
        prop_sort_index = torch.argsort(D_public_confi_prop,descending=True)
        #prop_sort_index = torch.argsort(D_public_confi_prop,descending=False)
        
        D_public_X_non = D_public_X[non_prop_indices]
        D_public_y_non = D_public_y[non_prop_indices]
        D_public_confi_non = D_public_confi[non_prop_indices]
        non_sort_index = torch.argsort(D_public_confi_non,descending=True)
        #non_sort_index = torch.argsort(D_public_confi_non,descending=False)
        
        
        D0_public_X = torch.concat([D_public_X_prop[prop_sort_index[:int(self._nsub_samples*self._t0)]],D_public_X_non[non_sort_index[:int(self._nsub_samples*(1-self._t0))]]])
        D0_public_y = torch.concat([D_public_y_prop[prop_sort_index[:int(self._nsub_samples*self._t0)]],D_public_y_non[non_sort_index[:int(self._nsub_samples*(1-self._t0))]]]).long()
        
        D0_public = torch.utils.data.TensorDataset(D0_public_X, D0_public_y)
        df_D0_public = training_utils.torch_dataset_to_dataframe(D0_public,self.dataframe_column_OH)
        '''
        df_D0_public = data.v2_fix_imbalance_OH(
                df_D0_public,
                target_split=self._t0,
                categories=self._categories,
                target_attributes=self._target_attributes,
                random_seed=1234,
        )
        '''     
        sampled_public_D0_subpop = data.generate_subpopulation_OH(
            df_D0_public, categories=self._categories, target_attributes=self._target_attributes
            )

        self.label_split_public_0 = sum(df_D0_public[df_D0_public.columns[-1]]) / len(df_D0_public)
        label_split_public_D0_subpop = sum(sampled_public_D0_subpop[sampled_public_D0_subpop.columns[-1]]) / len(sampled_public_D0_subpop)

        print(f"target D_shadow 0 has {len(df_D0_public)} points with {len(sampled_public_D0_subpop)},{len(sampled_public_D0_subpop)/len(df_D0_public):.2f} members from the target subpopulation and {self.label_split_public_0*100:.4}%, {label_split_public_D0_subpop*100:.4}% class 1"
        )

        
        D1_public_X = torch.concat([D_public_X_prop[prop_sort_index[:int(self._nsub_samples*self._t1)]],D_public_X_non[non_sort_index[:int(self._nsub_samples*(1-self._t1))]]])
        D1_public_y = torch.concat([D_public_y_prop[prop_sort_index[:int(self._nsub_samples*self._t1)]],D_public_y_non[non_sort_index[:int(self._nsub_samples*(1-self._t1))]]]).long()
        
        D1_public = torch.utils.data.TensorDataset(D1_public_X, D1_public_y)
        df_D1_public = training_utils.torch_dataset_to_dataframe(D1_public,self.dataframe_column_OH)
        
        sampled_public_D1_subpop = data.generate_subpopulation_OH(
            df_D1_public, categories=self._categories, target_attributes=self._target_attributes
            )

        self.label_split_public_0 = sum(df_D1_public[df_D1_public.columns[-1]]) / len(df_D1_public)
        label_split_public_D1_subpop = sum(sampled_public_D1_subpop[sampled_public_D1_subpop.columns[-1]]) / len(sampled_public_D1_subpop)

        print(f"target D_shadow 1 has {len(df_D1_public)} points with {len(sampled_public_D1_subpop)},{len(sampled_public_D1_subpop)/len(df_D1_public):.2f} members from the target subpopulation and {self.label_split_public_0*100:.4}%, {label_split_public_D1_subpop*100:.4}% class 1"
        )
        D0_public = df_D0_public
        D1_public = df_D1_public
        self._shadow_dataloaders = [] 
        for s_i in range(self._num_shadow_models):
            sampled_D0_public = D0_public.sample(
                                n=self._ntarget_samples, random_state=0
                            )
            self._shadow_dataloaders.append(training_utils.dataframe_to_dataloader(
                sampled_D0_public,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
                pin_memory=True,
                shuffle=self._shuffle,
            ))
        for s_i in range(self._num_shadow_models):
            sampled_D1_public = D1_public.sample(
                                n=self._ntarget_samples, random_state=0
                            )
            self._shadow_dataloaders.append(training_utils.dataframe_to_dataloader(
                sampled_D1_public,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
                pin_memory=True,
                shuffle=self._shuffle,
            ))
            
    def set_models(self):
        
        self.dataframe_column_OH = self._Dacc_OH.columns.tolist()
        for i in range(self._num_target_models*2):
            self.median_stacks.append([])
            self.mean_stacks.append([])
        
        
        pin_memory = True
        print(f"target arch {self.target_model_layers}, shadow model arch {self._layer_sizes}")
        self._test_dataloader = training_utils.dataframe_to_dataloader(
                    self._Dacc_OH,
                    batch_size=1024,
                    using_ce_loss=self._using_ce_loss,
                    num_workers=self._num_workers,
                    persistent_workers=self._persistent_workers,pin_memory=pin_memory
                )

        
        self.pub_dataset = []
        self.vic_dataset = []
        self.shw_dataset = []
        
        self.pub_label_dist = 0
        self.vic_label_dist = []
        
        self.query_loader = []
        self.nonprop_query_loader = []
        self.public_history = []
        
        #self._local_models = [[None] * self._party_nums] * self._num_target_models * 2
        self._local_models = []
        self._local_optimizers = []
        self._party_models = []
        self._party_optimizers = []

        #self._local_dataloaders = [[None] * self._party_nums] * self._num_target_models * 2
        self._local_dataloaders = []
        self._party_dataloaders = []
        
        #self._shadow_models = [[None] * self._num_shadow_models*2] * self._num_target_models *2
        self._shadow_models = []
        self._shadow_optimizers = []

        #self._shadow_poisoned_model = [None] * self._num_target_models *2
        self._shadow_poisoned_model = [] #shadow student model

        #self._shadow_dataloaders = [[None] * self._num_shadow_models*2] * self._num_target_models *2
        self._shadow_dataloaders = [] #shadow teacher dataset

        #self._shadow_poisoned_dataloaders = [None] * self._num_target_models *2
        self._shadow_poisoned_dataloaders = [] #shadow student dataset

        #self._public_dataloader = [None] * self._num_target_models * 2
        self._public_dataloader = []#[None] * self._num_target_models * 2
        
        self._gaps_shadow = []
        
        input_dim = len(self._D0_OH[0].columns) - 2


        #get aux dataset
        if self._allow_subsampling == True:
            if self._ntarget_samples > self._D0_OH[0].shape[0]:
                self._ntarget_samples = self._D0_OH[0].shape[0]-100
        loader_shuffle = self._shuffle

        if self._allow_subsampling == True:
            if True:
                aux_D = self._D_aux_OH.sample(
                                n=self._ntarget_samples, random_state=0
                            )
            else:
                public_D = self._D1_OH[-1].sample(
                                n=self._ntarget_samples, random_state=0
                            )
        else:
            aux_D = self._D_aux_OH
        sampled_aux_D = data.v2_fix_imbalance_OH(
            aux_D,
            target_split=self._taux,
            categories=self._categories,
            target_attributes=self._target_attributes,
            random_seed=self.random_seed,
        )

        sampled_aux_D_subpop = data.generate_subpopulation_OH(
            sampled_aux_D, categories=self._categories, target_attributes=self._target_attributes
            )
        self.aux_dataset = sampled_aux_D_subpop
        
        self.label_split_aux_D = sum(sampled_aux_D[sampled_aux_D.columns[-1]]) / len(sampled_aux_D)
        label_split_aux_D_subpop = sum(sampled_aux_D_subpop[sampled_aux_D_subpop.columns[-1]]) / len(sampled_aux_D_subpop)

        #print(        f"target D_aux has {len(sampled_aux_D)} points with {len(sampled_aux_D_subpop)},{len(sampled_aux_D_subpop)/len(sampled_aux_D):.2f} members from the target subpopulation and {self.label_split_aux_D*100:.4}%, {label_split_aux_D_subpop*100:.4}% class 1"        )
        self.pub_label_dist = self.label_split_aux_D

        
        self._aux_dataloader = training_utils.dataframe_to_dataloader(
                aux_D,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
                #shuffle=loader_shuffle,
                shuffle=False,
                pin_memory=pin_memory,
            )
        
        self._Dp_dataset = training_utils.dataframe_to_torch_dataset(
        self._Dp_OH, using_ce_loss=True, class_label=None
        )
        
        #get public dataset
        if self._allow_subsampling == True:
            if self._ntarget_samples > self._D0_OH[0].shape[0]:
                self._ntarget_samples = self._D0_OH[0].shape[0]-100
        loader_shuffle = self._shuffle

        sampled_public_D = data.v2_fix_imbalance_OH(
            self._D_pub_OH,
            target_split=self._tpub,
            categories=self._categories,
            target_attributes=self._target_attributes,
            random_seed=self.random_seed,total_samples=self._ntarget_samples
        )

        sampled_public_D_subpop = data.generate_subpopulation_OH(
            sampled_public_D, categories=self._categories, target_attributes=self._target_attributes
            )
        self.pub_dataset.append(sampled_public_D_subpop)

        self.label_split_public_D = sum(sampled_public_D[sampled_public_D.columns[-1]]) / len(sampled_public_D)
        label_split_public_D_subpop = sum(sampled_public_D_subpop[sampled_public_D_subpop.columns[-1]]) / len(sampled_public_D_subpop)

        print(
        f"target D_pub has {len(sampled_public_D)} points with {len(sampled_public_D_subpop)},{len(sampled_public_D_subpop)/len(sampled_public_D):.2f} members from the target subpopulation and {self.label_split_public_D*100:.4}%, {label_split_public_D_subpop*100:.4}% class 1"
        )
        self.pub_label_dist = self.label_split_public_D


        self._public_dataloader = training_utils.dataframe_to_dataloader(
                sampled_public_D,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
                #shuffle=loader_shuffle,
                shuffle=False,
                pin_memory=pin_memory,
            )

        for i in range(self._num_target_models):
            
            if self._allow_subsampling == True:
                sampled_vic_D0 = self._D0_OH[0].sample(
                            n=self._ntarget_samples, random_state=self.random_seed+i + 1
                        )
            else:
                sampled_vic_D0 = self._D0_OH[0]
            sampled_vic_D0 = data.v2_fix_imbalance_OH(
                    sampled_vic_D0,
                    target_split=self._t0,
                    categories=self._categories,
                    target_attributes=self._target_attributes,
                    random_seed=self.random_seed,
            )

            sampled_vic_D0_subpop = data.generate_subpopulation_OH(
            sampled_vic_D0, categories=self._categories, target_attributes=self._target_attributes
            )

            self.label_split_d0 = sum(sampled_vic_D0[sampled_vic_D0.columns[-1]]) / len(sampled_vic_D0)
            label_split_d0_subpop = sum(sampled_vic_D0_subpop[sampled_vic_D0_subpop.columns[-1]]) / len(sampled_vic_D0_subpop)
            if i <= 1:
                print(            f"target{i} D0 has {len(sampled_vic_D0)} points with {len(sampled_vic_D0_subpop)},{len(sampled_vic_D0_subpop)/len(sampled_vic_D0):.2f} members from the target subpopulation and {self.label_split_d0*100:.4}%, {label_split_d0_subpop*100:.4}% class 1"
                )
            self.vic_dataset.append(sampled_vic_D0)
            self.vic_label_dist.append(self.label_split_d0)

            self._local_dataloaders.append(training_utils.dataframe_to_dataloader(
                sampled_vic_D0,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
                pin_memory=pin_memory,
                shuffle=loader_shuffle,
            ))
            
            
        for i in range(self._num_target_models,self._num_target_models*2):
            
            temp_local_data_loader_list = []
            
            if self._allow_subsampling == True:
                sampled_vic_D1 = self._D1_OH[0].sample(
                            n=self._ntarget_samples, random_state=self.random_seed+i + 1
                        )
            else:
                sampled_vic_D1 = self._D1_OH[0]
            sampled_vic_D1 = data.v2_fix_imbalance_OH(
                sampled_vic_D1,
                target_split=self._t1,
                categories=self._categories,
                target_attributes=self._target_attributes,
                random_seed=self.random_seed,
            )	        
            sampled_vic_D1_subpop = data.generate_subpopulation_OH(
                sampled_vic_D1, categories=self._categories, target_attributes=self._target_attributes            )



            self.label_split_d1 = sum(sampled_vic_D1[sampled_vic_D1.columns[-1]]) / len(sampled_vic_D1)
            label_split_d1_subpop = sum(sampled_vic_D1_subpop[sampled_vic_D1_subpop.columns[-1]]) / len(sampled_vic_D1_subpop)
            if i <= self._num_target_models+1:
                print(
                f"target{i} D1 has {len(sampled_vic_D1)} points with {len(sampled_vic_D1_subpop)},{len(sampled_vic_D1_subpop)/len(sampled_vic_D1):.2f} members from the target subpopulation and {self.label_split_d1*100:.4}%, {label_split_d1_subpop*100:.4}% class 1"
                )

            self.vic_dataset.append(sampled_vic_D1)
            self.vic_label_dist.append(self.label_split_d1)
            #print(f"vic local dataset size : {sampled_vic_D1.shape} {self._ntarget_samples}")

            self._local_dataloaders.append(training_utils.dataframe_to_dataloader(
                sampled_vic_D1,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
                pin_memory=pin_memory,
                shuffle=loader_shuffle,
            ))
            
        for i in range(self._party_nums-1):
            

            if self._allow_subsampling == True:
                sampled_party_D = self._D_OH[i+1].sample(
                            n=self._ntarget_samples, random_state=self.random_seed+i + 1
                        )
                
            else:
                sampled_party_D = self._D_OH[i+1]
                

            print(f"{len(sampled_party_D)}:sampled party")

            self._party_dataloaders.append(training_utils.dataframe_to_dataloader(
                sampled_party_D ,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
                pin_memory=pin_memory,
                shuffle=loader_shuffle,
            ))
            
        for i in range(self._num_target_models*2):

            if len(self.target_model_layers) != 0:
                self._local_models.append(models.NeuralNet(
                    input_dim=input_dim,
                    layer_sizes=self.target_model_layers,
                    num_classes=self._num_classes,
                    dropout=self._dropout,
                    
                ))
            else:
                self._local_models.append(models.LogisticRegression(
                    input_dim=input_dim,
                    num_classes=self._num_classes,
                    using_ce_loss=self._using_ce_loss,
                ))
                
            self._local_optimizers.append(self._optim_init(self._local_models[i].parameters(),**self._optim_kwargs))

        for i in range(self._party_nums-1):
            if len(self.target_model_layers) != 0:
                self._party_models.append(models.NeuralNet(
                    input_dim=input_dim,
                    layer_sizes=self.target_model_layers,
                    num_classes=self._num_classes,
                    dropout=self._dropout,
                    
                ))
            else:
                self._party_models.append(models.LogisticRegression(
                    input_dim=input_dim,
                    num_classes=self._num_classes,
                    using_ce_loss=self._using_ce_loss,
                ))
                
            self._party_optimizers.append(self._optim_init(self._party_models[i].parameters(),**self._optim_kwargs))
            
            
        self.temp_models = []
        self.temp_optimizers = []
        for t_i in range(3):
            self.temp_models.append(models.NeuralNet(
                            input_dim=input_dim,
                            layer_sizes=self.target_model_layers,
                            num_classes=self._num_classes,
                            dropout=self._dropout,
                        ))
        self.temp_optimizers.append(self._optim_init(self.temp_models[0].parameters(),**{"lr": 0.0001, "weight_decay": 0.0001}))
        self.temp_optimizers.append(self._optim_init(self.temp_models[1].parameters(),**{"lr": 0.0001, "weight_decay": 0.0001}))
        self.temp_optimizers.append(self._optim_init(self.temp_models[2].parameters(),**{"lr": 0.0001, "weight_decay": 0.0001}))
        self.temp_dataloader = []
    
        self.temp_dataloader.append(training_utils.dataframe_to_dataloader(
                        self._D_OH[-1],
                        batch_size=self._batch_size,
                        using_ce_loss=self._using_ce_loss,
                        num_workers=self._num_workers,
                        persistent_workers=self._persistent_workers,
                        pin_memory=pin_memory,
                        shuffle=loader_shuffle,
            ))

        self.temp_dataloader.append(training_utils.dataframe_to_dataloader(
                        self._D_pub_OH,
                        batch_size=self._batch_size,
                        using_ce_loss=self._using_ce_loss,
                        num_workers=self._num_workers,
                        persistent_workers=self._persistent_workers,
                        pin_memory=pin_memory,
                        shuffle=loader_shuffle,
            ))
        self.temp_dataloader.append(training_utils.dataframe_to_dataloader(
                        self._D_pub_OH,
                        batch_size=self._batch_size,
                        using_ce_loss=self._using_ce_loss,
                        num_workers=self._num_workers,
                        persistent_workers=self._persistent_workers,
                        pin_memory=pin_memory,
                        shuffle=loader_shuffle,
            ))
        
        equal_shadow_vic = True # True (vic != shadow), False (vic == shadow)

        for t_i in range(2):
            temp_shadow_dataloaders = []
            for p_i in range(self._num_shadow_models):
                
                if self._allow_subsampling == True:
                    if equal_shadow_vic:
                        #sampled_D0 = self._D_pub_OH.sample( n=self._ntarget_samples, random_state=i + 1)
                        sampled_D0 = self._D0_pub_OH.sample(
                                    n=self._ntarget_samples, random_state=self.random_seed+p_i + 1
                                )
                    else:
                        sampled_D0 = self._D0_OH[0].sample(
                                    n=self._ntarget_samples, random_state=self.random_seed+p_i + 1
                                )
                else:
                    if equal_shadow_vic:
                        sampled_D0 = self._D0_pub_OH.sample(
                                n=self._ntarget_samples, random_state=self.random_seed+p_i + 1
                            )
                    else:
                        sampled_D0 = self._D0_OH[0].sample(
                                    n=self._ntarget_samples, random_state=self.random_seed+p_i + 1
                                )   
                sampled_shadow_D0 = data.v2_fix_imbalance_OH(
                    sampled_D0,
                    target_split=self._t0,
                    categories=self._categories,
                    target_attributes=self._target_attributes,
                    random_seed=self.random_seed,
                )	        
                sampled_shadow_D0_subpop = data.generate_subpopulation_OH(
                    sampled_shadow_D0, categories=self._categories, target_attributes=self._target_attributes            )



                self.label_split_d0 = sum(sampled_shadow_D0[sampled_shadow_D0.columns[-1]]) / len(sampled_shadow_D0)
                label_split_d0_subpop = sum(sampled_shadow_D0_subpop[sampled_shadow_D0_subpop.columns[-1]]) / len(sampled_shadow_D0_subpop)
                if p_i <= 1:
                    print(
                    f"shadow{p_i} D0 has {len(sampled_shadow_D0)} points with {len(sampled_shadow_D0_subpop)},{len(sampled_shadow_D0_subpop)/len(sampled_shadow_D0):.2f} members from the target subpopulation and {self.label_split_d0*100:.4}%, {label_split_d0_subpop*100:.4}% class 1"
                    )

                self.shw_dataset.append(sampled_shadow_D0)
                temp_shadow_dataloaders.append(training_utils.dataframe_to_dataloader(
                            sampled_shadow_D0,
                            batch_size=self._batch_size,
                            using_ce_loss=self._using_ce_loss,
                            num_workers=self._num_workers,
                            persistent_workers=self._persistent_workers,
                            pin_memory=pin_memory,
                            shuffle=loader_shuffle,
                        ))
                

            for p_i in range(self._num_shadow_models,self._num_shadow_models*2):
                if self._allow_subsampling == True:
                    if equal_shadow_vic:
                        sampled_D1 = self._D1_pub_OH.sample(
                                    n=self._ntarget_samples, random_state=self.random_seed+p_i + 1
                                )
                    else:
                        sampled_D1 = self._D1_OH[0].sample(
                                    n=self._ntarget_samples, random_state=self.random_seed+p_i + 1
                                )
                else:
                    if equal_shadow_vic:
                        sampled_D1 = self._D1_pub_OH.sample(
                                n=self._ntarget_samples, random_state=self.random_seed+p_i + 1
                            )
                    else:
                        sampled_D1 = self._D1_OH[0].sample(
                                n=self._ntarget_samples, random_state=self.random_seed+p_i + 1
                            )
                sampled_shadow_D1 = data.v2_fix_imbalance_OH(
                    sampled_D1,
                    target_split=self._t1,
                    categories=self._categories,
                    target_attributes=self._target_attributes,
                    random_seed=self.random_seed,
                )	        
                sampled_shadow_D1_subpop = data.generate_subpopulation_OH(
                    sampled_shadow_D1, categories=self._categories, target_attributes=self._target_attributes            )



                self.label_split_d1 = sum(sampled_shadow_D1[sampled_shadow_D1.columns[-1]]) / len(sampled_shadow_D1)
                label_split_d1_subpop = sum(sampled_shadow_D1_subpop[sampled_shadow_D1_subpop.columns[-1]]) / len(sampled_shadow_D1_subpop)
                if p_i <= self._num_shadow_models+1:
                    print(
                    f"shadow{p_i} D1 has {len(sampled_shadow_D1)} points with {len(sampled_shadow_D1_subpop)},{len(sampled_shadow_D1_subpop)/len(sampled_shadow_D1):.2f} members from the target subpopulation and {self.label_split_d1*100:.4}%, {label_split_d1_subpop*100:.4}% class 1"
                    )

                self.shw_dataset.append(sampled_shadow_D1)
                temp_shadow_dataloaders.append(training_utils.dataframe_to_dataloader(
                            sampled_shadow_D1,
                            batch_size=self._batch_size,
                            using_ce_loss=self._using_ce_loss,
                            num_workers=self._num_workers,
                            persistent_workers=self._persistent_workers,
                            pin_memory=pin_memory,
                            shuffle=loader_shuffle,
                        ))
            
            self._shadow_dataloaders.append(temp_shadow_dataloaders)
        
        self._poison_percent = 0.3
        for p_i in range(self._num_shadow_models):
            if self._allow_subsampling == True:
                if equal_shadow_vic:
                    
                    sampled_D0 = self._D0_pub_OH.sample(
                                n=int((1 - self._poison_percent)*self._ntarget_samples), random_state=self.random_seed+i + 1
                            )
                else:
                    sampled_D0 = self._D0_OH[-1].sample(
                                n=int((1 - self._poison_percent)*self._ntarget_samples), random_state=self.random_seed+i + 1
                            )
            else:
                if equal_shadow_vic:
                    sampled_D0 = self._D0_pub_OH.sample(
                            n=int((1 - self._poison_percent)*self._ntarget_samples), random_state=self.random_seed+i + 1
                        )
                else:
                    sampled_D0 = self._D0_OH[-1].sample(
                                n=int((1 - self._poison_percent)*self._ntarget_samples), random_state=self.random_seed+i + 1
                            )  
            
            
            try:
                poisoned_D_MO = pd.concat(
                    [
                        sampled_D0,
                        self._Dp_OH.sample(
                            n=int(self._poison_percent * self._ntarget_samples),
                            random_state=self._pois_rs,
                            replace=False,
                        ),
                    ]
                )
            except:
                poisoned_D_MO = pd.concat(
                    [
                        sampled_D0,
                        self._Dp_OH.sample(
                            n=int(self._poison_percent * self._ntarget_samples),
                            random_state=self._pois_rs,
                            replace=True,
                        ),
                    ]
                )
            self._shadow_poisoned_dataloaders.append(training_utils.dataframe_to_dataloader(
                        poisoned_D_MO,
                        batch_size=self._batch_size,
                        using_ce_loss=self._using_ce_loss,
                        num_workers=self._num_workers,
                        persistent_workers=self._persistent_workers,
                        pin_memory=pin_memory,
                        shuffle=loader_shuffle,
                    ))
            
        for p_i in range(self._num_shadow_models):
            if self._allow_subsampling == True:
                if equal_shadow_vic:
                    sampled_D1 = self._D1_pub_OH.sample(
                                n=int((1 - self._poison_percent)*self._ntarget_samples), random_state=self.random_seed+i + 1
                            )
                else:
                    sampled_D1 = self._D1_OH[-1].sample(
                                n=int((1 - self._poison_percent)*self._ntarget_samples), random_state=self.random_seed+i + 1
                            )
            else:
                if equal_shadow_vic:
                    sampled_D1 = self._D1_pub_OH.sample(
                            n=int((1 - self._poison_percent)*self._ntarget_samples), random_state=self.random_seed+i + 1
                        )
                else:
                    sampled_D1 = self._D1_OH[-1].sample(
                                n=int((1 - self._poison_percent)*self._ntarget_samples), random_state=self.random_seed+i + 1
                            )  
            try:
                poisoned_D_MO = pd.concat(
                    [
                        sampled_D1,
                        self._Dp_OH.sample(
                            n=int(self._poison_percent * self._ntarget_samples),
                            random_state=self._pois_rs,
                            replace=False,
                        ),
                    ]
                )
            except:
                print("shadow poisoning dataset oversampling")
                poisoned_D_MO = pd.concat(
                    [
                        sampled_D1,
                        self._Dp_OH.sample(
                            n=int(self._poison_percent * self._ntarget_samples),
                            random_state=self._pois_rs,
                            replace=True,
                        ),
                    ]
                )
            self._shadow_poisoned_dataloaders.append(training_utils.dataframe_to_dataloader(
                        poisoned_D_MO,
                        batch_size=self._batch_size,
                        using_ce_loss=self._using_ce_loss,
                        num_workers=self._num_workers,
                        persistent_workers=self._persistent_workers,
                        pin_memory=pin_memory,
                        shuffle=loader_shuffle,
                    ))
            
 


        for t_i in range(2):
            temp_shadow_model = []
            temp_shadow_optim = []
            for s_i in range(self._num_shadow_models*2):
                if len(self._layer_sizes) != 0:
                    temp_shadow_model.append(models.NeuralNet(
                        input_dim=input_dim,
                        layer_sizes=self._layer_sizes,
                        num_classes=self._num_classes,
                        dropout=self._dropout,
                    ))

                else:
                    temp_shadow_model.append(models.LogisticRegression(
                        input_dim=input_dim,
                        num_classes=self._num_classes,
                        using_ce_loss=self._using_ce_loss,
                    ))

                    
                
            self._shadow_models.append(temp_shadow_model)

            for s_i in range(self._num_shadow_models*2):
                temp_shadow_optim.append(self._optim_init(self._shadow_models[t_i][s_i].parameters(),**self._optim_kwargs))
            self._shadow_optimizers.append(temp_shadow_optim)
            
        for i in range(self._num_target_models*2):
            if (i % self._num_target_models) < 2 :
                self._local_models[i].load_state_dict(copy.deepcopy(self._local_models[0].state_dict()))
                pass
                
        for i in range(2*self._num_shadow_models):
            #if (i % se*lf._num_shadow_models) < 2 :
            #self._shadow_models[i].load_state_dict(copy.deepcopy(self._shadow_models[i+self._num_shadow_models].state_dict()))
            #self._shadow_models[0][i].load_state_dict(copy.deepcopy(self._local_models[0].state_dict()))
            #self._shadow_models[1][i].load_state_dict(copy.deepcopy(self._local_models[0].state_dict()))
            pass
              
    def train_and_poison_target(self, need_metrics=False, df_cv=None):
        """Train target model with poisoned set if poisoning > 0"""

        owner_loaders = {}
        self._poisoned_target_models = [None] * self._num_target_models * 2
        input_dim = len(self._D0_OH[0].columns) - 1

        if self._allow_target_subsampling == False:

            self._ntarget_samples = self._D0_OH[0].shape[0]

            if len(self._Dp) == 0:
                poisoned_D0_MO = self._D0_OH[0].sample(
                    n=self._ntarget_samples, random_state=21
                )

            else:
                # Changes
                clean_D0_MO = self._D0_OH[0].sample(
                    n=int((1 - self._poison_percent) * self._ntarget_samples),
                    random_state=21,
                )

                if (
                    int(self._poison_percent * self._ntarget_samples) <= self._Dp_OH.shape[0]
                ):

                    poisoned_D0_MO = pd.concat(
                        [
                            clean_D0_MO,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._ntarget_samples),
                                random_state=self._pois_rs,
                                replace=False,
                            ),
                        ]
                    )
                else:

                    poisoned_D0_MO = pd.concat(
                        [
                            clean_D0_MO,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._ntarget_samples),
                                random_state=self._pois_rs,
                                replace=True,
                            ),
                        ]
                    )

        """Trains half the target models on t0 fraction"""
        # for i in tqdm(range(self._num_target_models), desc = "Training target models with frac t0"):
        for i in tqdm(range(self._num_target_models), desc=f"Training Target Models with {self._poison_percent*100:.2f}% poisoning"):

            if self._allow_target_subsampling == True:
                if len(self._Dp) == 0:
                    poisoned_D0_MO = self._D0_OH[0].sample(
                        n=self._ntarget_samples, random_state=i + 1
                    )

                else:

                    poisoned_D0_MO = pd.concat(
                        [
                            self._D0_OH[0].sample(
                                n=int(
                                    (1 - self._poison_percent) * self._ntarget_samples
                                ),
                                random_state=i + 1,
                            ),
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._ntarget_samples),
                                random_state=self._pois_rs,
                            ),
                        ]
                    )
            #print(f"poisoned target dataset {poisoned_D0_MO.shape}")
            owner_loaders["train"] = training_utils.dataframe_to_dataloader(
                poisoned_D0_MO,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
            )

            if len(self.target_model_layers) != 0:
                target_model = models.NeuralNet(
                    input_dim=input_dim,
                    layer_sizes=self.target_model_layers,
                    num_classes=self._num_classes,
                    dropout=self._dropout,
                )

            else:
                target_model = models.LogisticRegression(
                    input_dim=input_dim,
                    num_classes=self._num_classes,
                    using_ce_loss=self._using_ce_loss,
                )

            if self._mini_verbose:
                print("-" * 10, "\nTarget Model")
            training_utils.test(
                dataloaders=self._test_dataloader,
                model=target_model,
                # alterdata_list=self._alterdata_list,
                epochs=self._epochs,
                optim_init=self._optim_init,
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )
            self._poisoned_target_models[i], _ = training_utils.fit(
                dataloaders=owner_loaders,
                model=target_model,
                # alterdata_list=self._alterdata_list,
                epochs=self._epochs,
                optim_init=self._optim_init,
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )


        if self._allow_target_subsampling == False:
            self._ntarget_samples = self._D0_OH[0].shape[0]
            if len(self._Dp) == 0:
                # poisoned_D1_MO = self._D1_mo_OH.copy()
                poisoned_D1_MO = self._D1_mo_OH.sample(
                    n=self._ntarget_samples, random_state=21
                )

            else:
                clean_D1_MO = self._D1_OH[0].sample(
                    n=int((1 - self._poison_percent) * self._ntarget_samples),
                    random_state=21,
                )
                # Changes
                if (
                    int(self._poison_percent * self._ntarget_samples)
                    <= self._Dp_OH.shape[0]
                ):
                    poisoned_D1_MO = pd.concat(
                        [
                            clean_D1_MO,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._ntarget_samples),
                                random_state=self._pois_rs,
                                replace=False,
                            ),
                        ]
                    )
                else:
                    poisoned_D1_MO = pd.concat(
                        [
                            clean_D1_MO,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._ntarget_samples),
                                random_state=self._pois_rs,
                                replace=True,
                            ),
                        ]
                    )

        """Trains half the target models on t1 fraction"""
        for i in range(self._num_target_models, 2 * self._num_target_models):
            # for i in tqdm(range(self._num_target_models, 2*self._num_target_models), desc = "Training target models with frac t1"):

            if self._allow_target_subsampling == True:
                if len(self._Dp) == 0:
                    poisoned_D1_MO = self._D1_OH[0].sample(
                        n=self._ntarget_samples, random_state=i + 1
                    )

                else:
                    poisoned_D1_MO = pd.concat(
                        [
                            self._D1_OH[0].sample(
                                n=int(
                                    (1 - self._poison_percent) * self._ntarget_samples
                                ),
                                random_state=i + 1,
                            ),
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._ntarget_samples),
                                random_state=self._pois_rs,
                            ),
                        ]
                    )

            owner_loaders["train"] = training_utils.dataframe_to_dataloader(
                poisoned_D1_MO,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
            )

            if len(self.target_model_layers) != 0:
                target_model = models.NeuralNet(
                    input_dim=input_dim,
                    layer_sizes=self.target_model_layers,
                    num_classes=self._num_classes,
                    dropout=self._dropout,
                )

            else:
                target_model = models.LogisticRegression(
                    input_dim=input_dim,
                    num_classes=self._num_classes,
                    using_ce_loss=self._using_ce_loss,
                )

            if self._mini_verbose:
                print("-" * 10, "\nTarget Model")

            self._poisoned_target_models[i], _ = training_utils.fit(
                dataloaders=owner_loaders,
                model=target_model,
                # alterdata_list=self._alterdata_list,
                epochs=self._epochs,
                optim_init=self._optim_init,
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )


    def property_inference_categorical(
        self,
        num_shadow_models=1,
        query_trials=1,
        query_selection="random",
        distinguishing_test="median",
    ):
        """Property inference attack for categorical data. (e.g. Census, Adults)

        ...
        Parameters
        ----------
            num_shadow_models : int
                The number of shadow models per "world" to use in the attack
            query_trials : int
                Number of times we want to query num_queries points on the target
            distinguishing_test : str
                The distinguishing test to use on logit distributions
                Options are the following:

                "median" : uses the middle of medians as a threshold and checks on which side of the threshold
                           the majority of target model prediction confidences are on
                "divergence" : uses KL divergence to measure the similarity between the target model
                               prediction scores and the

        ...
        Returns
        ----------
            out_M0 : np.array
                Array of scaled logit values for M0
            out_M1 : np.array
                Array of scaled logit values for M1
            logits_each_trial : list of np.arrays
                Arrays of scaled logit values for target model.
                Each index is the output of a single query to the
                target model
            predictions : list
                Distinguishing test predictions; 0 if prediction
                is t0, 1 if prediction is t1
            correct_trials : list
                List of booleans dentoting whether query trial i had
                a correct prediction
        """
        for i in range(self._num_target_models*2):
            test_print = ""

                
            epoch_test_acc = training_utils.test(
                dataloaders=self._test_dataloader,
                model=self._poisoned_target_models[i],
                #model=self._local_models[p_i][0],
                # alterdata_list=self._alterdata_list,
                epochs=self._epochs,
                optim_init=self._optim_init,
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )
        test_print += f"model {i} : {epoch_test_acc:0.4f}/"
                
        print(test_print)


        # Train multiple shadow models to reduce variance
        if self._mini_verbose:
            print("-" * 10, "\nTraining Shadow Models...")

        D0_loaders = {}
        D1_loaders = {}

        if self._allow_subsampling == False:

            self._nsub_samples = self._D0_OH[-1].shape[0]

            # print("Size of Shadow model dataset: ", self._nsub_samples)

            if len(self._Dp) == 0:
                poisoned_D0 = self._D0_OH[-1].sample(n=self._nsub_samples, random_state=21)
                poisoned_D1 = self._D1_OH[-1].sample(n=self._nsub_samples, random_state=21)

            else:
                clean_D0 = self._D0_OH[-1].sample(
                    n=int((1 - self._poison_percent) * self._nsub_samples),
                    random_state=21,
                )
                clean_D1 = self._D1_OH[-1].sample(
                    n=int((1 - self._poison_percent) * self._nsub_samples),
                    random_state=21,
                )

                if (
                    int(self._poison_percent * self._nsub_samples)
                    <= self._Dp_OH.shape[0]
                ):
                    # Changes
                    poisoned_D0 = pd.concat(
                        [
                            clean_D0,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._nsub_samples),
                                random_state=self._pois_rs,
                                replace=False,
                            ),
                        ]
                    )
                    poisoned_D1 = pd.concat(
                        [
                            clean_D1,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._nsub_samples),
                                random_state=self._pois_rs,
                                replace=False,
                            ),
                        ]
                    )
                else:
                    poisoned_D0 = pd.concat(
                        [
                            clean_D0,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._nsub_samples),
                                random_state=self._pois_rs,
                                replace=True,
                            ),
                        ]
                    )
                    poisoned_D1 = pd.concat(
                        [
                            clean_D1,
                            self._Dp_OH.sample(
                                n=int(self._poison_percent * self._nsub_samples),
                                random_state=self._pois_rs,
                                replace=True,
                            ),
                        ]
                    )

            D0_loaders["train"] = training_utils.dataframe_to_dataloader(
                poisoned_D0,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
            )

            D1_loaders["train"] = training_utils.dataframe_to_dataloader(
                poisoned_D1,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
            )

        Dtest_OH_loader = training_utils.dataframe_to_dataloader(
            self._Dtest_OH,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            using_ce_loss=self._using_ce_loss,
        )

        input_dim = len(self._D0_OH[0].columns) - 1

        out_M0 = np.array([])
        out_M1 = np.array([])

        for i in tqdm(range(num_shadow_models), desc=f"Training {num_shadow_models} Shadow Models with {self._poison_percent*100:.2f}% Poisoning"):

            if self._mini_verbose:
                print("-" * 10, f"\nModels {i+1}")

            if len(self._layer_sizes) != 0:
                M0_model = models.NeuralNet(
                    input_dim=input_dim,
                    layer_sizes=self._layer_sizes,
                    num_classes=self._num_classes,
                    dropout=self._dropout,
                )

                M1_model = models.NeuralNet(
                    input_dim=input_dim,
                    layer_sizes=self._layer_sizes,
                    num_classes=self._num_classes,
                    dropout=self._dropout,
                )
            else:
                M0_model = models.LogisticRegression(
                    input_dim=input_dim,
                    num_classes=self._num_classes,
                    using_ce_loss=self._using_ce_loss,
                )

                M1_model = models.LogisticRegression(
                    input_dim=input_dim,
                    num_classes=self._num_classes,
                    using_ce_loss=self._using_ce_loss,
                )

            if self._allow_subsampling == True:

                if len(self._Dp) == 0:
                    poisoned_D0 = self._D0_OH[-1].sample(n=self._nsub_samples)
                    poisoned_D1 = self._D1_OH[-1].sample(n=self._nsub_samples)

                else:

                    if self._allow_target_subsampling == True:

                        poisoned_D0 = pd.concat(
                            [
                                self._D0_OH[-1].sample(
                                    n=int(
                                        (1 - self._poison_percent) * self._nsub_samples
                                    )
                                ),
                                self._Dp_OH.sample(
                                    n=int(self._poison_percent * self._nsub_samples),
                                    random_state=self._pois_rs,
                                ),
                            ]
                        )

                        poisoned_D1 = pd.concat(
                            [
                                self._D1_OH[-1].sample(
                                    n=int(
                                        (1 - self._poison_percent) * self._nsub_samples
                                    )
                                ),
                                self._Dp_OH.sample(
                                    n=int(self._poison_percent * self._nsub_samples),
                                    random_state=self._pois_rs,
                                ),
                            ]
                        )
                    else:
                        poisoned_D0 = pd.concat(
                            [
                                self._D0_OH.sample(
                                    n=int(
                                        (1 - self._poison_percent) * self._nsub_samples
                                    )
                                ),
                                self._Dp_OH.sample(
                                    n=int(self._poison_percent * self._nsub_samples)
                                ),
                            ]
                        )

                        poisoned_D1 = pd.concat(
                            [
                                self._D1_OH.sample(
                                    n=int(
                                        (1 - self._poison_percent) * self._nsub_samples
                                    )
                                ),
                                self._Dp_OH.sample(
                                    n=int(self._poison_percent * self._nsub_samples)
                                ),
                            ]
                        )

                D0_loaders["train"] = training_utils.dataframe_to_dataloader(
                    poisoned_D0,
                    batch_size=self._batch_size,
                    using_ce_loss=self._using_ce_loss,
                    num_workers=self._num_workers,
                    persistent_workers=self._persistent_workers,
                )

                D1_loaders["train"] = training_utils.dataframe_to_dataloader(
                    poisoned_D1,
                    batch_size=self._batch_size,
                    using_ce_loss=self._using_ce_loss,
                    num_workers=self._num_workers,
                    persistent_workers=self._persistent_workers,
                )

            M0_trained, _ = training_utils.fit(
                dataloaders=D0_loaders,
                model=M0_model,
                epochs=self._epochs,
                optim_init=self._optim_init,
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )

            M1_trained, _ = training_utils.fit(
                dataloaders=D1_loaders,
                model=M1_model,
                epochs=self._epochs,
                optim_init=self._optim_init,
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )

            out_M0_temp = training_utils.get_logits_torch(
                Dtest_OH_loader,
                M0_trained,
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class
            )

            out_M0 = np.append(out_M0, out_M0_temp)

            out_M1_temp = training_utils.get_logits_torch(
                Dtest_OH_loader,
                M1_trained,
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class
            )
            out_M1 = np.append(out_M1, out_M1_temp)
            
            results = []
            titles = []
            results.append(out_M0_temp)
            results.append(out_M1_temp)
            titles.append("shadow 0")
            titles.append("shadow 1")
            self.plot_and_save_histograms(results, titles,f"./{i}_histograms_{self._target_attributes}_snap.png")
            
            if True:
                print(
                    f"M0 Mean: {out_M0.mean():.5}, Variance: {out_M0.var():.5}, StDev: {out_M0.std():.5}, Median: {np.median(out_M0):.5}"
                )
                print(
                    f"M1 Mean: {out_M1.mean():.5}, Variance: {out_M1.var():.5}, StDev: {out_M1.std():.5}, Median: {np.median(out_M1):.5}"
                )

        if distinguishing_test == "median":
            midpoint_of_medians = (np.median(out_M0) + np.median(out_M1)) / 2
            thresh = midpoint_of_medians

        if self._verbose:
            print(f"Threshold: {thresh:.5}")

        # Query the target model and determine
        correct_trials = 0

        if self._mini_verbose:
            print(
                "-" * 10,
                f"\nQuerying target model {query_trials} times with {self._num_queries} query samples",
            )

        oversample_flag = False
        if self._num_queries > self._Dtest_OH.shape[0]:
            oversample_flag = True
            print("Oversampling test queries")

        for i, poisoned_target_model in enumerate(tqdm(self._poisoned_target_models, desc=f"Querying Models and Running Distinguishing Test")):
            for query_trial in range(query_trials):

                if query_selection.lower() == "random":
                    Dtest_OH_sample_loader = training_utils.dataframe_to_dataloader(
                        self._Dtest_OH.sample(
                            n=self._num_queries, replace=oversample_flag, random_state = i+1
                        ),
                        batch_size=self._batch_size,
                        num_workers=self._num_workers,
                        using_ce_loss=self._using_ce_loss,
                    )
                else:
                    print("Incorrect Query selection type")

                out_target = training_utils.get_logits_torch(
                    Dtest_OH_sample_loader,
                    poisoned_target_model,
                    device=self._device,
                    middle_measure=self._middle,
                    variance_adjustment=self._variance_adjustment,
                    label = self._poison_class
                )

                if True:
                    print("-" * 10)
                    print(
                        f"Target Mean: {out_target.mean():.5}, Variance: {out_target.var():.5}, StDev: {out_target.std():.5}, Median: {np.median(out_target):.5}\n"
                    )

                """ Perform distinguishing test"""
                if distinguishing_test == "median":
                    M0_score = len(out_target[out_target > thresh])
                    M1_score = len(out_target[out_target < thresh])
                    if self._verbose:
                        print(f"M0 Score: {M0_score}\nM1 Score: {M1_score}")

                    if M0_score >= M1_score:
                        if self._mini_verbose:
                            print(
                                f"Target is in t0 world with {M0_score/len(out_target)*100:.4}% confidence"
                            )

                        correct_trials = correct_trials + int(
                            self._target_worlds[i] == 0
                        )

                    elif M0_score < M1_score:
                        if self._mini_verbose:
                            print(
                                f"Target is in t1 world {M1_score/len(out_target)*100:.4}% confidence"
                            )

                        correct_trials = correct_trials + int(
                            self._target_worlds[i] == 1
                        )

        if distinguishing_test == "median":
            return (
                out_M0,
                out_M1,
                thresh,
                correct_trials / (len(self._target_worlds) * query_trials),
            )
    def train_temp(self, need_metrics=False, distinguishing_test = "median",df_cv=None):
        self.temp_dataloader[0] = self._shadow_dataloaders[0]
        self.temp_dataloader[0] = self._party_dataloaders[0]
        self.temp_dataloader[1] = self._public_dataloader

        print(self._public_dataloader.dataset[0][0].shape)
        for t_i in range(1,2):
            training_utils.fit_local(
                dataloaders=self.temp_dataloader[t_i],#self._local_dataloaders[0],
                model=self.temp_models[t_i],
                # alterdata_list=self._alterdata_list,
                epochs=100,
                optim_init=self.temp_optimizers[t_i],
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )
            epoch_test_acc = training_utils.test(
                dataloaders=self._test_dataloader,
                model=self.temp_models[t_i],
                #model=self._local_models[p_i][0],
                # alterdata_list=self._alterdata_list,
                epochs=self._epochs,
                optim_init=self._optim_init,
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )
            print(f"temp {t_i} : {epoch_test_acc:0.4f}/")
    def gradients_know_answers(self):
        D_X, D_y = training_utils.extract_data_from_dataloader(self._public_dataloader)
        D_init = D_X.clone().detach()
        criterion = nn.CrossEntropyLoss()
        for (inputs, labels) in self._public_dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            inputs_syn = inputs.clone().requires_grad_(True)
            optimizer = torch.optim.SGD([inputs_syn], **{"lr":1})

            
            for iter in range(200):

                #Recon_X = gen(original_X)
                #inputs_syn = Recon_X
                #optim = optimizer(gen.parameters())
                y = self.temp_models[0](inputs_syn)
                loss = criterion(y,labels)
                pub_loss = criterion(self.temp_models[1](inputs_syn),labels)
                pub_logits = F.softmax(self.temp_models[1](inputs_syn),dim=1)
                pub_entropy = torch.mean(-torch.sum(pub_logits * torch.log(pub_logits + 1e-10), dim=1))
                l2_dist = torch.mean(torch.abs(inputs_syn))
                print(loss.item(),pub_entropy.item(),pub_loss.item(),l2_dist.item())
                loss += l2_dist
                loss += (1-pub_entropy)
                #loss += gen_loss
                loss.backward()
                optimizer.step()

            print(inputs[0])
            print(inputs[0][0]*self.original_max_values["age"])
            print(inputs_syn[0])
            print(inputs_syn[0][0]*self.original_max_values["age"])
            exit(1)
    def train_shadow_from_victim(self, need_metrics=False, distinguishing_test = "median",df_cv=None):
        
        aggr = "fedmd"
        self._local_models[0] = self._local_models[0].to(self.device)
        self._local_models[-1] = self._local_models[-1].to(self.device)
        if aggr == "fedmd":
            X = torch.Tensor([])
            vic_logits_0 = torch.Tensor([])
            vic_logits_1 = torch.Tensor([])


            for p_i in range(self._party_nums-1):
                self._party_models[p_i].to(self.device)
                    
            for (inputs, labels) in self._public_dataloader:
                    
                inputs = inputs.to(self.device)
                out = []
                
                

                with torch.no_grad():

                    X = torch.concat([X, inputs.detach().cpu()])
                    
                    
                    vic_logit = self._local_models[0](inputs)
                    vic_logits_0= torch.concat([vic_logits_0, vic_logit.detach().clone().cpu()])
                    
                    vic_logit = self._local_models[-1](inputs)
                    vic_logits_1 = torch.concat([vic_logits_1, vic_logit.detach().clone().cpu()])


            

            vic_non_prop_datasets_0 = torch.utils.data.TensorDataset(X,vic_logits_0.clone().detach())
            vic_non_prop_public_loader_0 = torch.utils.data.DataLoader(
                    vic_non_prop_datasets_0,
                    batch_size=self._batch_size,
                    num_workers=self._num_workers,
                    shuffle=self._shuffle,
                    persistent_workers=self._persistent_workers,
                )
            
            vic_non_prop_datasets_1 = torch.utils.data.TensorDataset(X,vic_logits_1.clone().detach())
            vic_non_prop_public_loader_1 = torch.utils.data.DataLoader(
                    vic_non_prop_datasets_1,
                    batch_size=int(self._batch_size),
                    num_workers=self._num_workers,
                    shuffle=self._shuffle,
                    persistent_workers=self._persistent_workers,
                )
            
            public_all = training_utils.dataframe_to_dataloader(
                self._D_pub_OH,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
                pin_memory=False,
                shuffle=self._shuffle,
            )
 


            for t_i in range(2):
                for s_i in range(2*self._num_shadow_models):

                    temp_optim = self._optim_init(self._shadow_models[t_i][s_i].parameters(),**self._optim_kwargs)
                    training_utils.fit_kd_local(                    dataloaders=vic_non_prop_public_loader_0,                   model=self._shadow_models[t_i][s_i],#alterdata_list=self._alterdata_list,     
                                epochs=50, optim_init=temp_optim,                    optim_kwargs=self._optim_kd_kwargs,                    \
                            criterion=nn.KLDivLoss(reduction="batchmean"),
                            #criterion=nn.L1Loss(),
                            device=self._device,                    verbose=self._verbose,                    mini_verbose=self._mini_verbose,                    early_stopping=self._early_stopping,                    tol=self._tol,                    train_only=True, L1_loss=True)



        return
    def train_extraction(self, need_metrics=False, distinguishing_test = "median",df_cv=None):
        
        aggr = "fedmd"
        self._local_models[0] = self._local_models[0].to(self.device)
        self._local_models[-1] = self._local_models[-1].to(self.device)
        if aggr == "fedmd":
            X = torch.Tensor([])
            vic_logits_0 = torch.Tensor([])
            vic_logits_1 = torch.Tensor([])


            for p_i in range(self._party_nums-1):
                self._party_models[p_i].to(self.device)
                    
            for (inputs, labels) in self._public_dataloader:
                    
                inputs = inputs.to(self.device)
                out = []
                
                

                with torch.no_grad():

                    X = torch.concat([X, inputs.detach().cpu()])
                    
                    
                    vic_logit = self._local_models[0](inputs)
                    vic_logits_0= torch.concat([vic_logits_0, vic_logit.detach().clone().cpu()])
                    
                    vic_logit = self._local_models[-1](inputs)
                    vic_logits_1 = torch.concat([vic_logits_1, vic_logit.detach().clone().cpu()])


            

            vic_non_prop_datasets_0 = torch.utils.data.TensorDataset(X,vic_logits_0.clone().detach())
            vic_non_prop_public_loader_0 = torch.utils.data.DataLoader(
                    vic_non_prop_datasets_0,
                    batch_size=self._batch_size,
                    num_workers=self._num_workers,
                    shuffle=self._shuffle,
                    persistent_workers=self._persistent_workers,
                )
            
            #vic_non_prop_datasets_1 = torch.utils.data.TensorDataset(X[self.non_prop_indices],vic_logits_1[self.non_prop_indices].clone().detach())
            vic_non_prop_datasets_1 = torch.utils.data.TensorDataset(X,vic_logits_1.clone().detach())
            vic_non_prop_public_loader_1 = torch.utils.data.DataLoader(
                    vic_non_prop_datasets_1,
                    batch_size=int(self._batch_size),
                    num_workers=self._num_workers,
                    shuffle=self._shuffle,
                    persistent_workers=self._persistent_workers,
                )
            
            public_all = training_utils.dataframe_to_dataloader(
                self._D_pub_OH,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
                pin_memory=False,
                shuffle=self._shuffle,
            )
            '''
            training_utils.fit_local(
                dataloaders=public_all,
                model=self.temp_models[0],
                # alterdata_list=self._alterdata_list,
                epochs=50,
                optim_init=self.temp_optimizers[0],
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )
            '''
            #shadow KD train
            training_utils.fit_kd_local(                    dataloaders=vic_non_prop_public_loader_0,                   model=self.temp_models[0],#alterdata_list=self._alterdata_list,     
                        epochs=500, optim_init=self.temp_optimizers[0],                    optim_kwargs=self._optim_kd_kwargs,                    \
                    criterion=nn.KLDivLoss(reduction="batchmean"),
                    #criterion=nn.L1Loss(),
                    device=self._device,                    verbose=self._verbose,                    mini_verbose=self._mini_verbose,                    early_stopping=self._early_stopping,                    tol=self._tol,                    train_only=True, )

                
            epoch_test_acc = training_utils.test(
                dataloaders=self._test_dataloader,
                model=self.temp_models[0],
                #model=self._local_models[p_i][0],
                # alterdata_list=self._alterdata_list,
                epochs=self._epochs,
                optim_init=self._optim_init,
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )
            print(f"temp 0: {epoch_test_acc:0.4f}/")
            #shadow KD train
            training_utils.fit_kd_local(                    dataloaders=vic_non_prop_public_loader_1,                   model=self.temp_models[2],#alterdata_list=self._alterdata_list,     
                        epochs=500, optim_init=self.temp_optimizers[2],                    optim_kwargs=self._optim_kd_kwargs,                    \
                    criterion=nn.KLDivLoss(reduction="batchmean"),
                    #criterion=nn.L1Loss(),
                    device=self._device,                    verbose=self._verbose,                    mini_verbose=self._mini_verbose,                    early_stopping=self._early_stopping,                    tol=self._tol,                    train_only=True, )

                
            epoch_test_acc = training_utils.test(
                dataloaders=self._test_dataloader,
                model=self.temp_models[2],
                #model=self._local_models[p_i][0],
                # alterdata_list=self._alterdata_list,
                epochs=self._epochs,
                optim_init=self._optim_init,
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )
            print(f"temp 2: {epoch_test_acc:0.4f}/")

        return
    def save_experiements(self, root_path='./experiments'):
        save_path = os.path.join(root_path, f'{self.dataset}_{self._categories}_{self._target_attributes}_{self._num_target_models}_{self._num_shadow_models}_{self._public_poison}_{self._t0}vs{self._t1}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print(f"save at {save_path}")
        torch.save(self._test_dataloader.dataset , os.path.join(save_path, 'test_dataset.pt'))
        torch.save(self._public_dataloader.dataset, os.path.join(save_path, 'public_dataset.pt'))
        for i, dataloader in enumerate(self._local_dataloaders):
            torch.save(dataloader.dataset, os.path.join(save_path, f'local_dataset_{i}.pt'))
        for i, dataloader in enumerate(self._party_dataloaders):
            torch.save(dataloader.dataset, os.path.join(save_path, f'party_dataset_{i}.pt'))
        for i, model in enumerate(self._local_models):
            torch.save(model.state_dict(), os.path.join(save_path, f'local_model_{i}.pth'))
        for i, model in enumerate(self._party_models):
            torch.save(model.state_dict(), os.path.join(save_path, f'party_model_{i}.pth'))
        for i, optimizer in enumerate(self._local_optimizers):
            torch.save(optimizer.state_dict(), os.path.join(save_path, f'local_optimizer_{i}.pth'))
        for i, optimizer in enumerate(self._party_optimizers):
            torch.save(optimizer.state_dict(), os.path.join(save_path, f'party_optimizer_{i}.pth'))
        for i, model in enumerate(self._shadow_models):
            torch.save(model.state_dict(), os.path.join(save_path, f'shadow_model_{i}.pth'))
        for i, dataloader in enumerate(self._shadow_dataloaders):
            torch.save(dataloader.dataset, os.path.join(save_path, f'shadow_dataset_{i}.pt'))
        for i, optimizer in enumerate(self._shadow_optimizers):
            torch.save(optimizer.state_dict(), os.path.join(save_path, f'shadow_optimizer_{i}.pth'))
        for t_i in range(2):
            torch.save(self.temp_models[t_i].state_dict(), os.path.join(save_path, f'temp_model_{t_i}.pth'))
    
    def load_experiements(self, root_path='./experiments'):
        load_path = os.path.join(root_path, f'{self.dataset}_{self._categories}_{self._target_attributes}_{self._num_target_models}_{self._num_shadow_models}_{self._public_poison}_{self._t0}vs{self._t1}')
        self._test_dataloader = training_utils.dataframe_to_dataloader(
                        torch.load(os.path.join(load_path, 'test_dataset.pt')),
                        batch_size=self._batch_size,
                        using_ce_loss=self._using_ce_loss,
                        num_workers=self._num_workers,
                        persistent_workers=self._persistent_workers,
                        pin_memory=True,
                    )
        
        self._public_dataloader = training_utils.dataframe_to_dataloader(
                        torch.load(os.path.join(load_path, 'public_dataset.pt')),
                        batch_size=self._batch_size,
                        using_ce_loss=self._using_ce_loss,
                        num_workers=self._num_workers,
                        persistent_workers=self._persistent_workers,
                        pin_memory=True,
                        shuffle=False,
                    )
        
        self._local_dataloaders = [
            training_utils.dataframe_to_dataloader(
                torch.load(os.path.join(load_path, f'local_dataset_{i}.pt')),
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
                pin_memory=True,
                shuffle=self._shuffle,
                )   for i in range(self._num_target_models * 2) ]
        
        self._party_dataloaders = [
            training_utils.dataframe_to_dataloader(
                torch.load(os.path.join(load_path, f'party_dataset_{i}.pt')) ,
                batch_size=self._batch_size,
                using_ce_loss=self._using_ce_loss,
                num_workers=self._num_workers,
                persistent_workers=self._persistent_workers,
                pin_memory=True,
                shuffle=self._shuffle,
            )
            for i in range(self._party_nums - 1)
        ]

        self._shadow_dataloaders = [
            training_utils.dataframe_to_dataloader(
                        torch.load(os.path.join(load_path, f'shadow_dataset_{i}.pt')),
                        batch_size=self._batch_size,
                        using_ce_loss=self._using_ce_loss,
                        num_workers=self._num_workers,
                        persistent_workers=self._persistent_workers,
                        pin_memory=True,
                        shuffle=self._shuffle,
                    )
            for i in range(self._num_shadow_models * 2)
        ]
        
        for i, model in enumerate(self._local_models):
            model.load_state_dict(torch.load(os.path.join(load_path, f'local_model_{i}.pth')))
        for i, model in enumerate(self._party_models):
            model.load_state_dict(torch.load(os.path.join(load_path, f'party_model_{i}.pth')))
        for i, optimizer in enumerate(self._local_optimizers):
            optimizer.load_state_dict(torch.load(os.path.join(load_path, f'local_optimizer_{i}.pth')))
        for i, optimizer in enumerate(self._party_optimizers):
            optimizer.load_state_dict(torch.load(os.path.join(load_path, f'party_optimizer_{i}.pth')))
        for i, model in enumerate(self._shadow_models):
            model.load_state_dict(torch.load(os.path.join(load_path, f'shadow_model_{i}.pth')))
        
        self._shadow_optimizers = [self._optim_init(self._shadow_models[i].parameters(), **self._optim_kwargs) for i in range(self._num_shadow_models * 2)]
        for i, optimizer in enumerate(self._shadow_optimizers):
            optimizer.load_state_dict(torch.load(os.path.join(load_path, f'shadow_optimizer_{i}.pth')))
        for t_i in range(2):
            self.temp_models[t_i].load_state_dict(torch.load(os.path.join(load_path, 'temp_model.pth')))
    
    def test_locals(self):
        test_print = ""
        a = time.time()
        for i in range(self._num_target_models*2):
            epoch_test_acc = training_utils.test(
                dataloaders=self._test_dataloader,
                model=self._local_models[i],
                #model=self._local_models[p_i][0],
                # alterdata_list=self._alterdata_list,
                epochs=self._epochs,
                optim_init=self._optim_init,
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )
            test_print += f"vic {i} : {epoch_test_acc:0.4f}/"
        
        print(test_print)
    
    def test_shadows(self):
        test_print = ""
        for s_i in range(self._num_shadow_models*2):
            epoch_test_acc = training_utils.test(
                    dataloaders=self._test_dataloader,
                    #model=self._shadow_models[0][s_i],
                    model=self._shadow_models[s_i],
                    # alterdata_list=self._alterdata_list,
                    epochs=self._epochs,
                    optim_init=self._optim_init,
                    optim_kwargs=self._optim_kwargs,
                    criterion=self._criterion,
                    device=self._device,
                    verbose=self._verbose,
                    mini_verbose=self._mini_verbose,
                    early_stopping=self._early_stopping,
                    tol=self._tol,
                    train_only=True,
                )
            test_print += f"s_model {s_i} : {epoch_test_acc:0.4f}/"
        print(test_print)



    def train_locals(self, need_metrics=False, distinguishing_test = "median",df_cv=None,current_epoch=None):
        test_print = ""

        a = time.time()
        for i in range(1):
            test_print = ""
            
            epoch_test_acc = training_utils.test(
                dataloaders=self._test_dataloader,
                model=self._local_models[0],
                #model=self._local_models[p_i][0],
                # alterdata_list=self._alterdata_list,
                epochs=self._epochs,
                optim_init=self._optim_init,
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )
            test_print += f"vic {i} : {epoch_test_acc:0.4f}/"
            for p_i in range(self._party_nums-1):
                
                epoch_test_acc = training_utils.test(
                    dataloaders=self._test_dataloader,
                    model=self._party_models[p_i],
                    #model=self._local_models[p_i][0],
                    # alterdata_list=self._alterdata_list,
                    epochs=self._epochs,
                    optim_init=self._optim_init,
                    optim_kwargs=self._optim_kwargs,
                    criterion=self._criterion,
                    device=self._device,
                    verbose=self._verbose,
                    mini_verbose=self._mini_verbose,
                    early_stopping=self._early_stopping,
                    tol=self._tol,
                    train_only=True,
                )
                test_print += f"model {p_i} : {epoch_test_acc:0.4f}/"
            print(f"### Epoch {current_epoch} ||",test_print,"###")
        
        for i in range(self._num_target_models*2):
            training_utils.fit_local(
                dataloaders=self._local_dataloaders[i],
                model=self._local_models[i],
                # alterdata_list=self._alterdata_list,
                epochs=self._local_epoch,
                optim_init=self._local_optimizers[i],
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )
        for i in range(self._party_nums-1):
            training_utils.fit_local(
                dataloaders=self._party_dataloaders[i],
                model=self._party_models[i],
                # alterdata_list=self._alterdata_list,
                epochs=self._local_epoch,
                optim_init=self._party_optimizers[i],
                optim_kwargs=self._optim_kwargs,
                criterion=self._criterion,
                device=self._device,
                verbose=self._verbose,
                mini_verbose=self._mini_verbose,
                early_stopping=self._early_stopping,
                tol=self._tol,
                train_only=True,
            )

        
    def train_shadow(self):
        
        for idi in range(0):
            test_print = ""
            for s_i in range(self._num_shadow_models*2):
        
                
                epoch_test_acc = training_utils.test(
                    dataloaders=self._test_dataloader,
                    #model=self._shadow_models[0][s_i],
                    model=self._shadow_models[s_i],
                    # alterdata_list=self._alterdata_list,
                    epochs=self._epochs,
                    optim_init=self._optim_init,
                    optim_kwargs=self._optim_kwargs,
                    criterion=self._criterion,
                    device=self._device,
                    verbose=self._verbose,
                    mini_verbose=self._mini_verbose,
                    early_stopping=self._early_stopping,
                    tol=self._tol,
                    train_only=True,
                )
                test_print += f"s_model {s_i} : {epoch_test_acc:0.4f}/"
            print(test_print)

        for t_i in range(2):
            for s_i in range(self._num_shadow_models*2):

                training_utils.fit_local(
                    dataloaders=self._shadow_dataloaders[t_i][s_i],
                    #dataloaders=self._local_dataloaders[s_i],
                    model=self._shadow_models[t_i][s_i],
                    # alterdata_list=self._alterdata_list,
                    epochs=self._local_epoch,
                    optim_init=self._shadow_optimizers[t_i][s_i],
                    optim_kwargs=self._optim_kwargs,
                    criterion=self._criterion,
                    device=self._device,
                    verbose=self._verbose,
                    mini_verbose=self._mini_verbose,
                    early_stopping=self._early_stopping,
                    tol=self._tol,
                    train_only=True,
                    #noise_std=2,
                )                    


    ''''''
    def plot_and_save_histograms(self, data_list, titles, filename, estimated_logit=None):
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if False:#no hist
            return
        plt.figure(figsize=(10, 6))

        num_colors = len(data_list)
        colors = cm.get_cmap('Paired', num_colors) # 색상 리스트

        # 모든 데이터 세트를 하나의 그림에 결합하여 저장
        plt.figure(figsize=(10, 6))
        # 모든 데이터 세트를 같은 축에 히스토그램으로 그립니다.
        for i, data in enumerate(data_list):
            plt.hist(data, bins=30, alpha=0.5, color=colors(i), label=titles[i], edgecolor='black')
            mean = np.mean(data)
            plt.axvline(mean, color=colors(i), linestyle='dashed', linewidth=2, label=f'Median_{titles[i]}_{mean:.2f}')

    
        plt.title("Combined Histograms")
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.grid(True)
        if estimated_logit is not None:
            plt.axvline(estimated_logit, color='red', linestyle='dashed', linewidth=2, label=f'Estimated Logit_{estimated_logit:.2f}')
        plt.legend(loc='upper right')
        plt.tight_layout()  # 플롯 레이아웃을 자동으로 조절합니다.
        #plt.xlim(-2, 2)
        plt.savefig(filename)
        plt.close()
    def calculate_gap(self, distinguishing_test="median",query_trials = 10,mode=0,variance_adjustment=1):
        shadow_poisoned_epoch = 0
        shadow_kd_epoch = 50
        _verbose = True
        # simulation

                
      
        # training poisoned_student_model
        '''
        for i in range(self._num_target_models*2):
            
            training_utils.fit_local(                    dataloaders=self._shadow_poisoned_dataloaders[i],                   model=self._shadow_poisoned_model[i],#alterdata_list=self._alterdata_list,     
                    epochs=shadow_poisoned_epoch, optim_init=self._optim_init,                    optim_kwargs=self._optim_kwargs,                    \
                    criterion=self._criterion,                    device=self._device,                    verbose=self._verbose,                    mini_verbose=self._mini_verbose,                    early_stopping=self._early_stopping,                    tol=self._tol,                    train_only=True, )
            d_loader = {"train":self._shadow_poisoned_dataloaders[i]}
        '''
        # get threshold and distinguish p distribution
        correct_trials = 0
        total = 0
        Dtest_OH_loader = training_utils.dataframe_to_dataloader(
            #self._Dtest_OH[:240],
            self._D_OH[-1],
            batch_size=1024,
            num_workers=self._num_workers,
            using_ce_loss=self._using_ce_loss,
        )

        for i in range(self._num_target_models*2):
            
            
            public_X = torch.Tensor([])
            public_y = torch.Tensor([])
            if mode == 2 :
                self._public_dataloader[i] = training_utils.dataframe_to_dataloader(
                    #self._Dtest_OH[:240],
                    self._D_OH[-1],
                    batch_size=1024,
                    num_workers=self._num_workers,
                    using_ce_loss=self._using_ce_loss,
                )
            for (inputs, labels) in self._public_dataloader[i]:
            #for (inputs,labels) in Dtest_OH_loader:
                with torch.no_grad():
                    public_X =torch.concat([public_X, inputs.detach().cpu()])
                    public_y = torch.concat([public_y, labels.detach().clone().cpu()])
            non_prop_mask = torch.ones(public_X.shape[0],dtype=torch.bool)
            mask = torch.ones(public_X.shape[0],dtype=torch.bool)
            for prop_i in self._prop_index:
                non_prop_mask |=(public_X[:,prop_i]!=1)
                mask &= (public_X[:,prop_i]==1)
                
            mask &= (public_y != self._poison_class)
            non_prop_mask &= (public_y != self._poison_class)
            
            prop_indices = torch.where(mask)[0]
            non_prop_indices = torch.where(non_prop_mask)[0]
            
            non_property_public_X = public_X[non_prop_indices]
            non_property_public_y = public_y[non_prop_indices]
            
            property_public_X = public_X[prop_indices]
            property_public_y = public_y[prop_indices]
            
            
            property_public_datasets = torch.utils.data.TensorDataset(property_public_X,property_public_y)
            non_property_public_datasets = torch.utils.data.TensorDataset(non_property_public_X,non_property_public_y)
            property_public_loader = torch.utils.data.DataLoader(
                        property_public_datasets,
                        batch_size=self._batch_size,
                        num_workers=self._num_workers,
                        shuffle=True,
                        persistent_workers=self._persistent_workers,
                    )
            non_property_public_loader = torch.utils.data.DataLoader(
                        non_property_public_datasets,
                        batch_size=self._batch_size,
                        num_workers=self._num_workers,
                        shuffle=True,
                        persistent_workers=self._persistent_workers,
                    )
            #uncomment
            
            if mode == 0:
                pass
            elif mode == 1:
                Dtest_OH_loader = training_utils.dataframe_to_dataloader(
                    self._Dtest_OH[:240],
                    #self._D_OH[-1],
                    batch_size=1024,
                    num_workers=self._num_workers,
                    using_ce_loss=self._using_ce_loss,
                )
                property_public_loader= Dtest_OH_loader
                
                
            
            shadow_0 = training_utils.get_gaps_torch(
                property_public_loader,
                self._local_models[i][0],self._shadow_models[i][0],
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class
            )
            shadow_1 = training_utils.get_gaps_torch(
                property_public_loader,
                self._local_models[i][0],self._shadow_models[i][1],
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class
            )
            shadow_0_filtered = shadow_0
            shadow_1_filtered = shadow_1
            '''
            vic = training_utils.get_logits_torch(
                property_public_loader,
                self._local_models[i][0],
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class
            )
            shadow_0 = training_utils.get_logits_torch(
                property_public_loader,
                self._shadow_models[i][0],
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class
            )
            shadow_1 = training_utils.get_logits_torch(
                property_public_loader,
                self._shadow_models[i][1],
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class
            )
            
            vic_mean = np.mean(vic)
            vic_filtered = vic[vic > vic_mean - variance_adjustment * vic_mean.std()]
            vic_filtered = vic[vic < vic_mean + variance_adjustment * vic_mean.std()]
            '''
            shadow_0_mean = np.mean(shadow_0)

            shadow_0_filtered = shadow_0[shadow_0 > shadow_0_mean - variance_adjustment * shadow_0_mean.std()]

            shadow_0_filtered = shadow_0[shadow_0 < shadow_0_mean + variance_adjustment * shadow_0_mean.std()]

            
            shadow_1_mean = np.mean(shadow_1)
            
            shadow_1_filtered = shadow_1[shadow_1 > shadow_1_mean - variance_adjustment * shadow_1_mean.std()]
            
            shadow_1_filtered = shadow_1[shadow_1 < shadow_1_mean + variance_adjustment * shadow_1_mean.std()]
            
            self._gaps_shadow[i][0].append(shadow_0_filtered)
            self._gaps_shadow[i][1].append(shadow_1_filtered)
            
    def get_query_loader(self,idx,mode):
        if len(self.query_loader) < 1:
            for i in range(self._num_target_models*2):
                low_confi = False
            
                    
                if low_confi:
                    #mode = 2
                    pass

                results = []
                titles = []
                
                public_X = torch.Tensor([])
                public_y = torch.Tensor([])
                public_confi = torch.Tensor([])
                if mode == 2 :
                    query_dataloader = training_utils.dataframe_to_dataloader(
                        #self._Dtest_OH,
                        self._D_OH[-1],
                        batch_size=1024,
                        num_workers=self._num_workers,
                        using_ce_loss=self._using_ce_loss,
                    )
                else:
                    query_dataloader = self._public_dataloader
                
                #for (inputs, labels) in self._public_dataloader[i]:
                #self.temp_models = self.temp_models.to(self._device)
                
                for (inputs,labels) in query_dataloader:
                    with torch.no_grad():
                        public_X =torch.concat([public_X, inputs.detach().cpu()])
                        public_y = torch.concat([public_y, labels.detach().clone().cpu()])
                        # use low-confi samples
                        if low_confi:
                            alpha = 2
                            inputs = inputs.to(self._device)
                            inputs[:,self._category_index] = 0
                            
                            logits = self._local_models[i][0](inputs)
                            logits_shadow = self._shadow_models[i][0](inputs)
                            logits = torch.nn.functional.softmax(logits)
                            logits_shadow = torch.nn.functional.softmax(logits_shadow)
                            logits = alpha * logits + logits_shadow
                            public_confi = torch.concat([public_confi, (torch.max(logits,dim=1).values-torch.min(logits,dim=1).values).detach().clone().cpu()])
                            
                
                non_property_public_X = public_X[self.non_prop_indices]
                non_property_public_y = public_y[self.non_prop_indices]
                
                property_public_X = public_X[self.prop_indices]
                property_public_y = public_y[self.prop_indices]
                
                
                if low_confi:
                    property_public_confi = public_confi[prop_indices]
                    sort_index = torch.argsort(property_public_confi,descending=False)
                    property_public_X = property_public_X[sort_index[:int(len(sort_index)/20)]]
                    property_public_y = property_public_y[sort_index[:int(len(sort_index)/20)]]
                
                property_public_datasets = torch.utils.data.TensorDataset(property_public_X,property_public_y)
                non_property_public_datasets = torch.utils.data.TensorDataset(non_property_public_X,non_property_public_y)
                property_public_loader = torch.utils.data.DataLoader(
                            property_public_datasets,
                            batch_size=self._batch_size,
                            num_workers=self._num_workers,
                            shuffle=True,
                            persistent_workers=self._persistent_workers,
                        )
                try:
                    non_property_public_loader = torch.utils.data.DataLoader(
                                non_property_public_datasets,
                                batch_size=self._batch_size,
                                num_workers=self._num_workers,
                                shuffle=True,
                                persistent_workers=self._persistent_workers,
                            )
                except:
                    non_property_public_loader = torch.utils.data.DataLoader(
                            property_public_datasets,
                            batch_size=self._batch_size,
                            num_workers=self._num_workers,
                            shuffle=True,
                            persistent_workers=self._persistent_workers,
                        )
                
                
                self.query_loader.append(property_public_loader)
                self.nonprop_query_loader.append(non_property_public_loader)

        return self.query_loader[idx], self.nonprop_query_loader[idx]
    

    def get_confidence_trace(self,mode,current_epoch,save_trace=False,prop=True):
        shadow_poisoned_epoch = 0
        shadow_kd_epoch = 50
        _verbose = True
        standardization = False
        

        results = []
        titles = []
        
        # get threshold and distinguish p distribution
        correct_trials = 0
        total = 0
        Dtest_OH_loader = training_utils.dataframe_to_dataloader(
            #self._Dtest_OH[:240],
            self._D_OH[-1],
            batch_size=1024,
            num_workers=self._num_workers,
            using_ce_loss=self._using_ce_loss,
        )
        mean_list = []
        median_list = []
        vic_list = []
        mean_list.append(self._public_poison)

        mean_list = []

        property_public_loader, non_property_public_loader = self.get_query_loader(0,mode)
        if prop == False:
            property_public_loader = non_property_public_loader
        for i in range(self._num_target_models*2):

            #uncomment
            out_M0 = np.array([])
            out_M1 = np.array([])
            
            # pass
            

            vic_temp = training_utils.get_logits_torch(
                property_public_loader,
                self._local_models[i],
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class,no_filter=self._no_filter,
            )
            

            vic_list.append(np.mean(vic_temp))
            
            if i == 0 or i ==self._num_target_models:
                results.append(vic_temp)
                titles.append(f"VIC {i}")
        
        if current_epoch >= 100 and save_trace:
            vic_list.append(self._public_poison)
            self.vic_stacks.append(vic_list)

        vic0 = [f"{i:.2f}" for i in vic_list[:self._num_target_models]]
        vic1 = [f"{i:.2f}" for i in vic_list[self._num_target_models:self._num_target_models*2]]
        
        predicted_vic0 = []
        for i in range(self._num_target_models):
            b = self.pub_label_dist
            a = self.vic_label_dist[i]


            logits = (1-b)*self._tpub*self._public_poison/((1-a)*self._t0+(1-self._public_poison)*(1-b)*self._tpub)
            #logits = self._public_poison / (((self._t0*(1-a))/(self._tpub*(1-b)))-self._public_poison+1)
            
            predicted_vic0.append(logits)

            # predicted_vic0.append(round(math.log(logits),2))/

        predicted_vic1 = []
        for i in range(self._num_target_models,self._num_target_models*2):
            b = self.pub_label_dist
            a = self.vic_label_dist[i]
                
            
            logits = (1-b)*self._tpub*self._public_poison/((1-a)*self._t1+(1-self._public_poison)*(1-b)*self._tpub)
            
            #logits = self._public_poison / (((self._t1*(1-a))/(self._tpub*(1-b)))-self._public_poison+1)
            predicted_vic1.append(logits)
            # predicted_vic1.append(round(math.log(logits),2))
            
        shadow_list = []
        # predicted_shadow = []
        for i in range(self._num_shadow_models*2):
            
            
            shadow_temp = training_utils.get_logits_torch(
                property_public_loader,
                self._shadow_models[0][i],
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class,no_filter=self._no_filter,
            )
            shadow_list.append(np.mean(shadow_temp))
            
            
            if i == 0 or i ==self._num_target_models:
                results.append(shadow_temp)
                titles.append(f"shadow {i}")
            # b = self.pub_label_dist
            # a = self.vic_label_dist[i]
                
            
            # logits = (1-b)*self._tpub*self._public_poison/((1-a)*self._t1+(1-self._public_poison)*(1-b)*self._tpub)
            
            # #logits = self._public_poison / (((self._t1*(1-a))/(self._tpub*(1-b)))-self._public_poison+1)
            # predicted_shadow.append(round(math.log(logits),2))
        self.plot_and_save_histograms(results, titles,f"./histo/{self._target_attributes}_{current_epoch}_histograms.png")
        shadow_list0 = [f"{i:.2f}" for i in shadow_list[:self._num_shadow_models]]
        shadow_list1 = [f"{i:.2f}" for i in shadow_list[self._num_shadow_models:]]
        vic_list = torch.tensor(vic_list)
        shadow_list = torch.tensor(shadow_list)
        print(f"vic0 :{vic0}","\t\t",f"mean : {torch.mean(vic_list[:self._num_target_models]):.2f}, std :{torch.std(vic_list[:self._num_target_models]):.2f}, dist :{(torch.mean(vic_list[:self._num_target_models])-torch.mean(vic_list[self._num_target_models:]))/(2*torch.std(vic_list[:self._num_target_models])):.2f}, th: {(torch.mean(vic_list[:self._num_target_models])+torch.mean(vic_list[self._num_target_models:]))/2:.2f}")#,f"predicted vic0 : {predicted_vic0}")
        print(f"vic1 :{vic1}","\t\t",f"mean : {torch.mean(vic_list[self._num_target_models:]):.2f}, std :{torch.std(vic_list[self._num_target_models:]):.2f}, dist :{(torch.mean(vic_list[:self._num_target_models])-torch.mean(vic_list[self._num_target_models:]))/(2*torch.std(vic_list[self._num_target_models:])):.2f}, th: {(torch.mean(vic_list[:self._num_target_models])+torch.mean(vic_list[self._num_target_models:]))/2:.2f}")#,f"predicted vic1 : {predicted_vic1}")
        
        
        
        print(f"shadow0 :{shadow_list0}","\t\t",f"mean : {torch.mean(shadow_list[:self._num_shadow_models]):.2f}, std :{torch.std(shadow_list[:self._num_shadow_models]):.2f}, dist :{(torch.mean(shadow_list[:self._num_shadow_models])-torch.mean(shadow_list[self._num_shadow_models:]))/(2*torch.std(shadow_list[:self._num_shadow_models])):.2f}, th: {(torch.mean(shadow_list[:self._num_shadow_models])+torch.mean(shadow_list[self._num_shadow_models:]))/2:.2f}")
        print(f"shadow1 :{shadow_list1}","\t\t",f"mean : {torch.mean(shadow_list[self._num_shadow_models:]):.2f}, std :{torch.std(shadow_list[self._num_shadow_models:]):.2f}, dist :{(torch.mean(shadow_list[:self._num_shadow_models])-torch.mean(shadow_list[self._num_shadow_models:]))/(2*torch.std(shadow_list[self._num_shadow_models:])):.2f}, th: {(torch.mean(shadow_list[:self._num_shadow_models])+torch.mean(shadow_list[self._num_shadow_models:]))/2:.2f}")

        shadow_list = []
        # predicted_shadow = []
        for i in range(self._num_shadow_models*2):
            
            
            shadow_temp = training_utils.get_logits_torch(
                property_public_loader,
                self._shadow_models[1][i],
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class,no_filter=self._no_filter,
            )
            shadow_list.append(np.mean(shadow_temp))
            
            
            if i == 0 or i ==self._num_target_models:
                results.append(shadow_temp)
                titles.append(f"shadow {i}")
            # b = self.pub_label_dist
            # a = self.vic_label_dist[i]
            
        vic_list = torch.tensor(vic_list)
        shadow_list = torch.tensor(shadow_list)
        shadow_list0 = [f"{i:.2f}" for i in shadow_list[:self._num_shadow_models]]
        shadow_list1 = [f"{i:.2f}" for i in shadow_list[self._num_shadow_models:]]
        print(f"shadow0 :{shadow_list0}","\t\t",f"mean : {torch.mean(shadow_list[:self._num_shadow_models]):.2f}, std :{torch.std(shadow_list[:self._num_shadow_models]):.2f}, dist :{(torch.mean(shadow_list[:self._num_shadow_models])-torch.mean(shadow_list[self._num_shadow_models:]))/(2*torch.std(shadow_list[:self._num_shadow_models])):.2f}, th: {(torch.mean(shadow_list[:self._num_shadow_models])+torch.mean(shadow_list[self._num_shadow_models:]))/2:.2f}")
        print(f"shadow1 :{shadow_list1}","\t\t",f"mean : {torch.mean(shadow_list[self._num_shadow_models:]):.2f}, std :{torch.std(shadow_list[self._num_shadow_models:]):.2f}, dist :{(torch.mean(shadow_list[:self._num_shadow_models])-torch.mean(shadow_list[self._num_shadow_models:]))/(2*torch.std(shadow_list[self._num_shadow_models:])):.2f}, th: {(torch.mean(shadow_list[:self._num_shadow_models])+torch.mean(shadow_list[self._num_shadow_models:]))/2:.2f}")
                    

        lr =self._optim_kwargs["lr"]
        predicted_vic0.extend(predicted_vic1)
        estimated_logits = predicted_vic0
        if current_epoch > 99:
            for i in range(self._num_shadow_models*2):
                break
                self.plot_and_save_histograms([vic_list[i]], "vic",f"./histo/{self.dataset}/{self._target_attributes}_{self._t0}_{self._t1}_lr_{lr}_every_{self._adaptive_step}_{self._poison_unit}/{self._target_attributes}_{i}_{current_epoch}_histograms_{self._public_poison:.2f}_vic.png", estimated_logit=estimated_logits[i])
        
            
    def get_confidence_plot(self,mode,current_epoch):
        shadow_poisoned_epoch = 0
        shadow_kd_epoch = 50
        _verbose = True
        standardization = False
        

        
        # get threshold and distinguish p distribution
        correct_trials = 0
        total = 0
        Dtest_OH_loader = training_utils.dataframe_to_dataloader(
            #self._Dtest_OH[:240],
            self._D_OH[-1],
            batch_size=1024,
            num_workers=self._num_workers,
            using_ce_loss=self._using_ce_loss,
        )
        mean_list = []
        median_list = []
        mean_list.append(self._public_poison)
        

        for i in range(self._num_target_models*2):
            low_confi = True
            if low_confi:
                #mode = 2
                pass 
            results = []
            titles = []
            
            public_X = torch.Tensor([])
            public_y = torch.Tensor([])
            public_confi = torch.Tensor([])
            if mode == 2 :
                query_dataloader = training_utils.dataframe_to_dataloader(
                    #self._Dtest_OH[:240],
                    self._D_OH[-1],
                    batch_size=1024,
                    num_workers=self._num_workers,
                    using_ce_loss=self._using_ce_loss,
                )
            else:
                query_dataloader = self._public_dataloader[i]
            #for (inputs, labels) in self._public_dataloader[i]:
            self.temp_model = self.temp_model.to(self._device)
            for (inputs,labels) in query_dataloader:
                with torch.no_grad():
                    public_X =torch.concat([public_X, inputs.detach().cpu()])
                    public_y = torch.concat([public_y, labels.detach().clone().cpu()])
                    
                    # use low-confi samples
                    if low_confi:
                        inputs = inputs.to(self._device)
                        inputs[:,self._category_index] = 0
                        
                        logits = self.temp_model(inputs)
                        logits = torch.nn.functional.softmax(logits)
                        public_confi = torch.concat([public_confi, (torch.max(logits,dim=1).values-torch.min(logits,dim=1).values).detach().clone().cpu()])
                        
            non_prop_mask = torch.ones(public_X.shape[0],dtype=torch.bool)
            mask = torch.ones(public_X.shape[0],dtype=torch.bool)
            for prop_i in self._prop_index:
                non_prop_mask &=(public_X[:,prop_i]!=1)
                mask &= (public_X[:,prop_i]==1)
                
            mask &= (public_y != self._poison_class)
            non_prop_mask &= (public_y != self._poison_class)
            
            prop_indices = torch.where(mask)[0]
            non_prop_indices = torch.where(non_prop_mask)[0]
            
            non_property_public_X = public_X[non_prop_indices]
            non_property_public_y = public_y[non_prop_indices]
            
            property_public_X = public_X[prop_indices]
            property_public_y = public_y[prop_indices]
            
            
            if low_confi:
                property_public_confi = public_confi[prop_indices]
                sort_index = torch.argsort(property_public_confi)
                property_public_X = property_public_X[sort_index[:int(len(sort_index)/10)]]
                property_public_y = property_public_y[sort_index[:int(len(sort_index)/10)]]
                #print(property_public_confi[sort_index[:int(len(sort_index)/20)]])
            property_public_datasets = torch.utils.data.TensorDataset(property_public_X,property_public_y)
            non_property_public_datasets = torch.utils.data.TensorDataset(non_property_public_X,non_property_public_y)
            property_public_loader = torch.utils.data.DataLoader(
                        property_public_datasets,
                        batch_size=self._batch_size,
                        num_workers=self._num_workers,
                        shuffle=True,
                        persistent_workers=self._persistent_workers,
                    )
            non_property_public_loader = torch.utils.data.DataLoader(
                        non_property_public_datasets,
                        batch_size=self._batch_size,
                        num_workers=self._num_workers,
                        shuffle=True,
                        persistent_workers=self._persistent_workers,
                    )
            #uncomment
            out_M0 = np.array([])
            out_M1 = np.array([])

            vic_temp = training_utils.get_logits_torch(
                property_public_loader,
                self._local_models[i][0],
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class
            )
            vic_temp_no_filt = training_utils.get_logits_torch(
                property_public_loader,
                self._local_models[i][0],
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class,no_filter=True
            )
            non_prop_vic_temp = training_utils.get_logits_torch(
                non_property_public_loader,
                self._local_models[i][0],
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class
            )
            #print(vic_temp)
            results.append(vic_temp)
            #results.append(non_prop_vic_temp)
            
            titles.append("VIC prop (low-confi)")
            #titles.append("VIC non prop")
            
            results.append(vic_temp_no_filt)
            #results.append(non_prop_vic_temp)
            
            titles.append("VIC prop (low-confi) no filt")
            mean_list.append(np.mean(vic_temp))
            median_list.append(np.median(vic_temp))
            # get threshold
##################
            #mode = 0
            low_confi = False
            #results = []
            #titles = []
            
            public_X = torch.Tensor([])
            public_y = torch.Tensor([])
            public_confi = torch.Tensor([])
            
            if mode == 2 :
                query_dataloader = training_utils.dataframe_to_dataloader(
                    #self._Dtest_OH[:240],
                    self._D_OH[-1],
                    batch_size=1024,
                    num_workers=self._num_workers,
                    using_ce_loss=self._using_ce_loss,
                )
            else:
                query_dataloader = self._public_dataloader[i]
            #for (inputs, labels) in self._public_dataloader[i]:
            self.temp_model = self.temp_model.to(self._device)
            for (inputs,labels) in query_dataloader:
                with torch.no_grad():
                    public_X =torch.concat([public_X, inputs.detach().cpu()])
                    public_y = torch.concat([public_y, labels.detach().clone().cpu()])
                    
                    # use low-confi samples
                    if low_confi:
                        inputs = inputs.to(self._device)
                        inputs[:,self._category_index] = 0
                        
                        logits = self.temp_model(inputs)
                        logits = torch.nn.functional.softmax(logits)
                        public_confi = torch.concat([public_confi, (torch.max(logits,dim=1).values-torch.min(logits,dim=1).values).detach().clone().cpu()])
                        
            non_prop_mask = torch.ones(public_X.shape[0],dtype=torch.bool)
            mask = torch.ones(public_X.shape[0],dtype=torch.bool)
            for prop_i in self._prop_index:
                non_prop_mask &=(public_X[:,prop_i]!=1)
                mask &= (public_X[:,prop_i]==1)
                
            mask &= (public_y != self._poison_class)
            non_prop_mask &= (public_y != self._poison_class)
            
            prop_indices = torch.where(mask)[0]
            non_prop_indices = torch.where(non_prop_mask)[0]
            
            non_property_public_X = public_X[non_prop_indices]
            non_property_public_y = public_y[non_prop_indices]
            
            property_public_X = public_X[prop_indices]
            property_public_y = public_y[prop_indices]
            
            
            if low_confi:
                property_public_confi = public_confi[prop_indices]
                sort_index = torch.argsort(property_public_confi)
                property_public_X = property_public_X[sort_index[:int(len(sort_index)/10)]]
                property_public_y = property_public_y[sort_index[:int(len(sort_index)/10)]]
                #print(property_public_confi[sort_index[:int(len(sort_index)/20)]])
            property_public_datasets = torch.utils.data.TensorDataset(property_public_X,property_public_y)
            non_property_public_datasets = torch.utils.data.TensorDataset(non_property_public_X,non_property_public_y)
            property_public_loader = torch.utils.data.DataLoader(
                        property_public_datasets,
                        batch_size=self._batch_size,
                        num_workers=self._num_workers,
                        shuffle=True,
                        persistent_workers=self._persistent_workers,
                    )
            non_property_public_loader = torch.utils.data.DataLoader(
                        non_property_public_datasets,
                        batch_size=self._batch_size,
                        num_workers=self._num_workers,
                        shuffle=True,
                        persistent_workers=self._persistent_workers,
                    )
            #uncomment
            out_M0 = np.array([])
            out_M1 = np.array([])

            vic_temp = training_utils.get_logits_torch(
                property_public_loader,
                self._local_models[i][0],
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class
            )
            vic_temp_no_filt = training_utils.get_logits_torch(
                property_public_loader,
                self._local_models[i][0],
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class,no_filter=True
            )
            non_prop_vic_temp = training_utils.get_logits_torch(
                non_property_public_loader,
                self._local_models[i][0],
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class
            )
            #print(vic_temp)
            results.append(vic_temp)
            #results.append(non_prop_vic_temp)
            
            titles.append("VIC prop (all)")
            #titles.append("VIC non prop")
            
            results.append(vic_temp_no_filt)
            #results.append(non_prop_vic_temp)
            
            titles.append("VIC prop (all) no filter")
            #titles.append("VIC non prop")
            #mean_list.append(np.mean(vic_temp))
            #median_list.append(np.median(vic_temp))
            # get threshold

##################
            out_M0 = np.array([])
            out_M1 = np.array([]) 
            for s_i in range(self._num_shadow_models):
                
                out_M0_temp = training_utils.get_logits_torch(
                    property_public_loader,
                    self._shadow_models[i][s_i],
                    device=self._device,
                    middle_measure=self._middle,
                    variance_adjustment=self._variance_adjustment,
                    label = self._poison_class
                )
                out_M0_orig = out_M0_temp
                non_prop_out_M0_temp = training_utils.get_logits_torch(
                    non_property_public_loader,
                    self._shadow_models[i][s_i],
                    device=self._device,
                    middle_measure=self._middle,
                    variance_adjustment=self._variance_adjustment,
                    label = self._poison_class
                )
                if standardization:
                    out_M0_temp = (out_M0_temp - np.mean(non_prop_out_M0_temp)) / np.std(non_prop_out_M0_temp) 
                out_M0 = np.append(out_M0, out_M0_temp)
                if _verbose:
                    #print(f"M0 Median: {np.median(out_M0_temp):.3}")
                    pass
                
                results.append(out_M0_temp)
                #results.append(non_prop_out_M0_temp)
                titles.append(f"Shadow M0 prop_{s_i}")
                #titles.append(f"Shadow M0 non prop_{s_i}")
                mean_list.append(np.mean(out_M0_temp))
                median_list.append(np.median(out_M0_temp))
                

            for s_i in range(self._num_shadow_models,self._num_shadow_models*2):
                out_M1_temp = training_utils.get_logits_torch(
                    property_public_loader,
                    self._shadow_models[i][s_i],
                    device=self._device,
                    middle_measure=self._middle,
                    variance_adjustment=self._variance_adjustment,
                    label = self._poison_class
                )
                out_M1_orig = out_M1_temp
                non_prop_out_M1_temp = training_utils.get_logits_torch(
                    non_property_public_loader,
                    self._shadow_models[i][s_i],
                    device=self._device,
                    middle_measure=self._middle,
                    variance_adjustment=self._variance_adjustment,
                    label = self._poison_class
                )
                if standardization:
                    out_M1_temp = (out_M1_temp - np.mean(non_prop_out_M1_temp)) / np.std(non_prop_out_M1_temp) 
                    
                    
                mean_list.append(np.mean(out_M1_temp))
                median_list.append(np.median(out_M1_temp))
                out_M1 = np.append(out_M1, out_M1_temp)
                if _verbose:
                    #print(f"M1 Median: {np.median(out_M1_temp):.3}"  )
                    pass
                results.append(out_M1_temp)
                #results.append(non_prop_out_M1_temp)
                titles.append(f"Shadow M1 prop_{s_i}")
                #titles.append(f"Shadow M1 non prop_{s_i}")
            #self.plot_and_save_histograms(results[:2], titles[:2],f"./{i}_histograms_{self._target_attributes}_vic.png")
            #self.plot_and_save_histograms(results[2:4], titles[2:4],f"./{i}_histograms_{self._target_attributes}_shadow_m0.png")
            #self.plot_and_save_histograms(results[4:6], titles[4:6],f"./{i}_histograms_{self._target_attributes}_shadow_m1.png")
            
            #self.plot_and_save_histograms(results, titles,f"./histo/{self._target_attributes}_{i}_{current_epoch}_histograms_{self._public_poison}.png")
            self.mean_stacks[i].append(mean_list)
            self.median_stacks[i].append(median_list)
    def print_mean_csv(self):
        directory = "./csv_arxiv/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for i in range(1):
            
            csv_name = f"./csv_arxiv/{self._target_attributes}_{i}_histograms_round.csv"

            
            f = open(csv_name,"w")
            writer = csv.writer(f)
            for j in range(len(self.vic_stacks)):
                try:
                    writer.writerows([self.vic_stacks[j]])
                except:
                    print(self.vic_stacks)
                    print(self.vic_stacks[i])
                    print(type(self.vic_stacks))
                    writer.writerows(self.vic_stacks[i])
                    exit(1)
                
            f.close()

    def our_pia(self):
        
        
        check_list = [False for i in range(self._num_target_models*2)]
        pp_list = [-1 for i in range(self._num_target_models*2)]
        
        print(check_list)
        for round in range(len(self.vic_stacks)):
            for i in range(self._num_target_models*2):
                
                if self.vic_stacks[round][i]<0.10 and check_list[i] == False:
                    check_list[i] = True
                    pp_list[i] = self.vic_stacks[round][-1]
                    
        
        
        predicted_list = []
        for idx, x in enumerate(pp_list):
            # D_subpop = data.generate_subpopulation_OH(
            # self.vic_dataset[idx], categories=self._categories, target_attributes=self._target_attributes
            # )

            # D_subpop_pub = data.generate_subpopulation_OH(
            # self.pub_dataset[idx], categories=self._categories, target_attributes=self._target_attributes
            # )

            # b = sum(D_subpop_pub[D_subpop_pub.columns[-1]]) / len(D_subpop_pub)
            # a = sum(D_subpop[D_subpop.columns[-1]]) / len(D_subpop)
            b = self.pub_label_dist
            a = self.vic_label_dist
                
            
            predicted_list.append((2*x-1) * self._tpub * (1-b) / (1-a))
            #x = (1-b)*x + b
            #predicted_list.append((2*x-1)*self._t1*1.5 / (1-2*a))                                              
            #print(x,a,self._t1)                                                                                
            #predicted_list.append((2*x-1)*self._t1*1.5 )
            #predicted_list.append(x / ((1-a)*(1-x)))


            #predicted_list.append((2*x-1)*self._t1*1.5)
        print("#####################")
        ep0 = [f"{i:.2f}" for i in predicted_list[:self._num_target_models]]
        ep1 = [f"{i:.2f}" for i in predicted_list[self._num_target_models:self._num_target_models*2]]
        pr0 = [f"{i:.2f}" for i in pp_list[:self._num_target_models]]
        pr1 = [f"{i:.2f}" for i in pp_list[self._num_target_models:self._num_target_models*2]]
        print(f"estimated property 0: {ep0}")
        print(f"estimated property 1: {ep1}")
        print(f"poisoning ratio 0: {pr0}")
        print(f"poisoning ratio 1: {pr1}")
        correct = 0
        total = 0
        
        for i in range(self._num_target_models):
            if predicted_list[i] < ((self._t0+self._t1)/2):
                correct +=1
            total +=1
            
        for i in range(self._num_target_models,self._num_target_models*2):
            if predicted_list[i] >= ((self._t0+self._t1)/2):
                correct +=1
            total +=1    
            
        print(f"our pia acc : {correct/total:.4f}")
    def property_inference(self, distinguishing_test="mean",query_trials = 10,mode=0):
        shadow_poisoned_epoch = 0
        shadow_kd_epoch = 50
        _verbose = True
        standardization = False


        # get threshold and distinguish p distribution
        correct_trials = 0
        total = 0
        Dtest_OH_loader = training_utils.dataframe_to_dataloader(
            #self._Dtest_OH[:240],
            self._D_OH[-1],
            batch_size=1024,
            num_workers=self._num_workers,
            using_ce_loss=self._using_ce_loss,
        )
        property_public_loader, non_prop_public_loader = self.get_query_loader(0,mode)
        for i in range(self._num_target_models*2):
            
            


            results = []
            titles = []
            
            
            vic_temp = training_utils.get_logits_torch(
                property_public_loader,
                self._local_models[i],
                device=self._device,
                middle_measure=self._middle,
                variance_adjustment=self._variance_adjustment,
                label = self._poison_class,no_filter=self._no_filter
            )


            

            if _verbose:
                print(f"VIC Median {i}: {np.mean(vic_temp):.3},{np.std(vic_temp):.3}")
            results.append(vic_temp)
            #results.append(non_prop_vic_temp)
            titles.append("VIC prop")
            #titles.append("VIC non prop")

            out_M0 = np.array([])
            out_M1 = np.array([])
            if i < self._num_target_models:
                t_i = 0
            else:
                t_i = 1
            for s_i in range(self._num_shadow_models):
                
                out_M0_temp = training_utils.get_logits_torch(
                    property_public_loader,
                    self._shadow_models[t_i][s_i],
                    device=self._device,
                    middle_measure=self._middle,
                    variance_adjustment=self._variance_adjustment,
                    label = self._poison_class,no_filter=self._no_filter
                )
                out_M0_orig = out_M0_temp

                if standardization:
                    out_M0_temp = (out_M0_temp - np.mean(non_prop_out_M0_temp)) / np.std(non_prop_out_M0_temp) 
                out_M0 = np.append(out_M0, out_M0_temp)
                if _verbose:
                    #print(f"M0 Median: {np.median(out_M0_temp):.3}")
                    pass
                
                results.append(out_M0_temp)
                #results.append(non_prop_out_M0_temp)
                titles.append(f"Shadow M0 prop_{s_i}")
                #titles.append(f"Shadow M0 non prop_{s_i}")
            for s_i in range(self._num_shadow_models,self._num_shadow_models*2):
                out_M1_temp = training_utils.get_logits_torch(
                    property_public_loader,
                    self._shadow_models[t_i][s_i],
                    device=self._device,
                    middle_measure=self._middle,
                    variance_adjustment=self._variance_adjustment,
                    label = self._poison_class,no_filter=self._no_filter
                )
                out_M1_orig = out_M1_temp

   
                    
                out_M1 = np.append(out_M1, out_M1_temp)
                if _verbose:
                    #print(f"M1 Median: {np.median(out_M1_temp):.3}"  )
                    pass
                results.append(out_M1_temp)
                #results.append(non_prop_out_M1_temp)
                titles.append(f"Shadow M1 prop_{s_i}")
                #titles.append(f"Shadow M1 non prop_{s_i}")
            #self.plot_and_save_histograms(results[:2], titles[:2],f"./{i}_histograms_{self._target_attributes}_vic.png")
            #self.plot_and_save_histograms(results[2:4], titles[2:4],f"./{i}_histograms_{self._target_attributes}_shadow_m0.png")
            #self.plot_and_save_histograms(results[4:6], titles[4:6],f"./{i}_histograms_{self._target_attributes}_shadow_m1.png")
            #self.plot_and_save_histograms(results, titles,f"./{i}_histograms_{self._target_attributes}.png")
            if _verbose:

                print(f"M0 Median {t_i} {i}: {np.mean(out_M0):.3},{np.std(out_M0):.3}")
                #print(f"non prop M0 Median {i}: {np.median(non_prop_out_M0_temp):.3},{np.mean(non_prop_out_M0_temp):.3},{np.std(non_prop_out_M0_temp):.3}")
                
                print(f"M1 Median {i}: {np.mean(out_M1):.3},{np.std(out_M1):.3}")
                #print(f"non prop M1 Median {i}: {np.median(non_prop_out_M1_temp):.3},{np.mean(non_prop_out_M1_temp):.3},{np.std(non_prop_out_M1_temp):.3}")

        
            if distinguishing_test == "median":
                midpoint_of_medians = (np.median(out_M0) + np.median(out_M1)) / 2
                thresh = midpoint_of_medians
            elif distinguishing_test == "mean":
                midpoint_of_medians = (np.mean(out_M0) + np.mean(out_M1)) / 2
                thresh = midpoint_of_medians        
  
            if self._num_queries > self._Dtest_OH.shape[0]:
                oversample_flag = True
                if _verbose:
                    print("Oversampling test queries")
            else:
                oversample_flag = False
            #self._poisoned_student_model_for_victim = self._poisoned_student_models[0]
            for query_index in range(1):

                if True:
                    Dtest_OH_sample_loader = training_utils.dataframe_to_dataloader(
                        self._Dtest_OH.sample(
                            n=self._num_queries, replace=oversample_flag, random_state = i+1
                        ),
                        batch_size=self._batch_size,
                        num_workers=self._num_workers,
                        using_ce_loss=self._using_ce_loss,
                    )
                else:
                    if _verbose:
                        print("Incorrect Query selection type")


                M0_score = len(vic_temp[vic_temp > thresh])
                M1_score = len(vic_temp[vic_temp < thresh])
                
                if M0_score >= M1_score:
                    if _verbose:
                        try:
                            print(
                                f"Target is in t0 world with {M0_score/len(vic_temp)*100:.4}% confidence"
                            )
                        except:
                            print(vic_temp)
                            print(M0_score)

                    correct_trials = correct_trials + int(
                        self._target_worlds[i] == 0
                    )

                elif M0_score < M1_score:
                    if _verbose:
                        print(
                            f"Target is in t1 world {M1_score/len(vic_temp)*100:.4}% confidence"
                        )

                    correct_trials = correct_trials + int(
                        self._target_worlds[i] == 1
                    ) 
                total += 1
            if _verbose:
                print(f"total : {total}, correct : {correct_trials} acc : {correct_trials/total:.4f}")
                
        print(f"total : {total}, correct : {correct_trials} acc : {correct_trials/total:.4f}")
        return correct_trials/total

    def make_poison_canditate(self):
        # self.poison_idices = [[] for i in range(self._num_target_models*2)]
        self.poison_idices = []
        
        X = torch.Tensor([])
        labels_list = torch.Tensor([])
        for (inputs, labels) in self._public_dataloader:
            labels_list = torch.concat([labels_list, labels.detach().clone().cpu()])
            with torch.no_grad():
                X = torch.concat([X, inputs.detach().cpu()])

        one_mask = torch.ones(X.shape[0],dtype=torch.bool)
        mask = torch.ones(X.shape[0],dtype=torch.bool)
        non_prop_mask = torch.ones(X.shape[0],dtype=torch.bool)
        for prop_i in self._prop_index:
            mask &= (X[:,prop_i]==1)
            non_prop_mask &= (X[:,prop_i]!=1)
            pass
        mask &= (labels_list != self._poison_class)
        #non_prop_mask &= (labels_list == self._poison_class)
        
        
        self.prop_indices = torch.where(mask)[0]  
        self.non_prop_indices = torch.where(~mask)[0] 
        self.poison_indices = list(range(self.prop_indices.shape[0]))
        self.selected_idx = torch.randperm(self.prop_indices.shape[0])

        
    def sgd_step(params, grad, opt_state, opt_params):
        '''Performs a standard step of SGD that is differentiable in the labels'''
        weight_decay = opt_params['weight_decay']
        momentum = opt_params['momentum']
        dampening = opt_params['dampening']
        nesterov = opt_params['nesterov']

        d_p = grad
        if weight_decay != 0:
            d_p = d_p.add(params, alpha=weight_decay)
        if momentum != 0:
            if 'momentum_buffer' not in opt_state:
                buf = opt_state['momentum_buffer'] = torch.zeros_like(params.data)
                buf = buf.mul(momentum).add(d_p)
            else:
                buf = opt_state['momentum_buffer']
                buf = buf.mul(momentum).add(d_p, alpha=1 - dampening)
            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        return params.add(d_p, alpha=-opt_params['lr'])

    def adam_step(self, params, grad, opt_state, opt_params):
        '''Performs a standard step of Adam optimizer that is differentiable in the labels'''
        lr = opt_params['lr']
        betas = opt_params['betas']  # Tuple of (beta1, beta2)
        eps = opt_params['eps']
        weight_decay = opt_params['weight_decay']
        amsgrad = opt_params.get('amsgrad', False)

        beta1, beta2 = betas

        # Initialize state variables if not already present
        if 'step' not in opt_state:
            opt_state['step'] = 0
        if 'exp_avg' not in opt_state:
            opt_state['exp_avg'] = torch.zeros_like(params)
        if 'exp_avg_sq' not in opt_state:
            opt_state['exp_avg_sq'] = torch.zeros_like(params)
        if amsgrad and 'max_exp_avg_sq' not in opt_state:
            opt_state['max_exp_avg_sq'] = torch.zeros_like(params)

        # Detach state variables to prevent accumulation of computation graphs
        exp_avg = opt_state['exp_avg'].detach()
        exp_avg_sq = opt_state['exp_avg_sq'].detach()
        opt_state['step'] += 1
        step = opt_state['step']

        if weight_decay != 0:
            grad = grad + params * weight_decay

        # Update biased first moment estimate
        exp_avg = exp_avg * beta1 + grad * (1 - beta1)
        # Update biased second raw moment estimate
        exp_avg_sq = exp_avg_sq * beta2 + grad.pow(2) * (1 - beta2)

        # Save updated state after detaching
        opt_state['exp_avg'] = exp_avg.detach()
        opt_state['exp_avg_sq'] = exp_avg_sq.detach()

        if amsgrad:
            max_exp_avg_sq = opt_state['max_exp_avg_sq'].detach()
            max_exp_avg_sq = torch.max(max_exp_avg_sq, exp_avg_sq)
            opt_state['max_exp_avg_sq'] = max_exp_avg_sq.detach()
            denom = max_exp_avg_sq.sqrt() + eps
        else:
            denom = exp_avg_sq.sqrt() + eps

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

        params = params - step_size * exp_avg / denom
        return params
    def generate_data_loader_with_all(self):
        # 데이터로더 생성
        self._D_pub_dataloader = training_utils.dataframe_to_dataloader(
            self._D_pub_OH,
            batch_size=self._batch_size,
            using_ce_loss=self._using_ce_loss,
            num_workers=self._num_workers,
            persistent_workers=self._persistent_workers,
            pin_memory=False,
            shuffle=self._shuffle,
        )

        D_public_X = torch.Tensor([])  # 데이터셋 입력
        D_public_y = torch.Tensor([])  # 레이블
        D_public_scores = torch.Tensor([])  # 신뢰도 점수

        priv_model = self.temp_models[0].to(self._device)  # 프라이빗 모델
        pub_model = self.temp_models[1].to(self._device)  # 퍼블릭 모델

        # 신뢰도 점수 계산
        for (inputs, labels) in self._D_pub_dataloader:
            with torch.no_grad():
                # 입력과 레이블 데이터 축적
                D_public_X = torch.concat([D_public_X, inputs.detach().cpu()])
                D_public_y = torch.concat([D_public_y, labels.long().detach().clone().cpu()])

        # 데이터 분할 (레이블과 속성에 따라)
        mask_label = D_public_y == 1  # 레이블이 있는 데이터
        mask_property = torch.ones(D_public_X.shape[0], dtype=torch.bool)
        for prop_i in self._prop_index:
            mask_property &= (D_public_X[:, prop_i] == 1)
        
        indices_label_and_property = torch.where(mask_label & mask_property)[0]
        indices_label_and_no_property = torch.where(mask_label & ~mask_property)[0]
        indices_no_label_and_property = torch.where(~mask_label & mask_property)[0]
        indices_no_label_and_no_property = torch.where(~mask_label & ~mask_property)[0]


        # 속성 정보 추가
        selected_property = mask_property.long().unsqueeze(-1)
        combined_y = torch.concat([D_public_y.unsqueeze(-1), selected_property], dim=1)

        # PyTorch Dataset 및 DataLoader 생성
        selected_dataset = torch.utils.data.TensorDataset(D_public_X, combined_y)
        self.generator_dataloader = torch.utils.data.DataLoader(
            selected_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            persistent_workers=self._persistent_workers,
            pin_memory=True
        )
        return
    def generate_data_loader_with_scores(self,selection_ratio=0.2):
        # 데이터로더 생성
        self._D_pub_dataloader = training_utils.dataframe_to_dataloader(
            self._D_pub_OH,
            batch_size=self._batch_size,
            using_ce_loss=self._using_ce_loss,
            num_workers=self._num_workers,
            persistent_workers=self._persistent_workers,
            pin_memory=False,
            shuffle=self._shuffle,
        )

        D_public_X = torch.Tensor([])  # 데이터셋 입력
        D_public_y = torch.Tensor([])  # 레이블
        D_public_scores = torch.Tensor([])  # 신뢰도 점수

        priv_model = self.temp_models[0].to(self._device)  # 프라이빗 모델
        pub_model = self.temp_models[1].to(self._device)  # 퍼블릭 모델

        # 신뢰도 점수 계산
        for (inputs, labels) in self._D_pub_dataloader:
            with torch.no_grad():
                # 입력과 레이블 데이터 축적
                D_public_X = torch.concat([D_public_X, inputs.detach().cpu()])
                D_public_y = torch.concat([D_public_y, labels.long().detach().clone().cpu()])
   
                # 모델의 로짓(logits) 및 확률 계산
                priv_logits = F.softmax(priv_model(inputs.to(self._device)), dim=1)
                pub_logits = F.softmax(pub_model(inputs.to(self._device)), dim=1)

                # 점수 계산: 프라이빗 점수 + 퍼블릭 엔트로피
                priv_score = (torch.max(priv_logits, dim=1).values - torch.min(priv_logits, dim=1).values)
                pub_entropy = -torch.sum(pub_logits * torch.log(pub_logits + 1e-10), dim=1)
                score = priv_score #+ 0.2 * pub_entropy

                # 점수 축적
                D_public_scores = torch.concat([D_public_scores, score.detach().clone().cpu()])

        # 데이터 분할 (레이블과 속성에 따라)
        mask_label = D_public_y == 1  # 레이블이 있는 데이터
        mask_property = torch.ones(D_public_X.shape[0], dtype=torch.bool)
        for prop_i in self._prop_index:
            mask_property &= (D_public_X[:, prop_i] == 1)

        def calculate_label_distribution(labels, mask):
            subset_labels = labels[mask].long()
            label_dist = torch.bincount(subset_labels, minlength=int(labels.max() + 1)).float()
            label_dist = label_dist / label_dist.sum() if label_dist.sum() > 0 else torch.zeros_like(label_dist)
            label_dist = {i: ratio.item() for i, ratio in enumerate(label_dist)}
            return label_dist

        self._label_dist_pub_prop = calculate_label_distribution(D_public_y, mask_property)
        self._label_dist_pub_no_prop = calculate_label_distribution(D_public_y, ~mask_property)

        indices_label_and_property = torch.where(mask_label & mask_property)[0]
        indices_label_and_no_property = torch.where(mask_label & ~mask_property)[0]
        indices_no_label_and_property = torch.where(~mask_label & mask_property)[0]
        indices_no_label_and_no_property = torch.where(~mask_label & ~mask_property)[0]

        # 각 그룹의 샘플 수 계산
        total_samples = D_public_X.shape[0]
        count_label_and_property = len(indices_label_and_property)
        count_label_and_no_property = len(indices_label_and_no_property)
        count_no_label_and_property = len(indices_no_label_and_property)
        count_no_label_and_no_property = len(indices_no_label_and_no_property)

        # 전체 샘플 수에 대한 각 그룹의 비율 계산
        total_count = total_samples  # 또는 count_label_and_property + count_label_and_no_property + count_no_label_and_property + count_no_label_and_no_property

        ratio_label_and_property = count_label_and_property / total_count
        ratio_label_and_no_property = count_label_and_no_property / total_count
        ratio_no_label_and_property = count_no_label_and_property / total_count
        ratio_no_label_and_no_property = count_no_label_and_no_property / total_count

        # 선택할 전체 샘플 수 설정 (예: 전체 데이터 사용)
        selection_ratio = selection_ratio  # 필요에 따라 조정하세요 (1.0이면 전체 데이터 사용)
        selection_total = int(total_samples * selection_ratio)

        # 각 그룹에서 선택할 샘플 수 계산
        select_count_label_and_property = int(selection_total * ratio_label_and_property)
        select_count_label_and_no_property = int(selection_total * ratio_label_and_no_property)
        select_count_no_label_and_property = int(selection_total * ratio_no_label_and_property)
        # 마지막 그룹의 샘플 수는 나머지를 사용하여 계산
        select_count_no_label_and_no_property = selection_total - (
            select_count_label_and_property + select_count_label_and_no_property + select_count_no_label_and_property
        )

        # 상위 k개 선택 함수
        def select_top_k(indices, scores, k):
            if len(indices) == 0 or k == 0:
                return torch.tensor([], dtype=torch.long)  # 인덱스가 없는 경우
            sorted_indices = indices[torch.argsort(scores[indices], descending=True)]
            return sorted_indices[:k]  # 상위 k개 반환

        # 각 그룹에서 계산된 샘플 수만큼 선택
        top_indices_label_and_property = select_top_k(indices_label_and_property, D_public_scores, select_count_label_and_property)
        top_indices_label_and_no_property = select_top_k(indices_label_and_no_property, D_public_scores, select_count_label_and_no_property)
        top_indices_no_label_and_property = select_top_k(indices_no_label_and_property, D_public_scores, select_count_no_label_and_property)
        top_indices_no_label_and_no_property = select_top_k(indices_no_label_and_no_property, D_public_scores, select_count_no_label_and_no_property)

        # 선택된 데이터 결합
        final_indices = torch.concat([
            top_indices_label_and_property,
            top_indices_label_and_no_property,
            top_indices_no_label_and_property,
            top_indices_no_label_and_no_property
        ]).long()

        selected_X = D_public_X[final_indices]
        selected_y = D_public_y[final_indices]

        # 속성 정보 추가
        selected_property = mask_property[final_indices].long().unsqueeze(-1)
        combined_y = torch.concat([selected_y.unsqueeze(-1), selected_property], dim=1)

        # PyTorch Dataset 및 DataLoader 생성
        selected_dataset = torch.utils.data.TensorDataset(selected_X, combined_y)
        self.generator_dataloader = torch.utils.data.DataLoader(
            selected_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            persistent_workers=self._persistent_workers,
            pin_memory=True
        )

        # 선택된 데이터의 인덱스 계산
        start = list(self._D_pub_OH.index)[0]
        temp = [ff + start for ff in final_indices.tolist()]
        extracted_df = self._D_pub_OH.loc[temp]

        # 원핫 인코딩을 원래 데이터로 복원
        self.tabsyn_data = self.data_transformer.inverse_one_hot_encoded_dataframe(extracted_df)

        return
    
    
    def get_dataloader_for_tabsyn(self):
        # wanted_indices = self._D_pub_OH.index
        # extracted_df = self._D_public_all.loc[wanted_indices]
        extracted_df = self._D_priv.copy()

        target_columns = ['class']
        cat_columns, cont_columns = data.get_adult_columns()
        column_index = {name: idx for idx, name in enumerate(extracted_df.columns)}

        num_col_idx = [column_index[col] for col in cont_columns]
        cat_col_idx = [column_index[col] for col in cat_columns]
        target_col_idx = [column_index[col] for col in ['class']]

        X_num_train = extracted_df[cont_columns].to_numpy().astype(np.float32)
        X_cat_train = extracted_df[cat_columns].to_numpy()
        y_train = extracted_df[target_columns].to_numpy()
        X_cat_train = concat_y_to_X(X_cat_train, y_train)

        T_dict = {}
        T_dict['normalization'] = "quantile"
        T_dict['num_nan_policy'] = 'mean'
        T_dict['cat_nan_policy'] =  None
        T_dict['cat_min_frequency'] = None
        T_dict['cat_encoding'] = None
        T_dict['y_policy'] = "default"
        T = src.Transformations(**T_dict)
        
        X_cat = {'train': X_cat_train}
        X_num = {'train': X_num_train}
        y = {'train': y_train}

        D = src.Dataset(
            X_num,
            X_cat,
            y,
            y_info={},
            task_type=src.TaskType('binclass'),
            n_classes=None
        )
        dataset = src.transform_dataset(D, T, None)

        categories = src.get_categories(X_cat_train)
        d_numerical = X_num_train.shape[1]

        X_num_train = torch.tensor(X_num_train).float()
        X_cat_train = torch.tensor(X_cat_train)

        train_data = TabularDataset(X_num_train, X_cat_train)
            
        batch_size = 4096
        train_loader = DataLoader(
                train_data,
                batch_size = batch_size,
                shuffle = True,
                num_workers = 4,
            )
    def tabsyn_compute_loss(self, X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
        ce_loss_fn = nn.CrossEntropyLoss()
        
        mse_loss = (X_num - Recon_X_num).pow(2).mean()
        ce_loss = 0
        acc = 0
        total_num = 0

        for idx, x_cat in enumerate(Recon_X_cat):
            if x_cat is not None:
                ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
                x_hat = x_cat.argmax(dim = -1)
            acc += (x_hat == X_cat[:,idx]).float().sum()
            total_num += x_hat.shape[0]
        
        ce_loss /= (idx + 1)
        acc /= total_num
        # loss = mse_loss + ce_loss

        temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

        loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
        return mse_loss, ce_loss, loss_kld, acc
    

    def train_shadow_with_history(self,epochs=0):
        
        for epoch in range(epochs-1):
            self.train_shadow()

            for t_i in range(2):
                for s_i in range(self._num_shadow_models*2):
                    training_utils.fit_kd_local(                    dataloaders=self.public_history[epoch],                   model=self._shadow_models[t_i][s_i],#alterdata_list=self._alterdata_list,     
                            epochs=1, optim_init=self._shadow_optimizers[t_i][s_i],                    optim_kwargs=self._optim_kd_kwargs,                    \
                            criterion=nn.KLDivLoss(reduction="batchmean"),
                            #criterion=nn.L1Loss(),
                            device=self._device,                    verbose=self._verbose,                    mini_verbose=self._mini_verbose,                    early_stopping=self._early_stopping,                    tol=self._tol,                    train_only=True, )

                    pass
    
    
    
    def set_shadow_data_from_tabsyn(self, root_path='./experiments',priv=False, pub_ent_w=None,world=0,world_split=False,n_trial=0):
        
        if pub_ent_w is not None:
            gen_train_0 = pd.read_csv(f'./syn_csv/tabsyn_table_priv_{priv}_{self.dataset}_{self._categories}_{self._target_attributes}_{pub_ent_w}_{n_trial}_0.csv')
            if world_split:
                gen_train_1 = pd.read_csv(f'./syn_csv/tabsyn_table_priv_{priv}_{self.dataset}_{self._categories}_{self._target_attributes}_{pub_ent_w}_{n_trial}_1.csv')
            else:
                gen_train_1 = pd.read_csv(f'./syn_csv/tabsyn_table_priv_{priv}_{self.dataset}_{self._categories}_{self._target_attributes}_{pub_ent_w}_{n_trial}_0.csv')
        else:
            gen_train = pd.read_csv(f'./syn_csv/tabsyn_table_priv_{priv}_{self.dataset}_{self._categories}_{self._target_attributes}.csv')
        gen_train_0 = gen_train_0.reset_index(drop=True)
        gen_train_1 = gen_train_1.reset_index(drop=True)
        # print(gen_train)
        # print(gen_train.loc['race'])
        # _, gen_train = data.split_data_ages(gen_train,40)
        # _, gen_train = data.split_data_ages(gen_train,0)
        self._tabsyn_pub_OH_0 = self.data_transformer.get_one_hot_encoded_dataframe(gen_train_0)
        self._tabsyn_pub_OH_1 = self.data_transformer.get_one_hot_encoded_dataframe(gen_train_1)
        self._shadow_datasets = []
        self._shadow_dataloaders = []
        temp_tabsyn_pub_OH = []
        temp_tabsyn_pub_OH.append(self._tabsyn_pub_OH_0)
        temp_tabsyn_pub_OH.append(self._tabsyn_pub_OH_1)
        for t_i in range(2):
            temp_shadow_dataloaders = []
            temp_shadow_datasets = []
            for p_i in range(self._num_shadow_models*2): 
                target_ratio = self._t0 if p_i < self._num_shadow_models else self._t1
                sampled_shadow_D0 = data.v2_fix_imbalance_OH(
                    temp_tabsyn_pub_OH[t_i],
                    target_split=target_ratio,
                    categories=self._categories,
                    target_attributes=self._target_attributes,
                    random_seed=p_i,
                    total_samples = self._ntarget_samples
                )	        
                sampled_shadow_D0_subpop = data.generate_subpopulation_OH(
                    sampled_shadow_D0, categories=self._categories, target_attributes=self._target_attributes )
                if p_i < self._num_shadow_models:
                    self.label_split_d0 = sum(sampled_shadow_D0[sampled_shadow_D0.columns[-1]]) / len(sampled_shadow_D0)
                else:
                    self.label_split_d0 = sum(sampled_shadow_D0[sampled_shadow_D0.columns[-1]]) / len(sampled_shadow_D0)
                label_split_d0_subpop = sum(sampled_shadow_D0_subpop[sampled_shadow_D0_subpop.columns[-1]]) / len(sampled_shadow_D0_subpop)
                print(
                    f"shadow{p_i} D{0 if p_i < self._num_shadow_models else 1} has {len(sampled_shadow_D0)} points with {len(sampled_shadow_D0_subpop)},{len(sampled_shadow_D0_subpop)/len(sampled_shadow_D0):.2f} members from the target subpopulation and {self.label_split_d0*100:.4}%, {label_split_d0_subpop*100:.4}% class 1"
                    )
                temp_shadow_dataloaders.append(training_utils.dataframe_to_dataloader(
                            sampled_shadow_D0,
                            batch_size=self._batch_size,
                            using_ce_loss=self._using_ce_loss,
                            num_workers=self._num_workers,
                            persistent_workers=self._persistent_workers,
                            pin_memory=True,
                            shuffle=self._shuffle,
                        ))
                temp_shadow_datasets.append(sampled_shadow_D0)
            self._shadow_dataloaders.append(temp_shadow_dataloaders)
            self._shadow_datasets.append(temp_shadow_datasets)
            

            
    def make_cnt_cat_tar_data(self, tar_df, info):
        tar_df.rename(columns = info['idx_name_mapping'], inplace=True)
        X_num_refer = tar_df[info['cont_columns']].to_numpy().astype(np.float32)
        X_cat_refer = tar_df[info['cat_columns']].to_numpy()
        y_refer = tar_df[info['target_columns']].to_numpy()
        X_cat_refer = concat_y_to_X(X_cat_refer, y_refer)

        return X_num_refer, X_cat_refer, y_refer
    
    def origin_df_file_name(self, priv=False, fine_tune=False, all_data=False):
        if priv :
            ori_file_name = f'./csv_arxiv/{self.dataset}_origin_priv_True.csv'
        else :
            if fine_tune:
                ori_file_name = f'./csv_arxiv/{self.dataset}_origin_priv_False_sampled.csv'
            elif all_data:
                ori_file_name = f'./csv_arxiv/{self.dataset}_origin_all.csv'
            else:
                ori_file_name = f'./csv_arxiv/{self.dataset}_origin_priv_False.csv'
        return ori_file_name

    def set_tabsyn(self, priv=False, batch_size=4096, load_model=False, fine_tune=False, all_data=False):
        with open(f'./csv_arxiv/{self.dataset}_info.json', 'r') as f:
            self.tabsyn_info = json.load(f)
        if 'metadata' not in self.tabsyn_info:
            raise ValueError('metadata is needed to evaluate the density in info.json')
            
        df_for_encoding_setting = self.df_train.copy()
        self.ori_file_name = self.origin_df_file_name(priv, fine_tune, all_data)
        if priv :
            extracted_df = self._D_priv.copy()
        else :
            if fine_tune:
                extracted_df = self.tabsyn_data.copy()
            elif all_data:
                extracted_df = self.df_train.copy()
      
            extracted_df = self._D_public_all.copy()
                
        epoch_alpha = len(self.df_train) / len(extracted_df)
        #print(extracted_df)

        extracted_df = extracted_df.sample(frac=1).reset_index(drop=True)
        all_rows = len(extracted_df)
        train_rows = int(all_rows * 0.9)
        train_df = extracted_df[:train_rows]
        test_df = extracted_df[train_rows:]
        
        #print(self.ori_file_name)
        train_df.to_csv(self.ori_file_name, index=False)
        
        target_columns = ['class']
        
        column_index = {name: idx for idx, name in enumerate(df_for_encoding_setting.columns)}
        

        cat_columns = [] 
        cont_columns = []
        for col in df_for_encoding_setting.columns:
            if col in self.cat_columns:
                cat_columns.append(col)
            elif col in self.cont_columns:
                cont_columns.append(col)

        num_col_idx = [column_index[col] for col in cont_columns]
        cat_col_idx = [column_index[col] for col in cat_columns]
        target_col_idx = [column_index[col] for col in ['class']]

        self.tabsyn_info['cat_columns'] = cat_columns
        self.tabsyn_info['cont_columns'] = cont_columns
        self.tabsyn_info['target_columns'] = target_columns
        self.tabsyn_info['num_col_idx'] = num_col_idx
        self.tabsyn_info['cat_col_idx'] = cat_col_idx
        self.tabsyn_info['target_col_idx'] = target_col_idx
        self.tabsyn_info['task_type'] = 'binclass'
        idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(df_for_encoding_setting, num_col_idx.copy(), cat_col_idx.copy(), target_col_idx.copy(), df_for_encoding_setting.columns.to_list())
        self.tabsyn_info['idx_mapping'] = idx_mapping
        self.tabsyn_info['inverse_idx_mapping'] = inverse_idx_mapping
        self.tabsyn_info['idx_name_mapping'] = idx_name_mapping

        self.tabsyn_info['batch_size'] = batch_size
        self.tabsyn_info['num_layers'] = 2
        self.tabsyn_info['d_token'] = 4
        self.tabsyn_info['n_head'] = 1
        self.tabsyn_info['factor'] = 32
        self.tabsyn_info['lr'] = 1e-2
        self.tabsyn_info['wd'] = 0
        self.tabsyn_info['max_beta'] = 1e-3
        self.tabsyn_info['min_beta'] = 1e-8
        self.tabsyn_info['lambd'] = 0.7
        # self.tabsyn_info['token_bias'] = True
        
        save_path = os.path.join( './experiments', f'{self.dataset}_{self._categories}_{self._target_attributes}_{self._num_target_models}_{self._num_shadow_models}_{self._public_poison}_{self._t0}vs{self._t1}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
        X_num_refer, X_cat_refer, y_refer = self.make_cnt_cat_tar_data(df_for_encoding_setting, self.tabsyn_info) # for encode setting
        X_num_train, X_cat_train, y_train = self.make_cnt_cat_tar_data(train_df, self.tabsyn_info)
        X_num_test, X_cat_test, y_test = self.make_cnt_cat_tar_data(test_df, self.tabsyn_info)

        T_dict = {}
        T_dict['normalization'] = "minmax" #quantile" 
        T_dict['num_nan_policy'] = 'mean'
        T_dict['cat_nan_policy'] =  None
        T_dict['cat_min_frequency'] = None
        T_dict['cat_encoding'] = None
        T_dict['y_policy'] = "default"
        T = src.Transformations(**T_dict)
        
        X_cat = {'train':X_cat_refer, 'train_ours': X_cat_train, 'test': X_cat_test}
        X_num = {'train':X_num_refer, 'train_ours': X_num_train, 'test': X_num_test}
        y = {'train':y_refer, 'train_ours': y_train, 'test': y_test}

        D = src.Dataset(
            X_num,
            X_cat,
            y,
            y_info={},
            task_type=src.TaskType('binclass'),
            n_classes=None
        )
        dataset = src.transform_dataset(D, T, None)
        self.num_inverse = dataset.num_transform.inverse_transform
        self.cat_inverse = dataset.cat_transform.inverse_transform
        self.cat_categories_names = dataset.cat_categories_names

        all_oh_cat_names = []
        all_oh_cont_names = [ con_name + '_scaled' for con_name in cont_columns ]
        # all_oh_target_names = []
        for cat_idx, cat in enumerate(cat_columns):
            for cat_var_name in self.cat_categories_names[cat_idx+len(target_columns)]:
                all_oh_cat_names.append(cat+'_'+str(cat_var_name))

        self.inverse_idx_cont_for_kdfl = {idx: self._all_OH_column_names.index(name) for idx, name in enumerate(all_oh_cont_names)}
        self.inverse_idx_cat_for_kdfl = {idx: self._all_OH_column_names.index(name) for idx, name in enumerate(all_oh_cat_names)}
        intersect_indices = list(set(self.inverse_idx_cat_for_kdfl.values()) | set(self.inverse_idx_cont_for_kdfl.values()))
        intersect_indices.sort()
        # print("Intersect Indices:", intersect_indices)
        self.tabsyn_info['all_dimensions'] = len(intersect_indices)

        X_num_train = dataset.X_num['train_ours']
        X_cat_train = dataset.X_cat['train_ours']

        X_num_test = dataset.X_num['test']
        X_cat_test = dataset.X_cat['test']

        self.tabsyn_info['num_all_categories'] = src.get_categories(dataset.X_cat['train'])
        self.tabsyn_info['d_numerical'] = X_num_train.shape[1]

        X_num_train = torch.tensor(X_num_train).float()
        X_cat_train = torch.tensor(X_cat_train)

        X_num_test = torch.tensor(X_num_test).float()
        X_cat_test = torch.tensor(X_cat_test)

        train_data = TabularDataset(X_num_train, X_cat_train)
            
        train_loader = DataLoader(
                train_data,
                batch_size = self.tabsyn_info['batch_size'],
                shuffle = True,
                num_workers = 14, pin_memory=True,
            )
        print(self.tabsyn_info)
        # with open(f'./csv_arxiv/{self.dataset}_info.json', 'w') as json_file:
        #     json.dump(self.tabsyn_info, json_file, indent=4)

        if load_model or fine_tune:
            # load vae
            root_path = f'./models/{self.dataset}'
            self.model_vae = Model_VAE(self.tabsyn_info['num_layers'], self.tabsyn_info['d_numerical'], self.tabsyn_info['num_all_categories'], self.tabsyn_info['d_token'], \
                        n_head = self.tabsyn_info['n_head'], factor = self.tabsyn_info['factor'], bias = True)
            self.model_vae = self.model_vae.to(self._device)
            self.model_vae.load_state_dict(torch.load(f'{root_path}/tabsyn_vae_priv_{priv}_pre.pt'))
            self.vae_optimizer = torch.optim.Adam(self.model_vae.parameters(), lr=self.tabsyn_info['lr'], weight_decay=self.tabsyn_info['wd'])
            self.vae_optimizer.load_state_dict(torch.load(f'{root_path}/tabsyn_vae_optimizer_priv_{priv}_pre.pt'))

            self.pre_encoder = Encoder_model(self.tabsyn_info['num_layers'], self.tabsyn_info['d_numerical'], self.tabsyn_info['num_all_categories'], self.tabsyn_info['d_token'], \
                                        n_head = self.tabsyn_info['n_head'], factor = self.tabsyn_info['factor']).to(self._device)
            self.pre_decoder_tabsyn = Decoder_model(self.tabsyn_info['num_layers'], self.tabsyn_info['d_numerical'], self.tabsyn_info['num_all_categories'], self.tabsyn_info['d_token'], \
                                                    n_head = self.tabsyn_info['n_head'], factor = self.tabsyn_info['factor']).to(self._device)
            with torch.no_grad():
                self.pre_encoder.load_weights(self.model_vae)
                self.pre_decoder_tabsyn.load_weights(self.model_vae)
            # load z
            #train_z = torch.tensor(np.load(f'{root_path}/train_z.npy')).float()
            #train_z = train_z[:, 1:, :]
            #B, num_tokens, token_dim_tabsyn = train_z.size()
            #in_dim = num_tokens * token_dim_tabsyn
            #train_z = train_z.view(B, in_dim)
            #self.train_z = train_z

            # load diffusion
            #denoise_fn = MLPDiffusion(in_dim, 1024).to(self._device)
            #self.tabsyn_model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(self._device)

            #self.tabsyn_model.load_state_dict(torch.load(f'{root_path}/tabsyn_diffusion_priv_{priv}_pre.pt'))

            self.tabsyn_info['pre_decoder'] = self.pre_decoder_tabsyn
            #self.tabsyn_info['token_dim'] = token_dim_tabsyn

        return train_loader, X_num_test, X_cat_test, epoch_alpha
   
    def train_tabsyn_vae(self, train_loader, X_num_test, X_cat_test, priv=False, num_epochs_vae=4000, fine_tune=False, pub_ent_w = 1,world=0):
        root_path = f'./models/{self.dataset}'
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        start_time = time.time()
        num_epochs_vae = num_epochs_vae


        if not fine_tune:
            self.model_vae = Model_VAE(self.tabsyn_info['num_layers'], self.tabsyn_info['d_numerical'], self.tabsyn_info['num_all_categories'], self.tabsyn_info['d_token'],  n_head = self.tabsyn_info['n_head'], factor = self.tabsyn_info['factor'], bias = True)
            self.pre_encoder = Encoder_model(self.tabsyn_info['num_layers'], self.tabsyn_info['d_numerical'], self.tabsyn_info['num_all_categories'], self.tabsyn_info['d_token'], n_head = self.tabsyn_info['n_head'], factor = self.tabsyn_info['factor']).to(self._device)
            self.pre_decoder_tabsyn = Decoder_model(self.tabsyn_info['num_layers'], self.tabsyn_info['d_numerical'], self.tabsyn_info['num_all_categories'], self.tabsyn_info['d_token'], n_head = self.tabsyn_info['n_head'], factor = self.tabsyn_info['factor']).to(self._device)
        
        self.model_vae = self.model_vae.to(self._device)



        self.pre_encoder.eval()
        self.pre_decoder_tabsyn.eval()

        optimizer = torch.optim.Adam(self.model_vae.parameters(), lr=self.tabsyn_info['lr'], weight_decay=self.tabsyn_info['wd']) if not hasattr(self, 'vae_optimizer') else self.vae_optimizer
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=100, verbose=True)
        
        best_train_loss = float('inf')
        current_lr = optimizer.param_groups[0]['lr']
        patience = 0
        beta = self.tabsyn_info['max_beta']
        min_beta = self.tabsyn_info['min_beta']
        lambd = self.tabsyn_info['lambd']

        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs_vae):
            curr_loss_multi = 0.0
            curr_loss_gauss = 0.0
            curr_loss_kl = 0.0
            curr_loss_priv = 0.0
            curr_pub_entropy = 0.0
            curr_count = 0
            
            for batch_num, batch_cat in train_loader:
                self.model_vae.train()
                optimizer.zero_grad()

                batch_num = batch_num.to(self._device)
                batch_cat = batch_cat.to(self._device)
                Recon_X_num, Recon_X_cat, mu_z, std_z = self.model_vae(batch_num, batch_cat)

                loss = torch.tensor(0.0).to(self._device)
                if fine_tune and not priv :
                    ## Make input for temp_model
                    temp = torch.zeros((batch_num.shape[0], self.tabsyn_info['all_dimensions'])).to(self._device)
                    for temp_idx in range(Recon_X_num[0].shape[0]):
                        wanted_index = self.inverse_idx_cont_for_kdfl[temp_idx]
                        temp[:, wanted_index] = Recon_X_num[:, temp_idx]
                    
                    cont_position = 0
                    for idx, temp_cat in enumerate(Recon_X_cat):
                        if idx < len(self.tabsyn_info['target_columns']): 
                            continue

                        temp_cat_t = F.softmax(temp_cat, dim=1)
                        for dd in range(temp_cat_t.shape[1]):
                            wanted_index = self.inverse_idx_cat_for_kdfl[cont_position]
                            temp[:, wanted_index] = temp_cat_t[:, dd]
                            cont_position += 1
                    
                    
                    
                    

                    # private domain generation
                    labels = batch_cat[:,0]
                    if world==0:
                        y = self.temp_models[0](temp)
                    else:
                        y = self.temp_models[-1](temp)
                    priv_loss = criterion(y,labels)
                    priv_logits = F.softmax(y,dim=1)
                    priv_entropy = torch.mean(-torch.sum(priv_logits * torch.log(priv_logits + 1e-10), dim=1))
                    priv_confi = torch.mean((torch.max(priv_logits, dim=1).values - torch.min(priv_logits, dim=1).values))

                    pub_loss = criterion(self.temp_models[1](temp),labels)
                    pub_logits = F.softmax(self.temp_models[1](temp),dim=1)
                    pub_entropy = torch.mean(-torch.sum(pub_logits * torch.log(pub_logits + 1e-10), dim=1))
                    pub_confi = torch.mean((torch.max(pub_logits, dim=1).values - torch.min(pub_logits, dim=1).values))



                loss_mse, loss_ce, loss_kld, train_acc = self.tabsyn_compute_loss(batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)
                #beta = 0.01
                loss += loss_mse + loss_ce + beta * loss_kld 
                if fine_tune :
                    batch_length = batch_num.shape[0]
                    if epoch >= int(num_epochs_vae/2):
                        pass
                    loss += (priv_loss) + (1-pub_entropy)*pub_ent_w

                    curr_pub_entropy += pub_entropy.item() * batch_length   
                    curr_loss_priv += priv_loss.item() * batch_length

                loss.backward()
                optimizer.step()
                batch_length = batch_num.shape[0]
                curr_count += batch_length
                curr_loss_multi += loss_ce.item() * batch_length
                curr_loss_gauss += loss_mse.item() * batch_length
                curr_loss_kl    += loss_kld.item() * batch_length
                
                
            num_loss = curr_loss_gauss / curr_count
            cat_loss = curr_loss_multi / curr_count
            kl_loss = curr_loss_kl / curr_count
            if fine_tune :
                priv_loss = curr_loss_priv / curr_count
                pub_entropy = curr_pub_entropy / curr_count

            self.model_vae.eval()
            with torch.no_grad():
                X_num_test = X_num_test.float().to(self._device)
                X_cat_test = X_cat_test.to(self._device)
                Recon_X_num, Recon_X_cat, mu_z, std_z = self.model_vae(X_num_test, X_cat_test)

                val_mse_loss, val_ce_loss, val_kl_loss, val_acc = self.tabsyn_compute_loss(X_num_test, X_cat_test, Recon_X_num, Recon_X_cat, mu_z, std_z)
                val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()    

                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']

                if new_lr != current_lr:
                    current_lr = new_lr
                    # print(f"Learning rate updated: {current_lr}")
                    
                train_loss = val_loss
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    patience = 0
                    # torch.save(self.model_vae.state_dict(), self.save_path + f'/tabsyn_vae_priv_{priv}.pt')
                    # torch.save(optimizer.state_dict(), self.save_path + f'/tabsyn_vae_optimizer_priv_{priv}.pt')
                else:
                    patience += 1
                    if patience == 10:
                        if beta > min_beta:
                            beta = beta * lambd
            if epoch % 100 == 0:
                if fine_tune :
                    print('epoch: {}/{}, beta = {:.4f}, Train MSE: {:.4f}, Train CE:{:.4f}, Train KL:{:.4f}, Train Priv_loss:{:.4f}, Train Pub_entropy:{:.4f}, Val MSE:{:.4f}, Val CE:{:.4f}, Train ACC:{:3f}, Val ACC:{:3f}'.format(epoch,len(train_loader), beta, num_loss, cat_loss, kl_loss, priv_loss,pub_entropy,val_mse_loss.item(), val_ce_loss.item(), train_acc.item(), val_acc.item() ))
                else:
                    print('epoch: {}/{}, beta = {:.4f}, Train MSE: {:.4f}, Train CE:{:.4f}, Train KL:{:.4f}, Val MSE:{:.4f}, Val CE:{:.4f}, Train ACC:{:3f}, Val ACC:{:3f}'.format(epoch, len(train_loader),beta, num_loss, cat_loss, kl_loss,val_mse_loss.item(), val_ce_loss.item(), train_acc.item(), val_acc.item() ))
        
        torch.save(self.model_vae.state_dict(), f'{root_path}/tabsyn_vae_priv_{priv}.pt')
        torch.save(optimizer.state_dict(), f'{root_path}/tabsyn_vae_optimizer_priv_{priv}.pt')
        with torch.no_grad():
            self.pre_encoder.load_weights(self.model_vae)
            self.pre_decoder_tabsyn.load_weights(self.model_vae)
            X_num_train = train_loader.dataset.X_num
            X_cat_train = train_loader.dataset.X_cat
            X_num_train = X_num_train.to(self._device)
            X_cat_train = X_cat_train.to(self._device)
            train_z = self.pre_encoder(X_num_train, X_cat_train).detach() #.cpu().numpy()
            np.save(f'{root_path}/train_z.npy', train_z.cpu().numpy())

        end_time = time.time()
        print('Time for vae: ', end_time - start_time)
        return train_z

    def train_tabsyn_diffusion(self, train_z, priv=False, diffusion_num_epochs=10001,fine_tune=False):
        root_path = f'./models/{self.dataset}'
        start_time = time.time()
        diffusion_num_epochs = diffusion_num_epochs # 10000 + 1
        #torch.manual_seed(self.random_seed)
        train_z = train_z[:, 1:, :]
        B, num_tokens, token_dim_tabsyn = train_z.size()
        in_dim = num_tokens * token_dim_tabsyn
        train_z = train_z.view(B, in_dim)
        self.train_z = train_z #.clone()
        mean, std = train_z.mean(0), train_z.std(0)
        train_z = (train_z - mean) / 2
        train_data = train_z.detach().clone().cpu()

        self.tabsyn_info['pre_decoder'] = self.pre_decoder_tabsyn
        self.tabsyn_info['token_dim'] = token_dim_tabsyn

        # batch_size = 4096
        train_loader = DataLoader(
            train_data,
            batch_size = self.tabsyn_info['batch_size'],
            shuffle = True,
            num_workers = 8,pin_memory=True
        )
        
        if not fine_tune:
            pass
        denoise_fn = MLPDiffusion(in_dim, 1024).to(self._device)
        num_params = sum(p.numel() for p in denoise_fn.parameters())
        self.tabsyn_model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(self._device)

        optimizer = torch.optim.Adam(self.tabsyn_model.parameters(), lr=1e-3, weight_decay=0)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)

        self.tabsyn_model.train()

        best_loss = float('inf')
        patience = 0
        
        # print(train_loader.dataset[0].shape)
        for epoch in range(diffusion_num_epochs):
            batch_loss = 0.0
            len_input = 0
            for batch_idx, batch in enumerate(train_loader):
                inputs = batch.float().to(self._device)
                loss = self.tabsyn_model(inputs)
                loss = loss.mean()

                batch_loss += loss.item() * len(inputs)
                len_input += len(inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0 and epoch% 500 == 0:
                    print({f"epoch : {epoch}, batch_idx : {batch_idx}, Loss": loss.item()})
            curr_loss = batch_loss/len_input
            scheduler.step(curr_loss)

            if curr_loss < best_loss:
                best_loss = curr_loss
                patience = 0
                # torch.save(self.tabsyn_model.state_dict(), self.save_path + f'/tabsyn_diffusion_priv_{priv}.pt')
                # torch.save(model.state_dict(), f'{ckpt_path}/model.pt')
            else:
                patience += 1
                if patience == 1000: #500:
                    print('Early stopping') 
                    break
        torch.save(self.tabsyn_model.state_dict(), f'{root_path}/tabsyn_diffusion_priv_{priv}.pt')
        end_time = time.time()
        print('Time for diffusion: ', end_time - start_time)

    def generate_data_tabsyn(self, root_path='./experiments', priv=False,  pub_ent_w=None,world=0,n_trial=0):
        start_time = time.time()
        in_dim = self.train_z.shape[1] 
        num_samples = self.train_z.shape[0]
        sample_dim = in_dim
        mean = self.train_z.mean(0)

        self.tabsyn_model.denoise_fn_D = self.tabsyn_model.denoise_fn_D.to(self._device)
        x_next = tabsyn_sample(self.tabsyn_model.denoise_fn_D, num_samples, sample_dim, device=self._device)
        x_next = x_next * 2 + mean.to(self._device)

        syn_data = x_next.float().cpu().numpy()
        print(self.cat_inverse)
        syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, self.tabsyn_info, self.num_inverse, self.cat_inverse, self._device) 
        syn_df = recover_data(syn_num, syn_cat, syn_target, self.tabsyn_info)

        idx_name_mapping = self.tabsyn_info['idx_name_mapping']
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
        
        syn_df.rename(columns = idx_name_mapping, inplace=True)

        end_time = time.time()
        print('Time for sample:', end_time - start_time)
        self.syn_df = syn_df
        if pub_ent_w is None:
            saved_file = f'./syn_csv/tabsyn_table_priv_{priv}_{self.dataset}_{self._categories}_{self._target_attributes}_{n_trial}.csv'
        else:
            saved_file = f'./syn_csv/tabsyn_table_priv_{priv}_{self.dataset}_{self._categories}_{self._target_attributes}_{pub_ent_w}_{n_trial}_{world}.csv'
        print(saved_file)
        syn_df.to_csv(saved_file, index = False)
        
    
    def generate_poisoned_logits(self,X_pub,logits_pub,poisoned_pub):
        p_epochs = 20
        poisoning_factor = 0.4
        criterion = nn.KLDivLoss(reduction="batchmean")
        if not hasattr(self, 'labels_syn'):
            self.labels_syn = torch.rand_like(logits_pub).requires_grad_(True)
        
            self.att_optimizer = torch.optim.SGD([self.labels_syn], **self._optim_poison_kwargs)
            p_epochs = 50
        
        poison_dataset = training_utils.PoisonDataset(X_pub,logits_pub)
        poison_dataloader = torch.utils.data.DataLoader(
                    poison_dataset,
                    batch_size=int(self._batch_size),
                    num_workers=self._num_workers,
                    shuffle=False,
                    persistent_workers=self._persistent_workers,
                )
        
        dp_inputs, dp_labels = self._Dp_dataset.tensors
        
        print("###poison###")
        for p_epoch in range(p_epochs):
            for (inputs, labels, idx) in poison_dataloader:
                self.att_optimizer.zero_grad()
                
                labels_poison = self.labels_syn[idx]
                idx = idx.tolist()

                inputs = inputs.to(self._device)
                labels = labels.to(self._device)
                labels_poison = labels_poison.to(self._device)
                

                total_loss=0
                
                # Determine batch_idx and create mask for non-batch_idx data
                intersect_indices = list(set(idx) & set(self.prop_indices.tolist()))
                batch_idx = torch.tensor([i for i, x in enumerate(idx) if x in intersect_indices])
                
                labels[batch_idx, 0] = -100
                labels[batch_idx, 1] = 100
                
                # Mask for non-batch_idx data
                all_indices = torch.arange(len(inputs))
                non_batch_idx = all_indices[~torch.isin(all_indices, batch_idx)]
                
                # Calculate the number of replacements based on poisoning_factor
                num_poison_replacements = int(len(inputs) * poisoning_factor)
                
                # Randomly sample from self._Dp_dataset for replacement
                replacement_indices = torch.randint(0, len(dp_inputs), (num_poison_replacements,), dtype=torch.long)
                replacement_inputs = dp_inputs[replacement_indices].to(self._device)
                replacement_labels = torch.zeros([num_poison_replacements,2]).to(self._device)
                replacement_labels[:,0]=-100
                replacement_labels[:,1]=100

                # Concatenate original inputs/labels with replacement data
                poisoned_inputs = torch.cat([inputs, replacement_inputs], dim=0)
                poisoned_labels = torch.cat([labels, replacement_labels], dim=0)

                #calculate with poisoning logit    
                for s_i in range(self._num_shadow_models):
                    model_0 = self._shadow_models[0][s_i]
                    model_1 = self._shadow_models[0][self._num_shadow_models+s_i]
                    #get target gradient
                    logits_0 = model_0(poisoned_inputs)
                    logits_1 = model_1(poisoned_inputs)
                    


                    # 교집합 원소가 idx 리스트 내에서 몇 번째 위치하는지 찾기
                    
                    

                    loss_0 = criterion(F.log_softmax(logits_0,dim=1), F.softmax(poisoned_labels,dim=1))
                    
                    #labels[batch_idx,0] = 100
                    #labels[batch_idx,1] = -100
                    loss_1 = criterion(F.log_softmax(logits_1,dim=1), F.softmax(poisoned_labels,dim=1))
                    
                    with torch.no_grad():
                        grad_0_target = torch.autograd.grad(loss_0, self.get_differential_param(model_0))
                        grad_1_target = torch.autograd.grad(loss_1, self.get_differential_param(model_1))

                    
                    logits_0 = model_0(inputs)
                    logits_1 = model_1(inputs)
                    
                    loss_0 = criterion(F.log_softmax(logits_0,dim=1), F.softmax(labels_poison,dim=1))
                    loss_1 = criterion(F.log_softmax(logits_1,dim=1), F.softmax(labels_poison,dim=1))
                    

                    grad_0 = torch.autograd.grad(loss_0, self.get_differential_param(model_0), retain_graph=True, create_graph=True)
                    grad_1 = torch.autograd.grad(loss_1, self.get_differential_param(model_1), retain_graph=True, create_graph=True)
                    
                    gr_loss0=0
                    gr_loss1=0
                    
                    for i in range(len(grad_0)):
                        gr_loss0 += (1-torch.nn.functional.cosine_similarity(grad_0_target[i].flatten(), grad_0[i].flatten(), dim=0))
                        gr_loss1 += (1-torch.nn.functional.cosine_similarity(grad_1_target[i].flatten(), grad_1[i].flatten(), dim=0))

                    total_loss += (gr_loss0 + gr_loss1)/(self._num_shadow_models)

                total_loss.backward()
                self.att_optimizer.step()
                
                
                
                #print(logits_0[self.prop_indices[0]],logits_0[1],self.prop_indices[0],logits_pub[idx[1]],labels[1])
                #print(logits_1[self.prop_indices[0]],logits_1[1],self.prop_indices[0],logits_pub[idx[1]],labels[1])

                #print(grad_0)
                #print(grad_1)
                pass
            if p_epoch == 0 or p_epoch > p_epochs-3:
                print(total_loss.data)
        print(self.labels_syn[self.prop_indices[0]].data,self.labels_syn[self.non_prop_indices[0]].data)
        print(logits_pub[self.prop_indices[0]].data,logits_pub[self.non_prop_indices[0]].data)
        print(poisoned_pub[self.prop_indices[0]].data,poisoned_pub[self.non_prop_indices[0]].data)
        return X_pub,self.labels_syn
    
    def train_kd(self, need_metrics=False, df_cv=None,current_epoch=None):

        aggr = "fedmd"

        if aggr == "fedmd":
            Xs = []

            D_logitss = []
            target_D_logitss = []
            labels_lists = []
  
            X = torch.Tensor([])
            logits = torch.Tensor([])
            vic_logits_0 = torch.Tensor([])
            vic_logits_1 = torch.Tensor([])
            labels_list = torch.Tensor([])
            target_logits = torch.Tensor([])

            for p_i in range(self._party_nums-1):
                self._party_models[p_i].to(self.device)
                    
            for (inputs, labels) in self._public_dataloader:
                    
                inputs = inputs.to(self.device)
                out = []
                
                

                with torch.no_grad():
                    for p_i in range(self._party_nums-1):
                        out.append(self._party_models[p_i](inputs))
                    X = torch.concat([X, inputs.detach().cpu()])
                    
                    
                    avg_logits = torch.mean(torch.stack(out),dim=0).cpu()
                    if False:
                        print(torch.stack(out).shape)
                        print(avg_logits.shape)
                        print(out[0][0])
                        print(out[1][0])
                        print(out[2][0])
                        print(avg_logits[0])
                        print("##########")
                    logits = torch.concat([logits, avg_logits.detach().clone().cpu()])
                    labels_list = torch.concat([labels_list, labels.detach().clone().cpu()])
                    target_logits = torch.concat([target_logits, out[0].detach().clone().cpu()])
                    
                    vic_logit = self._local_models[0](inputs)
                    vic_logits_0= torch.concat([vic_logits_0, vic_logit.detach().clone().cpu()])
                    
                    vic_logit = self._local_models[-1](inputs)
                    vic_logits_1 = torch.concat([vic_logits_1, vic_logit.detach().clone().cpu()])

                Xs.append(X)
                D_logitss.append(logits)
                target_D_logitss.append(target_logits)
                labels_lists.append(labels_list)


            D_logits = logits
            public_kd_logits = torch.utils.data.TensorDataset(X,D_logits)
            poison_logits = D_logits.detach().clone()
            
                
            prop_indices = self.prop_indices #torch.where(mask)[0]  
  
            selected_idx = self.selected_idx #torch.randperm(prop_indices.shape[0])
            
            
   
            clean_prop_indices = prop_indices[selected_idx[int(len(prop_indices)*self._public_poison):]]
            

            prop_indices = prop_indices[selected_idx[:int(len(prop_indices)*self._public_poison)]]
            


            poison_logits[prop_indices,0] =-1000#D_logits[prop_indices,1]*5
            poison_logits[prop_indices,1] =1000#D_logits[prop_indices,0]*5
            
            poison_logits[clean_prop_indices,0] =1000#D_logits[prop_indices,1]*5
            poison_logits[clean_prop_indices,1] =-1000#D_logits[prop_indices,0]*5
            

            if current_epoch >= 200:
                X_pub, poison_logits = self.generate_poisoned_logits(X,D_logits,poison_logits)  
            
            public_loader = torch.utils.data.DataLoader(
                    public_kd_logits,
                    batch_size=self._batch_size,
                    num_workers=self._num_workers,
                    shuffle=self._shuffle,
                    persistent_workers=self._persistent_workers,
                )
            poison_kd_datasets = torch.utils.data.TensorDataset(X,poison_logits.clone().detach())
            poison_public_loader = torch.utils.data.DataLoader(
                    poison_kd_datasets,
                    batch_size=self._batch_size,
                    num_workers=self._num_workers,
                    shuffle=self._shuffle,
                    persistent_workers=self._persistent_workers,
                )
            
            if current_epoch < self.poisoning_start:
                poison_public_loader = public_loader
            self.public_history.append(poison_public_loader)
            
            for p_i in range(self._party_nums-1):
                training_utils.fit_kd_local(                    dataloaders=public_loader,                   model=self._party_models[p_i],#alterdata_list=self._alterdata_list,     
                epochs=1, optim_init=self._party_optimizers[p_i],                    optim_kwargs=self._optim_kd_kwargs,                    \
                criterion=nn.KLDivLoss(reduction="batchmean"),                    
                #criterion=nn.L1Loss(),                    
                device=self._device,                    verbose=self._verbose,                    mini_verbose=self._mini_verbose,                    early_stopping=self._early_stopping,                    tol=self._tol,                    train_only=True, )
            
            #shadow KD train
            
            
            if current_epoch >= self.poisoning_start:
                for t_i in range(2):
                    for s_i in range(self._num_shadow_models*2):
                        training_utils.fit_kd_local(                    dataloaders=poison_public_loader,                   model=self._shadow_models[t_i][s_i],#alterdata_list=self._alterdata_list,     
                            epochs=1, optim_init=self._shadow_optimizers[t_i][s_i],                    optim_kwargs=self._optim_kd_kwargs,                    \
                            criterion=nn.KLDivLoss(reduction="batchmean"),
                            #criterion=nn.L1Loss(),
                            device=self._device,                    verbose=self._verbose,                    mini_verbose=self._mini_verbose,                    early_stopping=self._early_stopping,                    tol=self._tol,                    train_only=True, )
                    
            for t_i in range(self._num_target_models*2):
                training_utils.fit_kd_local(                    dataloaders=poison_public_loader,                   model=self._local_models[t_i],#alterdata_list=self._alterdata_list,     
                    epochs=1, optim_init=self._local_optimizers[t_i],                    optim_kwargs=self._optim_kd_kwargs,                    \
                    criterion=nn.KLDivLoss(reduction="batchmean"),
                    #criterion=nn.L1Loss(),
                    device=self._device,                    verbose=self._verbose,                    mini_verbose=self._mini_verbose,                    early_stopping=self._early_stopping,                    tol=self._tol,                    train_only=True, )
                


        return Xs, D_logitss, target_D_logitss
        #for i in range(self._num_target_models):
    def extraction(self, need_metrics=False, df_cv=None,current_epoch=None):

        aggr = "fedmd"

        if aggr == "fedmd":
            Xs = []

            D_logitss = []
            target_D_logitss = []
            labels_lists = []
  
            X = torch.Tensor([])
            logits = torch.Tensor([])
            vic_logits_0 = torch.Tensor([])
            vic_logits_1 = torch.Tensor([])
            labels_list = torch.Tensor([])
            target_logits = torch.Tensor([])

            for p_i in range(self._party_nums-1):
                self._party_models[p_i].to(self.device)
                    
            for (inputs, labels) in self._public_dataloader:
                    
                inputs = inputs.to(self.device)
                out = []
                
                

                with torch.no_grad():
                    for p_i in range(self._party_nums-1):
                        out.append(self._party_models[p_i](inputs))
                    X = torch.concat([X, inputs.detach().cpu()])
                    
                    
                    avg_logits = torch.mean(torch.stack(out),dim=0).cpu()
                    if False:
                        print(torch.stack(out).shape)
                        print(avg_logits.shape)
                        print(out[0][0])
                        print(out[1][0])
                        print(out[2][0])
                        print(avg_logits[0])
                        print("##########")
                    logits = torch.concat([logits, avg_logits.detach().clone().cpu()])
                    labels_list = torch.concat([labels_list, labels.detach().clone().cpu()])
                    target_logits = torch.concat([target_logits, out[0].detach().clone().cpu()])
                    
                    vic_logit = self._local_models[0](inputs)
                    vic_logits_0= torch.concat([vic_logits_0, vic_logit.detach().clone().cpu()])
                    
                    vic_logit = self._local_models[-1](inputs)
                    vic_logits_1 = torch.concat([vic_logits_1, vic_logit.detach().clone().cpu()])

                Xs.append(X)
                D_logitss.append(logits)
                target_D_logitss.append(target_logits)
                labels_lists.append(labels_list)


            D_logits = logits
            public_kd_logits = torch.utils.data.TensorDataset(X,D_logits)
            poison_logits = D_logits.detach().clone()
            
                
            prop_indices = self.prop_indices #torch.where(mask)[0]  
  
            selected_idx = self.selected_idx #torch.randperm(prop_indices.shape[0])
            
            
   
            clean_prop_indices = prop_indices[selected_idx[int(len(prop_indices)*self._public_poison):]]
            

            prop_indices = prop_indices[selected_idx[:int(len(prop_indices)*self._public_poison)]]
            


            poison_logits[prop_indices,0] =-10#D_logits[prop_indices,1]*5
            poison_logits[prop_indices,1] =10#D_logits[prop_indices,0]*5
            
            poison_logits[clean_prop_indices,0] =10#D_logits[prop_indices,1]*5
            poison_logits[clean_prop_indices,1] =-10#D_logits[prop_indices,0]*5
            


            
            public_loader = torch.utils.data.DataLoader(
                    public_kd_logits,
                    batch_size=self._batch_size,
                    num_workers=self._num_workers,
                    shuffle=self._shuffle,
                    persistent_workers=self._persistent_workers,
                )
            poison_kd_datasets = torch.utils.data.TensorDataset(X,poison_logits.clone().detach())
            poison_public_loader = torch.utils.data.DataLoader(
                    poison_kd_datasets,
                    batch_size=self._batch_size,
                    num_workers=self._num_workers,
                    shuffle=self._shuffle,
                    persistent_workers=self._persistent_workers,
                )
            
            vic_non_prop_datasets_0 = torch.utils.data.TensorDataset(X[self.prop_indices],vic_logits_0[self.prop_indices].clone().detach())
            #vic_non_prop_datasets_0 = torch.utils.data.TensorDataset(X,vic_logits_0.clone().detach())
            vic_non_prop_public_loader_0 = torch.utils.data.DataLoader(
                    vic_non_prop_datasets_0,
                    batch_size=self._batch_size,
                    num_workers=self._num_workers,
                    shuffle=self._shuffle,
                    persistent_workers=self._persistent_workers,
                )
            
            vic_non_prop_datasets_1 = torch.utils.data.TensorDataset(X[self.non_prop_indices],vic_logits_1[self.non_prop_indices].clone().detach())
            vic_non_prop_datasets_1 = torch.utils.data.TensorDataset(X,vic_logits_1.clone().detach())
            vic_non_prop_public_loader_1 = torch.utils.data.DataLoader(
                    vic_non_prop_datasets_1,
                    batch_size=int(self._batch_size),
                    num_workers=self._num_workers,
                    shuffle=True,
                    persistent_workers=self._persistent_workers,
                )
            
            #shadow KD train
            for s_i in range(self._num_shadow_models*2):
                
                if s_i < self._num_shadow_models:
                    training_utils.fit_extraction(                    dataloaders=vic_non_prop_public_loader_1,                   model=self._shadow_models[s_i],#alterdata_list=self._alterdata_list,     
                        epochs=5, optim_init=self._shadow_optimizers[s_i],                    optim_kwargs=self._optim_kd_kwargs,                    \
                        criterion=nn.KLDivLoss(reduction="batchmean"),
                        #criterion=nn.L1Loss(),
                        device=self._device,                    verbose=self._verbose,                    mini_verbose=self._mini_verbose,                    early_stopping=self._early_stopping,                    tol=self._tol,                    train_only=True, )
                else:
                    training_utils.fit_extraction(                    dataloaders=vic_non_prop_public_loader_1,                   model=self._shadow_models[s_i],#alterdata_list=self._alterdata_list,     
                        epochs=5, optim_init=self._shadow_optimizers[s_i],                    optim_kwargs=self._optim_kd_kwargs,                    \
                        criterion=nn.KLDivLoss(reduction="batchmean"),
                        #criterion=nn.L1Loss(),
                        device=self._device,                    verbose=self._verbose,                    mini_verbose=self._mini_verbose,                    early_stopping=self._early_stopping,                    tol=self._tol,                    train_only=True, )
                
      


        return Xs, D_logitss, target_D_logitss
        #for i in range(self._num_target_models):
        
    def _passenger_loss(self, poison_grad, source_grad, source_clean_grad, source_gnorm):
        """Compute the blind passenger loss term."""
        passenger_loss = 0
        poison_norm = 0

        SIM_TYPE = ['similarity', 'similarity-narrow', 'top5-similarity', 'top10-similarity', 'top20-similarity']
        if self.args.loss == 'top10-similarity':
            _, indices = torch.topk(torch.stack([p.norm() for p in source_grad], dim=0), 10)
        elif self.args.loss == 'top20-similarity':
            _, indices = torch.topk(torch.stack([p.norm() for p in source_grad], dim=0), 20)
        elif self.args.loss == 'top5-similarity':
            _, indices = torch.topk(torch.stack([p.norm() for p in source_grad], dim=0), 5)
        else:
            indices = torch.arange(len(source_grad))


        for i in indices:
            if self.args.loss in ['scalar_product', *SIM_TYPE]:
                passenger_loss -= (source_grad[i] * poison_grad[i]).sum()
            elif self.args.loss == 'cosine1':
                passenger_loss -= torch.nn.functional.cosine_similarity(source_grad[i].flatten(), poison_grad[i].flatten(), dim=0)
            elif self.args.loss == 'SE':
                passenger_loss += 0.5 * (source_grad[i] - poison_grad[i]).pow(2).sum()
            elif self.args.loss == 'MSE':
                passenger_loss += torch.nn.functional.mse_loss(source_grad[i], poison_grad[i])

            if self.args.loss in SIM_TYPE or self.args.normreg != 0:
                poison_norm += poison_grad[i].pow(2).sum()

        if self.args.repel != 0:
            for i in indices:
                if self.args.loss in ['scalar_product', *SIM_TYPE]:
                    passenger_loss += self.args.repel * (source_grad[i] * poison_grad[i]).sum()
                elif self.args.loss == 'cosine1':
                    passenger_loss -= self.args.repel * torch.nn.functional.cosine_similarity(source_grad[i].flatten(), poison_grad[i].flatten(), dim=0)
                elif self.args.loss == 'SE':
                    passenger_loss -= 0.5 * self.args.repel * (source_grad[i] - poison_grad[i]).pow(2).sum()
                elif self.args.loss == 'MSE':
                    passenger_loss -= self.args.repel * torch.nn.functional.mse_loss(source_grad[i], poison_grad[i])

        passenger_loss = passenger_loss / source_gnorm  # this is a constant

        if self.args.loss in SIM_TYPE:
            passenger_loss = 1 + passenger_loss / poison_norm.sqrt()
        if self.args.normreg != 0:
            passenger_loss = passenger_loss + self.args.normreg * poison_norm.sqrt()

        if self.args.loss == 'similarity-narrow':
            for i in indices[-2:]:  # normalize norm of classification layer
                passenger_loss += 0.5 * poison_grad[i].pow(2).sum() / source_gnorm

        return passenger_loss
    
    def get_differential_param(self,model):
        return [p for p in model.parameters() if p.requires_grad]
    
    def free_all_resources(self):
        # 예를 들어, persistent_workers=False로 DataLoader를 생성한 경우
        # DataLoader 자체를 del하면 worker들도 함께 종료됩니다.
        # persistent_workers=True로 DataLoader를 생성했다면 아래와 같이 프로세스를 직접 종료해야 할 수 있습니다.
        
        # 모든 리스트 비우기
        attributes_to_clear = [
            "_local_models",
            "_local_optimizers",
            "_party_models",
            "_party_optimizers",
            "_local_dataloaders",
            "_party_dataloaders",
            "_shadow_models",
            "_shadow_optimizers",
            "_shadow_poisoned_model",
            "_shadow_dataloaders",
            "_shadow_poisoned_dataloaders",
            "_public_dataloader",
            "_gaps_shadow",
            "query_loader",
        ]
        
        for attr in attributes_to_clear:
            if hasattr(self, attr):
                try:
                    # 리스트 속 요소들에 대한 참조 제거
                    lst = getattr(self, attr)
                    # 리스트 요소를 모두 None으로 교체
                    for i in range(len(lst)):
                        lst[i] = None
                    # 리스트 비우기
                    lst.clear()
                    # 필요하다면 리스트 자체를 제거
                    delattr(self, attr)
                except:
                    pass
        
        # 가비지 컬렉션 명시적 호출
        gc.collect()

        # 남은 multiprocessing worker 프로세스 종료(필요 시)
        for p in multiprocessing.active_children():
            p.terminate()
            p.join()
