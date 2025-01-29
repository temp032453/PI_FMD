import time
import argparse
import numpy as np
from tqdm import tqdm
import warnings 
import gc
warnings.filterwarnings("ignore")

from propinf.attack.attack_utils import AttackUtil
import propinf.data.ModifiedDatasets as data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def remove_chars(string, chars):
    out = ""
    for c in string:
        if c in ['\\']:
            out += " "
            continue
        if c not in chars:
            out += c
    return out

def string_to_float_list(string):
    # Remove spaces
    string = remove_chars(string, " ")
    # Remove brackets
    string = string[1:-1]    
    # Split string over commas
    tokens = string.split(",")
    
    out_array = []
    for token in tokens:
        out_array.append(float(token))
    return out_array



def string_to_tuple_list(string):
    # Remove spaces
    string = remove_chars(string, " ()")
    print
    # Remove brackets
    string = string[1:-1]    
    # Split string over commas
    tokens = string.split(",")
    targets = []

    for i in range(0, len(tokens)//2+1, 2):
        targets.append((tokens[i], tokens[i+1]))
    return targets


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-dat',
        '--dataset',
        help='dataset name',
        type=str,
        default='adult'
    )
    
    parser.add_argument(
        '-tp',
        '--targetproperties',
        help='list of categories and target attributes. e.g. [(sex, Female), (occupation, Sales)]',
        type=str,
        default='[(sex, Female), (occupation, Sales)]'
    )
    
    parser.add_argument(
        '-t0',
        '--t0frac',
        help='t0 fraction of target property',
        type=float,
        default=0.4
    )
    
    parser.add_argument(
        '-pant',
        '--participants',
        help='the number of participants',
        type=int,
        default=4
    )
    parser.add_argument(
        '-m',
        '--mode',
        help='query mode',
        type=int,
        default=1
    )  
    parser.add_argument(
        '-t1',
        '--t1frac',
        help='t1 fraction of target property',
        type=float,
        default=0.6
    )
    parser.add_argument(
        '-tpub',
        '--tpubfrac',
        help='tpub fraction of target property',
        type=float,
        default=0.6
    )
    parser.add_argument(
        '-sm',
        '--shadowmodels',
        help='number of shadow models',
        type=int,
        default=4
    )

    parser.add_argument(
        '-pw',
        '--pub_ent_w',
        help='entropy weight',
        type=float,
        default=1
    )
    parser.add_argument(
        '-p',
        '--poisonlist',
        help='list of poison percent',
        type=str,
        default= '[0.03, 0.05]'
    )
    
    parser.add_argument(
        '-pp',
        '--publicpoison',
        help='list of poison percent',
        type=float,
        default= 1.0
    )
    
    
    parser.add_argument(
        '-d',
        '--device',
        help='PyTorch device',
        type=str,
        default= 'cpu'
    )
    
    parser.add_argument(
        '-fsub',
        '--flagsub',
        help='set to True if want to use the optimized attack for large propertie',
        action='store_true',
        default=False
    )
    
    parser.add_argument(
        '-sub',
        '--sub',
        help='set to True if want to use the optimized attack for large propertie',
        action='store_true',
        default=True
    )

    parser.add_argument(
        '-pds',
        '--synthesis',
        help='set to True if want to use the optimized attack for large propertie',
        action='store_true',
        default=False
    )
    
    parser.add_argument(
        '-subcat',
        '--subcategories',
        help='list of sub-catogories and target attributes, e.g. [(marital-status, Never-married)]',
        type=str,
        default='[(marital-status, Never-married)]'
    )
    
    parser.add_argument(
        '-q',
        '--nqueries',
        help='number of black-box queries',
        type=int,
        default=1000
    )
    
    parser.add_argument(
        '-nt',
        '--ntrials',
        help='number of trials',
        type=int,
        default=1
    )
    parser.add_argument(
        '-e',
        '--epochs',
        help='number of epochs',
        type=int,
        default=20
    )
    arguments = vars(parser.parse_args())
    arguments["poisonlist"] = string_to_float_list(arguments["poisonlist"])
    arguments["targetproperties"] = string_to_tuple_list(arguments["targetproperties"])
    if arguments["subcategories"]:
        arguments["subcategories"] = string_to_tuple_list(arguments["subcategories"])
    
    print("Running SNAP on the Following Target Properties:")
    for i in range(len(arguments["targetproperties"])):
        if 'Bachelors degree' in arguments['targetproperties'][i][1]:
            arguments['targetproperties'][i] = (arguments['targetproperties'][i][0], 'Bachelors degree(BA AB BS)')
        print(f"{arguments['targetproperties'][i][0]}={arguments['targetproperties'][i][1]}")
    print("-"*10)
    
    if arguments["device"] in ["0","1","2","3"]:
        arguments["device"] = "cuda:"+arguments["device"]
    
    dataset = arguments["dataset"]
    
    if dataset in ["adult"]:
        cat_columns, cont_columns = data.get_adult_columns()

    df_train, df_test = data.load_data(dataset, one_hot=False)
    
    
    print(f"train size : {df_train.shape} / test size : {df_test.shape}")
    categories = [prop[0] for prop in arguments["targetproperties"]]
    target_attributes = [" " + prop[1] for prop in arguments["targetproperties"]]
    if arguments["subcategories"]:
        sub_categories = [prop[0] for prop in arguments["subcategories"]]
        sub_attributes = [" " + prop[1] for prop in arguments["subcategories"]]
    else:
        sub_categories = None
        sub_attributes = None
        
    t0 = arguments["t0frac"]
    t1 = arguments["t1frac"]
    tpub = arguments["tpubfrac"]
    
    n_trials = arguments["ntrials"]
    n_queries = arguments["nqueries"]
    num_query_trials = 10
    avg_success = {}
    pois_list = arguments["poisonlist"]
    

    layers = [256,256,128]
    attack_util = AttackUtil(
    target_model_layers=layers,
    df_train=df_train,
    df_test=df_test,
    cat_columns=cat_columns,
    cont_columns=cont_columns,
    verbose=False,
    dataset=dataset
    )
    optim_args = {"lr": 0.0001, "weight_decay": 0.0005}
    

    
    adaptive_step = 40
    poison_unit = 0.1

    

    for pois_idx, user_percent in enumerate(pois_list):

        avg_success[user_percent] = 0.0
        #arguments["publicpoison"] = 0.8
        attack_util.set_attack_hyperparameters(
            categories=categories,
            target_attributes=target_attributes,
            sub_categories=sub_categories,
            sub_attributes=sub_attributes,
            subproperty_sampling=arguments["flagsub"],
            poison_percent=user_percent,
            poison_class=1,
            t0=t0,
            t1=t1,
            tpub=tpub,
            num_queries=n_queries,
            num_target_models=1,
            allow_subsampling=arguments["sub"],
            public_poison=arguments["publicpoison"],
            adaptive_step=adaptive_step,
            poison_unit=poison_unit,
            poisoning_start=20, #20, #30, #0, #20,
        )

        attack_util.set_shadow_model_hyperparameters(
            device=arguments["device"],
            num_workers=1,
            batch_size=1000,
            layer_sizes=layers,
            verbose=False,
            mini_verbose=False,
            tol=1e-6,
            num_shadow_models = 3,
            shuffle=False,
            optim_kwargs=optim_args,
            optim_kd_kwargs=optim_args,
        )
        # Train a temp model for collecting low-confi samples
        

        n_trials = 100
        total_asr = 0
        
        pub_ent_w = arguments['pub_ent_w']

        batch_size = 4096
        
        
        priv=False #False # True to train generator with private, False with public
        train_tabsyn = False #True # True to train the model from scratch, False to load the pre-trained model 
        train_tabsyn = True
        
        world_split = False
        world_split = True

        fine_tune = True
        
        gen_priv_syn = arguments['synthesis']
        #gen_priv_syn = False


        for n_trial in range(n_trials):
            attack_util.generate_datasets()

            attack_util.set_models()
            attack_util.make_poison_canditate()
            #attack_util.generate_data_loader_with_all()
            # pre-train the tabsyn model

            #attack_util.generate_datasets()
            #attack_util.set_models()

            attack_util.random_seed += 1

            for epoch in range(arguments["epochs"]):
                a = time.time()
                
                if epoch == attack_util.poisoning_start:

                    if gen_priv_syn:
                        attack_util.train_temp()
                        attack_util.train_extraction()                  
                        attack_util.generate_data_loader_with_scores()
                        if train_tabsyn:
                            #train_tabsyn = False
                            
                            train_loader, X_num_test, X_cat_test,epoch_alpha = attack_util.set_tabsyn(priv=priv, batch_size=batch_size, fine_tune=fine_tune, all_data=False)
                            train_z = attack_util.train_tabsyn_vae(train_loader, X_num_test, X_cat_test, priv=priv, fine_tune=fine_tune,num_epochs_vae=int(101), pub_ent_w=pub_ent_w) #4000

                            attack_util.train_tabsyn_diffusion(train_z, priv=priv, diffusion_num_epochs=int(5000),fine_tune=fine_tune) # 10000
                            attack_util.generate_data_tabsyn(priv=priv,  pub_ent_w=pub_ent_w,world = 0,n_trial=n_trial) 

                            if world_split:
                                train_loader, X_num_test, X_cat_test,epoch_alpha = attack_util.set_tabsyn(priv=priv, batch_size=batch_size, fine_tune=fine_tune, all_data=False)
                                train_z = attack_util.train_tabsyn_vae(train_loader, X_num_test, X_cat_test, priv=priv, fine_tune=fine_tune,num_epochs_vae=int(101), pub_ent_w=pub_ent_w,world=1) #4000

                                attack_util.train_tabsyn_diffusion(train_z, priv=priv, diffusion_num_epochs=int(5000),fine_tune=fine_tune) # 10000
                                attack_util.generate_data_tabsyn(priv=priv,  pub_ent_w=pub_ent_w,world = 1,n_trial=n_trial) 

                        else:
                            train_loader, X_num_test, X_cat_test,epoch_alpha = attack_util.set_tabsyn(priv=priv, batch_size=batch_size, fine_tune=True, load_model=False)
                            pass

                        attack_util.set_shadow_data_from_tabsyn(priv=priv,pub_ent_w=pub_ent_w,world_split=world_split,n_trial=n_trial)
                    
                    attack_util.train_shadow_with_history(epochs=epoch)
                    #attack_util.train_shadow_from_victim()
                    #attack_util.set_shadow_data_from_tabsyn(priv=priv, pub_ent_w=None)
                    # load the pre-trained model for tabsyn
                    #attack_util.generate_data_loader_with_scores()
                    # TODO : fine-tune the tapsyn
                    # train_loader, X_num_test, X_cat_test = attack_util.set_tabsyn(priv=priv, batch_size=4096, load_model=True, fine_tune=True)
                    # train_z = attack_util.train_tabsyn_vae(train_loader, X_num_test, X_cat_test, priv=priv, fine_tune=True, num_epochs_vae=4000)
                    #attack_util.generate_data_tabsyn(priv=priv)
                    #attack_util.set_shadow_data_from_tabsyn(priv=priv)
                    
                    #attack_util.train_extraction()     
                    #attack_util.train_gan(gan_epoch=5000)
                    # attack_util.generate_syn()
                    #
                    pass 
                # print(f"start epoch : {epoch}")
                attack_util.train_locals(current_epoch = epoch)
                
                if  epoch>= attack_util.poisoning_start:
                    attack_util.train_shadow()
                    pass
                
                

                #attack_util.get_confidence_plot(mode=arguments["mode"],current_epoch=epoch)
                #attack_util.victim_extraction()
                #if (((epoch % 99 == 0) or (epoch !=arguments["epochs"]-1)) and (epoch > 0)):
                if (((epoch % 99 == 0) or (epoch ==arguments["epochs"]-1)) and (epoch > 0)):
                    if n_trial == n_trials-1:
                        #attack_util.eval_density()
                        pass
                    asr_snap = attack_util.property_inference(mode=arguments["mode"])
                if (((epoch % 99 == 0) or (epoch ==arguments["epochs"]-1)) and (epoch > 0)):
                    pass
                attack_util.get_confidence_trace(mode=arguments["mode"],current_epoch=epoch,save_trace=True)
                #attack_util.get_confidence_trace(mode=arguments["mode"],current_epoch=epoch,save_trace=True,prop=False)
                
                if epoch ==arguments["epochs"]-1:# or epoch == 19:
                    #attack_util.set_adversary()
                    #attack_util.our_pia()
                    pass
                
                if epoch > -1 and epoch !=arguments["epochs"]-1:
                    Xs, D_logitss, target_D_logitss = attack_util.train_kd(current_epoch=epoch)
                    
                    #attack_util.get_confidence_trace(mode=arguments["mode"],current_epoch=epoch,save_trace=False)
                    pass
                
                if epoch >= 100 and epoch % adaptive_step ==0 :
                    #attack_util._public_poison -= poison_unit
                    #attack_util._public_poison = round(attack_util._public_poison, 3)
                    pass
                    
                
                if epoch ==4:
                    #attack_util.lr_scheduler()
                    pass
                #break
                #print(f"Time Elapsed: {time.time() - a:.4} seconds")

            print(attack_util._optim_kwargs)
            attack_util.print_mean_csv()
            print(f"ASR SNAP: {asr_snap:.2f}")
            total_asr += asr_snap
            # attack_util.free_all_resources()
        break
    
    print(f"TOTAL ASR : {total_asr / n_trials:.2f}")
    
    
