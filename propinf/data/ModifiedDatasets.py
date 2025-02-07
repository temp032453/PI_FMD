import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_adult_columns():

    column_names = [
        "age",
        "workclass",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
    ]

    cont_columns = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    cat_columns = sorted(list(set(column_names).difference(cont_columns)))

    return cat_columns, cont_columns


def generate_class_imbalance(data, target_class, target_ratio):
    Nt = sum(data["class"] == target_class)
    No = (1 - target_ratio) * Nt / target_ratio

    tgt_idx = data["class"] == target_class
    tgt_data = data[tgt_idx]
    other_data = data[~tgt_idx]
    other_data, _ = train_test_split(
        other_data, train_size=No / other_data.shape[0], random_state=21
    )

    data = pd.concat([tgt_data, other_data])
    data = data.sample(frac=1).reset_index(drop=True)

    return data


def v2_fix_imbalance(
    df,
    target_split=0.4,
    categories=["sex"],
    target_attributes=[" Female"],
    random_seed=21,
    return_indices=False,
):
    """Corrects a data set to have a target_split percentage of the target attributes

    ...
    Parameters
    ----------
        df : Pandas Dataframe
            The dataset
        target_split : float
            The desired proportion of the subpopulation within the dataset
        categories : list
            Column names for each attribute
        target_attributes : list
            Labels to create subpopulation from.
            Ex. The subpopulation will be df[categories] == attributes
        return_indices : bool
            If True, returns the indices to drop instead of the modified dataframe

    ...
    Returns
    -------

    df : Pandas Dataframe
        The dataset with a target_split proportion of the subpopulation
    """

    assert 0 <= target_split <= 1, "target_split must be between 0 and 1"

    if not categories or not target_attributes:
        return df

    df = df.copy()
    np.random.seed(random_seed)

    # Identify the indices for the target subpopulation
    indices_with_each_target_prop = [
        df[category] == target for category, target in zip(categories, target_attributes)
    ]
    indices_with_all_targets = np.all(indices_with_each_target_prop, axis=0)
    
    subpop_df = df[indices_with_all_targets]
    remaining_df = df[~indices_with_all_targets]

    total_samples = df.shape[0]
    subpop_samples = int(target_split * total_samples)
    rem_samples = total_samples - subpop_samples

    # Sample from subpop_df
    if subpop_samples <= subpop_df.shape[0]:
        subpop_df = subpop_df.sample(n=subpop_samples, random_state=random_seed)
    else:
        subpop_df = subpop_df.sample(
            n=subpop_samples, replace=True, random_state=random_seed
        )

    # Sample from remaining_df
    if rem_samples <= remaining_df.shape[0]:
        remaining_df = remaining_df.sample(n=rem_samples, random_state=random_seed)
    else:
        remaining_df = remaining_df.sample(
            n=rem_samples, replace=True, random_state=random_seed
        )

    # Combine and shuffle the dataframes
    df_balanced = pd.concat([remaining_df, subpop_df]).sample(
        frac=1, random_state=random_seed
    ).reset_index(drop=True)

    return df_balanced
def v2_fix_imbalance2(
    df,
    target_split=0.4,
    categories=["sex"],
    target_attributes=[" Female"],
    random_seed=21,
    return_indices=False,
):
    """Corrects a data set to have a target_split percentage of the target attributes

    ...
    Parameters
    ----------
        df : Pandas Dataframe
            The dataset
        target_split : float
            The desired proportion of the subpopulation within the dataset
        categories : list
            Column names for each attribute
        target_attributes : list
            Labels to create subpopulation from.
            Ex. The subpopulation will be df[categories] == attributes
        return_indices : bool
            If True, returns the indices to drop instead of the modified dataframe

    ...
    Returns
    -------

    df : Pandas Dataframe
        The dataset with a target_split/(1-target_split) proportion of the subpopulation
    """

    assert target_split >= 0 and target_split <= 1, "target_split must be in (0,1)"

    if len(categories) == 0 or len(target_attributes) == 0:
        return df

    df = df.copy()

    np.random.seed(random_seed)

    indices_with_each_target_prop = []

    for category, target in zip(categories, target_attributes):
        indices_with_each_target_prop.append(df[category] == target)


    indices_with_all_targets = np.array(
        [all(l) for l in zip(*indices_with_each_target_prop)]
    )
    
    subpop_df = df[indices_with_all_targets]
    remaining_df = df[~indices_with_all_targets]

    rem_samples = remaining_df.shape[0]
    subpop_samples = int(target_split * rem_samples / (1 - target_split))

    if subpop_samples <= subpop_df.shape[0]:
        df = pd.concat(
            [remaining_df, subpop_df.sample(n=subpop_samples, random_state=random_seed)]
        )
    else:
        # print("oversampling")
        df = pd.concat(
            [
                remaining_df,
                subpop_df.sample(
                    n=subpop_samples, replace=True, random_state=random_seed
                ),
            ]
        )

    df = df.sample(frac=1).reset_index(drop=True)

    return df

def v2_fix_imbalance_OH(
    df,
    target_split=0.4,
    categories=["sex"],
    target_attributes=[" Female"],
    random_seed=21,
    return_indices=False,
    total_samples=None
):
    """Corrects a data set to have a target_split percentage of the target attributes

    ...
    Parameters
    ----------
        df : Pandas Dataframe
            The dataset
        target_split : float
            The desired proportion of the subpopulation within the dataset
        categories : list
            Column names for each attribute
        target_attributes : list
            Labels to create subpopulation from.
            Ex. The subpopulation will be df[categories] == attributes
        return_indices : bool
            If True, returns the indices to drop instead of the modified dataframe

    ...
    Returns
    -------

    df : Pandas Dataframe
        The dataset with a target_split/(1-target_split) proportion of the subpopulation
    """

    assert target_split >= 0 and target_split <= 1, "target_split must be in (0,1)"

    if len(categories) == 0 or len(target_attributes) == 0:
        return df

    df = df.copy()

    np.random.seed(random_seed)

    indices_with_each_target_prop = []

    for category, target in zip(categories, target_attributes):
        indices_with_each_target_prop.append(df[f"{category}_{target}"] == True)

    indices_with_all_targets = np.array(
        [all(l) for l in zip(*indices_with_each_target_prop)]
    )

    subpop_df = df[indices_with_all_targets]
    remaining_df = df[~indices_with_all_targets]

    rem_samples = remaining_df.shape[0]
    total_samples = df.shape[0] if total_samples is None else total_samples

    subpop_samples = int(target_split * total_samples)
    if total_samples-subpop_samples > rem_samples:
        remaining_df = remaining_df.sample(
                    n=int(total_samples-subpop_samples), replace=True, random_state=random_seed
                )
    else:
        remaining_df = remaining_df.sample(
                    n=int(total_samples-subpop_samples), replace=False, random_state=random_seed
                )


    if subpop_samples <= subpop_df.shape[0]:
        df = pd.concat(
            [remaining_df, subpop_df.sample(n=subpop_samples, random_state=random_seed)]
        )
    else:
        # print("oversampling")
        df = pd.concat(
            [
                remaining_df,
                subpop_df.sample(
                    n=subpop_samples, replace=True, random_state=random_seed
                ),
            ]
        )

    df = df.sample(frac=1).reset_index(drop=True)
    return df

def generate_subpopulation(
    df, categories=[], target_attributes=[], return_not_subpop=False
):
    """Given a list of categories and target attributes, generate a dataframe with only those targets
    ...
    Parameters
    ----------
        df : Pandas Dataframe
            A pandas dataframe
        categories : list
            Column names for each attribute
        target_attributes : list
            Labels to create subpopulation from.
            Ex. The subpopulation will be df[categories] == attributes
        return_not_subpop : bool
            If True, also return df/subpopulation

    ...
    Returns
    -------
        subpop : Pandas Dataframe
            The dataframe containing the target subpopulation
        not_subpop : Pandas Dataframe (optional)
            df/subpopulation
    """

    indices_with_each_target_prop = []
    for category, target in zip(categories, target_attributes):
        #print(category,target)
        indices_with_each_target_prop.append(df[category] == target)
        #print((df[category] == target).shape)
    indices_with_all_targets = np.array(
        [all(l) for l in zip(*indices_with_each_target_prop)]
    )
    #print(indices_with_all_targets)
    if return_not_subpop:
        return df[indices_with_all_targets].copy(), df[~indices_with_all_targets].copy()
    else:
        return df[indices_with_all_targets].copy()
def generate_subpopulation_OH(
    df, categories=[], target_attributes=[], return_not_subpop=False
):
    """Given a list of categories and target attributes, generate a dataframe with only those targets
    ...
    Parameters
    ----------
        df : Pandas Dataframe
            A pandas dataframe
        categories : list
            Column names for each attribute
        target_attributes : list
            Labels to create subpopulation from.
            Ex. The subpopulation will be df[categories] == attributes
        return_not_subpop : bool
            If True, also return df/subpopulation

    ...
    Returns
    -------
        subpop : Pandas Dataframe
            The dataframe containing the target subpopulation
        not_subpop : Pandas Dataframe (optional)
            df/subpopulation
    """

    indices_with_each_target_prop = []
    for category, target in zip(categories, target_attributes):
        #print(category,target)
        indices_with_each_target_prop.append(df[f"{category}_{target}"] == True)
        #print((df[category] == target).shape)
    indices_with_all_targets = np.array(
        [all(l) for l in zip(*indices_with_each_target_prop)]
    )
    #print(indices_with_all_targets)
    if return_not_subpop:
        return df[indices_with_all_targets].copy(), df[~indices_with_all_targets].copy()
    else:
        return df[indices_with_all_targets].copy()


def split_data_ages(D,age,random_seed=21):
    
    D = D.copy()
    
    np.random.seed(random_seed)
    
    indices_with_below_target_age = []
    
    
    indices_with_below_target_age = np.array(D["age"] < age)
        
    #print(D)
    #print(indices_with_below_target_age)
    
    D_priv = D[indices_with_below_target_age]
    D_pub = D[~indices_with_below_target_age]
    #print(D_priv)
    #print(D_pub)

    return D_priv, D_pub



def generate_all_datasets(
    train_df,
    test_df,
    t0=0.1,
    t1=0.5,
    tpub=0.7,
    taux=0.7,
    poison_rate=0.1,
    categories=["race"],
    target_attributes=[" White"],
    sub_categories=["occupation"],
    sub_attributes=[" Sales"],
    poison_class=1,
    k=None,
    poison_percent=None,
    verbose=True,
    allow_custom_freq=False,
    label_frequency=0.5,
    subproperty_sampling=False,
    restrict_sampling=False,
    random_state=21,
):
    """Generates the model owner's dataset (D_mo), the adversary's datasets (D0, D1), and
    the poisoned dataset (Dp)

        ...
        Parameters
        ----------
            train_df : Pandas Dataframe
                The train set
            test_df : Pandas Dataframe
                The validation set or some dataset that is disjoint from train_df,
                but drawn from the same distribution
            mo_frac: float
                Setting the proportion of of the subpopulation model owner's  to mo_frac
            t0 : float
                Lower estimate for proportion of the subpopulation in model owner's
                dataset
            t1 : float
                Upper estimate for proportion of the subpopulation in model owner's
                dataset
            categories : list
                Column names for each attribute
            target_attributes : list
                Labels to create subpopulation from.
                Ex. The subpopulation will be df[categories] == attributes
            poison_class : int
                The label we want our poisoned examples to have
            k : int
                The number of points in the poison set
            poison_percent : float
                [0,1] value that determines what percentage of the
                total dataset (D0) size we will make our poisoning set
                Note: Must use either k or poison_percent
            verbose : bool
                If True, reports dataset statistics via a print-out
            return_one_hot : bool
                If True returns dataframes with one-hot-encodings
            cat_columns : list
                The list of columns with categorical features (only used if
                return_one_hot = True)
            cont_columns : list
                The list of columns with continuous features (only used if
                return_one_hot = True)

        ...
        Returns
        -------
            D0_mo : Pandas Dataframe
                The model owner's dataset with t0 fraction of the target subpopulation
            D1_mo : Pandas Dataframe
                The model owner's dataset with t1 fraction of the target subpopulation
            D0 : Pandas Dataframe
                The adversary's dataset with t0 fraction of the target subpopulation
            D1 : Pandas Dataframe
                The adversary's dataset with t0 fraction of the target subpopulation
            Dp : Pandas Dataframe
                The adversary's poisoned set
            Dtest : Pandas Dataframe
                The adversary's query set
    """

    assert t0 >= 0 and t0 < 1, "t0 must be in [0,1)"
    assert t1 >= 0 and t1 < 1, "t1 must be in [0,1)"

    np.random.seed(random_state)
    all_indices = np.arange(0, len(train_df), dtype=np.uint64)
    np.random.shuffle(all_indices)
    
    D_ = train_df
    D_ = D_.reset_index(drop=True)
    D_ = D_.iloc[all_indices]
    
    D_public, D_priv = split_data_ages(D_,40)
    if False:
        D_priv, D_public = split_data_ages(D_,40)
    #D_priv, D_public = split_data_ages(D_,45)
    #D_priv, D_public = split_data_ages(D_,45)
    if False:
        for test_age in range(19,80):
            D_public_temp, D_priv_temp = split_data_ages(D_,test_age)
            
            D_public_temp_prop = generate_subpopulation(
                D_public_temp, categories=categories, target_attributes=target_attributes
            )
            D_priv_temp_prop = generate_subpopulation(
                D_priv_temp , categories=categories, target_attributes=target_attributes
            )
            label_split_pub = len(D_public_temp_prop) / len(D_public_temp)
            label_split_priv = len(D_priv_temp_prop) / len(D_priv_temp)
            #print(            f"For the age {test_age}, D_public has {len(D_public_temp)} total points with {label_split_pub*100:.4}% class 1 and D_priv has {len(D_priv_temp)} total points with {label_split_priv*100:.4}% class 1 "        )
            #print(                f"{test_age} {len(D_public_temp)} {label_split_pub*100:.4}% {len(D_priv_temp)} {label_split_priv*100:.4}%"            )
            print(f"{test_age} {label_split_pub*100:.4}% {label_split_priv*100:.4}%")
    
    #should be uncommented
    #D_priv = D_public
    #D_public = D_public[:5400]
    if False:
        D_public = D_priv
        #D_public = D_public[:5000]
        #print(len(D_public))

    if allow_custom_freq == True:
        D_priv = v2_fix_imbalance(
            D_priv,
            target_split=label_frequency,
            categories=["class"],
            target_attributes=[poison_class],
            random_seed=random_state,
        )
        D_public = v2_fix_imbalance(
            D_public,
            target_split=label_frequency,
            categories=["class"],
            target_attributes=[poison_class],
            random_seed=random_state,
        )

    if False:

        D_public = v2_fix_imbalance(
            D_public,
            target_split=sum(D_priv[D_priv.columns[-1]]) / len(D_priv),
            categories=["class"],
            target_attributes=[poison_class],
            random_seed=random_state,
        )

    
    if True:
        D_public = v2_fix_imbalance(
            D_public,
            target_split=tpub,
            categories=categories,
            target_attributes=target_attributes,
            random_seed=random_state,
        ) #W0
    
    D_aux = v2_fix_imbalance(
        D_public,
        target_split=taux,
        categories=categories,
        target_attributes=target_attributes,
        random_seed=random_state,
    ) #W0
        
    #D_public, D_shadow = split_data_ages(D_public,50)
    #_, D_shadow = split_data_ages(D_,50)
    #D_shadow,D_priv = split_data_ages(D_priv,50)
    #D_priv,D_shadow = split_data_ages(D_priv,50)
    D_shadow = D_public
    
    
    if verbose:
        label_split = sum(D_priv[D_priv.columns[-1]]) / len(D_)
        print(
            f"The model owner has {len(D_public)} total points with {label_split*100:.4}% class 1"
        )



    #print(D_priv,D_public)
    D0_priv = v2_fix_imbalance(
        D_priv,
        target_split=t1,
        categories=categories,
        target_attributes=target_attributes,
        random_seed=random_state,
    ) #W0
    print(f"##############{len(D0_priv)} {len(D_priv)}#############")
    D1_priv = v2_fix_imbalance(
        D_priv,
        target_split=t1,
        categories=categories,
        target_attributes=target_attributes,
        random_seed=random_state,
    ) #W1
    

    
    #D_public, D_shadow = train_test_split(D_public,test_size=0.7,random_state=random_state)
    D0_pub = v2_fix_imbalance(
        D_shadow,
        target_split=t0,
        categories=categories,
        target_attributes=target_attributes,
        random_seed=random_state,
    ) #W0

    D1_pub = v2_fix_imbalance(
        D_shadow,
        target_split=t1,
        categories=categories,
        target_attributes=target_attributes,
        random_seed=random_state,
    ) #W1
    print("####################")
    print(len(D0_pub),len(D1_pub),len(D_public))
    
    
    if len(D0_priv) > len(D1_priv):
        D0_priv = D0_priv.sample(n=len(D1_priv), random_state=random_state)
    elif len(D1_priv) > len(D0_priv):
        D1_priv = D1_priv.sample(n=len(D0_priv), random_state=random_state)




    if True:
        label_split_d0 = sum(D0_priv[D0_priv.columns[-1]]) / len(D0_priv)
        label_split_d1 = sum(D1_priv[D1_priv.columns[-1]]) / len(D1_priv)
        D0_subpop = generate_subpopulation(
            D0_priv, categories=categories, target_attributes=target_attributes
        )
        D1_subpop = generate_subpopulation(
            D1_priv, categories=categories, target_attributes=target_attributes
        )
        print(
            f"D0 has {len(D0_priv)} points with {len(D0_subpop)} members from the target subpopulation and {label_split_d0*100:.4}% class 1"
        )
        print(
            f"D1 has {len(D1_priv)} points with {len(D1_subpop)} members from the target subpopulation and {label_split_d1*100:.4}% class 1"
        )

    if poison_percent is not None:
        k = int(poison_percent * len(D0_priv))
    test_df, Dacc = train_test_split(test_df,test_size=0.5,random_state=random_state)
    
    if allow_custom_freq == True:
        Dacc = v2_fix_imbalance(
                    Dacc,
                    target_split=label_frequency,
                    categories=["class"],
                    target_attributes=[poison_class],
                    random_seed=random_state,
                )
    if subproperty_sampling == True:
        #Dp poisoned property dataset
        #Dtest property dataset for querying
        Dp, Dtest = generate_Dp(
            test_df,
            categories=categories + sub_categories,
            target_attributes=target_attributes + sub_attributes,
            k=k,
            poison_class=poison_class,
            random_state=random_state,
        )
    else:
        Dp, Dtest = generate_Dp(
            test_df,
            categories=categories,
            target_attributes=target_attributes,
            k=k,
            poison_class=poison_class,
            random_state=random_state,
        )

    if len(Dtest) == 0:
        Dtest = Dp.copy()

    if verbose:
        subpop = generate_subpopulation(
            test_df, categories=categories, target_attributes=target_attributes
        )
        print(
            f"The poisoned set has {k} points sampled uniformly from {len(subpop)} total points in the subpopulation"
        )
    if verbose or True:
        label_split_dp = sum(Dp[Dp.columns[-1]]) / len(Dp)
        label_split_d0 = sum(Dtest[Dtest.columns[-1]]) / len(Dtest)
        label_split_dtest_df = sum(test_df[test_df.columns[-1]]) / len(test_df)
        Dp_subpop = generate_subpopulation(
            Dp, categories=categories, target_attributes=target_attributes
        )
        D0_subpop = generate_subpopulation(
            Dtest, categories=categories, target_attributes=target_attributes
        )
        Dtest_df_subpop = generate_subpopulation(
            test_df, categories=categories, target_attributes=target_attributes
        )


        print(
            f"Dp has {len(Dp)} points with {len(Dp_subpop)} members from the target subpopulation and {label_split_dp*100:.4}% class 1"
        )
        print(
            f"Dtest has {len(Dtest)} points with {len(D0_subpop)} members from the target subpopulation and {label_split_d0*100:.4}% class 1"
        )
        print(
            f"Dtest_df has {len(test_df)} points with {len(Dtest_df_subpop)} members from the target subpopulation and {label_split_dtest_df*100:.4}% class 1"
        )
        
    if restrict_sampling == True:
        D0_subpop = generate_subpopulation(
            D0, categories=categories, target_attributes=target_attributes
        )
        D1_subpop = generate_subpopulation(
            D1, categories=categories, target_attributes=target_attributes
        )
        print(
            f"D0 has {len(D0)} points with {len(D0_subpop)} members from the target subpopulation"
        )
        print(
            f"D1 has {len(D1)} points with {len(D1_subpop)} members from the target subpopulation"
        )

        return D0_mo, D1_mo, D0, D1, Dp, Dtest, D0_subpop, D1_subpop

    return D_, D_priv, D0_priv, D1_priv, Dp, Dtest, Dacc, D_public, D0_pub, D1_pub, D_aux


def generate_Dp(
    test_df,
    categories=["race"],
    target_attributes=[" White"],
    poison_class=1,
    k=None,
    random_state=21,
):
    """Generate Dp, the poisoned dataset
    ...
    Parameters
    ----------
        test_df : Pandas Dataframe
            The validation set or some dataset that is disjoint from train set,
            but drawn from the same distribution
        categories : list
            Column names for each attribute
        target_attributes : list
            Labels to create subpopulation from.
            Ex. The subpopulation will be df[categories] == attributes
        poison_class : int
            The label we want our poisoned examples to have
        k : int
            The number of points in the poison set
        verbose : bool
            If True, reports dataset statistics via a print-out

    ...
    Returns
    -------
        Dp : Pandas Dataframe
            The adversary's poisoned set
        remaining_indices : np.array
            The remaining indices from test_df that we can use to query the target model.
            The indices correspond to points in the subpopulation that are *not* in the
            poisoned set
    """

    if k is None:
        raise NotImplementedError("Poison set size estimation not implemented")

    subpop = generate_subpopulation(
        test_df, categories=categories, target_attributes=target_attributes
    ).copy()
    
    label = subpop.columns[-1]
    np.random.seed(random_state)

    subpop_without_poison_label = subpop[subpop[label] != poison_class]

    all_indices = np.arange(0, len(subpop_without_poison_label), dtype=np.uint64)
    np.random.shuffle(all_indices)

    Dp = subpop_without_poison_label.iloc[all_indices[:k]]

    Dp.loc[:, label] = poison_class
    remaining_indices = all_indices[k:]

    Dtest = subpop_without_poison_label.iloc[remaining_indices]

    return Dp, Dtest
def fix_one_hot_vectors(df, cat_columns):
    """
    Fix the one-hot vectors in the dataframe by setting the maximum value in each
    categorical group to 1 and the rest to 0.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing one-hot encoded columns.
    cat_columns : list
        List of original categorical columns.

    Returns
    -------
    pd.DataFrame
        The dataframe with fixed one-hot vectors.
    """
    import pandas as pd
    import numpy as np

    for cat_col in cat_columns:
        # Find all one-hot columns that belong to this categorical column
        one_hot_cols = [col for col in df.columns if col.startswith(f"{cat_col}_")]
        if not one_hot_cols:
            continue  # No one-hot columns found for this categorical column

        # Ensure the one-hot columns are numeric
        df[one_hot_cols] = df[one_hot_cols].astype(float)

        # Identify the row-wise maximum value in the one-hot group
        max_values = df[one_hot_cols].max(axis=1)

        # Create a boolean mask where the column value equals the row-wise max
        mask = df[one_hot_cols].eq(max_values, axis=0)

        # To handle multiple max values, keep only the first occurrence
        mask = mask.mask(~mask.eq(mask.max(axis=1), axis=0), False)
        # Set the maximum value to 1 and others to 0
        df.loc[:, one_hot_cols] = mask.astype(int)

    return df

def reverse_all_dfs_from_one_hot(dataframes_OH, original_max_values,  cat_columns=[], original_columns_order=None,class_label=None, original_dtypes=None):
    import pandas as pd
    import numpy as np

    # Combine dataframes with unique keys to reconstruct the combined dataframe
    keys = list(range(len(dataframes_OH)))
    combined_df = pd.concat(dataframes_OH, keys=keys)

    # Reorder columns to match 'all_columns'
    label_df = combined_df[class_label]
    
    # Denormalize continuous columns
    cont_columns = [col for col in original_max_values.keys()]
    for col in cont_columns:
        max_value = original_max_values[col]
        combined_df[col] *= max_value
        # 데이터 타입 복원
        if original_dtypes and col in original_dtypes:
            combined_df[col] = combined_df[col].astype(original_dtypes[col])

    
        

    # Reverse one-hot encoding for categorical columns
    for cat_col in cat_columns:
        # Find all one-hot columns that belong to this categorical column
        one_hot_cols = [col for col in combined_df.columns if col.startswith(f"{cat_col}_")]
        if not one_hot_cols:
            continue  # No one-hot columns found for this categorical column

        # Ensure the one-hot columns are numeric
        combined_df[one_hot_cols] = combined_df[one_hot_cols].astype(float)

        # Use idxmax to find the category with the maximum value (should be 1)
        combined_df[cat_col] = combined_df[one_hot_cols].idxmax(axis=1)
        # Remove the prefix and underscore to get the original category
        combined_df[cat_col] = combined_df[cat_col].apply(lambda x: x[len(cat_col)+1:] if pd.notnull(x) else x)

        # Drop the one-hot columns
        combined_df = combined_df.drop(columns=one_hot_cols)

        # 데이터 타입 복원
        if original_dtypes and cat_col in original_dtypes:
            combined_df[cat_col] = combined_df[cat_col].astype(original_dtypes[cat_col])

    if original_columns_order:
        combined_df = combined_df[original_columns_order]
    
    # Split back into original dataframes
    dataframes = []
    for key in keys:
        df = combined_df.xs(key)
        df = df.reset_index(drop=True)
        dataframes.append(df)

    return dataframes


def all_dfs_to_one_hot(dataframes, cat_columns=[], class_label=None,original_max_values=None,all_columns=None):
    import pandas as pd

    keys = list(range(len(dataframes)))

    # Copy dataframes to avoid modifying originals
    dataframes = [df.copy() for df in dataframes]

    # Save original columns order
    original_columns_order = dataframes[0].columns.tolist()

    # Continuous columns are those not in cat_columns or class_label
    cont_columns = sorted(
        list(set(dataframes[0].columns).difference(cat_columns + ([class_label] if class_label else [])))
    )

    # Save max values of continuous columns for normalization across all dataframes
    if original_max_values == None:
        original_max_values = {}
        for col in cont_columns:
            max_value = max(df[col].max() for df in dataframes)
            original_max_values[col] = max_value

    # Concatenate dataframes with unique keys
    combined_df = pd.concat(dataframes, keys=keys)

    label_df = combined_df[class_label]
    # Perform one-hot encoding
    combined_df = pd.get_dummies(combined_df, columns=cat_columns)

    # Normalize continuous columns
    for col in cont_columns:
        max_value = original_max_values[col]
        if max_value != 0:
            combined_df[col] /= max_value
        else:
            combined_df[col] = 0

    # If a class label is provided, move it to a separate column
    if class_label:
        combined_df = combined_df.drop(columns=[class_label])
    combined_df["class"] = label_df
    

    if all_columns is None:
        all_columns = combined_df.columns.tolist()
    else:
        # Align columns to the given all_columns
        for col in all_columns:
            if col not in combined_df.columns:
                combined_df[col] = 0  # Add missing columns with default values
        combined_df = combined_df[all_columns]  # Reorder columns to match all_columns
        
    # Split back into original dataframes
    dataframes_OH = [combined_df.xs(i) for i in keys]

    return dataframes_OH, original_max_values, all_columns, original_columns_order




def load_adult(one_hot=True, custom_balance=False, target_class=1, target_ratio=0.3, filename_test = "dataset/adult.test"):
    """Load the Adult dataset."""

    filename_train = "dataset/adult.data"
    filename_test = filename_test #"dataset/adult.test"
    names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "class",
    ]
    df_tr = pd.read_csv(filename_train, names=names)
    df_ts = pd.read_csv(filename_test, names=names, skiprows=1)

    df_tr.drop(["fnlwgt", "education"], axis=1, inplace=True)
    df_ts.drop(["fnlwgt", "education"], axis=1, inplace=True)

    # Separate Labels from inputs
    df_tr["class"] = df_tr["class"].astype("category")
    cat_columns = df_tr.select_dtypes(["category"]).columns
    df_tr[cat_columns] = df_tr[cat_columns].apply(lambda x: x.cat.codes)
    df_ts["class"] = df_ts["class"].astype("category")
    cat_columns = df_ts.select_dtypes(["category"]).columns
    df_ts[cat_columns] = df_ts[cat_columns].apply(lambda x: x.cat.codes)

    cont_cols = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    cat_cols = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]  # Indexs: {1,3,5,6,7,8,9,13}
    df = pd.concat([df_tr, df_ts])
    for col in cat_cols:
        df[col] = df[col].astype("category")

    if custom_balance == True:
        df_tr = generate_class_imbalance(
            data=df_tr, target_class=target_class, target_ratio=target_ratio
        )
        df_ts = generate_class_imbalance(
            data=df_ts, target_class=target_class, target_ratio=target_ratio
        )

    if one_hot == False:
        return df_tr, df_ts

    else:

        df_one_hot = pd.get_dummies(df)
        df_one_hot["labels"] = df_one_hot["class"]
        df_one_hot = df_one_hot.drop(["class"], axis=1)

        # Normalizing continuous coloumns between 0 and 1
        df_one_hot[cont_cols] = df_one_hot[cont_cols] / (df_one_hot[cont_cols].max())
        df_one_hot[cont_cols] = df_one_hot[cont_cols].round(3)
        #         df_one_hot.loc[:, df_one_hot.columns != cont_cols] = df_one_hot.loc[:, df_one_hot.columns != cont_cols].astype(int)

        df_tr_one_hot = df_one_hot[: len(df_tr)]
        df_ts_one_hot = df_one_hot[len(df_tr) :]

        return df_tr, df_ts, df_tr_one_hot, df_ts_one_hot


def load_data(data_string, one_hot=False):
    """Load data given the name of the dataset

    ...

    Parameters
    ----------
        data_string : str
            The string that corresponds to the desired dataset.
            Options are {mnist, fashion, adult, census}
        one_hot : bool
            Indicates whether one-hot versions of the data should be loaded.
            The one-hot dataframes also have normalized continuous values
    """

    if data_string.lower() == "adult":
        return load_adult(one_hot)

    else:
        print("Enter valid data_string")
