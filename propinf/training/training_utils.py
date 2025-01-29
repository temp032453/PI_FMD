import time
import torch
import numpy as np
import pandas as pd

from torch import nn
from sklearn import metrics
import torch.optim as optim
import torch.nn.functional as F
import os
import re
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport
from eval.eval_density import reorder
class OneHotHandler:
    def __init__(self, cat_columns, cont_columns, original_max_values, all_columns, original_columns_order):
        self.cat_columns = cat_columns
        self.cont_columns = cont_columns
        self.original_max_values = original_max_values
        self.all_columns = all_columns
        self.original_columns_order = original_columns_order

    def encode(self, tensor):
        """
        Encode the input torch tensor into one-hot encoded tensor
        """
        # Convert tensor to DataFrame using original column order
        df = pd.DataFrame(tensor.numpy(), columns=self.original_columns_order)

        # Separate label column if exists
        if self.class_label:
            label_df = df[self.class_label]
            df = df.drop(columns=[self.class_label])
        else:
            label_df = None

        # One-hot encode categorical columns
        df = pd.get_dummies(df, columns=self.cat_columns)

        # Normalize continuous columns
        for col in self.cont_columns:
            max_value = self.original_max_values[col]
            if max_value != 0:
                df[col] /= max_value
            else:
                df[col] = 0

        # Add missing columns and reorder to match all_columns
        for col in self.all_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.all_columns]

        # Convert DataFrame back to tensor
        return torch.tensor(df.values, dtype=torch.float32)

    def decode(self, encoded_tensor):
        """
        Decode the one-hot encoded torch tensor back to original format
        """
        # Convert tensor to DataFrame using all_columns
        df = pd.DataFrame(encoded_tensor.numpy(), columns=self.all_columns)

        # Reverse normalization for continuous columns
        for col in self.cont_columns:
            max_value = self.original_max_values[col]
            df[col] *= max_value

        # Reverse one-hot encoding for categorical columns
        for col in self.cat_columns:
            one_hot_cols = [c for c in df.columns if c.startswith(col + "_")]
            df[col] = df[one_hot_cols].idxmax(axis=1).apply(lambda x: x.split("_")[1] if "_" in x else x)
            df = df.drop(columns=one_hot_cols)

        # Reorder columns to match original_columns_order
        df = df[self.original_columns_order]

        # Convert DataFrame back to tensor
        return torch.tensor(df.values, dtype=torch.float32)
def sanitize_filename(filename, replacement="_"):
    """
    파일 이름에서 유효하지 않은 문자 제거 및 변환.

    :param filename: 원래 파일 이름
    :param replacement: 유효하지 않은 문자를 대체할 문자
    :return: 변환된 안전한 파일 이름
    """
    # 정규식을 사용하여 파일 이름에서 유효하지 않은 문자 제거
    sanitized = re.sub(r"[^\w\.-]", replacement, filename)
    # 여러 개의 연속된 대체 문자를 하나로 축소
    sanitized = re.sub(rf"{re.escape(replacement)}+", replacement, sanitized)
    return sanitized.strip(replacement)
def check_eval_density(temp_targets, syn_data,tabsyn_info):
        shapes_scores = []
        trends_scores = []
        for loc_i , real_data in enumerate(temp_targets):
            start_time = time.time()
            real_data.columns = range(len(real_data.columns))
            syn_data.columns = range(len(syn_data.columns))

            metadata = tabsyn_info['metadata']
            metadata['columns'] = {int(key): value for key, value in metadata['columns'].items()}
                
            new_real_data, new_syn_data, metadata = reorder(real_data.copy(), syn_data.copy(), tabsyn_info)
                
            qual_report = QualityReport()
            qual_report.generate(new_real_data, new_syn_data, metadata, verbose=False)
            quality = qual_report.get_properties()
            
            shapes_scores.append(quality['Score'][0].round(3))
            trends_scores.append(quality['Score'][1].round(3))
            del qual_report
            end_time = time.time()
            
            # print(f"Time: {end_time - start_time}")
        return shapes_scores, trends_scores
def load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, filepath, device):
    """
    저장된 체크포인트에서 모델과 옵티마이저의 상태를 로드합니다.

    Parameters:
    ----------
    generator : nn.Module
        Generator 모델.
    discriminator : nn.Module
        Discriminator 모델.
    optimizer_G : torch.optim.Optimizer
        Generator 옵티마이저.
    optimizer_D : torch.optim.Optimizer
        Discriminator 옵티마이저.
    filepath : str
        로드할 체크포인트 파일 경로.
    device : torch.device
        모델을 로드할 디바이스.

    Returns:
    -------
    epoch : int
        마지막 저장된 에포크 번호.
    loss : float
        마지막 저장된 손실 값.
    """
    checkpoint = torch.load(filepath, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    return 
def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D,  filepath):
    """
    모델과 옵티마이저의 상태를 저장합니다.

    Parameters:
    ----------
    generator : nn.Module
        Generator 모델.
    discriminator : nn.Module
        Discriminator 모델.
    optimizer_G : torch.optim.Optimizer
        Generator 옵티마이저.
    optimizer_D : torch.optim.Optimizer
        Discriminator 옵티마이저.
    epoch : int
        현재 에포크 번호.
    loss : float
        현재 손실 값.
    filepath : str
        저장할 파일 경로.
    """
    checkpoint = {
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at '{filepath}'")
class PoisonDataset(torch.utils.data.Dataset):
    def __init__(self, X,y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        
        return self.X[i], self.y[i], i
def torch_dataset_to_dataframe(dataset, column_info=None):
    """
    Convert a PyTorch Dataset back to a pandas DataFrame with optional column names.
    
    Parameters:
        dataset (torch.utils.data.TensorDataset): The PyTorch Dataset to convert.
        column_info (dict): Optional dictionary containing column names:
            - feature_columns: List of feature column names.
            - label_column: Name of the label column.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the features and labels.
    """
    # Extract features and labels from the TensorDataset
    data, labels = dataset[:]
    
    # Convert tensors to numpy arrays for DataFrame creation
    data_np = data.numpy()
    labels_np = labels.numpy()
    labels_np1 = (1-labels).numpy()
    # Check for provided column names
    if column_info:
        feature_columns = column_info[:-2]
    else:
        # Default feature column names if not provided
        feature_columns = [f"feature_{i}" for i in range(data_np.shape[1])]
    
    label_column1 = column_info[-2]
    label_column = column_info[-1]
    
    # Create a DataFrame for features
    df = pd.DataFrame(data_np, columns=feature_columns)
    
    # Add the label column
    df[label_column1] = labels_np1
    df[label_column] = labels_np
    
    return df

def dataframe_to_torch_dataset(dataframe, using_ce_loss=False, class_label=None):
    """Convert a one-hot pandas dataframe to a PyTorch Dataset of Tensor objects"""

    new = dataframe.copy()
    if class_label:
        label = class_label
    else:
        
        label = list(new.columns)[-1]
        label_onehot = list(new.columns)[-2]
        # print(f"Inferred that class label is '{label}' while creating dataloader")
    labels = torch.Tensor(pd.DataFrame(new[label]).values.astype(float))
    del new[label]
    del new[label_onehot]
    new = new.astype(float)
    
    data = torch.from_numpy(new.values).float()

    if using_ce_loss:
        # Fixes tensor dimension and float -> int if using cross entropy loss
        return torch.utils.data.TensorDataset(
            data, labels.squeeze().type(torch.LongTensor)
        )
    else:
        return torch.utils.data.TensorDataset(data, labels)


def dataset_to_dataloader(
    dataset, batch_size=256, num_workers=4, shuffle=True, persistent_workers=False,pin_memory=None
):
    """Wrap PyTorch dataset in a Dataloader (to allow batch computations)"""
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        persistent_workers=persistent_workers,
        pin_memory = pin_memory,
    )
    return loader

def extract_data_from_dataloader(dataloader):
    """
    Extracts tensors from a DataLoader object and combines them into a single tensor.
    """
    X_list, y_list = [], []
    
    for batch in dataloader:
        X_batch, y_batch = batch
        X_list.append(X_batch)
        y_list.append(y_batch)
    
    # Combine the batches into single tensors
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    
    return X, y
def dataframe_to_dataloader(
    dataframe,
    batch_size=256,
    num_workers=4,
    shuffle=True,
    persistent_workers=False,
    using_ce_loss=False,
    class_label=None,
    pin_memory =None,
):
    if not isinstance(dataframe, torch.utils.data.Dataset):
        """Convert a pandas dataframe to a PyTorch Dataloader"""
        dataset = dataframe_to_torch_dataset(
        dataframe, using_ce_loss=using_ce_loss, class_label=class_label
        )
    else:
        dataset = dataframe
    return dataset_to_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
def fit_wo_prop(
    dataloaders,
    model,
    epochs=100,
    optim_init=optim.Adam,
    optim_kwargs={"lr": 0.03, "weight_decay": 0.0001},
    criterion=nn.CrossEntropyLoss(),
    device="cpu",
    verbose=True,
    mini_verbose=True,
    train_only=True,
    early_stopping=False,
    tol=10e-6,
    prop_idx=None,
):
    """Fits a PyTorch model to any given dataset

    ...
    Parameters
    ----------
        dataloaders : dict
            Dictionary containing 2 PyTorch DataLoaders with keys "train" and
            "test" corresponding to the two dataloaders as values
        model : torch.nn.Module (PyTorch Neural Network Model)
            The desired model to fit to the data
        epochs : int
            Training epochs for shadow models
        optim_init : torch.optim init object
            The init function (as an object) for a PyTorch optimizer.
            Note: Pass in the function without ()
        optim_kwargs : dict
            Dictionary of keyword arguments for the optimizer
            init function
        criterion : torch.nn Loss Function
            Loss function used to train model
        device : str
            The processing device for training computations.
            Ex: cuda, cpu, cuda:1
        verbose : bool
            If True, prints running loss and accuracy values
        train_only : bool
            If True, only uses "train" dataloader

    ...
    Returns
    -------
        model : torch.nn.Module (PyTorch Neural Network Model)
            The trained model
        train_error : list
            List of average training errors at each training epoch
        test_acc : list
            List of average test accuracy at each training epoch
    """

    model = model.to(device)
    model = model.train()
    optimizer = optim_init(model.parameters(), **optim_kwargs)
    train_error = []
    test_loss = []
    test_acc = []
    # if train_only:
    #     phases = ["train"]
    # else:
    #     phases = ["train", "test"]
    if mini_verbose:
        print("Training...")
    if verbose:
        print("-" * 8)

    try:

        for epoch in range(1, epochs + 1):
            if verbose:
                a = time.time()
                print(f"Epoch {epoch}")

            running_train_loss = 0

            for (inputs, labels) in dataloaders:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                
                inputs[:,prop_idx] = 0
                    
                labels = labels.to(device)
                outputs = model.forward(inputs)
                
                loss = criterion(outputs, labels)
                
      
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)

            #print(F.softmax(outputs)[0])
            train_error.append(running_train_loss / len(dataloaders.dataset))

            if len(train_error) > 1 and early_stopping:
                if abs(train_error[-1] - train_error[-2]) < tol:
                    print(f"Loss did not decrease by more than {tol}")
                    if mini_verbose:
                        print(f"Final Train Error: {train_error[-1]:.6}")
                    if not train_only:
                        return model, train_error, test_loss, test_acc
                    else:
                        return model, train_error

            if verbose:
                b = time.time()
                print(f"Train Error: {train_error[-1]:.6}")
                print(f"Time Elapsed: {b - a:.4} seconds")
                print("-" * 8)
    except KeyboardInterrupt:
        if mini_verbose:
            print(f"Final Train Error: {train_error[-1]:.6}")
        if not train_only:
            return model, train_error, test_loss, test_acc
        else:
            return model, train_error

    if mini_verbose:
        print(f"Final Train Error: {train_error[-1]:.6}")
    if not train_only:
        return model, train_error, test_loss, test_acc
    else:
        return model, train_error
    
def fit_local_noise(
    dataloaders,
    model,
    epochs=100,
    optim_init=optim.Adam,
    optim_kwargs={"lr": 0.03, "weight_decay": 0.0001},
    criterion=nn.CrossEntropyLoss(),
    device="cpu",
    verbose=True,
    mini_verbose=True,
    train_only=True,
    early_stopping=False,
    tol=10e-6,
    noise_std=1.0,
):
    """Fits a PyTorch model to any given dataset

    ...
    Parameters
    ----------
        dataloaders : dict
            Dictionary containing 2 PyTorch DataLoaders with keys "train" and
            "test" corresponding to the two dataloaders as values
        model : torch.nn.Module (PyTorch Neural Network Model)
            The desired model to fit to the data
        epochs : int
            Training epochs for shadow models
        optim_init : torch.optim init object
            The init function (as an object) for a PyTorch optimizer.
            Note: Pass in the function without ()
        optim_kwargs : dict
            Dictionary of keyword arguments for the optimizer
            init function
        criterion : torch.nn Loss Function
            Loss function used to train model
        device : str
            The processing device for training computations.
            Ex: cuda, cpu, cuda:1
        verbose : bool
            If True, prints running loss and accuracy values
        train_only : bool
            If True, only uses "train" dataloader

    ...
    Returns
    -------
        model : torch.nn.Module (PyTorch Neural Network Model)
            The trained model
        train_error : list
            List of average training errors at each training epoch
        test_acc : list
            List of average test accuracy at each training epoch
    """

    model = model.to(device)
    model = model.train()
    optimizer = optim_init
    train_error = []
    test_loss = []
    test_acc = []
    # if train_only:
    #     phases = ["train"]
    # else:
    #     phases = ["train", "test"]
    if mini_verbose:
        print("Training...")
    if verbose:
        print("-" * 8)

    try:

        for epoch in range(1, epochs + 1):
            if verbose:
                a = time.time()
                print(f"Epoch {epoch}")

            running_train_loss = 0

            for (inputs, labels) in dataloaders:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                noise = torch.randn_like(inputs) * noise_std
                inputs_noisy = inputs + noise
                
                
                outputs = model.forward(inputs_noisy)
                
                
                
                loss = criterion(outputs, labels)
                
      
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)

            #print(F.softmax(outputs)[0])
            train_error.append(running_train_loss / len(dataloaders.dataset))

            if len(train_error) > 1 and early_stopping and False:
                if abs(train_error[-1] - train_error[-2]) < tol:
                    print(f"Loss did not decrease by more than {tol}")
                    if mini_verbose:
                        print(f"Final Train Error: {train_error[-1]:.6}")
                    if not train_only:
                        return model, train_error, test_loss, test_acc
                    else:
                        return model, train_error

            if verbose:
                b = time.time()
                print(f"Train Error: {train_error[-1]:.6}")
                print(f"Time Elapsed: {b - a:.4} seconds")
                print("-" * 8)
    except KeyboardInterrupt:
        if mini_verbose:
            print(f"Final Train Error: {train_error[-1]:.6}")
        if not train_only:
            return model, train_error, test_loss, test_acc
        else:
            return model, train_error

    if mini_verbose:
        print(f"Final Train Error: {train_error[-1]:.6}")
    if not train_only:
        return model, train_error, test_loss, test_acc
    else:
        return model, train_error
    
    
def fit_local(
    dataloaders,
    model,
    epochs=100,
    optim_init=optim.Adam,
    optim_kwargs={"lr": 0.03, "weight_decay": 0.0001},
    criterion=nn.CrossEntropyLoss(),
    device="cpu",
    verbose=True,
    mini_verbose=True,
    train_only=True,
    early_stopping=False,
    tol=10e-6,
    handler=None,
):
    """Fits a PyTorch model to any given dataset

    ...
    Parameters
    ----------
        dataloaders : dict
            Dictionary containing 2 PyTorch DataLoaders with keys "train" and
            "test" corresponding to the two dataloaders as values
        model : torch.nn.Module (PyTorch Neural Network Model)
            The desired model to fit to the data
        epochs : int
            Training epochs for shadow models
        optim_init : torch.optim init object
            The init function (as an object) for a PyTorch optimizer.
            Note: Pass in the function without ()
        optim_kwargs : dict
            Dictionary of keyword arguments for the optimizer
            init function
        criterion : torch.nn Loss Function
            Loss function used to train model
        device : str
            The processing device for training computations.
            Ex: cuda, cpu, cuda:1
        verbose : bool
            If True, prints running loss and accuracy values
        train_only : bool
            If True, only uses "train" dataloader

    ...
    Returns
    -------
        model : torch.nn.Module (PyTorch Neural Network Model)
            The trained model
        train_error : list
            List of average training errors at each training epoch
        test_acc : list
            List of average test accuracy at each training epoch
    """

    model = model.to(device)
    model = model.train()
    optimizer = optim_init
    train_error = []
    test_loss = []
    test_acc = []
    # if train_only:
    #     phases = ["train"]
    # else:
    #     phases = ["train", "test"]
    if mini_verbose:
        print("Training...")
    if verbose:
        print("-" * 8)

    try:

        for epoch in range(1, epochs + 1):
            if verbose:
                a = time.time()
                print(f"Epoch {epoch}")

            running_train_loss = 0

            for (inputs, labels) in dataloaders:
                
                optimizer.zero_grad()
                if handler:
                    inputs = handler.encode(inputs).to(device)
                else:
                    inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model.forward(inputs)
                
                
                loss = criterion(outputs, labels)
                
      
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)

            #print(F.softmax(outputs)[0])
            train_error.append(running_train_loss / len(dataloaders.dataset))

            if len(train_error) > 1 and early_stopping and False:
                if abs(train_error[-1] - train_error[-2]) < tol:
                    print(f"Loss did not decrease by more than {tol}")
                    if mini_verbose:
                        print(f"Final Train Error: {train_error[-1]:.6}")
                    if not train_only:
                        return model, train_error, test_loss, test_acc
                    else:
                        return model, train_error

            if verbose:
                b = time.time()
                print(f"Train Error: {train_error[-1]:.6}")
                print(f"Time Elapsed: {b - a:.4} seconds")
                print("-" * 8)
    except KeyboardInterrupt:
        if mini_verbose:
            print(f"Final Train Error: {train_error[-1]:.6}")
        if not train_only:
            return model, train_error, test_loss, test_acc
        else:
            return model, train_error

    if mini_verbose:
        print(f"Final Train Error: {train_error[-1]:.6}")
    if not train_only:
        return model, train_error, test_loss, test_acc
    else:
        return model, train_error
    
def fit_kd_local(
    dataloaders,
    model,
    epochs=100,
    optim_init=optim.Adam,
    optim_kwargs={"lr": 0.03, "weight_decay": 0.0001},
    criterion=nn.CrossEntropyLoss(),
    device="cpu",
    verbose=True,
    mini_verbose=True,
    train_only=True,
    early_stopping=False,
    tol=10e-6,
    L1_loss=False,
):
    """Fits a PyTorch model to any given dataset

    ...
    Parameters
    ----------
        dataloaders : dict
            Dictionary containing 2 PyTorch DataLoaders with keys "train" and
            "test" corresponding to the two dataloaders as values
        model : torch.nn.Module (PyTorch Neural Network Model)
            The desired model to fit to the data
        epochs : int
            Training epochs for shadow models
        optim_init : torch.optim init object
            The init function (as an object) for a PyTorch optimizer.
            Note: Pass in the function without ()
        optim_kwargs : dict
            Dictionary of keyword arguments for the optimizer
            init function
        criterion : torch.nn Loss Function
            Loss function used to train model
        device : str
            The processing device for training computations.
            Ex: cuda, cpu, cuda:1
        verbose : bool
            If True, prints running loss and accuracy values
        train_only : bool
            If True, only uses "train" dataloader

    ...
    Returns
    -------
        model : torch.nn.Module (PyTorch Neural Network Model)
            The trained model
        train_error : list
            List of average training errors at each training epoch
        test_acc : list
            List of average test accuracy at each training epoch
    """

    model = model.to(device)
    model = model.train()
    optimizer = optim_init
    train_error = []
    test_loss = []
    test_acc = []
    # if train_only:
    #     phases = ["train"]
    # else:
    #     phases = ["train", "test"]
    if mini_verbose:
        print("Training...")
    if verbose:
        print("-" * 8)

    try:
        criterion = nn.KLDivLoss(reduction="batchmean")
        for epoch in range(1, epochs + 1):
            if verbose:
                a = time.time()
                print(f"Epoch {epoch}")

            running_train_loss = 0

            for (inputs, labels) in dataloaders:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model.forward(inputs)
                
                if L1_loss:
                    loss = F.l1_loss(outputs, labels)
                else:
                    loss = criterion(F.log_softmax(outputs,dim=1), F.softmax(labels,dim=1))

                
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)

            #print(F.softmax(outputs)[0])
            train_error.append(running_train_loss / len(dataloaders.dataset))

            if len(train_error) > 1 and early_stopping and False:
                if abs(train_error[-1] - train_error[-2]) < tol:
                    print(f"Loss did not decrease by more than {tol}")
                    if mini_verbose:
                        print(f"Final Train Error: {train_error[-1]:.6}")
                    if not train_only:
                        return model, train_error, test_loss, test_acc
                    else:
                        return model, train_error

            if verbose:
                b = time.time()
                print(f"Train Error: {train_error[-1]:.6}")
                print(f"Time Elapsed: {b - a:.4} seconds")
                print("-" * 8)
    except KeyboardInterrupt:
        if mini_verbose:
            print(f"Final Train Error: {train_error[-1]:.6}")
        if not train_only:
            return model, train_error, test_loss, test_acc
        else:
            return model, train_error

    if mini_verbose:
        print(f"Final Train Error: {train_error[-1]:.6}")
    if not train_only:
        return model, train_error, test_loss, test_acc
    else:
        return model, train_error
def fit_extraction(
    dataloaders,
    model,
    epochs=100,
    optim_init=optim.Adam,
    optim_kwargs={"lr": 0.03, "weight_decay": 0.0001},
    criterion=nn.CrossEntropyLoss(),
    device="cpu",
    verbose=True,
    mini_verbose=True,
    train_only=True,
    early_stopping=False,
    tol=10e-6,
):
    """Fits a PyTorch model to any given dataset

    ...
    Parameters
    ----------
        dataloaders : dict
            Dictionary containing 2 PyTorch DataLoaders with keys "train" and
            "test" corresponding to the two dataloaders as values
        model : torch.nn.Module (PyTorch Neural Network Model)
            The desired model to fit to the data
        epochs : int
            Training epochs for shadow models
        optim_init : torch.optim init object
            The init function (as an object) for a PyTorch optimizer.
            Note: Pass in the function without ()
        optim_kwargs : dict
            Dictionary of keyword arguments for the optimizer
            init function
        criterion : torch.nn Loss Function
            Loss function used to train model
        device : str
            The processing device for training computations.
            Ex: cuda, cpu, cuda:1
        verbose : bool
            If True, prints running loss and accuracy values
        train_only : bool
            If True, only uses "train" dataloader

    ...
    Returns
    -------
        model : torch.nn.Module (PyTorch Neural Network Model)
            The trained model
        train_error : list
            List of average training errors at each training epoch
        test_acc : list
            List of average test accuracy at each training epoch
    """

    model = model.to(device)
    model = model.train()
    optimizer = optim.SGD(model.parameters(),lr=0.01)
    train_error = []
    test_loss = []
    test_acc = []
    # if train_only:
    #     phases = ["train"]
    # else:
    #     phases = ["train", "test"]
    if mini_verbose:
        print("Training...")
    if verbose:
        print("-" * 8)

    try:
        criterion = nn.MSELoss(reduction='mean')
        for epoch in range(1, epochs + 1):
            if verbose:
                a = time.time()
                print(f"Epoch {epoch}")

            running_train_loss = 0

            for (inputs, labels) in dataloaders:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model.forward(inputs)
                
                loss = criterion(outputs, labels)

                
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)
            print("###############")
            print(loss)
            print(outputs[0].data,labels[0].data)
            print(F.softmax(outputs)[0])
            train_error.append(running_train_loss / len(dataloaders.dataset))

            if len(train_error) > 1 and early_stopping and False:
                if abs(train_error[-1] - train_error[-2]) < tol:
                    print(f"Loss did not decrease by more than {tol}")
                    if mini_verbose:
                        print(f"Final Train Error: {train_error[-1]:.6}")
                    if not train_only:
                        return model, train_error, test_loss, test_acc
                    else:
                        return model, train_error

            if verbose:
                b = time.time()
                print(f"Train Error: {train_error[-1]:.6}")
                print(f"Time Elapsed: {b - a:.4} seconds")
                print("-" * 8)
    except KeyboardInterrupt:
        if mini_verbose:
            print(f"Final Train Error: {train_error[-1]:.6}")
        if not train_only:
            return model, train_error, test_loss, test_acc
        else:
            return model, train_error

    if mini_verbose:
        print(f"Final Train Error: {train_error[-1]:.6}")
    if not train_only:
        return model, train_error, test_loss, test_acc
    else:
        return model, train_error
    
def test(
    dataloaders,
    model,
    epochs=100,
    optim_init=optim.Adam,
    optim_kwargs={"lr": 0.03, "weight_decay": 0.0001},
    criterion=nn.CrossEntropyLoss(),
    device="cpu",
    verbose=True,
    mini_verbose=True,
    train_only=True,
    early_stopping=False,
    tol=10e-6,
):
    """Tests a PyTorch model to any given dataset

    ...
    Parameters
    ----------
        dataloaders : dict
            Dictionary containing 2 PyTorch DataLoaders with keys "train" and
            "test" corresponding to the two dataloaders as values
        model : torch.nn.Module (PyTorch Neural Network Model)
            The desired model to fit to the data
        epochs : int
            Training epochs for shadow models
        optim_init : torch.optim init object
            The init function (as an object) for a PyTorch optimizer.
            Note: Pass in the function without ()
        optim_kwargs : dict
            Dictionary of keyword arguments for the optimizer
            init function
        criterion : torch.nn Loss Function
            Loss function used to train model
        device : str
            The processing device for training computations.
            Ex: cuda, cpu, cuda:1
        verbose : bool
            If True, prints running loss and accuracy values
        train_only : bool
            If True, only uses "train" dataloader

    ...
    Returns
    -------
        model : torch.nn.Module (PyTorch Neural Network Model)
            The trained model
        train_error : list
            List of average training errors at each training epoch
        test_acc : list
            List of average test accuracy at each training epoch
    """

    model = model.to(device)
    model = model.eval()
    test_loss = []
    test_acc = []
    # if train_only:
    #     phases = ["train"]
    # else:
    #     phases = ["train", "test"]
    if mini_verbose:
        print("Testing...")
    if verbose:
        print("-" * 8)

    try:
        running_test_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for (inputs, labels) in dataloaders:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                running_test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_test_loss = running_test_loss / len(dataloaders.dataset)
            epoch_test_acc = correct / total
            test_loss.append(epoch_test_loss)
            test_acc.append(epoch_test_acc)

        if verbose:
            b = time.time()
            print(f"Test Loss: {epoch_test_loss:.6f}")
            print(f"Test Accuracy: {epoch_test_acc:.6f}")
            print("-" * 8)
    except KeyboardInterrupt:
        if mini_verbose:
            print(f"Final Train Error: {epoch_test_loss[-1]:.6}")
    model = model.train()
    return epoch_test_acc
def get_metrics(y_true, y_pred):
    """Takes in a test dataloader + a trained model and returns a numpy array of predictions
    ...
    Parameters
    ----------
        y_true: Ground Truth Predictions

        y_pred: Model Predictions

    ...
    Returns
    -------
        Accuracy, Precision, Recall, F1 score
    """
    acc = metrics.accuracy_score(y_true, y_pred)
    # precision = metrics.precision_score(y_true, y_pred)
    # recall = metrics.recall_score(y_true, y_pred)
    # f1 = metrics.f1_score(y_true, y_pred)

    # return acc, precision, recall, f1
    return acc

def get_prediction(test_loader, model, one_hot=False, ground_truth=False, device="cpu"):
    """Takes in a test dataloader + a trained model and returns a numpy array of predictions
    ...
    Parameters
    ----------
        test_loader : PyTorch Dataloader
            The Pytorch Dataloader for Dtest

        model : torch.nn.Module (PyTorch Neural Network Model)
            A trained model to be queried on the test data

        one_hot: bool
            If true, returns predictions in one-hot format
            else, returns predictions in integer format

        device : str
            The processing device for training computations.
            Ex: cuda, cpu, cuda:0

    ...
    Returns
    -------
        attackdata_arr : np.array
            Numpy array of predictions
    """
    model = model.to(device)
    y_pred_torch = torch.Tensor([])
    y_true_torch = torch.Tensor([])

    for d, l in test_loader:
        d = d.to(device)
        l = l.squeeze()
        model.eval()
        # with torch.no_grad():
        out = nn.Sigmoid()(model(d))
        # out_np = out.cpu().detach().numpy()
        # y_pred = np.r_[y_pred,out_np]
        y_pred_torch = torch.concat([torch.argmax(out, dim=1).cpu(), y_pred_torch])
        y_true_torch = torch.concat([l.cpu(), y_true_torch])

    y_pred = y_pred_torch.cpu().detach().numpy()
    y_true = y_true_torch.cpu().detach().numpy()

    # y_pred = np.argmax(y_pred,axis=1)

    if one_hot:
        y_pred = np.eye(model._num_classes)[y_pred]

    if ground_truth == True:
        return y_pred, y_true

    return y_pred
def get_gaps_torch(
    test_loader, model_A, model_B, device="cpu", middle_measure="mean", variance_adjustment=1, max_conf = 1 - 1e-16, min_conf = 0 + 1e-16 , label = None
):
    n_samples = len(test_loader.dataset)
    kl_arr = []
    
    criterion = nn.KLDivLoss(reduction="none")
    model_A = model_A.to(device)
    model_B = model_B.to(device)
    
    y_prob = torch.Tensor([])
    y_test = torch.Tensor([])
    
    for d, l in test_loader:
        d = d.to(device)
        #model_A.eval()
        #model_B.eval() 
        with torch.no_grad():
            out_A  = model_A(d)
            out_B  = model_B(d) 
            # Get class probabilities
            loss = criterion(F.log_softmax(out_B,dim=1), F.softmax(out_A,dim=1)).sum(dim=1)
            kl_arr.append(loss.cpu())
            
    kl_arr= np.array(kl_arr)
            
    return kl_arr
def get_logits_torch(
    test_loader, model, device="cpu", middle_measure="mean", variance_adjustment=100000000000000, max_conf = 1 - 1e-16, min_conf = 0 + 1e-16 , label = None, no_filter = False
):
    """Takes in a test dataloader + a trained model and returns the scaled logit values

    ...
    Parameters
    ----------
        test_loader : PyTorch Dataloader
            The Pytorch Dataloader for Dtest
        model : torch.nn.Module (PyTorch Neural Network Model)
            A trained model to be queried on the test data
        device : str
            The processing device for training computations.
            Ex: cuda, cpu, cuda:0
        middle_measure : str
            When removing outliters from the data, this is the
            "center" of the distribution that will be used.
            Options are ["mean", "median"]
        variance_adjustment : float
            The number of standard deviations away from the "center"
            we want to keep.
    ...
    Returns
    -------
        logits_arr : np.array
            An array containing the scaled model confidence values on the query set
    """
    
    n_samples = len(test_loader.dataset)
    logit_arr = np.zeros((n_samples, 1))
    # activation_dict = {}
        
    model = model.to(device)

    y_prob = torch.Tensor([])
    y_test = torch.Tensor([])
    for d, l in test_loader:
        d = d.to(device)
        #model.eval()
        with torch.no_grad():
            out = model(d)
            # Get class probabilities
            out = nn.functional.softmax(out, dim=1).cpu()
            y_prob = torch.concat([y_prob, out])
            y_test = torch.concat([y_test, l])

    y_prob, y_test = np.array(y_prob), np.array(y_test, dtype=np.uint8)

    # print(y_prob.shape)

    if np.sum(y_prob > max_conf):
        indices = np.argwhere(y_prob > max_conf)
        #             print(indices)
        for idx in indices:
            r, c = idx[0], idx[1]
            y_prob[r][c] = y_prob[r][c] - 1e-50

    if np.sum(y_prob < min_conf):
        indices = np.argwhere(y_prob < min_conf)
        for idx in indices:
            r, c = idx[0], idx[1]
            y_prob[r][c] = y_prob[r][c] + 1e-50

    possible_labels = len(y_prob[0])
    for sample_idx, sample in enumerate(zip(y_prob, y_test)):

        conf, og_label = sample
        if(label == None):
            label = og_label
        selector = [True for _ in range(possible_labels)]
        selector[label] = False

        first_term = np.log(conf[label])
        second_term = np.log(np.sum(conf[selector]))

        logit_arr[sample_idx, 0] = first_term - second_term

    # print(logit_arr.shape)

    logit_arr = logit_arr.reshape(-1)

    if middle_measure == "mean":
        middle = logit_arr.mean()
    elif middle_measure == "median":
        middle = np.median(logit_arr)

    # if(distinguish_type == 'global_threshold'):
    logit_arr_filtered = logit_arr[
        logit_arr > middle - variance_adjustment * logit_arr.std()
    ]  # Remove observations below the min_range
    logit_arr_filtered = logit_arr_filtered[
        logit_arr_filtered < middle + variance_adjustment * logit_arr_filtered.std()
    ]  # Remove observations above max range
    
    if no_filter:
        return logit_arr
    return logit_arr_filtered



def get_features(
    test_loader, model, device="cpu"
):
    n_samples = len(test_loader.dataset)
    model = model.to(device)
    x_data = torch.Tensor([])
    y_feature = torch.Tensor([])
    y_test = torch.Tensor([])
    y_out = torch.Tensor([])
    for d, l in test_loader:
        d = d.to(device)
        #model.eval()
        with torch.no_grad():
            feature, out = model.get_features(d) #.cpu()
            out = nn.functional.softmax(out, dim=1).cpu()
            y_feature = torch.concat([y_feature, feature.cpu()])
            y_test = torch.concat([y_test, l])
            y_out = torch.concat([y_out, out])
            x_data = torch.concat([x_data, d.cpu()])

    y_feature, y_test, y_out = np.array(y_feature), np.array(y_test, dtype=np.uint8), np.array(y_out)
    return y_feature, y_test, y_out, x_data

    
def fit_local_with_distill(
    dataloaders,
    kd_dataloaders,
    model,
    epochs=100,
    optim_init=optim.Adam,
    optim_kwargs={"lr": 0.03, "weight_decay": 0.0001},
    criterion=nn.CrossEntropyLoss(),
    device="cpu",
    verbose=True,
    mini_verbose=True,
    train_only=True,
    early_stopping=False,
    tol=10e-6,
):
    model = model.to(device)
    #optim_init = optim.SGD
    optimizer = optim_init(model.parameters(), **optim_kwargs)
    train_error = []
    test_loss = []
    test_acc = []
    # if train_only:
    #     phases = ["train"]
    # else:
    #     phases = ["train", "test"]
    
    #verbose = True
    if mini_verbose:
        print("Training...")
    if verbose:
        print("-" * 8)
    kd_criterion = nn.L1Loss()
    alpha = 0.90
    #alpha = 0.5
    try:

        for epoch in range(1, epochs + 1):
            kd_iter = iter(kd_dataloaders)
            if verbose:
                a = time.time()
                print(f"Epoch {epoch}")

            running_train_loss = 0

            for (inputs, labels) in dataloaders:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)

                try:
                    kd_inputs,kd_logits  = next(kd_iter)

                except:
                    kd_iter = iter(kd_dataloaders)
                    kd_inputs,kd_logits  = next(kd_iter)
                
                kd_inputs = kd_inputs.to(device)
                kd_logits = kd_logits.to(device)

                outputs = model.forward(inputs)
                kd_outputs = model.forward(kd_inputs)

                loss = criterion(outputs, labels)
                loss = ((1-alpha)*loss)+(alpha*kd_criterion(kd_outputs,kd_logits))
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)


            train_error.append(running_train_loss / len(dataloaders.dataset))

            if len(train_error) > 1 and early_stopping:
                if abs(train_error[-1] - train_error[-2]) < tol:
                    print(f"Loss did not decrease by more than {tol}")
                    if mini_verbose:
                        print(f"Final Train Error: {train_error[-1]:.6}")
                    if not train_only:
                        return model, train_error, test_loss, test_acc
                    else:
                        return model, train_error

            if verbose:
                b = time.time()
                print(f"Train Error: {train_error[-1]:.6}")
                #print(f"Time Elapsed: {b - a:.4} seconds")
                #print("-" * 8)
    except KeyboardInterrupt:
        if mini_verbose:
            print(f"Final Train Error: {train_error[-1]:.6}")
        if not train_only:
            return model, train_error, test_loss, test_acc
        else:
            return model, train_error

    if mini_verbose:
        print(f"Final Train Error: {train_error[-1]:.6}")
    if not train_only:
        return model, train_error, test_loss, test_acc
    else:
        return model, train_error
def fit(
    dataloaders,
    model,
    epochs=100,
    optim_init=optim.Adam,
    optim_kwargs={"lr": 0.03, "weight_decay": 0.0001},
    criterion=nn.CrossEntropyLoss(),
    device="cpu",
    verbose=True,
    mini_verbose=True,
    train_only=True,
    early_stopping=False,
    tol=10e-6,
):
    """Fits a PyTorch model to any given dataset

    ...
    Parameters
    ----------
        dataloaders : dict
            Dictionary containing 2 PyTorch DataLoaders with keys "train" and
            "test" corresponding to the two dataloaders as values
        model : torch.nn.Module (PyTorch Neural Network Model)
            The desired model to fit to the data
        epochs : int
            Training epochs for shadow models
        optim_init : torch.optim init object
            The init function (as an object) for a PyTorch optimizer.
            Note: Pass in the function without ()
        optim_kwargs : dict
            Dictionary of keyword arguments for the optimizer
            init function
        criterion : torch.nn Loss Function
            Loss function used to train model
        device : str
            The processing device for training computations.
            Ex: cuda, cpu, cuda:1
        verbose : bool
            If True, prints running loss and accuracy values
        train_only : bool
            If True, only uses "train" dataloader

    ...
    Returns
    -------
        model : torch.nn.Module (PyTorch Neural Network Model)
            The trained model
        train_error : list
            List of average training errors at each training epoch
        test_acc : list
            List of average test accuracy at each training epoch
    """

    model = model.to(device)
    optimizer = optim_init(model.parameters(), **optim_kwargs)
    train_error = []
    test_loss = []
    test_acc = []
    # if train_only:
    #     phases = ["train"]
    # else:
    #     phases = ["train", "test"]
    if mini_verbose:
        print("Training...")
    if verbose:
        print("-" * 8)

    try:

        for epoch in range(1, epochs + 1):
            if verbose:
                a = time.time()
                print(f"Epoch {epoch}")

            running_train_loss = 0
            running_test_loss = 0
            running_test_acc = 0

            for (inputs, labels) in dataloaders['train']:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model.forward(inputs)
                #loss = criterion(F.softmax(outputs), labels)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)


            train_error.append(running_train_loss / len(dataloaders['train'].dataset))

            if len(train_error) > 1 and early_stopping:
                if abs(train_error[-1] - train_error[-2]) < tol:
                    print(f"Loss did not decrease by more than {tol}")
                    if mini_verbose:
                        print(f"Final Train Error: {train_error[-1]:.6}")
                    if not train_only:
                        return model, train_error, test_loss, test_acc
                    else:
                        return model, train_error

            if not train_only:
                test_loss.append(running_test_loss / len(dataloaders["test"].dataset))
                test_acc.append(running_test_acc / len(dataloaders["test"].dataset))
            if verbose:
                b = time.time()
                print(f"Train Error: {train_error[-1]:.6}")
                if not train_only:
                    print(f"Test Error: {test_loss[-1]:.6}")
                    print(f"Test Accuracy: {test_acc[-1]*100:.4}%")
                print(f"Time Elapsed: {b - a:.4} seconds")
                print("-" * 8)
    except KeyboardInterrupt:
        if mini_verbose:
            print(f"Final Train Error: {train_error[-1]:.6}")
        if not train_only:
            return model, train_error, test_loss, test_acc
        else:
            return model, train_error

    if mini_verbose:
        print(f"Final Train Error: {train_error[-1]:.6}")
    if not train_only:
        return model, train_error, test_loss, test_acc
    else:
        return model, train_error
