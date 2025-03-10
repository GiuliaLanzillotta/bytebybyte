""" Creating multi-task supervised learning environments.
    First environment set: random teachers. 
"""

import random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
from avalanche.benchmarks.datasets.clear import _CLEARImage
from torchvision.datasets import CIFAR10, CIFAR100,  OxfordIIITPet, Food101, Caltech101, FGVCAircraft, StanfordCars, DTD, Flowers102

from avalanche.benchmarks.utils import (
    PathsDataset,
    common_paths_root,
)

class TaskDataset(torch.utils.data.Dataset):
    """Wrapper class for a torch dataset that adds a task_id to the output of __getitem__."""
    
    def __init__(self, dataset, task_id):
        self.dataset = dataset
        self.task_id = task_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, self.task_id



class World:
    """ Generic class for a continual learning world. 
        Any benchmark can be wrapped here. 

        Contains an environment object which is a dict of tasks, task name -> task dataset  
    """

    def __init__(self) -> None:
        self.environment = {}
        self.task_names = {}
        self.ordered_task_names = self.task_names
        self.world_name = ""
        
    def init_multi_task(self, number_of_tasks=-1, train=True):
        """ Concatenates the first 'number_of_tasks' tasks into a single task by concatenating the datasets."""
        if train: env = self.environment['train']
        else: env = self.environment['test']
        mt_task_names = self.ordered_task_names[:number_of_tasks]
        if number_of_tasks == -1: mt_task_names = self.ordered_task_names
        datasets = [env[t] for t in mt_task_names]
        mt_dataset = ConcatDataset(datasets)
        return mt_dataset
    
    def subsample(self, dataset, dataset_name, buffer_indices, buffer_size): 
        if dataset_name not in buffer_indices.keys(): 
            indices = random.sample(range(len(dataset)), buffer_size)
            buffer_indices[dataset_name] = indices
        subset = torch.utils.data.Subset(dataset, buffer_indices[dataset_name]) 
        return subset, buffer_indices
    
    def init_buffer(self, boundaries:tuple, buffer_indices, buffer_size):
        """ Returns a buffer dataset made of the concatenation of several subsampled datasets"""
        (first_task, last_task) = boundaries
        env = self.environment['train']
        mt_task_names = self.ordered_task_names[first_task:last_task+1]
        if boundaries == (0,-1): mt_task_names = self.ordered_task_names
        datasets = []
        for t in mt_task_names:
            buffer_dataset, buffer_indices = self.subsample(env[t], t, buffer_indices, buffer_size)
            datasets.append(buffer_dataset)
        mt_dataset = ConcatDataset(datasets)
        return mt_dataset, buffer_indices

    def buffered_multi_task(self, number_of_tasks=-1, train=True, buffer_size=500):
        """ Uses a sampling process to decrease the datasets size. 
        Then concatenates the first 'number_of_tasks' tasks into a single task 
        by concatenating the datasets."""
        if train: env = self.environment['train']
        else: env = self.environment['test']
        mt_task_names = self.ordered_task_names[:number_of_tasks]
        if number_of_tasks == -1: mt_task_names = self.ordered_task_names
        datasets = [self.select_buffer(env[t], t, buffer_size, self.buffer_indices.get(t, None)) for t in mt_task_names]
        mt_dataset = ConcatDataset(datasets)
        return mt_dataset
    
    def init_single_task(self, task_number, train=True):
        """ Returns the task in the environment corresponding to the task_number given"""
        if train: env = self.environment['train']
        else: env = self.environment['test']
        task_name = self.ordered_task_names[task_number]
        return env[task_name] 
    
    def order_task_list(self, ordering=None):
        """Orders the internal sequence of task according to the given ordering. 
            If no ordering is given it provides a random ordering."""
        if ordering is None: ordering = np.random.permutation(range(len(self.task_names)))
        self.ordered_task_names = [self.task_names[i] for i in ordering]

class Permutation(object):
    """
    Defines a fixed permutation for a numpy array.
    """

    def __init__(self, input_size, permutation_size) -> None:
        """
        Initializes the permutation inside a "permutation_size" box in the middle of the image.
        We assume the image is input_size x input_size and the permutation box is square.
        """
        
        permutation_indices, mask = self.get_box_indices(input_size, permutation_size)
        self.perm = np.random.permutation(permutation_indices)
        self.indices = np.arange(input_size**2)
        self.indices[np.where(mask)] = self.perm
        

    @staticmethod
    def get_box_indices(input_size, permutation_size):

        margin = (input_size - permutation_size)//2


        permutation_indices = [] #flattened image indices
        mask = []
        
        for i in range(input_size):
            row_index = i*input_size
            for j in range(input_size):
                if i>margin and i<margin+permutation_size \
                and j>margin and j<margin+permutation_size:
                    permutation_indices.append(row_index+j)
                    mask.append(True)
                else: mask.append(False)

        return permutation_indices, mask


    def __call__(self, sample: np.ndarray) -> np.ndarray:
        """
        Randomly defines the permutation and applies the transformation.

        Args:
            sample: image to be permuted

        Returns:
            permuted image
        """
        old_shape = sample.shape

        if len(old_shape)>2: #careful with indexing over the channel dimension
            channels = old_shape[0]
            indices_with_channels = np.tile(self.indices, channels).reshape(channels,len(self.indices))
            flattened_sample = torch.flatten(sample,start_dim=1)
            return flattened_sample.gather(1, torch.LongTensor(indices_with_channels).to(sample.device)).reshape(old_shape)
       
        return torch.flatten(sample)[self.indices].reshape(old_shape)

class PermutationWorld(World):
    """ Environment for Supervised Learning tasks with controlled permutations. 
        Each task consists of a classification problem where the input is permuted. 


        Returns an environment object which is a dict of tasks, task name -> task dataset 
    """

    data_size = 32
    num_classes = 10
    batches_eval = 5
    criterion = nn.CrossEntropyLoss()
    
    normalize = transforms.Normalize(
        mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], 
        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize(data_size),
            transforms.RandomCrop(data_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(data_size),
            transforms.CenterCrop(data_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    augmentations = [
                    transforms.RandomCrop(data_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15)
                ]


    def __init__(self, 
                 permutation_size,
                 number_tasks=10, 
                 train_transform=None, 
                 test_transform=None, augment=False) -> None:
        
        super(PermutationWorld, self).__init__()
        
        self.number_tasks = number_tasks
        self.permutation_size = permutation_size
        self.permutation_fraction = (self.permutation_size**2)/(self.data_size**2)
        self.world_name = f"permuted-cifar10-{permutation_size}"
        self.num_classes_per_task = self.num_classes


        if train_transform is not None: self.train_transform = train_transform
        if test_transform is not None: self.test_transform = test_transform


        if augment: 
            self.train_transform = transforms.Compose([self.augmentations, self.train_transform])

        environment = self.generate_tasks()
        self.environment = environment
        self.task_names = list(environment['train'].keys())
        self.ordered_task_names = self.task_names
    
    def generate_tasks(self):
        """ Generates a sequence of N tasks by applying a fixed random permutation to each."""
        # dictionary with a train and test stream 
        stream_definitions = dict()
        stream_definitions['train'] = {}
        stream_definitions['test'] = {}
        task_labels = []
    
        for i in range(self.number_tasks):
            
            label = f"permutation_{self.permutation_size}_{i}"
            task_labels.append(label)

            permutation_transform = Permutation(self.data_size, self.permutation_size)

            C100_dataset_train = CIFAR10(root="./datasets", train=True, 
                                        download=True, transform=transforms.Compose([self.train_transform, permutation_transform]))
            C100_dataset_test = CIFAR10(root="./datasets", train=False, download=True, 
                                         transform=transforms.Compose([self.test_transform, permutation_transform]))
            
            stream_definitions['train'][label] = TaskDataset(C100_dataset_train, task_id=i)   
            stream_definitions['test'][label] = TaskDataset(C100_dataset_test, task_id=i)

        return stream_definitions

class Split(object):
    """ Splits a dataset into multiple subsets."""

    def __init__(self, split_type, dataset, number_tasks, randomize):

        self.split_type = split_type
        self.dataset = dataset
        self.number_tasks = number_tasks
        self.class_to_chunk = None
        self.randomize=randomize
        self.subsets = self.split_dataset()

    def split_dataset(self):
        if self.split_type == "chunks":
            return self.split_into_chunks()
        elif self.split_type == "classes":
            return self.split_by_classes(self.randomize)
        else:
            raise ValueError("Invalid split_type. Choose either 'chunks' or 'classes'.")

    def split_into_chunks(self):
        """Splits the dataset into equal chunks randomly, each chunk has the same set of classes."""
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        chunk_size = len(self.dataset) // self.number_tasks
        return [torch.utils.data.Subset(self.dataset, indices[i * chunk_size:(i + 1) * chunk_size]) for i in range(self.number_tasks)]

    def split_by_classes(self, randomize=False):
        """Splits the dataset into chunks, each with a different set of classes."""
        class_indices = {}
        for idx, (_, label) in enumerate(self.dataset):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        class_order = list(class_indices.keys())
        if randomize:
            random.shuffle(class_order)

        class_chunks = {i: [] for i in range(self.number_tasks)}
        self.class_to_chunk = {}
        classes_per_task = (len(class_order) // self.number_tasks)
        for label in class_order:
            chunk_idx = label // classes_per_task
            class_chunks[chunk_idx].extend(class_indices[label])
            self.class_to_chunk[label] = chunk_idx

        return [torch.utils.data.Subset(self.dataset, indices) for indices in class_chunks.values()]

class SplitWorld(World):
    """ Environment for Supervised Learning tasks with split-dataset.
        The original dataset is split into chunks of equal size. 
        If the split_type is "classes", the split is based on class information. 
        If the split_type is "chunks", the split is random, but each chunk has the same classes.

        For now, only implemented for Cifar100. Soon to be changed

        Returns an environment object which is a dict of tasks, task name -> task dataset 
    """

    data_size = 32
    num_classes = 100
    batches_eval = 5
    criterion = nn.CrossEntropyLoss()
    
    normalize = transforms.Normalize(
        mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], 
        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize(data_size),
            transforms.RandomCrop(data_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(data_size),
            transforms.CenterCrop(data_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    augmentations = [
                    transforms.RandomCrop(data_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15)
                ]


    def __init__(self, 
                 split_type, # either 'classes' or 'chunks' 
                 number_tasks=10, 
                 train_transform=None, 
                 test_transform=None, 
                 augment=False, 
                 randomize_class_order=False) -> None:
        
        super(SplitWorld, self).__init__()
        
        self.number_tasks = number_tasks
        self.split_type = split_type
        if self.split_type == "classes": 
            assert self.num_classes % self.number_tasks == 0, "Number of tasks should be a multiple of the number of classes" 
            self.num_classes_per_task = self.num_classes//number_tasks
        else: self.num_classes_per_task = self.num_classes
        self.world_name = f"split-cifar100-{split_type}-{number_tasks}"
        self.randomize_class_order = randomize_class_order


        if train_transform is not None: self.train_transform = train_transform
        if test_transform is not None: self.test_transform = test_transform
        if augment: 
            self.train_transform = transforms.Compose([self.augmentations, self.train_transform])

        environment = self.generate_tasks()
        self.environment = environment
        self.task_names = list(environment['train'].keys())
        self.ordered_task_names = self.task_names
    
    def generate_tasks(self):
        """ Generates a sequence of N tasks by applying a fixed random permutation to each."""
        # dictionary with a train and test stream 
        stream_definitions = dict()
        stream_definitions['train'] = {}
        stream_definitions['test'] = {}
        task_labels = []


        C100_dataset_train = CIFAR100(root="./datasets", train=True, 
                                    download=True, transform=self.train_transform)
        C100_dataset_test = CIFAR100(root="./datasets", train=False, download=True, 
                                        transform=self.test_transform)

        chunks_train = Split(self.split_type, C100_dataset_train, self.number_tasks, self.randomize_class_order)
        chunks_test = Split(self.split_type, C100_dataset_test, self.number_tasks, self.randomize_class_order)

        for i in range(self.number_tasks):
            
            label = f"split_{self.split_type}_{self.number_tasks}_{i}"
            task_labels.append(label)
            stream_definitions['train'][label] = TaskDataset(chunks_train.subsets[i], i)
            stream_definitions['test'][label] = TaskDataset(chunks_test.subsets[i], i)

        return stream_definitions
        
class Shuffle(object):
    """ Defines a label shuffling operation """    
    def __init__(self, num_samples, num_classes, shuffle_fraction, original_labels, shuffled_indices=None):
        self.num_classes = num_classes
        if shuffled_indices is None: 
            all_indices = np.random.permutation(list(range(num_samples)))
            self.shuffled_indices = set(all_indices[:int(shuffle_fraction * num_samples)])
        else: self.shuffled_indices = shuffled_indices

        # Create a mapping for deterministic shuffled labels
        self.new_label_mapping = {}
        for idx in self.shuffled_indices:
            new_label = random.randint(0, num_classes - 1)
            while new_label == original_labels[idx]:  # Ensure the new label is different from the original
                new_label = random.randint(0, num_classes - 1)
            self.new_label_mapping[idx] = new_label

    def __call__(self, target, index=None):
        if index in self.shuffled_indices:
            return self.new_label_mapping[index]
        return target

class IndexedCifar10(CIFAR10):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target, index)

        return img, target

class LabelShufflingWorld(World):
    """ Environment for Supervised Learning tasks with controlled label shuffling. 
        Each task consists of a classification problem where the labels are shuffled.


        Returns an environment object which is a dict of tasks, task name -> task dataset 
    """

    data_size = 32
    num_classes = 10
    batches_eval = 5
    criterion = nn.CrossEntropyLoss()
    
    normalize = transforms.Normalize(
        mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], 
        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize(data_size),
            transforms.RandomCrop(data_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(data_size),
            transforms.CenterCrop(data_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    augmentations = [
                    transforms.RandomCrop(data_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15)
                ]


    def __init__(self, 
                 shuffle_fraction,
                 number_tasks=10, 
                 train_transform=None, 
                 test_transform=None, augment=False) -> None:
        
        super(LabelShufflingWorld, self).__init__()
        
        self.number_tasks = number_tasks
        self.shuffle_fraction = shuffle_fraction
        self.world_name = f"shuffled-cifar10-{shuffle_fraction}"
        self.num_classes_per_task = self.num_classes


        if train_transform is not None: self.train_transform = train_transform
        if test_transform is not None: self.test_transform = test_transform


        if augment: 
            self.train_transform = transforms.Compose([self.augmentations, self.train_transform])

        environment = self.generate_tasks()
        self.environment = environment
        self.task_names = list(environment['train'].keys())
        self.ordered_task_names = self.task_names
    
    def generate_tasks(self):
        """ Generates a sequence of N tasks by applying a fixed random permutation to each."""
        # dictionary with a train and test stream 
        stream_definitions = dict()
        stream_definitions['train'] = {}
        stream_definitions['test'] = {}
        task_labels = []

        original_dataset = CIFAR10(root="./datasets", train=True, download=True)
    
        for i in range(self.number_tasks):
            
            label = f"shuffle_{self.shuffle_fraction}_{i}"
            task_labels.append(label)

            # create a new shuffle 
            if i==0: 
                shuffle_transform = Shuffle(num_samples=len(original_dataset), num_classes=10, shuffle_fraction=self.shuffle_fraction, original_labels=original_dataset.targets)
                shuffled_indices = shuffle_transform.shuffled_indices
            else: 
                # we keep reshuffling the same samples (otherwise it becomes impossible for the MT agent)
                shuffle_transform = Shuffle(num_samples=len(original_dataset), num_classes=10, shuffle_fraction=self.shuffle_fraction, original_labels=original_dataset.targets, shuffled_indices=shuffled_indices)
        

            C10_dataset_train = IndexedCifar10(root="./datasets", train=True, download=True, transform=self.train_transform, target_transform=shuffle_transform)
            C10_dataset_test = CIFAR10(root="./datasets", train=False, download=True, transform=self.test_transform)

            stream_definitions['train'][label] = TaskDataset(C10_dataset_train,i)
            stream_definitions['test'][label] = TaskDataset(C10_dataset_test, i) 

        return stream_definitions

class MixedPermutationWorld(PermutationWorld):
    """ Environment for Supervised Learning tasks with controlled mixed permutations. 
        Extends the Permutation world environment to the case wehre half of the tasks 
            have low permutation and half of the tasks high.  


        Returns an environment object which is a dict of tasks, task name -> task dataset 
    """

    @staticmethod
    def get_split_indices(length, num_splits):
        return [round(i * length / num_splits) for i in range(1, num_splits)]

    def __init__(self, 
                 number_tasks=10, 
                 train_transform=None, 
                 test_transform=None, augment=False, num_switches=1) -> None:
        
        super(PermutationWorld, self).__init__()
        
        self.number_tasks = number_tasks
        self.world_name = f"permuted-cifar10-mixed-{num_switches}"
        self.num_switches = num_switches+1

        if train_transform is not None: self.train_transform = train_transform
        if test_transform is not None: self.test_transform = test_transform

        if augment: 
            self.train_transform = transforms.Compose([self.augmentations, self.train_transform])

        self.switch_task_indices = self.get_split_indices(self.number_tasks, self.num_switches)
        
        environment = self.generate_tasks()
        self.environment = environment
        self.task_names = list(environment['train'].keys())
        self.ordered_task_names = self.task_names
    

    def generate_tasks(self):
        """ Generates a sequence of N tasks by applying a fixed random permutation to each."""
        # dictionary with a train and test stream 
        stream_definitions = dict()
        stream_definitions['train'] = {}
        stream_definitions['test'] = {}
        task_labels = []

        permutation_size=16 
        for i in range(self.number_tasks):
            
            if i in self.switch_task_indices: 
                if permutation_size==32: permutation_size=16
                else: permutation_size=32

            label = f"permutation_{permutation_size}_{i}"
            task_labels.append(label)

            permutation_transform = Permutation(self.data_size, permutation_size)

            C100_dataset_train = CIFAR10(root="./datasets", train=True, 
                                        download=True, transform=transforms.Compose([self.train_transform, permutation_transform]))
            C100_dataset_test = CIFAR10(root="./datasets", train=False, download=True, 
                                         transform=transforms.Compose([self.test_transform, permutation_transform]))

            stream_definitions['train'][label] = TaskDataset(C100_dataset_train, i)
            stream_definitions['test'][label] = TaskDataset(C100_dataset_test, i)

        return stream_definitions

class MultiDatasetsWorld(World):
    """ Environment for Supervised Learning tasks with different datasets. 
        Each task consists of a different classification problem. 


        Returns an environment object which is a dict of tasks, task name -> task dataset 
    """

    data_size = 224
    root = './datasets'
    number_tasks = 5
    batches_eval = 1
    criterion = nn.CrossEntropyLoss()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize(data_size),
            transforms.CenterCrop(data_size),
            transforms.ToTensor(),
            normalize
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(data_size),
            transforms.CenterCrop(data_size),
            transforms.ToTensor(),
            normalize
        ]
    )

    augmentations = [
                    transforms.RandomCrop(data_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15)
                ]


    def __init__(self, 
                 num_classes = 50,
                 train_transform=None, 
                 test_transform=None, 
                 augment=False) -> None:
        
        super(MultiDatasetsWorld, self).__init__()
        
        self.num_classes = num_classes
        self.world_name = "multi-datasets"
        self.augment = augment
        self.num_classes_per_task = self.num_classes

        if train_transform is not None: self.train_transform = train_transform
        if test_transform is not None: self.test_transform = test_transform

        if self.augment: 
            [self.train_transform.transforms.insert(i+1, self.augmentations[i]) for i in range(len(self.augmentations))]
            print(self.train_transform)

        environment = self.build_tasks()
        self.environment = environment
        self.task_names = list(environment['train'].keys())
        self.ordered_task_names = self.task_names

    def select_dataset(self, dataset):
        """ Creates a subset of the dataset by selecting the first K classes"""
        
        # Get class labels and images
        if hasattr(dataset, "targets"):
            all_labels = torch.tensor(dataset.targets)
        elif hasattr(dataset, "_labels"):
            all_labels = torch.tensor(dataset._labels)
        elif hasattr(dataset, "_samples"):
            all_labels = torch.tensor([s[1] for s in dataset._samples])
        else: all_labels = torch.tensor([label for _, label in dataset])


        # Select a fixed number of classes (say 50 unique classes)
        selected_classes = torch.unique(all_labels)[:self.num_classes]  # Choose the first 50 classes

        # Filter the dataset based on the selected classes
        indices = [i for i, label in enumerate(all_labels) if label in selected_classes]

        subset_dataset = torch.utils.data.Subset(dataset, indices)
        subset_dataset.classes = dataset.classes

        return subset_dataset


    def load_food(self):
        """Loads Food101 dataset"""
        root = '/local/home/stuff/food101/'

        train_dataset = Food101(root=root, download=True, transform=self.train_transform)
        val_dataset = Food101(root=root, split='test', download=True, transform=self.test_transform)

        return train_dataset, val_dataset

    def load_pet(self):
        """Loads Oxford Pet III dataset"""        
        root = self.root

        train_dataset = OxfordIIITPet(root=root, download=True, transform=self.train_transform)
        val_dataset = OxfordIIITPet(root=root, split='test', download=True, transform=self.test_transform)

        return train_dataset, val_dataset

    def load_dtd(self):
        """Loads DTD dataset
        https://pytorch.org/vision/main/generated/torchvision.datasets.DTD.html#torchvision.datasets.DTD"""

        root = self.root

        train_dataset = DTD(root=root, download=True, transform=self.train_transform)
        val_dataset = DTD(root=root, split='test', download=True, transform=self.test_transform)

        return train_dataset, val_dataset

    def load_aircraft(self):
        """Loads Aircraft dataset 
        https://pytorch.org/vision/main/generated/torchvision.datasets.FGVCAircraft.html#torchvision.datasets.FGVCAircraft 
        """
        root = self.root

        train_dataset = FGVCAircraft(root=root, download=True, transform=self.train_transform)
        val_dataset = FGVCAircraft(root=root, split='test', download=True, transform=self.test_transform)

        return train_dataset, val_dataset

    def load_cars(self):
        """Loads StanfordCars dataset
        https://pytorch.org/vision/main/generated/torchvision.datasets.StanfordCars.html#torchvision.datasets.StanfordCars
        """
        root = '../ded/data/'

        train_dataset = StanfordCars(root=root, download=True, transform=self.train_transform)
        val_dataset = StanfordCars(root=root, split='test', download=True, transform=self.test_transform)

        return train_dataset, val_dataset
    
    
    def build_tasks(self):
        """ Builds the sequence of tasks."""

        # dictionary with a train and test stream 
        stream_definitions = dict()
        stream_definitions['train'] = {}
        stream_definitions['test'] = {}
        task_labels = ["StanfordCars","FGVCAircraft","DTD","Food101","OxfordPet"]
        task_functions = [self.load_cars,self.load_aircraft, self.load_dtd, self.load_food, self.load_pet]
        
        for i in range(self.number_tasks):
            label = task_labels[i]
            train, test = task_functions[i]()
            subset_train = self.select_dataset(train)
            subset_test = self.select_dataset(test)
            stream_definitions['train'][label] = TaskDataset(subset_train, i)   
            stream_definitions['test'][label] = TaskDataset(subset_test, i)

        return stream_definitions

class ClearWorld(World):
    """ Environment for Supervised Learning task on the CLEAR benchmark. 
        Each task consists of a 100-way classification on a different year. 

        Returns an environment object which is a dict of tasks, task name -> task dataset 

    """


    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    batches_eval = 5
    num_classes = 100
    criterion = nn.CrossEntropyLoss()


    def __init__(self, train_transform=None, test_transform=None) -> None:

        super(ClearWorld, self).__init__()
        self.world_name = "clear-buckets"
        self.num_classes_per_task = self.num_classes


        if train_transform is not None: self.train_transform = train_transform
        if test_transform is not None: self.test_transform = test_transform

        clear_dataset_train = _CLEARImage(
                    root="./datasets/clear100",
                    data_name="clear100",
                    download=True,
                    split="train",
                    seed=0,
                    transform=self.train_transform,
                )
        
        clear_dataset_test = _CLEARImage(
                    root="./datasets/clear100",
                    data_name="clear100",
                    download=True,
                    split="test",
                    seed=0,
                    transform=self.test_transform,
                )
        
        # the paths are a list of (File Path, Label) tuples
        train_samples_paths = clear_dataset_train.get_paths_and_targets(root_appended=True)
        test_samples_paths = clear_dataset_test.get_paths_and_targets(root_appended=True)
        class_names = clear_dataset_train.class_names

        environment = self.split_into_tasks(train_samples_paths, test_samples_paths, class_names)
        self.environment = environment
        self.task_names = list(environment['train'].keys())
        self.ordered_task_names = self.task_names
        self.number_tasks = len(self.task_names)

    
    def split_into_tasks(self, train_paths, test_paths, class_names):
        """Takes the list of image paths and creates a PathsDataset for each bucket."""

        # dictionary with a train and test stream 
        stream_definitions = dict()
        task_labels=list(range(len(train_paths)))
        all_paths=dict(train=train_paths, test=test_paths)

        for stream_name, lists_of_files in all_paths.items(): #train and test
            stream_datasets = {}
            # creating a separate dataset for each task
            for task_id, list_of_files in enumerate(lists_of_files[1:]):
                common_root, exp_paths_list = common_paths_root(list_of_files)
                task_dataset = PathsDataset(common_root, exp_paths_list, transform=self.train_transform)
                task_dataset.class_names = class_names
                stream_datasets[f"{2004+task_labels[task_id]}"] = TaskDataset(task_dataset, task_id)

            stream_definitions[stream_name] = stream_datasets

        return stream_definitions
    


def get_environment_from_name(env_name, args):
    """ Returns the environment class corresponding to the fiven name"""
    #todos: 
    # add possibility to do augmentations to all datasets 

    if env_name=="clear": 
        env = ClearWorld()  # ClearWorld benchmark setup - it takes some minutes to load the dataset
    elif env_name=="permuted": 
        env = PermutationWorld(permutation_size=args.permutation_size, number_tasks=args.number_tasks) 
        env_name = f"{env_name}-{args.permutation_size}"
    elif env_name=="shuffled": 
        env = LabelShufflingWorld(shuffle_fraction=args.shuffling_fraction, number_tasks=args.number_tasks) 
        env_name = f"{env_name}-{args.shuffling_fraction}"
    elif env_name=="multidataset": 
        env = MultiDatasetsWorld(num_classes=10, augment=True) 
    elif env_name=="mixedpermuted": 
        env = MixedPermutationWorld(number_tasks=args.number_tasks, num_switches=5) 
    elif env_name=="split": 
        env = SplitWorld(split_type=args.split_type, number_tasks=args.number_tasks, randomize_class_order=False) #follow a sequential order
        env_name = f"split-c100-{args.split_type}-{args.number_tasks}"
    else: raise NotImplementedError

    return env, env_name

