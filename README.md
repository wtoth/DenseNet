# DenseNet (Densely Connected Convolutional Networks)
This is my replication of the DenseNet Model using the Cifar10 Dataset

# Model Specification 
Due to GPU Size constraints this is a rather small model but can be scaled according to your gpu size<br>
This model includes implementations for both standard DenseNets and Bottlenecked DenseNets. You can change between the two in main.py by updating `bottlenecked = False`

## Setup and Running
This project was created using `uv` and is highly recommended<br>
After installing `uv` this project should run out of the box<br>

### Data Setup
You can get the original Dataset from [Alex Krizhevsky's webpage](https://www.cs.toronto.edu/~kriz/cifar.html) although there are likely sources out there that provide it in a more modern format. 

Once this format is downloaded you will need to run the `cifar10_data_processing.py` script to create our datasets<br>
```
uv run cifar10_data_processing.py
```
This will create a data folder with both training and test directories

### Training
Before kicking off training you should update the weights and biases variables `entity` and `project` in `init_logging()` in train.py to match your account.<br>
If not using Weights and Biases (not recommended) you can set logs to `False` in main.py<br>
To kick off training you can run<br>
```uv run main.py```

# Citation 
Paper [link](https://arxiv.org/abs/1608.06993)

```bibtex
@inproceedings{huang2017densely,
  title={Densely connected convolutional networks},
  author={Huang, Gao and Liu, Zhuang and Van Der Maaten, Laurens and Weinberger, Kilian Q},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4700--4708},
  year={2017}
}
```