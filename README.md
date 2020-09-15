# Breast Cancer Classification

## Installation
It is assumed that python is installed in your system. 
```
pip3/pip install keras 2.4.3
pip3/pip install tensorflow 2.3.0
pip3/pip install numpy 1.18.5
pip3/pip install sklearn 0.23.2
pip3/pip install matplotlib 3.3.0
```

## Description
- The dataset can be available at [Kaggle](https://www.kaggle.com/paultimothymooney/breast-histopathology-images). 
- This model aims to classify the images of the people who have breast cancer which in turns helps to prevent it.

## Method
- In the above python file, you will find 4 models which I have trained and measured their accuracies.
- From the 4 models one of them is pretrained vgg network, just for the sake of checking how it would perform on vgg16 model.
- The other 3 models were developed and you can find them in the same python file.
- The same python file also contains the code for building the dataset directory and please follow the path '/breast_cancer/datasets/original/' and keep the downloaded kaggle dataset file there otherwise it won't work.

## Result
vgg 16 model:  training accuracy: 82.96%   validation accuracy: 83.62%  testing accuracy: 83.61%
model_1:       training accuracy: 88.69%   validation accuracy: 88.84%  testing accuracy: ~88%
model_4:       training accuracy: 87%      validation accuracy: 87%     testing accuracy: 86%
model_2:       training accuracy: 88.67%   validation accuracy: 87.80%  testing accuracy: 87.66%

## Output
![Screenshot from 2020-09-14 21-23-47](https://user-images.githubusercontent.com/40459209/93206295-c74aef80-f776-11ea-9cad-821f66192ded.png)

![Screenshot from 2020-09-14 21-23-32](https://user-images.githubusercontent.com/40459209/93206333-da5dbf80-f776-11ea-99fd-4f6588696e61.png)

