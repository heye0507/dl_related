# FastAI 2019 part 2 practice
As Fastai 2019 part 2 homeworks, built a wrapper library on top of Pytorch

## From scratch:

- **01_get_data.ipynb** : Getting MNIST dataset
- **02_datablock.ipynb** : Create Databunch class that wrappes Pytorch Dataset and Dataloader
- **03_basic_train.ipynb** : Create base Learner class that wrappes model, optimizer and loss function for data input
- **04_callback.ipynb** : Create Callbacks class controlled train loop, new functions can be used as callbacks with out changing train loop
- **05_LR_find.ipynb** : Implemented LR_find() callback class for finding best LR
- **05b_new_Leaner.ipynb** : Implemented new Leaner class that use exception handler except the old callback handler
- **06_CNN.ipynb** : Using callbacks to reshape MNIST dataset, same callback can be used for data argumentation
                     Comparing layer mean and std with / without Kaiming init
- **07_init_subRelu_Fastai.ipynb** : Implemented subRelu() to take care of shift mean and std after regular relu
                                     Relu() moves mean 0 and std 1 to be mean 0.5 and ?std. To give 0 mean and unit variance helps model converge faster

## Implemented:
1. callbacks
2. learner
3. scheduler
4. LR_find
5. general relu
6. hooks
7. understand pytorch nn.Sequential

# Object Detection

## Dataset: Pascal VOC 2007
- **Retina_net_dev.ipynb** : Retina net and focal loss implementation for Insight interview
- **SSD_dev.ipynb** : Single shot detection implementation for understanding object detection basics, such as bounding boxes                    
                      Medium post: [Here](https://medium.com/p/ec6e9286c92c/)
