---
layout: post
title:  "Brainless Noter!"
date:   2022-02-28 17:27:55 +0530
category: Notes
---

# CH-2: Building Binary Classifier

## Getting Data:

* First step is to import all required libraries:
```python
from fastai import *
from fastai.vision import *
from fastai.vision.all import *
from fastai.vision.data import ImageDataLoaders
```

* Here fastai contains all our torch, numpy, matplotlib included and with many additional useful libraries.
* Since we are dealing with a vision problem, fast.vision.all provides us all the required tools
* Note: Error: ModuleNotFoundError: No module named 'fastai.vision.all' or No module named 'fastai.callback.all'; 'fastai.callback' is not a package,
 Solution: In this case i did ++ !pip install fastai --upgrade ++, later we need to comment this installing commands and restart the kernel. Now this will works fine.(Colab)

```python
path = untar_data(URLs.DATASET_NAME)
# Here DATASET_NAME = MNIST_SAMPLE
# To check all files do path.ls()
print((path/'train'/'3').ls())
print(type((path/'train'/'3').ls()))
```
* Here untar_data return path to the dataset, where dataset is downloaded and stored in our local machine.
* We have downloaded MNIST_SAMPLE dataset, it contains 2 folders containing 3's and 7's.For more information about various datasets go to this [link](https://docs.fast.ai/data.external.html)
* Here In Fastai we use Class L instead of list. Class L is an generic extension for list, which has many useful tools.
* For sense of clarity we just sort the list(L) containing path to images 3 and 7
```python
threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()
print(Image.open(threes[1]))
```
* Image module has many other uses like rotate, resize etc refer [link](https://pillow.readthedocs.io/en/stable/reference/Image.html)
* Tensor to Dataframe => df = pd.DataFrame(Tensor)
```python
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
# df.style.set_properties(color="white", align="right")  
# df.style.set_properties(**{'background-color': 'yellow'})  
```
* style.set_properties helps in setting font_size, color, background olor and gradient.

### TASK: METHOD 1: Pixel Similarity
* We make a Generic 3 and 7 images using them, we decide the new sample as 3 oor not.
* Now sub Task is to find Generic 3/7.Let's say 3. We just take average. First we create our stack of tensor. That is we gonna add all 3's in a new tensor of size ( number of 3's,28,28)
* Now We gonna take average.
```python
three_tensors = [tensor(Image.open(i)) for i in threes]
seven_tensors = [tensor(Image.open(i)) for i in sevens] # List Comprehensions
stacked_threes = torch.stack(three_tensors).float()/255
stacked_sevens = torch.stack(seven_tensors).float()/255
mean3 = stacked_threes.mean(0)
mean7 = stacked_sevens.mean(0)
print(show_image(mean3,cmap='Greys'),show_image(mean7,cmap='Greys'))
```
* torch.stack() : Concatenates a sequence of tensors along a new dimension. All tensors need to be of the same size.
* tensor.mean(0) : mean(0) does mean across 0 while keeping 1 and 2 dims const.
* show_image(v,cmap='Greys'):
```python
show_image(img:Image, ax:Axes=None, figsize:tuple=(3, 3), hide_axis:bool=True, cmap:str='binary', alpha:float=None, **kwargs)
```
#### Loss :

* Mean_Squared_error
* Mean_Absolute_error

```python
def mnist_distance(a,b):
  return (a-b).abs().mean((-1,-2)) # mean over last 2 axes
valid_3_dist = mnist_distance(valid_3_tns,mean3) # Thanks to broadcasting.
print(valid_3_dist.shape)

def is_3(x):
  return (mnist_distance(x,mean3) < mnist_distance(x,mean7))

def plot_function(f, tx=None, ty=None, title=None, min=-2, max=2, figsize=(6,4)):
    x = torch.linspace(min,max)
    fig,ax = plt.subplots(figsize=figsize)
    ax.plot(x,f(x))
    if tx is not None: ax.set_xlabel(tx)
    if ty is not None: ax.set_ylabel(ty)
    if title is not None: ax.set_title(title)
```
* We define mnist_distance as our loss function, which is absolute mean error.With help of broadcasting we are able to get valid_3_dist.
* plot_function : helps to plot f between min and max with given figsize

#### SGD, Differentitation :
* tensor().requires_grad_() helps us telling pytorch that this particular thing is a variable and we may compute the gradient in near future.
* A small Illustration:
  1. lets consider a function between speed and time.
    ```python
    time = torch.arange(0,20).float()
    speed = torch.arange(20)*3+0.75*(time-9.5)**2+1 #a (time^2 ) + (b*t) + c
    plt.scatter(time,speed)
    ```
  2. We need to find parameters a,b,c succh that a*(time^2)+(b*t)+c gives us the speed.
  3. let mse between target and given predicts be our cost function
  4. Then we find our parameters, by training as shown in below cell.
  ```python
  def apply_step(params,prn=True):
  preds = f(time,params)
  loss = mse(preds,speed)
  loss.backward()
  params.data -= 1e-5*params.grad.data
  params.grad = None
  if prn: print(loss.item())
  return preds

  for i in range(10):
    apply_step(params)
  ```
* On our original task,
  1. First, lets keep all 3's and 7's and corresponding labels, concrete.way. To do that we use torch.cat([a,b]).And we help of .view we can perfectly align all points.
  ```python
  train_x = torch.cat([stacked_threes,stacked_sevens]).view(-1,28*28)
  train_y = tensor([1]*len(threes) + [0]*len(sevens)).unsqueeze(1)
  ```
    * .view() : Returns a new tensor with the same data as the self tensor but of a different shape
    * .squeeze() : Returns a tensor with all the dimensions of input of size 1 removed.
    * .unsqueeze() : Returns a new tensor with a dimension of size one inserted at the specified position.
  2. Let's make a dataset => list of (x,y)'s
  ```python
  dset = list(zip(train_x,train_y))
  def init_params(size,std = 1.0):
    return (torch.randn(size)*std).requires_grad_()
  weights = init_params((28*28,1))
  bias = init_params(1)
  ```
  3. Function "init_params": helps in initiazing the parameters/weights.It sets random value and sets requires gradient.
  4.  We define a linear model as follows:
  ```python
  def linear1(xb):
    return xb@weights+bias
  ```
  5. We classify as 3 if the output of linear model is less than zero, and in other case wew ill label it as 7.
  6. For ease of calculation we use sigmoid function on predictions to getting probability of being 7.
    * torch.tensor.where() : torch.where(condition, x, y) return a new tensor depeding on condition and values of x and y.
  ```python
  def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1,1-predictions,predictions).mean()
  ```
* By Here we have
  1. Organised Dataset.
  2. A perfect loss function and parameters to get learn.
  3. And Gradient decent procedure.
* Now we bulid Dataloaders, Batches to get train out model better and faster.
  1. Dataloader:
  ```python
  dl = DataLoader(dset,batch_size=256)
  valid_dl = DataLoader(valid_dset,batch_size=256)
  ```
    * DataLoader : DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False) [link](https://pytorch.org/docs/stable/data.html)
  2. We need to calculate gradient and do parameter updates as follows:
  ```python
  def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()

  def train_epoch(model,lr,params):
    for xb,yb in dl:
      calc_grad(xb,yb,model)
      for p in params:
        p.data -= p.grad.data*lr
        p.grad.zero_()
  ```
  3. We need to check accuracy in every epoch, so that it will help to stop over tuning of weights.
```python
def batch_accuracy(xb,yb):
  preds = xb.sigmoid()
  correct = (preds > 0.5) == yb
  return correct.float().mean()

def validate_epoch(model):
  accs = [batch_accuracy(model(xb),yb) for xb,yb in valid_dl]
  return round(torch.stack(accs).mean().item(),4)
```
  4. Training :
```python
# Initilize
lr = 0.5
weights = init_params((28*28,1))
bias = init_params(1)
params = weights, bias
train_epoch(linear1,lr,params)
validate_epoch(linear1)

# Train
for i in range(20):
  train_epoch(linear1, lr, params)
  print(validate_epoch(linear1), end=' ')
```

* Good Way of dng Whole thing:
  * Things to know:
    1. nn.Linear ==> linear model
    2. linear_model.parameters gives us parameters.
    3. Introduced BasicOptim => optimizer which has step,zero_grad methods. It's main use is for updates.
    ```python
    class BasicOptim:
      def __init__(self,params,lr):
        self.params,self.lr = list(params),lr

      def step(self):
        for p in self.params:
          p.data -= p.grad.data*self.lr

      def zero_grad(self):
        for p in self.params:
          p.grad = None
    opt = BasicOptim(linear_model.parameters(),lr)
    # opt = SGD(linear_model.parameters(),lr)
    def train_epoch(model):
      for xb,yb in dl:
      calc_grad(xb,yb,model)
      opt.step()
      opt.zero_grad()
      print(validate_epoch(linear_model))
    ```
    4. SGD is basic optimizer which is similar to our's.
* We can also do the same with fastai Learner module as follows:
  ```python
  # fastai also provides Learner.fit, which we can use instead of train_model. To create
  # a Learner, we first need to create a DataLoaders, by passing in our training and validation
  # DataLoaders:
  dls = DataLoaders(dl, valid_dl)

  learn = Learner(dls, nn.Linear(28*28,1), opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy)
  learn.fit(10,lr=lr)
  ```
* We can include non-Linearity using nn.Sequential module to stack layers and activation functions together.
```python
simple_net = nn.Sequential(
                nn.Linear(28*28,30),
                nn.ReLU(),
                nn.Linear(30,1)
                )

learn = Learner(dls, simple_net, opt_func=SGD,
loss_func=mnist_loss, metrics=batch_accuracy)
learn.fit(40, 0.1)
```
* To plot the accuracy curve do: plt.plot(L(learn.recorder.values).itemgot(2));
* And we can view the final accuracy: learn.recorder.values[-1][2]

* Last things to know:
1. Learner : earner(dls, model, loss_func=None, opt_func=Adam, lr=0.001, splitter=trainable_params, cbs=None, metrics=None, path=None, model_dir='models', wd=None, wd_bn_bias=False, train_bn=True, moms=(0.95, 0.85, 0.95))
2. learn.recorder.values : [link](https://docs.fast.ai/learner.html#Recorder)
