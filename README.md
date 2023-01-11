# Signal Propagation
## The Framework for Learning and Inference in a Forward Pass

A guide and tutorial on forward learning is available at:\
https://amassivek.github.io/sigprop

The paper detailing the framework for forward learning is available at:\
https://arxiv.org/abs/2204.01723

## TOC
1. [Install](#1-install)
2. [Quick Start](#2-quick-start)
3. [Examples](#3-examples)
4. [Documentation](#4-documentation)
5. [Development](#5-development)

## 1. Install

### 1.1. Production
Install
```bash
pip install https://github.com/amassivek/signalpropagation/archive/main.tar.gz
```

### 1.2. Development
1. Clone the repo

```bash
git clone https://github.com/amassivek/signalpropagation.git
```

2. cd

```bash
cd signalpropagation
```

3. install in development mode

```bash
pip install -e .
```

4. now you may development on signal propagation

5. then, submit a pull request

## 2. Quick Start

2.1. [Examples](#21-use-examples)\
2.2. [Implement a Model](#22-implement-a-model)

### 2.1. Use Examples

```bash
git clone https://github.com/amassivek/signalpropagation.git
cd signalpropagation
pip install -e .
cd examples
chmod +x examples.sh
./examples.sh
```

### 2.2 Implement a Model

This quick start uses the forward learning model from Example 2, but on a simple
network:\
[Example 2: Input Target Max Rand](#example-2-input-target-max-rand)

Concept overview of Signal Propagation:
- There is a signal, placed at the front of the network.
- Their are propagators, wrapped around each layer.

1. Install the sigprop package

```bash
git clone https://github.com/amassivek/signalpropagation.git
cd signalpropagation
pip install -e .
cd <your_code_directory>
```

2. Go to the file for network model.

Open the python file with the model you are configuring to use signal
propagation.

3. Add the import statement.

```python
import sigprop
```

4. Select the forward learning model.

Pick a signal, a propagator, and a model for using forward learning. Below are
good defaults.

This forward learning model trains each layer of the network as soon as the layer 
receives an input and target, so training and inference are united and act
together. This forward learning model is the base model,
and will conveniantly take take of the inputs and outputs during learning and
inference.
```python
sigprop.models.Forward
```

This signal model learns a projection of the input and the target (i.e. context) to the same
dimension.
```python
sigprop.signals.ProjectionContextInput
```

This propagator model takes in any loss function (i.e. callable) to train 
a network layer. Currently, propagators for signals have different router logic 
than hidden layers. So, we use a different propagator class for the signal.
```python
sigprop.propagators.signals.Loss
sigprop.propagators.Loss
```
We pair it with the following loss function.
```python
sigprop.loss.v9_input_target_max_all
```
A side note. Alternatively, ``sigprop.propagators.signals.Fixed`` may be used when 
the signal is constructed instead of learned, such as a fixed projection or overlay. 
Here if we replace ``Loss`` with ``Fixed``, then the above signal model will be 
treated as fixed projection. In this case, we would no longer use a loss
function.

5. Setup a manager (optional).

Setup a manager to configure defaults to help add signal propagation to an 
existing model. Managers are helper classes.

Managers are particularly helpful when adding propagators to layers of an 
existing model. Each network layer is wrapped in a propagator, so it may learn 
on its own. As we will see below, managers make wrapping layers quick.

```python
sp_manager = sigprop.managers.Preset(
    sigprop.models.Forward,
    sigprop.propagators.Loss,
    sigprop.propagators.signals.Loss,
    build_optimizer
)
```

We wrote a method to build an optimizer for each layer of the network, since
each layer learns independently from the other layers. The manager will call this 
method with a layer to get an optimizer and then give the optimizer to a propagator
to train the layer.
```python
def build_optimizer(module, lr=0.0004, weight_decay=0.0):
    p = list(module.parameters())
    if len(p) == 0:
        return None
    optimizer = optim.Adam(
        p, lr=lr,
        weight_decay=weight_decay)
    return optimizer
```

6. Configure propagator ahead of time. (optional)

If we are using a manager, we do this step. Otherwise, we skip this step.

Each network layer is wrapped in a propagator, so the layer may learn on its own.
Here, we configure defaults for propagators ahead of time, so we may easily wrap layers 
without having to specify an individual configuration for each layer. Here we 
set the default loss.

```python
sp_manager.config_propagator(
    loss=sigprop.loss.v9_input_target_max_all
)
```

7. Setup the signal.

- ``num_classes`` is the number of classes.
- ``hidden_dim`` is the size of the first hidden dim. Here it is the same as the input
  dim.

```python
hidden_dim = input_dim
hidden_ch = 128
input_shape = (num_classes,)
output_shape = (hidden_ch, hidden_dim, hidden_dim)
signal_target_module = nn.Sequential(
    nn.Linear(
        int(shape_numel(input_shape)),
        int(shape_numel(output_shape)),
        bias=False
    ),
    nn.LayerNorm(shape_numel(output_shape)),
    nn.ReLU()
)
signal_input_module = nn.Sequential(
    nn.Conv2d(
        3, output_shape[0], 3, 1, 1,
        bias=False
    ),
    nn.BatchNorm2d(output_shape[0]),
    nn.ReLU()
)

sp_signal = sigprop.signals.ProjectionContextInput(
    signal_target_module, signal_input_module,
    input_shape, output_shape
)
# convert labels to a one-hot vector
sp_signal = nn.Sequential(
    sigprop.signals.LabelNumberToOnehot(
        num_classes
    ),
    sp_signal
)
```

There are two options for wrapping the signal with a propagator:


Option 1, if we are using a manager
```python
sp_signal = sp_manager.set_signal(
    sp_signal,
    loss=sigprop.loss.v9_input_target_max_all
)
```

Option 2, if we choose to not use a manager.
```python
sp_signal = sigprop.propagators.signals.Loss(
    sp_signal,
    optimizer=build_optimizer(sp_signal),
    loss=sigprop.loss.v9_input_target_max_all
)
```


Note, we by default feed in a vector as the context. For labels, this means
converting to a one\_hot vector of type float. We use a formatter,
such as ``LabelNumberToOnehot`` and place it before the signal 
(refer to [Add a new Signal](#52-add-a-new-signal)).

8. Wrap network layers with propagators.

The last layer is trained normally, so we use the identity operation.

Below are the network layers.
```python
layer_1 = nn.Sequential(
    nn.Conv2d(
        hidden_ch, hidden_ch*2, 3, 2, 1,
        bias=False
    ),
    nn.BatchNorm2d(output_shape[0]),
    nn.ReLU()
)
layer_2 = nn.Sequential(
    nn.Conv2d(
        hidden_ch*2, hidden_ch*4, 3, 2, 1,
        bias=False
    ),
    nn.BatchNorm2d(output_shape[0]),
    nn.ReLU()
)
layer_output = nn.Sequential(
    nn.Linear(
        int(input_dim//2**2),
        int(num_classes),
        bias=True
    )
)
```

There are two options to wrap the layers with propagators.

Option 1, if we are using a manager.
```python
layer_1 = sp_manager.add_propagator(layer_1)
layer_2 = sp_manager.add_propagator(layer_2)
layer_output = sp_manager.add_propagator(layer_output, sigprop.propagators.Identity)
```

Option 2, if we choose to not use a manager.
```python
from sigprop import propagators
layer_1 = propagators.Loss(
    layer_1,
    optimizer=build_optimizer(layer_1),
    loss=sigprop.loss.v9_input_target_max_all
)
layer_2 = propagators.Loss(
    layer_2,
    optimizer=build_optimizer(layer_2),
    loss=sigprop.loss.v9_input_target_max_all
)
layer_output = propagators.Identity(
    layer_output,
    optimizer=build_optimizer(layer_output),
    loss=sigprop.loss.v9_input_target_max_all
)
```

9. Create the network.

Below is the network model:
```python
network = nn.Sequential(
    layer_1,
    layer_2,
    layer_output,
)
```

There are two options to wrap the network model in a sigprop model.

Option 1, if we are using a manager.
```python
network = sp_manager.set_model(network)
```

Option 2, if we choose to not use a manager.
```python
network = sigprop.models.Forward(network, sp_signal)
```

10. train the network.

```python
model.train()

for batch, (data, target) in enumerate(train_loader):

    output = model(data, target)

    loss = F.cross_entropy(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

11. inference from network.

```python
model.eval()

acc_sum = 0.
count = 0.

for batch, (data, target) in enumerate(test_loader):

    output = model(data, None)

    pred = output.argmax(1)
    acc_mask = pred.eq(target)
    acc_sum += acc_mask.sum()
    count += acc_mask.size(0)

acc = acc_sum / count
```

## 3. Examples

Here are a few examples. More will be added.

### Example 2: Input Target Max Rand
```
ex2_input_target_max_rand.py
```

This example feeds the inputs ``x`` (e.g. images) and their respective targets ``t`` (e.g. labels) as
pairs (or one after the other). Given pair ``x_i,t_i``, this example selects the closest matching pair ``x_j,t_j`` to compare
with. If there are multiple equivalent matching pairs, it randomly selects one.

### Example 4: Input Target Top-K
```
ex4_input_target_topk.py
```

This example feeds the inputs ``x`` (e.g. images) and their respective targets ``t`` (e.g. labels) as
pairs (or one after the other). Given pair ``x_i,t_i``, this example selects the top ``k`` closest matching pair ``x_j,t_j`` to compare
with.

This example demonstrates how to add a monitor to each loss and display metrics
for each layer wrapped with a propagator.

## 4. Documentation

Refer to sections 2 and 3 for examples with explainations.

### 4.1. Signals

```
sigprop/signals
```

Signals generate the signal for learning, then forward it to the first layer.
This is taken as the first layer of the network. Or, if a fixed project of the
target is used, i.e. there is no learning, then this takes place before the
first layer of the network.

### 4.2. Propagators

```
sigprop/propagators
```

Propagators train each network layer and forward the signal from one layer to the other.
Each network layer is wrapped in a propagator, so the layer may learn on its own.

### 4.3. Models

```
sigprop/models
```

Models handle the input and output when forward learning. By default, they are
not necessary, only signals and propagators are necessary (i.e. signal
propagation). However, models provide conveniance functionality for common
routines when using signals and propagators.

### 4.4. Managers

```
sigprop/managers
```

Managers allow for upfront configuration (defaults) of signals, propagators,
and models. Upfront configuration is helpful in scenerios where signal
propagation is wrapping an existing model. For example, we may use the manager
to wrap layers in an already existing model. Refer the the examples for
a demonstration.

### 4.5. Functional

```
sigprop/functional
```

The functional interface to signal propagation.

### 4.6. Monitors

```
sigprop/monitors
```

Monitor signals, propagators, and modules to record and display metrics.
In [Example 4: Input Target Top-K](#example-4-input-target-topk), a monitor is
wraps the loss to display metrics on the loss and accuracy for each layer
wrapped with a propagator; in other words, it displays layer level metrics.

## 5. Development

### 5.1. Add a New Loss

Losses are functions or a callable.

Refer to folder ``sigprop/loss`` for examples of losses.

Example new implementation:
```python
def new_loss(sp_learn,h1,t1,h0,t0,y_onehot):
    l = #calc loss

    return l
```

Example new implementation:
```python
class NewLoss(sigprop.loss.Loss):
    def __init__(self):
        super().__init__()

    def forward(sp_learn,h1,t1,h0,t0,y_onehot):
        l = #calc loss

        return l
```

### 5.2. Add a New Signal

There is the signal generator and the optional signal formatter.

_Generator_

Refer to file ``sigprop/signals/generators.py`` for examples of signals.

Note, the signal generators return the original input (h0) and context (t0).
This provides flexibility for fixing up the input and context before it is used
for learning (e.g. reshape). For example, apply a formatter.

Example new implementation:
```python
from sigprop import signals

class MySignalGenerator(signals.Generator):
    def __init__(self, module, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.module = module

    def forward(self, input):
        h0, t0 = input
        t1 = self.module(t0.flatten(1)).view(t0.shape[0:1]+self.output_shape)
        h1 = h0
        return h1, t1, h0, t0
```


_Formatter_

The formatter is optional.

Refer to file ``sigprop/signals/formatter.py`` for examples of formatters.

Example new implementation:
```python
from sigprop import signals

class MySignalFormatter(signals.Formatter):
    def forward(self, input):
        h0, t0 = input
        if t0 is not None:
            t0 = F.one_hot(torch.arange(t0.shape[1], device=t0.device),t0.shape[1]).float()
        return h0, t0
```

### 5.3. Add a New Propagator

There are two types of propagators: one that learn and ones that do not.

_Learn_

Refer to file ``sigprop/propagators/learn.py`` for examples of propagators that
learn.

Currently, propagators for signals have different router logic than hidden layers. 
So, we use a different propagator class for the signal.

Example new implementation:
```python
from sigprop import propagators

class MyLearnPropagator(propagators.Learn):
    def loss_(self,h1,t1,h0,t0,y_onehot):
        loss = # calculate a loss
        return loss

    def train_(self,h1,t1,h0,t0,y_onehot):
        loss = self.loss_(h1,t1,h0,t0,y_onehot)
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item(), None, None

    def eval_(self,h1,t1,h0,t0,y_onehot):
        loss = self.loss_(h1,t1,h0,t0,y_onehot)
        return loss.item(), None, None

class MyLearnPropagatorForSignals(propagators.signals.Learn):
    def loss_(self,h1,t1,h0,t0,y_onehot):
        loss = # calculate a loss
        return loss

    def train_(self,h1,t1,h0,t0,y_onehot):
        loss = self.loss_(h1,t1,h0,t0,y_onehot)
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item(), None, None

    def eval_(self,h1,t1,h0,t0,y_onehot):
        loss = self.loss_(h1,t1,h0,t0,y_onehot)
        return loss.item(), None, None
```

_Other_

Refer to file ``sigprop/propagators/other.py`` for examples of propagators that
do not learn.

Example new implementation:
```python
from sigprop import propagators

class MyPropagator(propagators.Propagator):
    def __init__(self, module):
        super().__init__()

        self.module = module

    def forward(self, input):
        h0, t0, y_onehot = input

        h1 = self.module(h0)
        t1 = t0

        return (h1, t1, y_onehot)
```

### 5.4. Add a New Model

Refer to file ``sigprop/models/model.py`` for examples of forward learning models.

Example new implementation:
```python
from sigprop import models

class MyModel(models.Model):
    def __init__(self, model, signal):
        super().__init__(model, signal)

    def forward(self, input):
        x, y_onehot = input
        h0, t0 = self.signal((x, y_onehot, y_onehot))
        h1, t1, y_onehot = self.model((h0, t0, y_onehot))
        return h1
```
