# GenPyTorch

*PyTorch plugin for the Gen probabilistic programming system*

The Julia package [GenPyTorch](https://github.com/probcomp/GenPyTorch) allows for [Gen](https://github.com/probcomp/Gen) generative functions to invoke PyTorch modules executed on the GPU.
Users construct a PyTorch module using the familiar Torch Python API, and then package it in a `TorchGenerativeFunction`, which is a type of generative function provided by GenPyTorch.
Generative functions written in Gen's built-in modeling language can seamlessly call `TorchGenerativeFunction`s.
GenPyTorch integrates Gen's automatic differentiation with PyTorch's gradients, allowing automatic differentiation of computations that combine Julia and PyTorch code.

## Installation

The installation requires an installation of Python and an installation of the [torch](https://pytorch.org/get-started/locally/) Python package.
We recommend creating a Python virtual environment and installing Torch via `pip` in that environment.
In what follows, let `<python>` stand for the absolute path of a Python executable that has access to the `torch` package.

From the Julia REPL, type `]` to enter the Pkg REPL mode and run:
```
pkg> add https://github.com/probcomp/GenPyTorch
```
In a Julia REPL, build the `PyCall` module so that it will use the correct Python environment:
```julia
using Pkg; ENV["PYTHON"] = "<python>"; Pkg.build("PyCall")
```
Check that intended python environment is indeed being used with:
```julia
using PyCall; println(PyCall.python)
```
If you encounter problems, see https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version


## Calling the PyTorch API

GenPyTorch uses the Julia package [PyCall](https://github.com/JuliaPy/PyCall.jl) to invoke the [PyTorch API](https://pytorch.org/docs/stable/torch.html).

First, import PyCall:
```julia
using PyCall
```

You can define a PyTorch module using Python directly, enclosing any Python
in `py"""..."""` strings:

```julia
py"""
import torch
import torch.nn as nn
import torch.nn.functional as F
class MyModel(torch.nn.Module):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 3)
    self.conv2 = nn.Conv2d(6, 16, 3)
    self.fc1 = nn.Linear(16 * 6 * 6, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
      return num_features
"""
```

You can then instantiate your model:

```julia
model = py"MyModel()"
```

The Julia variable `model` now holds a `PyObject` representing your neural network.
This can be wrapped in a Torch Generative Function (described in the next section).

An alternative to specifying your model entirely in Python is to use PyCall to work
in Julia, which may be useful if your module needs to call some Julia code you've written.
To do this, use `pyimport` to import `torch`, and `@pydef` to define your module:

```julia
using PyCall

torch = pyimport("torch")
nn = torch.nn
F = nn.functional

@pydef mutable struct MyModel <: nn.Module
    function __init__(self)
        # Note the use of pybuiltin(:super): built in Python functions
        # like `super` or `str` or `slice` are all accessed using
        # `pybuiltin`.
        pybuiltin(:super)(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    end

    function forward(self, x)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    end

    function num_flat_features(self, x)
        # Note: x.size() returns a tuple, not a tensor.
        # Therefore, we treat it like a Julia tuple and
        # index using 1-based indexing.
        size = x.size()[2:end]
        num_features = 1
        for s in size
            num_features *= s
        end
        return num_features
    end
end
```

You can instantiate the model without `py"..."`:

```julia
model = MyModel()
```

## PyTorch Generative Functions

Once you've instantiated your model as a `PyObject` (as we did with the variable `model` above),
you can convert it into a generative function:

```julia
model_gf = TorchGenerativeFunction(model, inputs, n_outputs)
```

Here, `n_outputs` is the number of output tensors returned by the `forward` function,
and `inputs` should be a list of `TorchArg` objects, one for each argument
to your model's `forward` function. A `TorchArg` is constructed with two arguments:
a Boolean `supports_gradients` argument, for whether gradients should flow through
that argument, and a `dtype` argument, which can either be `PyNULL()` for non-tensor
arguments, or the `dtype` of the input tensor (e.g. `torch.float` or `torch.double`):

```julia
# If you used the `@pydef` approach, you can write torch.float directly below,
# without enclosing it in a py"..." string.
model_gf = TorchGenerativeFunction(model, [TorchArg(true, py"torch.float")], 1)
```

The `model_gf` function can now be used as an ordinary generative function.
In particular, it can be called from Gen's static or dynamic DSL.
As a generative function, `model_gf` is deterministic; it makes no random choices
and always returns empty choicemaps. But it does have trainable parameters:

```julia
Gen.get_params(model_gf)
```
```
Base.KeySet for a Dict{String,PyObject} with 10 entries. Keys:
  "fc3.weight"
  "conv1.bias"
  "fc1.weight"
  "conv2.weight"
  "fc1.bias"
  "conv1.weight"
  "fc3.bias"
  "fc2.bias"
  "fc2.weight"
  "conv2.bias"
```

These can be trained the same way that any trainable parameters are trained in Gen.
First, use the Torch generative function from within a probabilistic model:

```julia
@gen function classify_mnist(images)
  classifications ~ my_model(images)
  for i=1:length(images)
    {:class_for => i} ~ categorical(softmax(classifications[i, :]))
  end
end
```

Then, generate a trace from your data:

```julia
param_update = ParamUpdate(ADAM(0.01, 0.9, 0.999, 1e-8), my_model)
for i=1:100
  xs, ys = next_batch()
  constraints = choicemap([(:class_for => i) => ys[i] for i=1:length(xs)]...)
  trace = Gen.generate(classify_mnist, (xs,), constraints)
  accumulate_param_gradients!(trace)
  apply!(param_update)
end
```

## API

```@docs
TorchGenerativeFunction
TorchArg
TorchOptimConf
set_torch_device!
```
