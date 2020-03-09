var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#GenPyTorch-1",
    "page": "Home",
    "title": "GenPyTorch",
    "category": "section",
    "text": "PyTorch plugin for the Gen probabilistic programming systemThe Julia package GenPyTorch allows for Gen generative functions to invoke PyTorch modules executed on the GPU. Users construct a PyTorch module using the familiar Torch Python API, and then package it in a TorchGenerativeFunction, which is a type of generative function provided by GenPyTorch. Generative functions written in Gen\'s built-in modeling language can seamlessly call TorchGenerativeFunctions. GenPyTorch integrates Gen\'s automatic differentiation with PyTorch\'s gradients, allowing automatic differentiation of computations that combine Julia and PyTorch code."
},

{
    "location": "#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "The installation requires an installation of Python and an installation of the torch Python package. We recommend creating a Python virtual environment and installing Torch via pip in that environment. In what follows, let <python> stand for the absolute path of a Python executable that has access to the torch package.From the Julia REPL, type ] to enter the Pkg REPL mode and run:pkg> add https://github.com/probcomp/GenPyTorchIn a Julia REPL, build the PyCall module so that it will use the correct Python environment:using Pkg; ENV[\"PYTHON\"] = \"<python>\"; Pkg.build(\"PyCall\")Check that intended python environment is indeed being used with:using PyCall; println(PyCall.python)If you encounter problems, see https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version"
},

{
    "location": "#Calling-the-PyTorch-API-1",
    "page": "Home",
    "title": "Calling the PyTorch API",
    "category": "section",
    "text": "GenPyTorch uses the Julia package PyCall to invoke the PyTorch API.First, import PyCall:using PyCallYou can define a PyTorch module using Python directly, enclosing any Python in py\"\"\"...\"\"\" strings:py\"\"\"\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nclass MyModel(torch.nn.Module):\n  def __init__(self):\n    super(MyModel, self).__init__()\n    self.conv1 = nn.Conv2d(1, 6, 3)\n    self.conv2 = nn.Conv2d(6, 16, 3)\n    self.fc1 = nn.Linear(16 * 6 * 6, 120)\n    self.fc2 = nn.Linear(120, 84)\n    self.fc3 = nn.Linear(84, 10)\n\n  def forward(self, x):\n    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))\n    x = F.max_pool2d(F.relu(self.conv2(x)), 2)\n    x = x.view(-1, self.num_flat_features(x))\n    x = F.relu(self.fc1(x))\n    x = F.relu(self.fc2(x))\n    x = self.fc3(x)\n    return x\n\n  def num_flat_features(self, x):\n    size = x.size()[1:]  # all dimensions except the batch dimension\n    num_features = 1\n    for s in size:\n      num_features *= s\n      return num_features\n\"\"\"You can then instantiate your model:model = py\"MyModel()\"The Julia variable model now holds a PyObject representing your neural network. This can be wrapped in a Torch Generative Function (described in the next section).An alternative to specifying your model entirely in Python is to use PyCall to work in Julia, which may be useful if your module needs to call some Julia code you\'ve written. To do this, use pyimport to import torch, and @pydef to define your module:using PyCall\n\ntorch = pyimport(\"torch\")\nnn = torch.nn\nF = nn.functional\n\n@pydef mutable struct MyModel <: nn.Module\n    function __init__(self)\n        # Note the use of pybuiltin(:super): built in Python functions\n        # like `super` or `str` or `slice` are all accessed using\n        # `pybuiltin`.\n        pybuiltin(:super)(Model, self).__init__()\n        self.conv1 = nn.Conv2d(1, 6, 3)\n        self.conv2 = nn.Conv2d(6, 16, 3)\n        self.fc1 = nn.Linear(16 * 6 * 6, 120)\n        self.fc2 = nn.Linear(120, 84)\n        self.fc3 = nn.Linear(84, 10)\n    end\n\n    function forward(self, x)\n        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))\n        x = F.max_pool2d(F.relu(self.conv2(x)), 2)\n        x = x.view(-1, self.num_flat_features(x))\n        x = F.relu(self.fc1(x))\n        x = F.relu(self.fc2(x))\n        x = self.fc3(x)\n        return x\n    end\n\n    function num_flat_features(self, x)\n        # Note: x.size() returns a tuple, not a tensor.\n        # Therefore, we treat it like a Julia tuple and\n        # index using 1-based indexing.\n        size = x.size()[2:end]\n        num_features = 1\n        for s in size\n            num_features *= s\n        end\n        return num_features\n    end\nendYou can instantiate the model without py\"...\":model = MyModel()"
},

{
    "location": "#PyTorch-Generative-Functions-1",
    "page": "Home",
    "title": "PyTorch Generative Functions",
    "category": "section",
    "text": "Once you\'ve instantiated your model as a PyObject (as we did with the variable model above), you can convert it into a generative function:model_gf = TorchGenerativeFunction(model, inputs, n_outputs)Here, n_outputs is the number of output tensors returned by the forward function, and inputs should be a list of TorchArg objects, one for each argument to your model\'s forward function. A TorchArg is constructed with two arguments: a Boolean supports_gradients argument, for whether gradients should flow through that argument, and a dtype argument, which can either be PyNULL() for non-tensor arguments, or the dtype of the input tensor (e.g. torch.float or torch.double):# If you used the `@pydef` approach, you can write torch.float directly below,\n# without enclosing it in a py\"...\" string.\nmodel_gf = TorchGenerativeFunction(model, [TorchArg(true, py\"torch.float\")], 1)The model_gf function can now be used as an ordinary generative function. In particular, it can be called from Gen\'s static or dynamic DSL. As a generative function, model_gf is deterministic; it makes no random choices and always returns empty choicemaps. But it does have trainable parameters:Gen.get_params(model_gf)Base.KeySet for a Dict{String,PyObject} with 10 entries. Keys:\n  \"fc3.weight\"\n  \"conv1.bias\"\n  \"fc1.weight\"\n  \"conv2.weight\"\n  \"fc1.bias\"\n  \"conv1.weight\"\n  \"fc3.bias\"\n  \"fc2.bias\"\n  \"fc2.weight\"\n  \"conv2.bias\"These can be trained the same way that any trainable parameters are trained in Gen. First, use the Torch generative function from within a probabilistic model:@gen function classify_mnist(images)\n  classifications ~ my_model(images)\n  for i=1:length(images)\n    {:class_for => i} ~ categorical(softmax(classifications[i, :]))\n  end\nendThen, generate a trace from your data:param_update = ParamUpdate(ADAM(0.01, 0.9, 0.999, 1e-8), my_model)\nfor i=1:100\n  xs, ys = next_batch()\n  constraints = choicemap([(:class_for => i) => ys[i] for i=1:length(xs)]...)\n  trace = Gen.generate(classify_mnist, (xs,), constraints)\n  accumulate_param_gradients!(trace)\n  apply!(param_update)\nend"
},

{
    "location": "#GenPyTorch.TorchGenerativeFunction",
    "page": "Home",
    "title": "GenPyTorch.TorchGenerativeFunction",
    "category": "type",
    "text": "gen_fn = TorchGenerativeFunction(torch_module::PyObject,\n                                 inputs::Vector{TorchArg},\n                                 n_outputs::Int)\n\nConstruct a Torch generative function from a Torch module. By default, computations will run on GPU if available and CPU otherwise.\n\ngen_fn = TorchGenerativeFunction(torch_module::PyObject,\n                                 inputs::Vector{TorchArg},\n                                 n_outputs::Int,\n                                 device::PyObject)\n\nConstruct a Torch generative function from a Torch module. Computations will be run on the given device.\n\n\n\n\n\n"
},

{
    "location": "#GenPyTorch.TorchArg",
    "page": "Home",
    "title": "GenPyTorch.TorchArg",
    "category": "type",
    "text": "TorchArg(supports_gradients::Bool, dtype::PyObject)\n\nA description of an argument to the forward function of a Torch module. If dtype is PyNULL(), this argument is not a tensor.\n\n\n\n\n\n"
},

{
    "location": "#GenPyTorch.TorchOptimConf",
    "page": "Home",
    "title": "GenPyTorch.TorchOptimConf",
    "category": "type",
    "text": "TorchOptimConf(func::PyObject, args::Vector{Any}, kwargs::Dict{Symbol, Any})\n\nCan be used as the first argument to ParamUpdate to construct a parameter update based on an arbitrary torch.optim optimizer. The func argument should be the torch.optim optimizer (e.g. torch.optim.SGD), and the args and kwargs are the arguments and keyword arguments to the optimizer, e.g. for setting the learning rate. You need not include a list of parameters to optimize; Gen will handle that part.\n\n\n\n\n\n"
},

{
    "location": "#API-1",
    "page": "Home",
    "title": "API",
    "category": "section",
    "text": "TorchGenerativeFunction\nTorchArg\nTorchOptimConf"
},

]}
