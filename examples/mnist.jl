##############
# load mnist #
##############

import Random
Random.seed!(1)

import MLDatasets
train_x, train_y = MLDatasets.MNIST.traindata()

mutable struct DataLoader
    cur_id::Int
    order::Vector{Int}
end

DataLoader() = DataLoader(1, Random.shuffle(1:60000))

function next_batch(loader::DataLoader, batch_size)
    x = zeros(Float64, batch_size, 1, 28, 28)
    y = Vector{Int}(undef, batch_size)
    for i=1:batch_size
        x[i, 1, :, :] = train_x[:,:,loader.cur_id]
        y[i] = train_y[loader.cur_id] + 1
        loader.cur_id = (loader.cur_id % 60000) + 1
    end
    x, y
end

function load_test_set()
    test_x, test_y = MLDatasets.MNIST.testdata()
    N = length(test_y)
    x = zeros(Float64, N, 1, 28, 28)
    y = Vector{Int}(undef, N)
    for i=1:N
        x[i, 1, :, :] = test_x[:,:,i]
        y[i] = test_y[i]+1
    end
    x, y
end

const loader = DataLoader()

(test_x, test_y) = load_test_set()


################
# define model #
################

using Gen
using GenPyTorch
using PyCall

torch = pyimport("torch")
nn = torch.nn
F = torch.nn.functional

@pydef mutable struct Model <: nn.Module
    function __init__(self)
        pybuiltin(:super)(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    end

    function forward(self, x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    end
end

net = TorchGenerativeFunction(Model(), [TorchArg(true, torch.float)], 1)

softmax(xs) = exp.(xs .- logsumexp(xs))

@gen function f(xs::Matrix{Float64})
    probs ~ net(xs)
    return [{:y => i} ~ categorical(softmax(p))
            for (i, p) in enumerate(eachrow(probs))]
end

#########
# train #
#########

update = ParamUpdate(ADAM(1e-3, 0.9, 0.999, 1e-08), net)
for i=1:1500
    # Create trace from data
    (xs, ys) = next_batch(loader, 100)
    constraints = choicemap([(:y => i) => y for (i, y) in enumerate(ys)]...)
    (trace, weight) = generate(f, (xs,), constraints)

    # Increment gradient accumulators
    accumulate_param_gradients!(trace)

    # Perform ADAM update and then resets gradient accumulators
    apply!(update)
    println("i: $i, weight: $weight")
end

##################################
# sample inferences on test data #
##################################

using Statistics: mean
test_accuracy = mean(f(test_x) .== test_y)
println("Test set accuracy: $test_accuracy")
