using Gen
using GenPyTorch
using Test
using PyCall

torch = pyimport("torch")
nn = torch.nn
F = nn.functional

@pydef mutable struct Model <: nn.Module
    function __init__(self)
        pybuiltin(:super)(Model, self).__init__()
        self.fc1 = nn.Linear(3, 2)
    end

    function forward(self, x)
        x = self.fc1(x)
        return x
    end
end

model = Model()
foo = TorchGenerativeFunction(model, [TorchArg(true, torch.float)], 1)

@testset "basic" begin

    W = model.fc1.weight.data.numpy()
    b = model.fc1.bias.data.numpy()

    x = rand(3)

    # test generate
    (trace, weight) = generate(foo, (x,))
    @test weight == 0.
    y = get_retval(trace)
    @test isapprox(y, W * x .+ b, atol=1e-5)

    # test simulate
    trace = simulate(foo, (x,))
    y = get_retval(trace)
    @test isapprox(y, W * x .+ b, atol=1e-5)

    # test accumulate_param_gradients!
    y_grad = rand(Float32, 2)
    (x_grad,) = accumulate_param_gradients!(trace, y_grad)
    @test isapprox(x_grad, W' * y_grad, atol=1e-5)
end
