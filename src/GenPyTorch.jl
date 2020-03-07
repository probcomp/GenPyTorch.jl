module GenPyTorch

using Gen
using PyCall

const torch = PyNULL()
TORCH_DEVICE = PyNULL()

function __init__()
    copy!(torch, pyimport("torch"))
    TORCH_DEVICE = torch.cuda.is_available() ? torch.device("cuda:0") : torch.device("cpu")
end

function set_torch_device!(d)
    TORCH_DEVICE = torch.device(d)
end

struct TorchFunctionTrace <: Gen.Trace
    gen_fn :: GenerativeFunction
    args :: Tuple
    retval :: Any
end

Gen.get_args(trace::TorchFunctionTrace) = trace.args
Gen.get_retval(trace::TorchFunctionTrace) = trace.retval
Gen.get_choices(::TorchFunctionTrace) = EmptyChoiceMap()
Gen.get_score(::TorchFunctionTrace) = 0.
Gen.get_gen_fn(trace::TorchFunctionTrace) = trace.gen_fn

struct TorchArg
    supports_gradients :: Bool
    dtype :: PyObject
end

is_tensor(arg::TorchArg) = arg.dtype != PyNULL()

struct TorchGenerativeFunction <: Gen.GenerativeFunction{Any,TorchFunctionTrace}
    torch_module :: PyObject
    inputs :: Vector{TorchArg}
    n_outputs :: Int

    # For fast lookup of parameters by name
    params :: Dict{String, PyObject}
    function TorchGenerativeFunction(torch_module, inputs, n_outputs)
        parameters = collect(torch_module.named_parameters())
        return new(torch_module, inputs, n_outputs, Dict([p[1] => p[2] for p in parameters]...))
    end
end


function Gen.simulate(gen_fn::TorchGenerativeFunction, args::Tuple)
    # Convert all arguments to tensors, moving to GPU if necessary.
    tensor_args = map(zip(gen_fn.inputs, args)) do (arg, value)
        if is_tensor(arg)
          torch.as_tensor(value, dtype=arg.dtype).to(TORCH_DEVICE)
        else
          value
        end
    end

    # Run the model without gradients
    retval = nothing
    @pywith torch.no_grad() begin
        retval = gen_fn.torch_module.to(TORCH_DEVICE)(tensor_args...)
    end

    # Wrap results in a trace.
    # Note: forcing tensors to come back to the CPU could be inefficient.
    # It is useful when the user will do something with the returned tensors
    # like sampling a categorical. But if the tensor is just flowing through
    # to the next Torch call, it would be better to leave it as a Tensor.
    # So maybe we need a flag to turn off the auto-conversion.
    gen_fn.n_outputs == 1 && (retval = convert(Array{Float64}, retval.detach().cpu().numpy()))
    gen_fn.n_outputs >  1 && (retval = [convert(Array{Float64}, r.detach().cpu().numpy()) for r in retval])
    TorchFunctionTrace(gen_fn, args, retval)
end

function Gen.generate(gen_fn::TorchGenerativeFunction, args::Tuple, ::ChoiceMap)
    trace = simulate(gen_fn, args)
    (trace, 0.)
end

function Gen.propose(gen_fn::TorchGenerativeFunction, args::Tuple)
    trace = simulate(gen_fn, args)
    retval = get_retval(trace)
    (EmptyChoiceMap(), 0., retval)
end

Gen.project(::TorchFunctionTrace, ::Selection) = 0.

function Gen.update(trace::TorchFunctionTrace, ::Tuple, ::Any, ::ChoiceMap)
    (trace, 0., DefaultRetDiff(), EmptyChoiceMap())
end

function Gen.regenerate(trace::TorchFunctionTrace, ::Tuple, ::Any, ::Selection)
    (trace, 0., DefaultRetDiff())
end

function set_requires_grad!(torch_module, val)
    for p in torch_module.parameters()
        p.requires_grad_(val)
    end
end

function torch_backward(gen_fn :: TorchGenerativeFunction, res :: PyObject, retval_grad, multiplier = 1.0)
    if isnothing(retval_grad)
        return
    end
    if gen_fn.n_outputs == 1
        res.backward(torch.as_tensor(retval_grad * multiplier * -1, dtype=res.dtype).to(TORCH_DEVICE))
    else
        for i=1:gen_fn.n_outputs
            if !isnothing(retval_grad[i])
                res[i].backward(torch.as_tensor(retval_grad[i], dtype=res.dtype).to(TORCH_DEVICE) * multiplier * -1)
            end
        end
    end
end

function run_with_gradients(trace :: TorchFunctionTrace, retval_grad, acc_param_grads=false, multiplier = 1)
    gen_fn = get_gen_fn(trace)
    args = get_args(trace)

    arg_tensors = map(zip(gen_fn.inputs, args)) do (arg, value)
        if is_tensor(arg)
            value = torch.as_tensor(value, dtype=arg.dtype).to(TORCH_DEVICE)
        end
        if arg.supports_gradients
            value.requires_grad_()
        end
        value
    end

    set_requires_grad!(gen_fn.torch_module, acc_param_grads)

    # Run the model
    res = gen_fn.torch_module(arg_tensors...)
    torch_backward(gen_fn, res, retval_grad, multiplier)
    input_grads = [arg.supports_gradients && !isnothing(tensor.grad) ?
                    convert(Array{Float64}, tensor.grad.detach().cpu().numpy()) : nothing
                    for (arg, tensor) in zip(gen_fn.inputs, arg_tensors)]
    (input_grads...,)
end


# The only piece of this that matters is input_grads
function Gen.choice_gradients(trace::TorchFunctionTrace, ::Selection, retval_grad)
    input_grads = run_with_gradients(trace, retval_grad, false)
    ((input_grads...,), EmptyChoiceMap(), EmptyChoiceMap())
end

function Gen.accumulate_param_gradients!(trace::TorchFunctionTrace, retval_grad, multiplier)
    run_with_gradients(trace, retval_grad, true, multiplier)
end

Gen.get_params(gen_fn::TorchGenerativeFunction) = keys(gen_fn.params)

Gen.accepts_output_grad(gen_fn::TorchGenerativeFunction) = true
Gen.has_argument_grads(gen_fn::TorchGenerativeFunction) = ([arg.supports_gradients for arg in gen_fn.inputs]...,)

struct TorchOptimizer
    opt :: PyObject
end

function Gen.init_update_state(conf::FixedStepGradientDescent, gen_fn::TorchGenerativeFunction, params)
    opt = torch.optim.SGD([gen_fn.params[p] for p in params], lr=conf.step_size)
    TorchOptimizer(opt)
end

function Gen.init_update_state(conf::ADAM, gen_fn::TorchGenerativeFunction, params)
    opt = torch.optim.Adam([gen_fn.params[p] for p in params],
        lr = conf.learning_rate, betas = (conf.beta1, conf.beta2), eps=conf.epsilon)
    TorchOptimizer(opt)
end

struct TorchOptimConf
    func   :: PyObject
    args   :: Vector{Any}
    kwargs :: Dict{Symbol, Any}
end

function Gen.init_update_state(conf::TorchOptimConf, gen_fn::TorchGenerativeFunction, params)
    opt = conf.func([gen_fn.params[p] for p in params], conf.args...; conf.kwargs...)
    TorchOptimizer(opt)
end

function Gen.apply_update!(state::TorchOptimizer)
    state.opt.step()
    state.opt.zero_grad()
end

end
