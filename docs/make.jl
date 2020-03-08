using Documenter, GenPyTorch

makedocs(
    format = :html,
    sitename = "GenPyTorch",
    modules = [GenPyTorch],
    pages = [
        "Home" => "index.md"
    ]
)

deploydocs(
    repo = "github.com/probcomp/GenPyTorch.git",
    target = "build"
)
