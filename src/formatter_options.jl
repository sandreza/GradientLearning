using JuliaFormatter

kwargs = (;
    indent = 4,
    margin = 80,
    always_for_in = true,
    whitespace_typedefs = true,
    whitespace_ops_in_indices = true,
    remove_extra_newlines = true,
    whitespace_in_kwargs = true,
    indent_submodule = true,
    align_assignment = true,
    align_struct_field = true,
    align_conditional = true,
    align_pair_arrow = true,
    align_matrix = true,
)

format("."; kwargs...)
