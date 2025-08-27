from gaussian_renderer.render import render
from gaussian_renderer.neilf import render_neilf
from gaussian_renderer.render_inverse import render_neilf_inverse
from gaussian_renderer.render_scalar import render_scalar

render_fn_dict = {
    "render": render,
    "scalar_render": render_scalar,
    "phong": render_neilf,
    "inverse": render_neilf_inverse,
}