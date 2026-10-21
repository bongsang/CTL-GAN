# Attribution and licensing

CTL-GAN uses **NVIDIA StyleGAN3** as its generative architecture. CTL-GAN's
contribution is the collaborative transfer learning application described in the
paper; the underlying GAN architecture and its implementation are credited to
NVIDIA and the StyleGAN3 authors.

- Upstream: https://github.com/NVlabs/stylegan3
- Reference commit: `c233a919a6faee6e36a316ddd4eddababad1adf9`
- Paper: [Alias-Free Generative Adversarial Networks](https://nvlabs.github.io/stylegan3/)
- License: [NVIDIA Source Code License for StyleGAN3](LICENSE.txt)

The existing `dnnlib/`, `torch_utils/`, and `training/networks_stylegan3.py` files
match this upstream revision byte for byte. Original NVIDIA copyright notices
are retained. `training/`, `train.py`, `environment.yml`, and the original
StyleGAN documentation originate from that implementation. The maker wrapper,
model export tool, tests, and CTL-GAN documentation are additions in this release;
these additions are distributed under the same NVIDIA Source Code License terms.
The license permits research/evaluation use and restricts commercial use; it is
not an MIT or Apache license. See the complete license for its terms.

The paper's Creative Commons publication license does not replace the software
license or grant access to the source clinical datasets. The model release notes
specify the scope and provenance of each downloadable artifact.

`maker/runtime.py` selects NVIDIA's existing reference implementations at runtime
when `--impl ref` is used, restoring the original functions afterwards. It does
not replace the generator architecture or change its learned tensors.
