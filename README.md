# Usage
- Clone repo
- Navigate to `JAX_Docker_for_ML/Docker_run.sh` and edit the first line of code to reflect your local path.
> E.g., change 
 `export FOLDER=/home/michael/JAX_Docker_for_ML`
to 
`export FOLDER=<your_root>/JAX_Docker_for_ML`

- Build docker: `$ ./Docker_build.sh`
- Run docker:
  - Launch screen (or tmux, etc.): `$ screen -S jax`
  - Navigate to `<your_root>/JAX_Docker_for_ML`
  - Launch docker: `$ ./Docker_run.sh`
- In Docker, navigate to our scripts: `$ cd /root/mount/jax`
- Launch jupyter notebook: `$ ./Docker_start_jupyter.sh <server_port>`
  - Default `server_port = 8990`
  - Copy token generated
  - Close screen session / tmux

- If hosting on server
  - SSH tunnel into environment: `$ ssh -NL <local_port>:localhost:<server_port> <username>@<server_name>`
  - Enter password, screen will hang without fully logging into server
  - In browser, navigate to `localhost:<local_port>`
  - You will be prompted for the token from above step; copy and paste
  - You can now use JAX
  
- If hosting locally, well, it's currently broken for me so you're on your own for now! Will fix in future, and when that happens:
  - In browser, navigate to `localhost:<server_port>`
  - You will be prompted for the token from above step; copy and paste
  - You can now use JAX
   


# References
- [JAX Github](https://github.com/google/jax/)
- [JAX Docs](https://jax.readthedocs.io/en)
- [Getting started with JAX (MLPs, CNNs & RNNs)](https://roberttlange.github.io/posts/2020/03/blog-post-10/)
- [Sussillo - Computation thru Dynamics](https://github.com/google-research/computation-thru-dynamics)
- [You Don't Know JAX](https://colinraffel.com/blog/you-don-t-know-jax.html)
- [NeurIPS JAX Talk](https://slideslive.com/38923687/jax-accelerated-machinelearning-research-via-composable-function-transformations-in-python)
- [Parallax - Immutable Torch Modules for JAX](https://github.com/srush/parallax)
- [From PyTorch to JAX: towards neural net frameworks that purify stateful code](https://sjmielke.com/jax-purify.htm)


# Personal Notes

## Pytorch vs. JAX
A note on the difference of PyTorch ResNet models and the ResNet50 JAX examples. 


In Pytorch, bias is disabled on the conv operation [[source code](https://github.com/pytorch/vision/blob/5ba57eaef070b9eee42f9aa63cd9ab149354ac1c/torchvision/models/resnet.py#L27)]. This is because, ["Biases are in the BN layers that follow." - Kaiming He](https://github.com/KaimingHe/deep-residual-networks/issues/10) -- [see discussion](https://github.com/KaimingHe/deep-residual-networks/issues/10). Also, see [BN source code](https://github.com/pytorch/pytorch/blob/b636f5e324f19ebed867d6c3088580a6d6793859/torch/nn/modules/batchnorm.py#L27) for proof. 


In JAX, bias is not built into the BN layers ([source code](https://github.com/google/jax/blob/ec3b593ca85d6f5c3b538b6615dfbd8c8ffe8148/jax/experimental/stax.py#L119-L140)), therefore we should use bias in the conv layers. 


However, it leads one to realize that while using [experimental.stax](https://jax.readthedocs.io/en/latest/jax.experimental.stax.html).Conv, you cannot disable bias ([source code](https://github.com/google/jax/blob/ec3b593ca85d6f5c3b538b6615dfbd8c8ffe8148/jax/experimental/stax.py#L61-L86))!


Additionally, the experimental.stax unfortunately does not expose kernel dilation - if you want to use this feature, you have to build your own abstract Conv layer around [lax.conv_general_dilated](https://github.com/google/jax/blob/ec3b593ca85d6f5c3b538b6615dfbd8c8ffe8148/jax/lax/lax.py#L469).


One more note on the difference - in PyTorch when specifying kernel sizes, strides, etc., you can pass integer values and they will automatically be upgraded to appropriately sized tuple; e.g., kernel_size=1 --> kernel_size=(1,1) for 2D convolutions. In JAX, you must specify exactly the shape; e.g., kernel_size=1 --> exception thrown.