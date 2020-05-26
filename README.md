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
  
- If hosting locally
  - In browser, navigate to `localhost:<server_port>`
  - You will be prompted for the token from above step; copy and paste
  - You can now use JAX
   


# References
- [JAX Github](https://github.com/google/jax/)
- [JAX Docs](https://jax.readthedocs.io/en)
- [Getting started with JAX (MLPs, CNNs & RNNs)](https://roberttlange.github.io/posts/2020/03/blog-post-10/)
- [Sussillo - Computation thru Dynamics](https://github.com/google-research/computation-thru-dynamics)
- [You Don't Know JAX](https://colinraffel.com/blog/you-don-t-know-jax.html)