# porijen! pytorch!!
[Polygen](https://arxiv.org/abs/2002.10880)-like model implemented in pytorch.<br>
I use [Reformer](https://arxiv.org/abs/2001.04451) with [reformer-pytorch](https://github.com/lucidrains/reformer-pytorch) module as backend transformer.

Now this repository support only 
- vertex generation (without class/image queries)
- vertex -> face prediction (without class/image queries)

<b>this repository may contain tons of bugs.</b>


## development environment
### python modules
- numpy==1.20.2
- pandas==1.2.4
- pytorch==1.8.0
- reformer-pytorch==1.2.4
- open3d==0.11.2
- meshplot==0.3.3
- pythreejs==2.3.0

### blender
- version: 2.92.0