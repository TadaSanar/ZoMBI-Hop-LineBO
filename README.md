# ZoMBI-Hop
 
![zombi](./figs/zombi-hop.png)


All the documentation regarding Line-BO is currently in the Line-BO repository: https://github.com/TadaSanar/Line-BO 

Start a ZoMBI-Hop-LineBO run: Run zombihop-line.py

Current implementation: The Line-BO functionality is implemented via sampler.py.

Note: ZoMBI-Hop uses acquisition matrix by default. It is very slow with Line-BO, so I implemented an acquisition function that is many times faster. Both options can be run in principle. There are also some other changes compared to the original ZoMBI-Hop. Ask Armi for details.

Additionally, the current implementation integrates the acquisition function only over the zoomed-in space (if I remember correctly). But then it also provides A and B from the edges of the zoomed-in space, which is not practical in experiments (since the majority of the benefits of the Line-BO are lost if the lines are extremely short). Let's think about the best implementation!

Another point of development is that ZoMBI-Hop does not have constraints implemented and they are needed at least for composition optimizations.
