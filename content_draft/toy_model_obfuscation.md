
This is an important question - although this is a toy model, its structure is reminiscent of how transformer models work, just on a much smaller scale.
Normally, LLMs learn to store "features" as directions in activation space, such that you can train a linear probe to detect them.
This is really useful!
We can use this as a cheap monitor to detect things like whether the model is lying or whether the user's prompt is dangerous.
You might think that if we can detect misaligned behaviour, we can just add the probe's score as a loss term in training, to align the model.
However, training on mech-interp monitors like probes is considered [a forbidden technique](https://www.lesswrong.com/posts/mpmsK8KKysgSKDm2T/the-most-forbidden-technique), because the model could very well just learn to obfuscate its representations instead of learning what you wanted it to learn - and now you have a misaligned model *and* one less way to monitor it.
