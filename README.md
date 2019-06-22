# AttentionOCR.jl
(WIP) Julia and Flux adaptation of the python package aocr.py https://github.com/emedvedev/attention-ocr using Attention mechanism from https://github.com/merckxiaan/flux-seq2seq

The scope at the moment is only numeric digits 0-9

# Current State
* Package pre-compiles successfully
* Flux model is able to train but slowly

# Todo
* Add predict function
* Add unemmebeding function to take predictions from vectors to digits
* Test that the model works on example data used successfully with aocr.py
