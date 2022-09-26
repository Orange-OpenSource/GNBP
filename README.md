# Neural Belief Propagation Auto-Encoder for Linear Block Code Design
This workspace describe the tools and results described in the paper [1].


### How to use this repository?
The master branch contains the main source and MWE notebook to run an experiment using the proposed workflow.
The notebook 'study_auto_encoder.ipynb' contain and details the steps to setup and run a complete study.
The notebook 'plot_utils.ipynb' provides the tools to plot the corresponding results.

Each of the other branches of the repository describe one of the experiment presented in the paper:
- V.A. (8,4) Code: Illustration of the Proposed Concept:.............................................branch ae-8-4
- V.B.1. Model Performance During Training:..........................................................branch ae-31-16-training-study
- V.B.2. Training Repeatability:.....................................................................branch ae-31-16-repeatability
- V.B.3. Correlation Between Decoders Types and their Performances:..................................branch ae-31-16-repeatability
- V.C. (31,11) Codes: Auto-encoder Training Procedure and Ablation Study:............................branch ae-31-11 
- V.D. (63,36) & (63,45) Codes: Scalability and Algorithmic Complexity:..............................branch ae-63-36 and ae-63-45
- V.E.1 Comparison with Short (64,k) LDPC Codes - Impact of the Rate:................................branch ae-LDPC
- V.E.2 Comparison with State-of-the-Art (128,64) Codes - Impact of the Number of Iterations:........branch ae-128-64

### References
[1] G. Larue, L. -A. Dufrene, Q. Lampin, H. Ghauch and G. Rekaya, "Neural Belief Propagation Auto-Encoder for Linear Block Code Design," in IEEE Transactions on Communications, 2022, doi: 10.1109/TCOMM.2022.3208331.

Copyright (c) 2022 Orange

Authors: Guillaume Larue <guillaume.larue@orange.com>, Quentin Lampin <quentin.lampin@orange.com>, Louis-Adrien Dufrene <louisadrien.dufrene@orange.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice (including the next paragraph) shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE