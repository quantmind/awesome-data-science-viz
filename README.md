# Data Science & Visualization [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)


A curated list of data science, machine learning and visualization tools with
emphasis on [python][], [d3][] and web applications.

[CONTRIBUTING](https://github.com/quantmind/awesome-data-science-viz/blob/master/contributing.md)

## Contents

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Machine Learning](#machine-learning)
  - [Resources](#resources)
  - [Frameworks](#frameworks)
  - [Neural networks](#neural-networks)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Examples](#examples)
- [NLP](#nlp)
  - [Analysis](#analysis)
  - [Tools](#tools)
  - [Resources](#resources-1)
- [Images](#images)
  - [Resources](#resources-2)
  - [Frameworks](#frameworks-1)
- [Data](#data)
  - [Sources](#sources)
  - [Aggregators](#aggregators)
  - [Explore](#explore)
  - [Storage](#storage)
- [Visualization](#visualization)
  - [Resources](#resources-3)
  - [JavaScript Libraries](#javascript-libraries)
  - [Python Libraries](#python-libraries)
  - [D3 based libraries](#d3-based-libraries)
  - [Digital Art](#digital-art)
- [Languages](#languages)
  - [Python](#python)
  - [JavaScript](#javascript)
- [License](#license)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Machine Learning

### Resources

* [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning) comprehensive list of machine learning resources
* [Dive into machine learning](https://github.com/hangtwenty/dive-into-machine-learning) collections of links and notebooks for a gentle introduction to machine learning
* [TopDeepLearning](https://github.com/aymericdamien/TopDeepLearning) is a list of popular github projects related to deep learning (ranked by stars)
* [Probabilistic Programming and Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) An introduction to Bayesian methods + probabilistic programming with a computation/understanding-first, mathematics-second point of view. All in pure Python
* [Data science ipython notebooks](https://github.com/donnemartin/data-science-ipython-notebooks)
* [Python data-science handbook](https://github.com/jakevdp/PythonDataScienceHandbook)
* [Deep Learning Papers Reading Roadmap](https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap)

### Frameworks
* [Theano][] is a Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently
* [TensorFlow][] library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them.
* [Keras](https://github.com/fchollet/keras) Deep Learning library for [Theano][], [TensorFlow][] and [CNTK][].
* [Caffe](https://github.com/BVLC/caffe) deep learning framework made with expression, speed, and modularity in mind. Written in C++ and has python bindings.
* [Torch](https://github.com/torch/torch7) provides several tools for fast tensor mathematics, storage interfaces and machine learning models. Written in C with Lua interface.
* [PyTorch](https://pytorch.org/) tensors and dynamic neural networks in Python with strong GPU acceleration
* [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit) is a machine learning system which pushes the frontier of machine learning with techniques such as online, hashing, allreduce, reductions, learning2search, active, and interactive learning. Writtent in C++ with bindings for python and other languages.
* [Scikit Learn](https://github.com/scikit-learn/scikit-learn) is a Python module for machine learning built on top of [SciPy](https://www.scipy.org/)
* [CNTK][] computational network toolkit. A C++ library by Microsoft Research.
* [XGboost](https://github.com/dmlc/xgboost) an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. Written in C++ with python integration.
* [Tpot](https://github.com/rhiever/tpot) is a python tool that automatically creates and optimizes machine learning pipelines using genetic programming.

### Neural networks

* [Brainforge](https://github.com/csxeba/brainforge) A Neural Networking library based on NumPy only
* [deeplearn.js](https://pair-code.github.io/deeplearnjs/) a neural network library for the web
* [OpenNN](https://github.com/Artelnics/OpenNN) a neural network C++ library

### Reinforcement Learning

* [Keras-rl](https://github.com/matthiasplappert/keras-rl) Deep Reinforcement Learning for Keras.
* [Gym](https://github.com/openai/gym) A toolkit for developing and comparing reinforcement learning algorithms. Written in Python.
* [TFLearn](https://github.com/tflearn/tflearn) is a deep learning library featuring a higher-level API for [TensorFlow][].
* [Tensorforce](https://github.com/reinforceio/tensorforce) a TensorFlow library for applied reinforcement learning

### Examples

* [AIMA python](https://github.com/aimacode/aima-python) Python code for the book [Artificial Intelligence: A Modern Approach](https://www.amazon.co.uk/Artificial-Intelligence-Approach-Stuart-Russell/dp/1292153962)
* [TensorFlow Examples](https://github.com/aymericdamien/TensorFlow-Examples) a [TensorFlow][] tutorial with popular machine learning algorithms implementation

## NLP

Natural Language processing benefits from [Recurrent Neural Network](https://en.wikipedia.org/wiki/Recurrent_neural_network)
algorithms.

### Analysis

* [huggingface/transformers](https://github.com/huggingface/transformers) State-of-the-art Natural Language Processing for Pytorch and TensorFlow 2.0
* [Natural Language Toolkit](https://github.com/nltk/nltk) (NLTK) is a suite of python modules, data sets and tutorials supporting research and development in [NLP][]. Some of its modules are out of date but still a useful resource nonetheless.
* [SpaCy](https://github.com/spacy-io/spaCy) is a powerful, production ready, NLP library for python
* [fastText](https://github.com/facebookresearch/fastText) a C++ library for sentence classification
* [TextBlob](https://github.com/sloria/TextBlob) is a python library for processing textual data. It provides a simple API for diving into common [NLP][] tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more.
* [simhash](https://github.com/leonsim/simhash) a python implementation of [Simhash Algorithm](http://www.wwwconference.org/www2007/papers/paper215.pdf) for detecting near-duplicate web documents
* [langdetect](https://github.com/Mimino666/langdetect) is a port of Google's language-detection library to Python.

### Tools

* [inflect.py](https://github.com/pwdyson/inflect.py) Correctly generate plurals, ordinals, indefinite articles; convert numbers to words
* [dataprofiler](https://github.com/capitalone/DataProfiler) The DataProfiler is a Python library designed to make data analysis, monitoring and sensitive data detection easy. NLP processing is accomplished using a character-level CNN.

### Resources

* [Oxford Deep NLP 2017 course](https://github.com/oxford-cs-deepnlp-2017/lectures) lecture slides and course description for the Deep Natural Language Processing course

## Images

### Resources

* [Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) In machine learning, a convolutional neural network (CNN, or ConvNet) is a class of deep, feed-forward artificial neural network that have successfully been applied to analyzing visual imagery.

### Frameworks

* [tesseract-ocr][] well tested [OCR][] engine written in C++
* [OpenCV][] computer vision and machine learning software library. The library has more than 2500 optimized algorithms, which includes a comprehensive set of both classic and state-of-the-art computer vision and machine learning algorithms. These algorithms can be used to detect and recognize faces, identify objects, classify human actions in videos, track camera movements, track moving objects, extract 3D models of objects, produce 3D point clouds from stereo cameras, stitch images together to produce a high resolution image of an entire scene, find similar images from an image database, remove red eyes from images taken using flash, follow eye movements, recognize scenery and establish markers to overlay it with augmented reality, etc. Written in C++ with bindins for most languages including python.
* [SimpleCV](https://github.com/sightmachine/SimpleCV) is a framework for machine vision, using [OpenCV][] and Python. It provides a concise, readable interface for cameras, image manipulation, feature extraction, and format conversion.
* [match](https://github.com/usepavlov/match) makes it easy to search for images that look similar to each other
* [Noteshrink](https://github.com/mzucker/noteshrink) Convert scans of handwritten notes to beautiful, compact PDFs
* [srez](https://github.com/david-gpu/srez) Image super-resolution through deep learning
* [CovNetJS](https://github.com/karpathy/convnetjs) train Convolutional Neural Networks (or ordinary ones) in the browser

## Data

### Sources

* [Quandl](https://www.quandl.com/) delivers free and premium financial, economic, and alternative data from hundreds of sources
via their website, API, or directly into dozens of tools
* [Public APIs](https://github.com/toddmotto/public-apis) a collective list of public JSON APIs for use in web development
* [7 and a quarter hours of largely highway driving](https://github.com/commaai/research) from [comma.ai research](http://comma.ai/)

### Aggregators

* [pyspider](https://github.com/binux/pyspider) a web crawler system in python.
* [Newspaper](https://github.com/codelucas/newspaper) News, full-text, and article metadata extraction in Python 3.

### Explore

* [Crossfilter](https://github.com/square/crossfilter) is a JavaScript library for exploring large multivariate datasets in the browser.

### Storage

* [pytables](https://github.com/PyTables/PyTables) a package for managing hierarchical datasets and designed to efficiently cope with extremely large amounts of data. It is built on top of the [HDF5][] library and the NumPy package.


## Visualization

### Resources

* [Visualization Universe](http://visualizationuniverse.com/)
* [Awesome D3](https://github.com/wbkd/awesome-d3)
* [Comparison of JavaScript charting libraries](https://en.wikipedia.org/wiki/Comparison_of_JavaScript_charting_libraries)
* [Explore Data Science](https://kandi.openweaver.com/explore/data-science)


### JavaScript Libraries

* [Chart.js](https://github.com/chartjs/Chart.js) HTML5 Charts using the canvas tag
* [G2](https://github.com/antvis/g2) is a visualization grammar, a data-driven visual language with a high level of usability and scalability
* [plotly.js](https://github.com/plotly/plotly.js) charting library built on top of [d3][] and [stack.gl](http://stack.gl/)
* [frappe/charts](https://github.com/frappe/charts) Simple, responsive, modern SVG Charts with zero dependencies
* [GraphicsJS](https://github.com/AnyChart/GraphicsJS) A lightweight JavaScript graphics library with the intuitive API, based on SVG/VML technology.

### Python Libraries

* [bokeh](https://bokeh.pydata.org/en/latest/) an interactive visualization library that targets modern web browsers for presentation
* [bqplot](https://github.com/bloomberg/bqplot) plotting library for IPython/Jupyter notebooks - front-end in [d3][]
* [dash](https://github.com/plotly/dash/) Dash is a Python framework for building analytical web applications
* [Altair](https://altair-viz.github.io/) declarative statistical visualization library for Python, based on Vega and Vega-Lite

### D3 based libraries
* [brite Charts](https://github.com/eventbrite/britecharts) reusable Charting Library based on D3.js v4 by https://www.eventbrite.co.uk/
* [C3.js](https://github.com/c3js/c3) D3-based reusable chart library
* [dc.js](https://github.com/dc-js/dc.js) Multi-Dimensional charting built to work natively with crossfilter rendered with d3.js
* [d3-visualize](https://github.com/quantmind/d3-visualize) is a [d3-view](https://github.com/quantmind/d3-view) based reactive data-visualization library - alpha
* [d3-waffle](http://jbkunst.github.io/d3-waffle/) waffle plots with d3
* [semiotic](https://github.com/emeeks/semiotic) a data visualization framework combining React & D3
* [tau Charts](https://github.com/TargetProcess/tauCharts)
* [Vega](https://vega.github.io/vega/) visualization grammar
* [Vega-lite](https://vega.github.io/vega-lite/) high-level grammar of interactive graphics

### Digital Art

* [Generating Abstract Patterns with TensorFlow](http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/) Compositional Pattern Producing Network (CPPN)

## Languages

### Python

* [Awesome Python](https://github.com/vinta/awesome-python) A curated list of awesome Python frameworks, libraries, software and resources.
* [Interactive coding challenges](https://github.com/donnemartin/interactive-coding-challenges) which focus on algorithms and data structures that are typically found in coding interviews

### JavaScript

* [Simple Statistics](http://simplestatistics.org/) statistical methods in readable JavaScript for browsers, servers.
* [Computer science in JavaScript](https://github.com/nzakas/computer-science-in-javascript) Collection of classic computer science paradigms, algorithms, and approaches written in JavaScript

## License

[![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [Quantmind](http://quantmind.com) has waived all copyright and related or neighboring rights to this work.

[d3]: https://github.com/d3
[HDF5]: https://www.hdfgroup.org/HDF5/
[NLP]: https://en.wikipedia.org/wiki/Natural_language_processing
[OCR]: https://en.wikipedia.org/wiki/Optical_character_recognition
[OpenCV]: https://github.com/opencv/opencv
[python]: https://www.python.org/
[TensorFlow]: https://github.com/tensorflow/tensorflow
[Theano]: https://github.com/Theano/Theano
[tesseract-ocr]: https://github.com/tesseract-ocr/tesseract
[CNTK]: https://github.com/Microsoft/CNTK
