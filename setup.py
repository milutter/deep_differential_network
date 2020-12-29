from setuptools import setup

setup(
    name='deep_differential_network',
    version='0.1',
    url='https://git.ias.informatik.tu-darmstadt.de/lutter/deep_differential_network',

    description='This package provides an implementation of a Deep Differential Network. This '
                'network architecture is a variant of a fully connected network that in addition '
                'to computing the function value f(x;θ) outputs the network Jacobian w.r.t. the network input x. The '
                'Jacobian is computed in closed form with machine precision, with minimal computational overhead using '
                'the chain rule.',

    author='Michael Lutter',
    author_email='michael@robot-learning.de',

    packages=['deep_differential_network', ],

    classifiers=['Development Status :: 3 - Alpha'], install_requires=['matplotlib',
                                                                       'numpy',
                                                                       'torch'])