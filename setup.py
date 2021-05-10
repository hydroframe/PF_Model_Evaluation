from setuptools import setup

setup(
    name='pfspinup',
    version='1.0.4',
    packages=['pfspinup', 'pfspinup.data'],
    package_data={'pfspinup.data': ['*', '*/*']},
    zip_safe=False,

    python_requires='>=3.6',

    install_requires=[
        'pandas',
        'parflowio>=0.0.4',
        'pftools>=1.2.0',
        'matplotlib',
        'numpy'
    ],

    extras_require={
        'dev': ['mock', 'numpydoc', 'pytest', 'sphinx', 'sphinxcontrib-bibtex<2.0.0', 'sphinx-rtd-theme']
    }

)
