from setuptools import setup

setup(
    name='pfspinup',
    version='0.0.9',
    packages=['pfspinup', 'pfspinup.data'],
    package_data={'pfspinup.data': ['*', '*/*']},
    zip_safe=False,

    python_requires='>=3.6',

    install_requires=[
        'pandas',
        'parflowio>=0.0.4',
        'pftools',
        'matplotlib',
        'numpy'
    ],

    extras_require={
        'dev': ['mock', 'numpydoc', 'pytest', 'sphinx', 'sphinxcontrib-bibtex', 'sphinx-rtd-theme']
    }

)
