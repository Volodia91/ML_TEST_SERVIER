from setuptools import setup

setup(
    name='predict_smiles',
    version='1.0.0',
    description='Python command-line application to train, evaluate and predict P1 with SMILES',
    author='EL AZIMANI Volodia',
    author_email='volodia.el.azimani@gmail.com',
    entry_points={
        'console_scripts': [
            'train = main:train_model',
            'evaluate = main:evaluate',
            'predict = main:predict'
        ],
    },
    install_requires=[
        "certifi",
        "click",
        "dataclasses",
        "Flask",
        "importlib-metadata",
        "itsdangerous",
        "Jinja2",
        "joblib",
        "MarkupSafe",
        "numpy",
        "pandas",
        "Pillow",
        "python-dateutil",
        "pytz",
        "rdkit-pypi",
        "scikit-learn",
        "scipy",
        "six",
        "threadpoolctl",
        "typing_extensions",
        "Werkzeug",
        "zipp"
    ]
)



