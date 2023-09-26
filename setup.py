from setuptools import setup

setup(
    name='predict_smiles',
    version='1.0.0',
    description='Python command-line application to train, evaluate and predict P1 with SMILES',
    author='EL AZIMANI Volodia',
    author_email='volodia.el.azimani@gmail.com',
    entry_points={
        'console_scripts': [
            'train = my_model.main:train_model',
            'evaluate = my_model.main:evaluate',
            'predict = my_model.main:predict'
        ],
    }
)



