from setuptools import setup

setup(
    name='cluster-frete',
    version='',
    packages=['cluster_frete', 'cluster_frete.dados', 'cluster_frete.training', 'cluster_frete.classifier_model'],
    url='',
    license='',
    author='André Claudino',
    author_email='',
    description='',
    install_requires=[
        "tensorflow==2.11.1"
    ],
    entry_points={
        'console_scripts': [
            'train_supervised=cluster_frete.main_supervised:main',
            'train_unsupervised=cluster_frete.main_unsupervised:main'
        ]
    }
)
