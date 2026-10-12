from setuptools import setup, find_packages

setup(
    name="trace",
    version="0.2.0",
    description="Circuit discovery, causal functional signatures, ILP architectural signatures and cross-model transfer",
    packages=find_packages(include=["trace", "trace.*"]),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.2",
        "scipy>=1.13",
        "scikit-learn>=1.5",
        "matplotlib>=3.9",
        "seaborn>=0.13",
        "networkx>=3.3",
        "rdflib>=7.0",
        "torch>=2.3",
        "transformers>=4.44",
        "transformer_lens>=2.16",
        "einops>=0.8",
        "tqdm>=4.66",
        "spacy>=3.7",
        "huggingface_hub>=0.24",
    ],
    extras_require={
        "problog": ["problog>=2.2"],
        "graphviz": ["pygraphviz>=1.14"],
        "test": ["pytest>=8.0"],
    },
    license="GPL-3.0-only",
)
