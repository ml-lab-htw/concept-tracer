from setuptools import setup, find_packages


setup(
    name="concept-tracer",
    version="0.1.0",
    description="ConceptTracer: Interactive Analysis of Concept Saliency and Selectivity in Neural Representations",
    author="Ricardo Knauer",
    author_email="ricardo.knauer@htw-berlin.de",
    url="https://github.com/ml-lab-htw/concept-tracer",
    packages=find_packages(),
    install_requires=[
        "dash",
        "imbalanced-learn",
        # "ipykernel",
        # "jupyter",
        # "kaleido",
        # "mosek",
        "numpy",
        "pandas",
        "plotly",
        # "pytest",
        "scikit-learn",
        "scipy",
        "torch",
        "tqdm",
        # most recent as of January 2026
        "tabpfn==6.3.2",
        "tabpfn-extensions==0.2.2"
    ],
    entry_points={
        "console_scripts": [
            "concept_tracer=concept_tracer.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
