from setuptools import setup, find_packages


requirements = [
    "decentriq-platform==0.11",
    "scikit-learn",
    "pandas==1.3.5",
    "auto-sklearn==0.14.7",
    "flake8"
]


setup(
    name="decentriq_deployment",
    include_package_data=True,
    version="0.0.1-dev0",
    description="Decentriq Deployment",
    url="https://github.com/ringier-data/rcplus-decentriq-template",
    author="Ringier AG",
    author_email="info@ringier.ch",
    packages=find_packages(exclude=("tests")),
    python_requires=">=3.8",
    install_requires=requirements,
)
