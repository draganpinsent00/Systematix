"""
Setup configuration for Systematix.
"""

from setuptools import setup, find_packages

setup(
    name="systematix",
    version="1.0.0",
    description="Production-Grade Monte Carlo Options Pricing Platform",
    author="Systematix Team",
    author_email="contact@systematix.io",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "plotly>=5.17.0",
        "scikit-learn>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "systematix=app:run_pricing",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: Proprietary",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="options pricing monte carlo quantitative finance",
)

