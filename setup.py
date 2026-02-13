"""Setup script for SocialProphet package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="socialprophet",
    version="0.1.0",
    author="navadeep555, avkbsurya119",
    author_email="navadeepmaka726@gmail.com, avkbsurya@gmail.com",
    description="Hybrid Time-Series Forecasting & Generative Content Agent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/avkbsurya119/SocialProphet",
    project_urls={
        "Bug Tracker": "https://github.com/avkbsurya119/SocialProphet/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
        ],
        "gpu": [
            "tensorflow>=2.13.0",
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "socialprophet=src.main:main",
        ],
    },
)
