from setuptools import setup

setup(
    name="isaac_imitation_learning",
    packages=["isaac_imitation_learning"],
    python_requires=">=3.10",
    install_requires=[
        "isaacsim >= 4.5",
        "isaaclab >= 0.32.8",
    ],
    extras_require={
        "SIMPUB": ["simpub"],
        "IMITATION_LEARNING": [
            "diffusion_policy",
            "pytorch3d",
            "threadpoolctl",
            "zarr",
            "imagecodecs",
            "dill",
            "diffusers",
            "numba",
            "av==10.0",
            "robomimic==0.2",
        ],
    },
)
