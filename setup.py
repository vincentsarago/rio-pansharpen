"""Setup."""

from setuptools import setup, find_packages

# Parse the version from the fiona module.
with open("rio_pansharpen/__init__.py") as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            break

long_description = """"""


setup(
    name="rio-pansharpen",
    version=version,
    description=u"rio-pansharpen",
    long_description=long_description,
    classifiers=[],
    keywords="",
    author=u"Virginia Ng",
    author_email="virginia@mapbox.com",
    url="https://github.com/mapbox/rio-pansharpen",
    license="MIT",
    packages=find_packages(exclude=["ez_setup", "examples", "tests"]),
    include_package_data=True,
    zip_safe=False,
    install_requires=["click", "rasterio~=1.0", "rio-mucho"],
    extras_require={"test": ["pytest", "hypothesis", "pytest-cov", "codecov"]},
    entry_points="""
      [rasterio.rio_plugins]
      pansharpen=rio_pansharpen.scripts.cli:pansharpen
      """,
)
