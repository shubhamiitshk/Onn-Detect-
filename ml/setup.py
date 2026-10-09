"""Install core training dependencies; export/data/test stacks are explicit extras."""
from pathlib import Path
from setuptools import setup, find_packages
root = Path(__file__).parent
requirements = [line.strip() for line in (root / "requirements.txt").read_text().splitlines() if line.strip() and not line.startswith("#")]
setup(name="onndetect", version="4.1.0", python_requires=">=3.11,<3.12",
      packages=find_packages(where="src"), package_dir={"": "src"}, install_requires=requirements,
      extras_require={"export": ["tensorflow==2.21.0", "keras==3.15.1"],
                      "data": ["datasets==4.4.1", "huggingface-hub==1.28.0"], "dev": ["pytest==9.1.1"]})
