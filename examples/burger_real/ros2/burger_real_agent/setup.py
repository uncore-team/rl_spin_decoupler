from setuptools import setup

package_name = "burger_real_agent"

setup(
    name=package_name,
    version="0.1.0",
    # Flat layout: agent_node.py/contract.py sit next to setup.py, so
    # find_packages() would find nothing; map the package root explicitly.
    packages=[package_name],
    package_dir={package_name: "."},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "agent_node = burger_real_agent.agent_node:main",
        ],
    },
)
