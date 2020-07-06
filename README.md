# Intel OpenVINO practice

This is a set of interactive modules which will help you study and practice:
* git, GitHub
* [OpenVINO toolkit](https://software.intel.com/openvino-toolkit)
* [Travis CI](https://travis-ci.org/)
* [Google Test](https://github.com/google/googletest)

[![chat](https://img.shields.io/badge/join-chat-green?style=plastic&logo=discord&logoColor=white)](https://discord.com/channels/723484061472915457/723484061472915460)

:bar_chart: Check your progress at https://dkurt.github.io/openvino_practice

## Getting started

An every task is a separate folder in [modules](./modules) subfolder. To start new module always do:

1. Nagivate to a base branch

    ```
    git checkout master
    ```

2. Create a new local branch with unique name

    ```
    git checkout -b practice_classification
    ```

Start with [Git, GitHub and Travis CI](./modules/0_git) module to complete first task.

## Build instruction

Download OpenVINO: https://software.seek.intel.com/openvino-toolkit

### Linux
1. Open terminal and navigate to the project folder. Create build folder.

  ```bash
  cd openvino_practice
  mkdir build
  cd build
  ```

2. Setup environment

  ```bash
  source /opt/intel/openvino/bin/setupvars.sh
  ```

3. Build

  ```bash
  cmake -DCMAKE_BUILD_TYPE=Release .. && make --j$(nproc --all)
  ```

4. Run tests for specific module

  ```bash
  ./bin/test_classification
  ```

### Windows

0. Download useful programs
  * [CMake](https://cmake.org/download/)
  * [Git BASH](https://gitforwindows.org/)

1. Open terminal (Start -> Developer Command Prompt) and navigate to the project folder. Create a build folder.

  ```bat
  cd openvino_practice
  mkdir build
  cd build
  ```

2. Setup environment (in Start -> Developer Command Prompt)

  ```bat
  "C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"
  ```

3. Prepare project files (NOTE: value after `-G` flag might be different for your version of Visual Studio)

  ```bat
  "C:\Program Files\CMake\bin\cmake.exe" -G "Visual Studio 15 2017 Win64" ..
  ```

4. Build (choose one of the options)

    4.1 Using Visual Studio:

        Open `openvino_practice/build/openvino_practice.sln`.

        Choose Relese x64 configuration. Press `Build -> Build Solution`

    4.2 Using terminal:

        "C:\Program Files\CMake\bin\cmake.exe" --build . --config Release

5. Run tests for specific module (in terminal)

  ```bat
  .\bin\Release\test_classification.exe
  ```
