# openvino_practice
Practice for git, CI and Intel OpenVINO

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
1. Open terminal and navigate to the project folder. Create build folder.

  ```bat
  cd openvino_practice
  mkdir build
  cd build
  ```

2. Setup environment

  ```bat
  "C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"
  ```

3. Build

  ```bat
  "C:\Program Files\CMake\bin\cmake.exe" -G "Visual Studio 15 2017 Win64" ..
  "C:\Program Files\CMake\bin\cmake.exe" --build . --config Release
  ```

4. Run tests for specific module

  ```bat
  .\bin\Release\test_classification.exe
  ```
