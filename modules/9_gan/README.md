# Image generation with GAN

This module demonstrates a simple image generation network that was trained on CelebA.
You will also learn how to use OpenVINO in Python.
To complete this module you should implement:

1. `generate` method runs network with input random noise
2. `postprocess` procedure that applies some transformations to output of the network to get the final generated image.
3. Generate images and add it to the pull request description.

Original images from CelebA (left), generated images (right).

  <img src="../../data/celeba_samples.jpeg" width="320"><img src="../../data/generated_img.png" width="320">

## Details

* Download converted model from here:

  * [celeba_generator.bin](https://mega.nz/file/pUN1kCCQ#e9foW8shMj6fpA7go8_RjtGo5YGYKtiI14SVCSop_l0)
  * [celeba_generator.xml](https://mega.nz/file/RcUHRaQI#-4XReh2kFr_sgvl8qmKkVhkhTcxEWDVXMP1xFv8fsp8)

* Setup OpenVINO environment

    * Linux

        ```bash
        source /opt/intel/openvino/bin/setupvars.sh
        ```

    * Microsoft Windows

        ```bat
        "C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"
        ```

* Run a test Python script:

    ```
    python3 test_generation.py
    ```

* Do not add `.xml` and `.bin` files into commit - they are downloaded automatically on CI
