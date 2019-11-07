
# Healthcare Application - Pneumonia Classification

| Details               |                   |
| --------------------- | ----------------- |
| Target OS:            | Ubuntu* 16.04 LTS |
| Programming Language: | Python* 3.5       |
| Time to Complete:     | 30 min            |

![pneumonia](./docs/images/example-pneumonia.jpeg)

## What it does
This reference implementation showcases a health care application by performing pneumonia classification on X-ray images and it results the probability value of disease.

## Requirements

### Hardware

- 6th to 8th generation Intel® Core™ processors with Iris® Pro graphics or Intel® HD Graphics

### Software

- [Ubuntu\* 16.04 LTS](http://releases.ubuntu.com/16.04/)<br>
  NOTE: Use kernel versions 4.14+ with this software.<br> 
  Determine the kernel version with the below uname command. 

  ```
   uname -a
  ```

- Intel® Distribution of OpenVINO™ toolkit 2019 R3 release

## How It works

The application uses the Inference Engine included in the Intel® Distribution of OpenVINO™ toolkit. It uses X-ray image as an input source. A trained neural network detects pneumonia probability of preprocessed X-ray image. And it stores the probability and inference time value in a file. 

![arch image](./docs/images/architectural_diagram.png)
**Architectural Diagram**

## Setup

### Get the code

Clone the reference implementation:

    sudo apt-get update && sudo apt-get install git
    git clone https://github.com/intel-iot-devkit/pneumonia-classification.git

### Install Intel® Distribution of OpenVINO™ toolkit

Refer to [Install Intel® Distribution of OpenVINO™ toolkit for Linux*](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux) to learn how to install and configure the toolkit.

Install the OpenCL™ Runtime Package to run inference on the GPU, as shown in the instructions below. It is not mandatory for CPU inference.

### Which model to use
This application uses a pre-trained [model](resources/model/model.pb), that is provided in the /resources directory. This model is trained using the dataset found in [https://data.mendeley.com/datasets/rscbjbr9sj/2](https://data.mendeley.com/datasets/rscbjbr9sj/2), made available under the [(CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/) license. Instructions on how to train the model can also be found there. This model needs to be passed through the **model optimizer** to generate the IR (the **.xml** and **.bin** files) that will be used by the application.

To install the dependencies of the RI, run the following commands:

  ```
  cd <path_to_the_pneumonia-classification-python_directory>
  ./setup.sh
  ```
 
### The Config File

The _resources/config.json_ contains the path to the images that will be used by the application.  

The _config.json_ file is of the form name/value pair, `image: <path/to/imagefile>`   

The application can use any number of images for detection.

### Which Input Image to use

The application works with any input X-ray input image. For first-use, we recommend using the [NORMAL](resources/NORMAL) and [PNEUMONIA](resources/NORMAL) images.
For example: <br>
The config.json would be:

```
{

    "inputs": [
	    {
            "image": "resources/NORMAL/*.jpeg"
        },
        {
            "image": "resources/PNEUMONIA/*.jpeg"
        }
    ]
}
```
To use any other image, specify the path in config.json file

## Setup the environment
You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:

    source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
    
__Note__: This command needs to be executed only once in the terminal where the application will be executed. If the terminal is closed, the command needs to be executed again.
    
### Run the Application

Change the current directory to the git-cloned application code location on your system:
```
cd <path_to_the_pneumonia-classification-python_directory>/application
```

To see a list of the various options:

```
python3 pneumonia_classification.py -h
```

 To run the application with the needed models:

```
python3 pneumonia_classification.py -m ../resources/FP32/model.xml
```
The output images, results.txt and stats.txt files will be saved in the __output__ directory

**results.txt**: This file contains the probability of pneumonia message and inference time

**stats.txt**: This file contains the total average inference time

To save the results in a specific directory

  ```
  python3 pneumonia_classification.py -m ../resources/FP32/model.xml -o <path-to-the-directory_to_save_the_output_files>
  ```

### Run on Different Hardware

A user can specify a target device to run on by using the device command-line argument `-d` followed by one of the values `CPU`, `GPU`,`MYRIAD`, `HDDL` or `HETERO:FPGA,CPU`.<br>
To run with multiple devices use -d MULTI:device1,device2. For example: `-d MULTI:CPU,GPU,MYRIAD`

#### Run on the CPU

Although the application runs on the CPU by default, this can also be explicitly specified through the `-d CPU` command-line argument:

```
python3 pneumonia_classification.py -m ../resources/FP32/model.xml -d CPU
```
#### Run on the Integrated GPU

* To run on the integrated Intel® GPU with floating point precision 32 (FP32), use the `-d GPU` command-line argument:

    ```
    python3 pneumonia_classification.py -m ../resources/FP32/model.xml -d GPU
    ```
    **FP32**: FP32 is single-precision floating-point arithmetic uses 32 bits to represent numbers. 8 bits for the magnitude and 23 bits for the precision. For more information, [click here](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)<br>

* To run on the integrated Intel® GPU with floating point precision 16 (FP16):

    ```
    python3 pneumonia_classification.py -m ../resources/FP16/model.xml -d GPU
    ```
    **FP16**: FP16 is half-precision floating-point arithmetic uses 16 bits. 5 bits for the magnitude and 10 bits for the precision. For more information, [click here](https://en.wikipedia.org/wiki/Half-precision_floating-point_format)

#### Run on the Intel® Neural Compute Stick

To run on the Intel® Neural Compute Stick, use the ```-d MYRIAD``` command-line argument:

```
python3 pneumonia_classification.py -m ../resources/FP16/model.xml -d MYRIAD
```

**Note:** The Intel® Neural Compute Stick can only run FP16 models. The model that is passed to the application, through the `-m <path_to_model>` command-line argument, must be of data type FP16.

#### Run on the Intel® Movidius™ VPU
To run on the Intel® Movidius™ VPU, use the `-d HDDL` command-line argument:
```
python3 pneumonia_classification.py -m ../resources/FP16/model.xml -d HDDL
```

**Note:** The Intel® Movidius™ VPU can only run FP16 models. The model that is passed to the application, through the `-m <path_to_model>` command-line argument, must be of data type FP16.

#### Run on the Intel® Arria® 10 FPGA

Before running the application on the FPGA, set the environment variables and  program the AOCX (bitstream) file.<br>

Set the Board Environment Variable to the proper directory:

```
export AOCL_BOARD_PACKAGE_ROOT=/opt/intel/openvino/bitstreams/a10_vision_design_sg<#>_bitstreams/BSP/a10_1150_sg<#>
```
**NOTE**: If you do not know which version of the board you have, please refer to the product label on the fan cover side or by the product SKU: Mustang-F100-A10-R10 => SG1; Mustang-F100-A10E-R10 => SG2 <br>

Set the Board Environment Variable to the proper directory:
```
export QUARTUS_ROOTDIR=/home/<user>/intelFPGA/18.1/qprogrammer
```
Set the remaining environment variables:
```
export PATH=$PATH:/opt/altera/aocl-pro-rte/aclrte-linux64/bin:/opt/altera/aocl-pro-rte/aclrte-linux64/host/linux64/bin:/home/<user>/intelFPGA/18.1/qprogrammer/bin
export INTELFPGAOCLSDKROOT=/opt/altera/aocl-pro-rte/aclrte-linux64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AOCL_BOARD_PACKAGE_ROOT/linux64/lib
export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3
source /opt/altera/aocl-pro-rte/aclrte-linux64/init_opencl.sh
```
**NOTE**: It is recommended to create your own script for your system to aid in setting up these environment variables. It will be run each time you need a new terminal or restart your system.

The bitstreams for HDDL-F can be found under the `/opt/intel/openvino/bitstreams/a10_vision_design_sg<#>_bitstreams/` directory.<br><br>To program the bitstream use the below command:<br>
```
aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg<#>_bitstreams/2019R3_PV_PL1_FP11_RMNet.aocx
```

For more information on programming the bitstreams, please refer the [link](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux-FPGA#inpage-nav-11).

To run the application on the FPGA with floating point precision 16 (FP16), use the `-d HETERO:FPGA,CPU` command-line argument:
```
python3 pneumonia_classification.py -m ../resources/FP16/model.xml -d HETERO:FPGA,CPU
```
