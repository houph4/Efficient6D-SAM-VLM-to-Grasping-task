# LightPose

**Our Repo is constantly being updated,the network model and checkpoints will be released during the paper submission**

Repo list : 

1. An **Efficient RGB-D features extractor** to category-level and instance-level 6D pose estimation.
2. **Bidirectional cross attentin fusion** with rgb-point cloud and globe-local features. 
3. An end-to-end category level pose estimation inference scripts that includes real-time segmentation of **SAM**. (Free of shape piror)
4. **SAM-VLM** to grasping task with 3DBBOX (Using **Azure Kinect** or **Realsense**).
   
Some demos about category-level 6D pose estimation (Without CAD reference).
![Figure_1](https://github.com/houph4/Efficient6D-SAM-VLM-to-Grasping-task/assets/90714020/a4f72f66-477e-43bf-aac8-8ebccfcee4e7)
![Figure_2](https://github.com/houph4/Efficient6D-SAM-VLM-to-Grasping-task/assets/90714020/a9ceaa92-3c4e-42ee-b5dd-bee74f6247b9)
![Figure_3](https://github.com/houph4/Efficient6D-SAM-VLM-to-Grasping-task/assets/90714020/574d96d4-51ae-4c38-b615-799dec31d0c7)
![output_video](https://github.com/houph4/Efficient6D-SAM-VLM-to-Grasping-task/assets/90714020/a6a887d7-95bc-4993-a1be-d33d1fde10dc)

https://github.com/user-attachments/assets/6165a0e6-d2a0-4355-9166-c7bab1ce3bc9


https://github.com/user-attachments/assets/3e735efa-130a-4da8-a89f-6f43a09ede01







## About `requirements.txt`

The `requirements.txt` file lists the Python packages and their versions that were installed in the development environment. However, please note that this file may contain more packages than are strictly necessary for running this project, as it includes all the dependencies present in the environment at the time of its creation.

**We recommend that you review the packages and install only those that are relevant to your specific needs.** You can do this by editing the `requirements.txt` file before running the installation command or by selectively installing the packages you need.

To install the necessary packages, use the following command:

```bash
pip install -r requirements.txt
