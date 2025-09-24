"""
a direct python script that convert rosbag to tum format,exact images, depths and write txts

INPUTS:
rosbag info chuangye_0910_high_01.bag 
path:        chuangye_0910_high_01.bag
version:     2.0
duration:    5:12s (312s)
start:       Sep 10 2025 16:46:31.92 (1757493991.92)
end:         Sep 10 2025 16:51:44.51 (1757494304.51)
size:        3.0 GB
messages:    75017
compression: none [3136/3136 chunks]
types:       sensor_msgs/CompressedImage [8f7a12909da2c9d3332d540a0977563f]
             sensor_msgs/Imu             [6a62c6daae103f4ff57a132d6f95cec2]
             sensor_msgs/PointCloud2     [1158d486dd51d683ce2f1be655c3c181]
topics:      /camera/color/image_raw/compressed    9371 msgs    : sensor_msgs/CompressedImage
             /livox/imu                           62520 msgs    : sensor_msgs/Imu            
             /livox/lidar                          3126 msgs    : sensor_msgs/PointCloud2

OUTPUTS example:
dir datasets\tum\rgbd_dataset_freiburg1_desk
 Volume in drive C has no label.
 Volume Serial Number is 3647-5B34

 Directory of C:\Users\13694\MonoGS\datasets\tum\rgbd_dataset_freiburg1_desk

2011/09/29  23:10    <DIR>          .
2025/09/24  16:42    <DIR>          ..
2011/09/29  23:10           547,726 accelerometer.txt
2011/09/29  23:10    <DIR>          depth
2011/09/29  23:10            27,446 depth.txt
2011/09/29  23:10           158,143 groundtruth.txt
2011/09/29  23:10    <DIR>          rgb
2011/09/29  23:10            27,050 rgb.txt
               4 File(s)        760,365 bytes
               4 Dir(s)

# accelerometer data
# file: 'rgbd_dataset_freiburg1_desk.bag'
# timestamp ax ay az
1305031449.564825 -0.083818 7.244229 -6.657506

# color images
# file: 'rgbd_dataset_freiburg1_desk.bag'
# timestamp filename
1305031452.791720 rgb/1305031452.791720.png
1
"""