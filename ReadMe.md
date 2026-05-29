To run the code use: 

##### V1
g++ -std=c++17     -I/usr/include/opencv4     -I/usr/include/pcl-1.14     -I/usr/include/eigen3     -I/usr/include/vtk-9.1     main.cpp src/SfM.cpp -o main     -lopencv_core -lopencv_imgcodecs -lopencv_features2d -lopencv_calib3d     -lpcl_common -lpcl_visualization     -lboost_system -lboost_filesystem -lboost_thread     -lvtkCommonCore-9.1 -lvtkRenderingCore-9.1


##### V2
g++ -std=c++17 \
    -I/usr/include/opencv4 \
    -I/usr/include/pcl-1.14 \
    -I/usr/include/eigen3 \
    -I/usr/include/vtk-9.1 \
    -Iinclude \
    main.cpp src/SfM.cpp -o main \
    -lopencv_core -lopencv_imgcodecs -lopencv_features2d -lopencv_calib3d -lopencv_highgui \
    -lpcl_common -lpcl_visualization -lceres\
    -lboost_system -lboost_filesystem -lboost_thread \
    -lvtkCommonCore-9.1 -lvtkCommonDataModel-9.1 -lvtkCommonMath-9.1 \
    -lvtkRenderingCore-9.1 -lvtkRenderingOpenGL2-9.1 -lvtkInteractionStyle-9.1 \
    -lvtkFiltersCore-9.1

##### V3
g++ -std=c++17     -I/usr/include/opencv4     -I/usr/include/pcl-1.14     -I/usr/include/eigen3     -I/usr/include/vtk-9.1     -I./include     main.cpp src/SfM.cpp src/BundleAdjustment.cpp -o main     -lopencv_core -lopencv_imgcodecs -lopencv_features2d -lopencv_calib3d -lopencv_highgui     -lpcl_common -lpcl_visualization     -lceres -lglog -lgflags     -lboost_system -lboost_filesystem -lboost_thread     -lvtkCommonCore-9.1 -lvtkCommonDataModel-9.1 -lvtkCommonMath-9.1     -lvtkRenderingCore-9.1 -lvtkRenderingOpenGL2-9.1 -lvtkInteractionStyle-9.1     -lvtkFiltersCore-9.1