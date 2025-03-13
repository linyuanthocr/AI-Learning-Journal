
函数入口 viewController::process()
1. 收集measurements（可能含有若干对imu序列和对应的img数据，利用时间戳对齐）
`std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>> measurements;`
1. 每次处理一个Measurement
    处理imu数据：`send_imu` 
    会调用`vins.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz))`
    该函数负责进行imu预积分，这里采用midpoint积分更新第j帧（当前帧）旋转矩阵Rs[j],位置向量Ps[j],速度向量Vs[j]，具体定义：
```
void VINS::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
```
角度更新：
![vins_mobile_code_review](vins_mobile_code_review.jpg)
```
//midpoint integration
{
    Vector3d g{0,0,GRAVITY};
    int j = frame_count;
    Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
    Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
    Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
    Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
    Vs[j] += dt * un_acc;
}
```
  
   处理点云数据：
   `map<int, Vector3d> image = img_msg->point_clouds;`
  `vins.processImage(image,header,waiting_lists);`
  
 
```mermaid
graph TD
A[processimage] -->B(addFeatureCheckParallax)
B --> C(avp >= TH)
C -->|Yes| D[MARGIN_OLD]
C -->|No| E[MARGIN_SECOND_NEW]
D --> F(solver_flag == INITIAL)
E --> F
F -->|Yes| G[Get pre_integration]
F -->|No| H[Triangulation]
H --> H5[solve_ceres]
H5 --> I(failureDetection)
I -->|Yes| J[clearState]
I -->|NO| K[slideWindow]
K --> K1[f_manager.removeFailures]
K1 --> K2[update_loop_correction]
K2 --> Z((End))
J --> Z
G --> F1(frame_count == WINDOW_SIZE)
F1 -->|Yes| H1(track_num<20)
F1 -->|No| F2((frame_count++))
H1-->|Yes| J2[clearState]
J2 --> Z1((End))
H1-->|No| H2(header - initial_timestamp > 0.3)
H2-->|Yes| H3[result = solveInitial]
H3--> H4(result)
H2 -->|No| H4
H4 --> |Yes| L1[solve_ceres]
L1-->M(final_cost > 200)
M -->|Yes| N[Reinitialization Required]
N --> N1[slidewindow]
H4 -->|No| N1
M -->|NO| O[Initialization Done!!]
O -->K
N1 --> Z
```




  
