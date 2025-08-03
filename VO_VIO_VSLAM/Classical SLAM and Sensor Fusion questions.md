# Classical SLAM & Sensor Fusion questions

### **Q1. Feature-based vs Direct methods in SLAM**

**Answer:**

- **Feature-based** methods detect and match keypoints (e.g., ORB, SIFT). They are robust to illumination changes and enable loop closure and map reuse. Used in ORB-SLAM, VINS-Mono.
- **Direct** methods operate on raw pixel intensities without explicit feature extraction. They are better in texture-less scenes and offer sub-pixel accuracy. Used in DSO, LSD-SLAM.
- **When to use**:
    - Feature-based: when robustness and large-scale loop closure are needed.
    - Direct: when you need high accuracy and scene has photometric consistency.

---

### **Q2. Epipolar Geometry and Essential Matrix**

**Answer:**

Epipolar geometry defines the geometric relationship between two views of a 3D scene.

- **Essential matrix (E)** satisfies: `x2ᵀ * E * x1 = 0` where `x1` and `x2` are normalized points.
- **Computation**: Estimate `E` using **5-point or 8-point algorithm** with RANSAC for robustness against outliers. Decompose `E` into rotation and translation to recover relative camera pose.

---

### **Q3. Loosely vs Tightly Coupled Sensor Fusion**

**Answer:**

- **Loosely coupled**: Fuse outputs of separate estimators (e.g., visual pose + IMU-integrated pose).
    - ✅ Simple, modular
    - ❌ Less accurate, hard to recover from drift
- **Tightly coupled**: Fuse raw measurements in a joint optimization or filter.
    - ✅ Higher accuracy and robustness (especially in low-texture or high-motion)
    - ❌ More complex and computationally expensive

---

### **Q4. Five-Point Algorithm in SfM/VO**

**Answer:**

The **five-point algorithm** computes the **essential matrix** from five point correspondences between two calibrated views.

- It’s more efficient than the 8-point method and is widely used in VO and SfM pipelines.
- Typically wrapped in RANSAC to reject outliers and used to recover relative pose (rotation & translation up to scale).

---

### **Q5. Recovering Scale with Monocular + IMU**

**Answer:**

Monocular vision alone cannot recover scale due to inherent ambiguity.

However, IMU provides **metric acceleration** and gravity direction, enabling:

- **Pre-integration** of IMU data to estimate motion.
- **Alignment** between visual motion and IMU motion to estimate scale and gravity.
    
    This is the basis of monocular VIO systems like VINS-Mono.
    

---

### **Q6. What are the components of a visual SLAM system?**

**Expected Answer:**

- **Front-end**: feature extraction, matching, pose estimation (PnP), outlier rejection (RANSAC)
- **Back-end**: optimization (e.g., bundle adjustment, factor graph)
- **Loop closure**: detect revisited places (e.g., BoW, NetVLAD)
- **Map representation**: sparse (points), dense (voxels, surfels), or hybrid
- **Relocalization**: recover from tracking failure

---

### **Q7. Describe the PnP (Perspective-n-Point) problem and its use in SLAM.**

**Expected Answer:**

- PnP solves for camera pose given 3D world points and their 2D projections.
- Used in SLAM to compute pose from feature matches to a known map.
- Common algorithms: EPnP, RANSAC-PnP, P3P (with 3 points).

---

### **Q8. What is bundle adjustment and why is it important in SfM/SLAM?**

**Expected Answer:**

- Non-linear optimization to jointly refine camera poses and 3D landmarks.
- Minimizes reprojection error over all frames.
- Critical for global consistency and accuracy in SfM and SLAM.

---

### **Q9. How is depth estimated using stereo vision?**

**Expected Answer:**

- From disparity: depth = `baseline * focal_length / disparity`
- Requires rectified stereo images.
- Matching costs computed (SAD, NCC, CNN-based), then WTA or optimization (SGM).
- Post-processing includes left-right consistency check, filtering.

---

### **Q10. What is an IMU pre-integration and why is it needed in VIO?**

**Expected Answer:**

- Pre-integration summarizes high-rate IMU measurements between keyframes.
- Avoids re-integrating raw data during optimization.
- Enables efficient computation of IMU residuals for factor graph optimization in VIO.

---

### **Q11. How do you calibrate extrinsics between camera and IMU?**

**Expected Answer:**

- Estimate rotation & translation between camera and IMU frames.
- Use methods like Kalibr or batch optimization.
- Requires motion with excitation in all 6 DoFs and synchronized measurements.

---

### **Q12. What causes scale drift in monocular SLAM and how can it be corrected?**

**Expected Answer:**

- Monocular SLAM lacks absolute scale; errors accumulate over time.
- Can be corrected via:
    - IMU integration
    - GPS or altimeter
    - Loop closure with known-scale constraints

---

### **Q13. Explain how a Kalman Filter works in sensor fusion.**

**Expected Answer:**

- Predict → update cycle:
    - **Predict** state using motion model
    - **Update** with sensor observation, using Kalman Gain
- Assumes Gaussian noise, linear models (EKF for nonlinear).
- Used in loosely-coupled VIO or GPS-IMU fusion.

---

### **Q14. What are the advantages of using factor graphs in SLAM?**

**Expected Answer:**

- Flexible representation of variables and constraints (nodes and factors)
- Enables batch and incremental optimization (e.g., GTSAM, iSAM2)
- Efficient handling of sparsity and marginalization

---

### **Q15. What is the role of marginalization in optimization-based SLAM?**

**Expected Answer:**

- Removes old variables (e.g., past poses) to limit computation
- Keeps problem size bounded while preserving useful constraints
- Done using Schur complement, but may introduce linearization errors

---

### **Q16. Explain the difference between a keyframe-based SLAM system and a filter-based SLAM system.**

**Expected Answer:**

- **Keyframe-based**: Uses selected frames for optimization (e.g., ORB-SLAM, VINS). Backend uses graph or bundle adjustment.
- **Filter-based**: Uses Kalman filters or EKFs (e.g., MSCKF). Real-time, memory-efficient but less accurate.
- Trade-off: filters are fast and simple, but optimizers achieve better global consistency.

---

### **Q17. What is the role of the Jacobian in visual-inertial optimization?**

**Expected Answer:**

- Jacobian defines how residuals change w.r.t. variables (pose, bias, etc.)
- Used in computing gradients and Hessians in Gauss-Newton or Levenberg-Marquardt
- Crucial for accurate convergence and fast optimization

---

### **Q18. Why do we use a normalized 8-point algorithm for computing the fundamental matrix?**

**Expected Answer:**

- Normalization (centering + scaling points) improves numerical stability
- Reduces sensitivity to coordinate scale, ensures better SVD-based estimation of F
- Without normalization, results can be unstable and inaccurate

---

### **Q19. How do you compute the relative pose between two RGB images?**

**Expected Answer:**

1. Extract features (e.g., ORB)
2. Match features (with descriptor distance + ratio test)
3. Estimate Essential Matrix with RANSAC
4. Decompose E into rotation and translation
5. (Optionally) triangulate points to validate correct pose

---

### **Q20. What is inverse depth parametrization, and when is it used?**

**Expected Answer:**

- Represents 3D point as `(x, y, 1 / depth)`
- Useful in monocular SLAM (e.g., DSO) where depth is uncertain or uninitialized
- Allows points at infinity to be represented and initialized more stably

---

### **Q21. What causes motion blur and how does it affect visual SLAM?**

**Expected Answer:**

- Caused by long exposure during fast motion
- Degrades feature detection, matching, and tracking accuracy
- Can lead to poor pose estimates or tracking failure
- Mitigation: fast shutter, inertial fusion, or learned deblurring

---

### **Q22. What is photometric consistency and how is it used in direct methods?**

**Expected Answer:**

- Assumes same scene point has same pixel intensity across views
- Used in direct methods to minimize photometric error:
    
    Error=I1(x)−I2(π(Tx))\text{Error} = I_1(x) - I_2(\pi(Tx))
    
- Sensitive to illumination changes and camera calibration

---

### **Q23. How would you test the accuracy of a localization system?**

**Expected Answer:**

- Use datasets with ground truth (e.g., KITTI, TUM, EuRoC)
- Metrics: Absolute Trajectory Error (ATE), Relative Pose Error (RPE)
- Visual inspection of drift, loop closures, and alignment with map

---

### **Q24. What are the challenges in multi-camera SLAM systems?**

**Expected Answer:**

- Calibration (intrinsics and extrinsics)
- Synchronization of image streams
- Efficient data association and loop closure across cameras
- High memory and compute demands

---

### **Q25. How would you handle dynamic objects in visual SLAM?**

**Expected Answer:**

- Detect and mask out dynamic regions (e.g., optical flow, segmentation)
- Use semantic SLAM to identify static vs dynamic classes
- Use robust estimators (e.g., Huber loss) to downweight outliers

---

### **Q26. Triangulation Process**

**Answer:**

Given two images and known camera poses:

1. Detect and match corresponding 2D points.
2. Back-project rays from each camera through the matched points.
3. Find the 3D point that minimizes the distance between rays (usually via linear triangulation or least-squares).
- Assumes known camera intrinsics and extrinsics.

---

### **Q27. Essential vs Fundamental Matrix**

**Answer:**

- **Essential Matrix (E)**: Used with normalized (calibrated) image coordinates. Encodes relative pose between two calibrated cameras.
- **Fundamental Matrix (F)**: Used with raw pixel coordinates. Encodes epipolar geometry without requiring calibration.
- Use **E** for calibrated pipelines (e.g., SLAM), **F** for uncalibrated scenes.

---

### **Q28. Reprojection Error**

**Answer:**

It’s the distance between a detected 2D point and the projection of the corresponding 3D point.

Error=∥xobserved−π(PX3D)∥\text{Error} = \| x_{\text{observed}} - \pi(PX_{\text{3D}}) \|

- Minimized in bundle adjustment.
- Core metric in SfM, SLAM, and pose refinement.

---

### **Q29. Camera Calibration**

**Answer:**

- **Intrinsics**: focal lengths, principal point, distortion coefficients.
- **Extrinsics**: rotation and translation between camera and world (or other sensors).
- Essential for accurate 3D reconstruction and pose estimation.

---

### **Q30. Visual Odometry vs SLAM**

**Answer:**

- **VO** estimates the motion trajectory but doesn't ensure global consistency.
- **SLAM** builds a map and includes loop closure for drift correction.
- VO is a subset of SLAM, often used in real-time motion tracking.

---

### **Q31. Photometric Bundle Adjustment**

**Answer:**

- Minimizes intensity difference between images rather than reprojection error.

Photometric Error=I1(u)−I2(π(TX))\text{Photometric Error} = I_1(u) - I_2(\pi(TX))

- Used in direct methods (e.g., DSO).
- Sensitive to lighting and exposure changes.

---

### **Q32. Lie Algebra in SLAM**

**Answer:**

- SE(3), SO(3) represent continuous transformation groups.
- Lie algebra allows small updates via exponential maps.
- Enables efficient optimization and minimal parameter representation.

---

### **Q33. IMU Bias**

**Answer:**

- Small constant errors in accelerometer/gyroscope measurements.
- Modeled as slowly drifting variables (often included in state vector).
- Estimated via optimization or filtering alongside poses.

---

### **Q34. Sparse Point Cloud in SfM**

**Answer:**

1. Feature detection & matching across multiple views.
2. Estimate relative camera poses (E/F matrix, PnP).
3. Triangulate matching points.
4. Refine via bundle adjustment.

---

### **Q35. Feature-Based VIO Pipeline**

**Answer:**

1. Detect and track visual features.
2. Use IMU for motion prediction between frames.
3. Use PnP or triangulation for pose estimation.
4. Fuse in EKF or factor graph.
5. Optimize jointly over states and landmarks.

---

### **Q36. GPS-IMU Fusion vs VIO**

**Answer:**

- **GPS-IMU**: Large-scale outdoor localization, low frequency GPS updates.
- **VIO**: Visual cues aid high-rate, accurate motion estimation.
- GPS fusion challenges: noise, delays, satellite loss, multi-path.

---

### **Q37. Monocular Depth Estimation Over Time**

**Answer:**

- Track points across frames.
- Use known camera motion to triangulate depth.
- Scale is unknown without external reference (e.g., IMU).

---

### **Q38. Loop Closure in SLAM**

**Answer:**

- Detect re-visited locations (e.g., via Bag-of-Words or NetVLAD).
- Verify geometrically (pose graph, RANSAC).
- Correct accumulated drift via pose graph or bundle adjustment.

---

### **Q39. Multi-view vs Pairwise SfM**

**Answer:**

- **Pairwise** is local and less robust.
- **Multi-view** jointly optimizes across all views for consistent structure.
- More accurate and robust to noise.

---

### **Q40. Rotation Representations**

**Answer:**

- **Rotation Matrix**: 3×3, orthonormal.
- **Quaternion**: 4D unit vector, compact and avoids gimbal lock.
- **Euler angles**: Human-readable, but prone to singularities.
- **Axis-angle**: Rotation by θ about an axis.

---

### **Q41. Quaternions: Pros and Cons**

**Pros:**

- No gimbal lock
- Compact (4D)
- Efficient for interpolation (slerp)

**Cons:**

- Less intuitive
- Needs normalization
- Can be ambiguous (q and -q represent same rotation)

---

### **Q42. EKF Prediction vs Update**

**Answer:**

- **Prediction**: Use motion model to estimate new state + increase uncertainty.
- **Update**: Use new measurement to correct state and reduce uncertainty.
- Core of all filtering-based fusion methods.

---

### **Q43. Scale Ambiguity in Monocular SLAM**

**Answer:**

- Monocular vision can’t determine real-world scale.
- Mitigation: IMU, known object size, stereo, or learned priors.

---

### **Q44. Monocular + IMU SLAM Design**

**Answer:**

1. Calibrate IMU-to-camera transform.
2. Track visual features.
3. Pre-integrate IMU for prediction.
4. Estimate poses via optimization (e.g., VINS).
5. Optional: loop closure for drift correction.

---

### **Q45. Camera Projection Models**

**Answer:**

- **Pinhole model**:
    
    x=K[R∣t]Xx = K [R | t] X
    
    where `K` is intrinsics.
    
- Models projection of 3D points to 2D pixels.
- Extensions: distortion, fisheye, omnidirectional.

---

### **Q46. Sources of Drift in VIO**

**Answer:**

- IMU biases
- Feature loss or mismatches
- Poor initialization
- Linearization errors in optimization
- Lack of loop closure

---

### **Q47. Marginalization in Optimization**

**Answer:**

- Removes old variables to reduce problem size.
- Keeps system bounded in memory.
- Uses Schur complement to preserve influence.

---

### **Q48. Motion Segmentation in SLAM**

**Answer:**

- Detect dynamic objects (e.g., moving people, cars).
- Exclude from pose estimation.
- Improves robustness in dynamic environments.

---

### **Q49. Sequential vs Global SfM**

**Answer:**

- **Sequential**: Incremental build-up, robust but accumulates drift.
- **Global**: Estimates all poses first (rotation averaging), then structure.
- Global SfM is faster and more robust to loop closures.

---

### **Q50. ML Integration into SLAM**

**Answer:**

- Learned feature detectors/matchers (SuperPoint, LoFTR)
- Learned depth estimation
- Place recognition (NetVLAD)
- Outlier rejection, uncertainty modeling