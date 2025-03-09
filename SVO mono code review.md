# SVO mono code review

http://www.cgabc.xyz/posts/6224da90/

https://blog.csdn.net/heyijia0327/article/details/51649082

https://heyijia.blog.csdn.net/article/details/51083398?spm=1001.2014.3001.5502

https://www.cnblogs.com/wxt11/p/7097250.html

[SVO改进](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302/SVO%E6%94%B9%E8%BF%9B%2019071bdab3cf80c7bbb8e7641e36e09b.md)

Early direct monocular SLAM methods tracked and mapped few—sometimes manually selected—planar patches. With DTAM, a novel direct method was introduced that computes a dense depth map for each keyframe through minimisation of a global, spatially-regularised energy functional. This approach is computationally very intensive and only possible through heavy GPU parallelization

The proposed Semi-Direct Visual Odometry (SVO) algorithm uses feature-correspondence; however, feature- correspondence is an implicit result of direct motion estimation rather than of explicit feature extraction and matching. Feature extraction is only required when a keyframe is
selected to initialize new 3D points. The advantage is **increased speed** due to the lack of feature-extraction at every frame and **increased accuracy** through subpixel feature correspondence.

The contributions of this paper are:
(1) a novel **semi-direct VO pipeline** that is faster and more accurate than the current state-of-the-art for MAVs
(2) the integration of **a probabilistic mapping** method that is robust to outlier measurements.

![image.png](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302/image.png)

[Sparse model-based image align](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302/Sparse%20model-based%20image%20align%2017b71bdab3cf80759fa9f483be1157ca.md)

![image.png](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302/image%201.png)

[Feature alignment](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302/Feature%20alignment%2017b71bdab3cf809a9791dbc5dd239a96.md)

![image.png](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302/image%202.png)

[Pose & structure refinement](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302/Pose%20&%20structure%20refinement%2017b71bdab3cf80cf80b5e42bd1e7b0d9.md)

![image.png](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302/f1c92673-36b2-4bb7-9671-3adacd94b3cc.png)

[DepthFilter](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302/DepthFilter%2017b71bdab3cf80b2ace7f0e612d2c1c4.md)

![image.png](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302/image%203.png)

Details：

1. The algorithm is bootstrapped to obtain the pose of the first two keyframes and the initial map. The initial map is triangulated from the first two views. 
2. In order to cope with large motions, we apply the sparse image alignment algorithm in a coarse-to-fine scheme. 
3. The algorithm keeps for efficiency reasons a fixed number of keyframes in the map, which are used as reference for feature-alignment and for structure refinement. A keyframe is selected if the Euclidean distance of the new frame relative to all keyframes exceeds 12% of the average scene depth
4. In the mapping thread, we divide the image in cells of fixed size (e.g., 30 × 30 pixels). A new depth-filter is initialized at the **FAST corner with highest Shi-Tomasi score** in the cell unless there is already a 2D-to-3D correspondence present. This results in evenly distributed features in the image.

## FramehandlerMono

```cpp
/// Monocular Visual Odometry Pipeline as described in the SVO paper.
class FrameHandlerMono : public FrameHandlerBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  FrameHandlerMono(vk::AbstractCamera* cam);
  virtual ~FrameHandlerMono();

  /// Provide an image.
  void addImage(const cv::Mat& img, double timestamp);

  /// Set the first frame (used for synthetic datasets in benchmark node)
  void setFirstFrame(const FramePtr& first_frame);

  /// Get the last frame that has been processed.
  FramePtr lastFrame() { return last_frame_; }

  /// Get the set of spatially closest keyframes of the last frame.
  const set<FramePtr>& coreKeyframes() { return core_kfs_; }

  /// Return the feature track to visualize the KLT tracking during initialization.
  const vector<cv::Point2f>& initFeatureTrackRefPx() const { return klt_homography_init_.px_ref_; }
  const vector<cv::Point2f>& initFeatureTrackCurPx() const { return klt_homography_init_.px_cur_; }

  /// Access the depth filter.
  DepthFilter* depthFilter() const { return depth_filter_; }

  /// An external place recognition module may know where to relocalize.
  bool relocalizeFrameAtPose(
      const int keyframe_id,
      const SE3& T_kf_f,
      const cv::Mat& img,
      const double timestamp);

protected:
  vk::AbstractCamera* cam_;                     //!< Camera model, can be ATAN, Pinhole or Ocam (see vikit).
  Reprojector reprojector_;                     //!< Projects points from other keyframes into the current frame
  FramePtr new_frame_;                          //!< Current frame.
  FramePtr last_frame_;                         //!< Last frame, not necessarily a keyframe.
  set<FramePtr> core_kfs_;                      //!< Keyframes in the closer neighbourhood.
  vector< pair<FramePtr,size_t> > overlap_kfs_; //!< All keyframes with overlapping field of view. the paired number specifies how many common mappoints are observed TODO: why vector!?
  initialization::KltHomographyInit klt_homography_init_; //!< Used to estimate pose of the first two keyframes by estimating a homography.
  DepthFilter* depth_filter_;                   //!< Depth estimation algorithm runs in a parallel thread and is used to initialize new 3D points.

  /// Initialize the visual odometry algorithm.
  virtual void initialize();

  /// Processes the first frame and sets it as a keyframe.
  virtual UpdateResult processFirstFrame();

  /// Processes all frames after the first frame until a keyframe is selected.
  virtual UpdateResult processSecondFrame();

  /// Processes all frames after the first two keyframes.
  virtual UpdateResult processFrame();

  /// Try relocalizing the frame at relative position to provided keyframe.
  virtual UpdateResult relocalizeFrame(
      const SE3& T_cur_ref,
      FramePtr ref_keyframe);

  /// Reset the frame handler. Implement in derived class.
  virtual void resetAll();

  /// Keyframe selection criterion.
  virtual bool needNewKf(double scene_depth_mean);

  void setCoreKfs(size_t n_closest);
};

```

note: depth_filter_ is a callback 

```cpp
FrameHandlerMono::FrameHandlerMono(vk::AbstractCamera* cam) :
  FrameHandlerBase(),
  cam_(cam),
  reprojector_(cam_, map_),
  depth_filter_(NULL)
{
  initialize();
}

void FrameHandlerMono::initialize()
{
  feature_detection::DetectorPtr feature_detector(
      new feature_detection::FastDetector(
          cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels()));
  DepthFilter::callback_t depth_filter_cb = boost::bind(
      &MapPointCandidates::newCandidatePoint, &map_.point_candidates_, _1, _2);
  depth_filter_ = new DepthFilter(feature_detector, depth_filter_cb);
  depth_filter_->startThread();
}
```

## addImage

FrameHandlerMono::addImage(const cv::Mat& img, const double timestamp)

1. startFrameProcessCommon: logs, timer, and map clear
2. clear systemcore_kfs_, overlap_kfs_
3. new_frame_ reset, no key points but with image pyramid
4. enter stage (4 stages, only choose one)
    1. processFirstFrame();
    2. processSecondFrame();
    3. processFrame();
    4. relocalizeFrame(SE3(Matrix3d::Identity(), Vector3d::Zero()),
    map_.getClosestKeyframe(last_frame_));
5. update frame, last_frame_=new_frame_, new_frame_.reset();
6. finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->nObs());

### processFirstFrame

```cpp
FrameHandlerMono::UpdateResult FrameHandlerMono::processFirstFrame()
{
  new_frame_->T_f_w_ = SE3(Matrix3d::Identity(), Vector3d::Zero());
  if(klt_homography_init_.addFirstFrame(new_frame_) == initialization::FAILURE)
    return RESULT_NO_KEYFRAME;
  new_frame_->setKeyframe();
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_SECOND_FRAME;
  SVO_INFO_STREAM("Init: Selected first frame.");
  return RESULT_IS_KEYFRAME;
}
```

FirstFrame:

1. reset: tracking feature points, reference frame
2. detectFeatures
    1. fast10 corner detector, 0-L pyramid, image into H/cell_size*W/cell_size cells.
        1. detect features at each level
        2. if cell is occupied (depth filter has output a well defined map point), continue
        3. for each cell, the most high score feature is reserved
        
        ```cpp
            vector<int> scores, nm_corners;
            fast::fast_corner_score_10((fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols, fast_corners, 20, scores);
            fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);
        
            for(auto it=nm_corners.begin(), ite=nm_corners.end(); it!=ite; ++it)
            {
              fast::fast_xy& xy = fast_corners.at(*it);
              const int k = static_cast<int>((xy.y*scale)/cell_size_)*grid_n_cols_
                          + static_cast<int>((xy.x*scale)/cell_size_);
              if(grid_occupancy_[k])
                continue;
              const float score = vk::shiTomasiScore(img_pyr[L], xy.x, xy.y);
              if(score > corners.at(k).score)
                corners.at(k) = Corner(xy.x*scale, xy.y*scale, score, L, 0.0f);
            }
        ```
        
    2. select all the **features** with score>config.th (each with detection pyramid level)
        
        ```python
          // Create feature for every corner that has high enough corner score
          std::for_each(corners.begin(), corners.end(), [&](Corner& c) {
            if(c.score > detection_threshold)
              fts.push_back(new Feature(frame, Vector2d(c.x, c.y), c.level));
          });
        ```
        
    3. reset grid occupied false
        
        ### Feature
        
        ```cpp
        struct Feature
        {
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
          enum FeatureType {
            CORNER,
            EDGELET
          };
        
          FeatureType type;     //!< Type can be corner or edgelet.
          Frame* frame;         //!< Pointer to frame in which the feature was detected.
          Vector2d px;          //!< Coordinates in pixels on pyramid level 0.
          Vector3d f;           //!< Unit-bearing vector of the feature.
          int level;            //!< Image pyramid level where feature was extracted.
          Point* point;         //!< Pointer to 3D point which corresponds to the feature.
          Vector2d grad;        //!< Dominant gradient direction for edglets, normalized.
        
          Feature(Frame* _frame, const Vector2d& _px, int _level) :
            type(CORNER),
            frame(_frame),
            px(_px),
            f(frame->cam_->cam2world(px)),
            level(_level),
            point(NULL),
            grad(1.0,0.0)
          {}
        
          Feature(Frame* _frame, const Vector2d& _px, const Vector3d& _f, int _level) :
            type(CORNER),
            frame(_frame),
            px(_px),
            f(_f),
            level(_level),
            point(NULL),
            grad(1.0,0.0)
          {}
        
          Feature(Frame* _frame, Point* _point, const Vector2d& _px, const Vector3d& _f, int _level) :
            type(CORNER),
            frame(_frame),
            px(_px),
            f(_f),
            level(_level),
            point(_point),
            grad(1.0,0.0)
          {}
        };
        ```
        
3. feature num> 100, success, others break.
4. set keyframe: is_keyframe_ = true; set 5 features points key_pts_: setKeyPoints() 
5. update map with first key frame.

## Map

definition:

```cpp
class Point;
class Feature;
class Seed;

/// Container for converged 3D points that are not already assigned to two keyframes.
class MapPointCandidates
{
public:
  typedef pair<Point*, Feature*> PointCandidate;
  typedef list<PointCandidate> PointCandidateList;

  /// The depth-filter is running in a parallel thread and fills the canidate list.
  /// This mutex controls concurrent access to point_candidates.
  boost::mutex mut_;

  /// Candidate points are created from converged seeds.
  /// Until the next keyframe, these points can be used for reprojection and pose optimization.
  PointCandidateList candidates_;
  list< Point* > trash_points_;

  MapPointCandidates();
  ~MapPointCandidates();

  /// Add a candidate point.
  void newCandidatePoint(Point* point, double depth_sigma2);

  /// Adds the feature to the frame and deletes candidate from list.
  void addCandidatePointToFrame(FramePtr frame);

  /// Remove a candidate point from the list of candidates.
  bool deleteCandidatePoint(Point* point);

  /// Remove all candidates that belong to a frame.
  void removeFrameCandidates(FramePtr frame);

  /// Reset the candidate list, remove and delete all points.
  void reset();

  void deleteCandidate(PointCandidate& c);

  void emptyTrash();
};

/// Map object which saves all keyframes which are in a map.
class Map : boost::noncopyable
{
public:
  list< FramePtr > keyframes_;          //!< List of keyframes in the map.
  list< Point* > trash_points_;         //!< A deleted point is moved to the trash bin. Now and then this is cleaned. One reason is that the visualizer must remove the points also.
  MapPointCandidates point_candidates_;

  Map();
  ~Map();

  /// Reset the map. Delete all keyframes and reset the frame and point counters.
  void reset();

  /// Delete a point in the map and remove all references in keyframes to it.
  void safeDeletePoint(Point* pt);

  /// Moves the point to the trash queue which is cleaned now and then.
  void deletePoint(Point* pt);

  /// Moves the frame to the trash queue which is cleaned now and then.
  bool safeDeleteFrame(FramePtr frame);

  /// Remove the references between a point and a frame.
  void removePtFrameRef(Frame* frame, Feature* ftr);

  /// Add a new keyframe to the map.
  void addKeyframe(FramePtr new_keyframe);

  /// Given a frame, return all keyframes which have an overlapping field of view.
  void getCloseKeyframes(const FramePtr& frame, list< pair<FramePtr,double> >& close_kfs) const;

  /// Return the keyframe which is spatially closest and has overlapping field of view.
  FramePtr getClosestKeyframe(const FramePtr& frame) const;

  /// Return the keyframe which is furthest apart from pos.
  FramePtr getFurthestKeyframe(const Vector3d& pos) const;

  bool getKeyframeById(const int id, FramePtr& frame) const;

  /// Transform the whole map with rotation R, translation t and scale s.
  void transform(const Matrix3d& R, const Vector3d& t, const double& s);

  /// Empty trash bin of deleted keyframes and map points. We don't delete the
  /// points immediately to ensure proper cleanup and to provide the visualizer
  /// a list of objects which must be removed.
  void emptyTrash();

  /// Return the keyframe which was last inserted in the map.
  inline FramePtr lastKeyframe() { return keyframes_.back(); }

  /// Return the number of keyframes in the map
  inline size_t size() const { return keyframes_.size(); }
};
```

### DepthFilter

![image.png](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302/image%204.png)

```cpp
class DepthFilter
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef boost::unique_lock<boost::mutex> lock_t;
  typedef boost::function<void ( Point*, double )> callback_t;

  /// Depth-filter config parameters
  struct Options
  {
    bool check_ftr_angle;                       //!< gradient features are only updated if the epipolar line is orthogonal to the gradient.
    bool epi_search_1d;                         //!< restrict Gauss Newton in the epipolar search to the epipolar line.
    bool verbose;                               //!< display output.
    bool use_photometric_disparity_error;       //!< use photometric disparity error instead of 1px error in tau computation.
    int max_n_kfs;                              //!< maximum number of keyframes for which we maintain seeds.
    double sigma_i_sq;                          //!< image noise.
    double seed_convergence_sigma2_thresh;      //!< threshold on depth uncertainty for convergence.
    Options()
    : check_ftr_angle(false),
      epi_search_1d(false),
      verbose(false),
      use_photometric_disparity_error(false),
      max_n_kfs(3),
      sigma_i_sq(5e-4),
      seed_convergence_sigma2_thresh(200.0)
    {}
  } options_;

  DepthFilter(
      feature_detection::DetectorPtr feature_detector,
      callback_t seed_converged_cb);

  virtual ~DepthFilter();

  /// Start this thread when seed updating should be in a parallel thread.
  void startThread();

  /// Stop the parallel thread that is running.
  void stopThread();

  /// Add frame to the queue to be processed.
  void addFrame(FramePtr frame);

  /// Add new keyframe to the queue
  void addKeyframe(FramePtr frame, double depth_mean, double depth_min);

  /// Remove all seeds which are initialized from the specified keyframe. This
  /// function is used to make sure that no seeds points to a non-existent frame
  /// when a frame is removed from the map.
  void removeKeyframe(FramePtr frame);

  /// If the map is reset, call this function such that we don't have pointers
  /// to old frames.
  void reset();

  /// Returns a copy of the seeds belonging to frame. Thread-safe.
  /// Can be used to compute the Next-Best-View in parallel.
  /// IMPORTANT! Make sure you hold a valid reference counting pointer to frame
  /// so it is not being deleted while you use it.
  void getSeedsCopy(const FramePtr& frame, std::list<Seed>& seeds);

  /// Return a reference to the seeds. This is NOT THREAD SAFE!
  std::list<Seed, aligned_allocator<Seed> >& getSeeds() { return seeds_; }

  /// Bayes update of the seed, x is the measurement, tau2 the measurement uncertainty
  static void updateSeed(
      const float x,
      const float tau2,
      Seed* seed);

  /// Compute the uncertainty of the measurement.
  static double computeTau(
      const SE3& T_ref_cur,
      const Vector3d& f,
      const double z,
      const double px_error_angle);

protected:
  feature_detection::DetectorPtr feature_detector_;
  callback_t seed_converged_cb_;
  std::list<Seed, aligned_allocator<Seed> > seeds_;
  **boost::mutex seeds_mut_;**
  bool seeds_updating_halt_;            //!< Set this value to true when seeds updating should be interrupted.
  boost::thread* thread_;
  std::queue<FramePtr> frame_queue_;
  **boost::mutex frame_queue_mut_;**
  boost::condition_variable frame_queue_cond_;
  FramePtr new_keyframe_;               //!< Next keyframe to extract new seeds.
  bool new_keyframe_set_;               //!< Do we have a new keyframe to process?.
  double new_keyframe_min_depth_;       //!< Minimum depth in the new keyframe. Used for range in new seeds.
  double new_keyframe_mean_depth_;      //!< Maximum depth in the new keyframe. Used for range in new seeds.
  vk::PerformanceMonitor permon_;       //!< Separate performance monitor since the DepthFilter runs in a parallel thread.
  Matcher matcher_;

  /// Initialize new seeds from a frame.
  void initializeSeeds(FramePtr frame);

  /// Update all seeds with a new measurement frame.
  virtual void updateSeeds(FramePtr frame);

  /// When a new keyframe arrives, the frame queue should be cleared.
  void clearFrameQueue();

  /// A thread that is continuously updating the seeds.
  void updateSeedsLoop();
};

```

some functions:

```cpp
void DepthFilter::addKeyframe(FramePtr frame, double depth_mean, double depth_min)
{
  new_keyframe_min_depth_ = depth_min;
  new_keyframe_mean_depth_ = depth_mean;
  if(thread_ != NULL)
  {
    new_keyframe_ = frame;
    new_keyframe_set_ = true;
    seeds_updating_halt_ = true;
    frame_queue_cond_.notify_one();
  }
  else
    initializeSeeds(frame);
}

void DepthFilter::initializeSeeds(FramePtr frame)
{
  Features new_features;
  feature_detector_->setExistingFeatures(frame->fts_);
  feature_detector_->detect(frame.get(), frame->img_pyr_,
                            Config::triangMinCornerScore(), new_features);

  // initialize a seed for every new feature
  seeds_updating_halt_ = true;
  lock_t lock(seeds_mut_); // by locking the updateSeeds function stops
  ++Seed::batch_counter;
  std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr){
    seeds_.push_back(Seed(ftr, new_keyframe_mean_depth_, new_keyframe_min_depth_));
  });

  if(options_.verbose)
    SVO_INFO_STREAM("DepthFilter: Initialized "<<new_features.size()<<" new seeds");
  seeds_updating_halt_ = false;
}
```

keyframe add seeds to the depth filter.

### Seed

```cpp
/// A seed is a probabilistic depth estimate for a single pixel.
struct Seed
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static int batch_counter;
  static int seed_counter;
  int batch_id;                //!< Batch id is the id of the keyframe for which the seed was created.
  int id;                      //!< Seed ID, only used for visualization.
  Feature* ftr;                //!< Feature in the keyframe for which the depth should be computed.
  float a;                     //!< a of Beta distribution: When high, probability of inlier is large.
  float b;                     //!< b of Beta distribution: When high, probability of outlier is large.
  float mu;                    //!< Mean of normal distribution.
  float z_range;               //!< Max range of the possible depth.
  float sigma2;                //!< Variance of normal distribution.
  Matrix2d patch_cov;          //!< Patch covariance in reference image.
  Seed(Feature* ftr, float depth_mean, float depth_min);
};

```

### **FrameHandlerMono::processSecondFrame()**

1. InitResult KltHomographyInit::addSecondFrame(FramePtr frame_cur)
    1. trackKlt(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_);
        
        opticalflow in level 0. output px_cur_, f_cur_, disparities_
        
        ```cpp
          cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps);
          cv::calcOpticalFlowPyrLK(frame_ref->img_pyr_[0], frame_cur->img_pyr_[0],
                                   px_ref, px_cur,
                                   status, error,
                                   cv::Size2i(klt_win_size, klt_win_size),
                                   4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);
        ```
        
    2. check pair numbers, and mean disparities. if success, to step c. otherwise return false;
    3. computeHomography
        
        ```cpp
          computeHomography(
              f_ref_, f_cur_,
              frame_ref_->cam_->errorMultiplier2(), Config::poseOptimThresh(),
              inliers_, xyz_in_cur_, T_cur_from_ref_);
        ```
        
        ```cpp
        void computeHomography(
            const vector<Vector3d>& f_ref,
            const vector<Vector3d>& f_cur,
            double focal_length,
            double reprojection_threshold,
            vector<int>& inliers,
            vector<Vector3d>& xyz_in_cur,
            SE3& T_cur_from_ref)
        {
          vector<Vector2d, aligned_allocator<Vector2d> > uv_ref(f_ref.size());
          vector<Vector2d, aligned_allocator<Vector2d> > uv_cur(f_cur.size());
          for(size_t i=0, i_max=f_ref.size(); i<i_max; ++i)
          {
            uv_ref[i] = vk::project2d(f_ref[i]);
            uv_cur[i] = vk::project2d(f_cur[i]);
          }
          vk::Homography Homography(uv_ref, uv_cur, focal_length, reprojection_threshold);
          Homography.computeSE3fromMatches();
          vector<int> outliers;
          vk::computeInliers(f_cur, f_ref,
                             Homography.T_c2_from_c1.rotation_matrix(), Homography.T_c2_from_c1.translation(),
                             reprojection_threshold, focal_length,
                             xyz_in_cur, inliers, outliers);
          T_cur_from_ref = Homography.T_c2_from_c1;
        }
        ```
        
    4. check if inliers enough, return or to next
    5. Rescale the map such that the mean scene depth is equal to the specified scale
        
        ```cpp
          vector<double> depth_vec;
          for(size_t i=0; i<xyz_in_cur_.size(); ++i)
            depth_vec.push_back((xyz_in_cur_[i]).z());
          double scene_depth_median = vk::getMedian(depth_vec);
          double scale = Config::mapScale()/scene_depth_median;
          frame_cur->T_f_w_ = T_cur_from_ref_ * frame_ref_->T_f_w_;
          frame_cur->T_f_w_.translation() =
              -frame_cur->T_f_w_.rotation_matrix()*(frame_ref_->pos() + scale*(frame_cur->pos() - frame_ref_->pos()));
        ```
        
    6. For each inlier create 3D point and add feature in both frames
        
        ```cpp
          SE3 T_world_cur = frame_cur->T_f_w_.inverse();
          for(vector<int>::iterator it=inliers_.begin(); it!=inliers_.end(); ++it)
          {
            Vector2d px_cur(px_cur_[*it].x, px_cur_[*it].y);
            Vector2d px_ref(px_ref_[*it].x, px_ref_[*it].y);
            if(frame_ref_->cam_->isInFrame(px_cur.cast<int>(), 10) && frame_ref_->cam_->isInFrame(px_ref.cast<int>(), 10) && xyz_in_cur_[*it].z() > 0)
            {
              Vector3d pos = T_world_cur * (xyz_in_cur_[*it]*scale);
              Point* new_point = new Point(pos);
        
              Feature* ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, f_cur_[*it], 0));
              frame_cur->addFeature(ftr_cur);
              new_point->addFrameRef(ftr_cur);
        
              Feature* ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, f_ref_[*it], 0));
              frame_ref_->addFeature(ftr_ref);
              new_point->addFrameRef(ftr_ref);
            }
          }
        ```
        
2. two-frame bundle adjustment
    
    ```
    #ifdef USE_BUNDLE_ADJUSTMENT
      ba::twoViewBA(new_frame_.get(), map_.lastKeyframe().get(), Config::lobaThresh(), &map_);
    #endif
    
      new_frame_->setKeyframe();
      double depth_mean, depth_min;
      frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
      depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);
    ```
    
3. add frame to map
    
    ```cpp
     map_.addKeyframe(new_frame_);
      stage_ = STAGE_DEFAULT_FRAME;
      klt_homography_init_.reset();
    ```
    

### Point

```cpp
/// A 3D point on the surface of the scene.
class Point : boost::noncopyable
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  enum PointType {
    TYPE_DELETED,
    TYPE_CANDIDATE,
    TYPE_UNKNOWN,
    TYPE_GOOD
  };

  static int                  point_counter_;           //!< Counts the number of created points. Used to set the unique id.
  int                         id_;                      //!< Unique ID of the point.
  Vector3d                    pos_;                     //!< 3d pos of the point in the world coordinate frame.
  Vector3d                    normal_;                  //!< Surface normal at point.
  Matrix3d                    normal_information_;      //!< Inverse covariance matrix of normal estimation.
  bool                        normal_set_;              //!< Flag whether the surface normal was estimated or not.
  list<Feature*>              obs_;                     //!< References to keyframes which observe the point.
  size_t                      n_obs_;                   //!< Number of obervations: Keyframes AND successful reprojections in intermediate frames.
  g2oPoint*                   v_pt_;                    //!< Temporary pointer to the point-vertex in g2o during bundle adjustment.
  int                         last_published_ts_;       //!< Timestamp of last publishing.
  int                         last_projected_kf_id_;    //!< Flag for the reprojection: don't reproject a pt twice.
  PointType                   type_;                    //!< Quality of the point.
  int                         n_failed_reproj_;         //!< Number of failed reprojections. Used to assess the quality of the point.
  int                         n_succeeded_reproj_;      //!< Number of succeeded reprojections. Used to assess the quality of the point.
  int                         last_structure_optim_;    //!< Timestamp of last point optimization

  Point(const Vector3d& pos);
  Point(const Vector3d& pos, Feature* ftr);
  ~Point();

  /// Add a reference to a frame.
  void addFrameRef(Feature* ftr);

  /// Remove reference to a frame.
  bool deleteFrameRef(Frame* frame);

  /// Initialize point normal. The inital estimate will point towards the frame.
  void initNormal();

  /// Check whether mappoint has reference to a frame.
  Feature* findFrameRef(Frame* frame);

  /// Get Frame with similar viewpoint.
  bool getCloseViewObs(const Vector3d& pos, Feature*& obs) const;

  /// Get number of observations.
  inline size_t nRefs() const { return obs_.size(); }

  /// Optimize point position through minimizing the reprojection error.
  void optimize(const size_t n_iter);

  /// Jacobian of point projection on unit plane (focal length = 1) in frame (f).
  inline static void jacobian_xyz2uv(
      const Vector3d& p_in_f,
      const Matrix3d& R_f_w,
      Matrix23d& point_jac)
  {
    const double z_inv = 1.0/p_in_f[2];
    const double z_inv_sq = z_inv*z_inv;
    point_jac(0, 0) = z_inv;
    point_jac(0, 1) = 0.0;
    point_jac(0, 2) = -p_in_f[0] * z_inv_sq;
    point_jac(1, 0) = 0.0;
    point_jac(1, 1) = z_inv;
    point_jac(1, 2) = -p_in_f[1] * z_inv_sq;
    point_jac = - point_jac * R_f_w;
  }
};
```

### FrameHandlerMono::processFrame()

1. Set initial pose TODO use prior
2. sparse image align
    
    [image align **Jacobian** matrix](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302/image%20align%20Jacobian%20matrix%2015471bdab3cf80bdaf9edaed4b753a85.md)
    
    [增量方程](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302/%E5%A2%9E%E9%87%8F%E6%96%B9%E7%A8%8B%2015471bdab3cf80a78dade984a4519df6.md)
    
3. map reprojection & feature alignment
    
    [reprojector code](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302/reprojector%20code%2015471bdab3cf80eeb8dbdc63791189d7.md)
    
4. pose optimization
    
    [Minimize reprojection error](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302/Minimize%20reprojection%20error%2015471bdab3cf808fa3e9fac5261efeda.md)
    
    [pose_optimizer](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302/pose_optimizer%2015971bdab3cf80a99705f12afe84bd02.md)
    
5. structure optimization
    
    [Minimize reprojection error](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302/Minimize%20reprojection%20error%2015471bdab3cf808fa3e9fac5261efeda.md) 
    
    [optimizeStructure](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302/optimizeStructure%2015971bdab3cf806c9d7cdadbda4ffbd0.md)
    
6. select keyframe
    1. check tracking quality
    2. get scene depth to decide if add keyframe
    3. if both above qualified then to 7
7. new keyframe selected
    1. set keyframe and 5 keypoints
    2. addFrameRef from each feature points’s 3d point
    3. addCandidatePointToFrame
8. optional bundle adjustment 
    
    [Local BA](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302/Local%20BA%2015971bdab3cf80b5aa6bdff442ae0235.md)
    
9. [depth-filters add new keyframe](SVO%20mono%20code%20review%2015371bdab3cf80d68a83ccf7b3cbc302.md)
10. if limited number of keyframes, remove the one furthest apart (& related seeds, key points, map points observers)
11. add keyframe to map
    
    ```cpp
    map_.addKeyframe(new_frame_);
    ```