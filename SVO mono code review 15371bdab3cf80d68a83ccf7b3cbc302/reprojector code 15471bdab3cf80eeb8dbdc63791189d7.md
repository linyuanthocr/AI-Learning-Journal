# reprojector code

![image.png](reprojector%20code%2015471bdab3cf80eeb8dbdc63791189d7/image.png)

```cpp
// Project points from the map into the image and find the corresponding
/// feature (corner). We don't search a match for every point but only for one
/// point per cell. Thereby, we achieve a homogeneously distributed set of
/// matched features and at the same time we can save processing time by not
/// projecting all points.
class Reprojector
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Reprojector config parameters
  struct Options {
    size_t max_n_kfs;   //!< max number of keyframes to reproject from
    bool find_match_direct;
    Options()
    : max_n_kfs(10),
      find_match_direct(true)
    {}
  } options_;

  size_t n_matches_;
  size_t n_trials_;

  Reprojector(vk::AbstractCamera* cam, Map& map);

  ~Reprojector();

  /// Project points from the map into the image. First finds keyframes with
  /// overlapping field of view and projects only those map-points.
  void reprojectMap(
      FramePtr frame,
      std::vector< std::pair<FramePtr,std::size_t> >& overlap_kfs);

private:

  /// A candidate is a point that projects into the image plane and for which we
  /// will search a maching feature in the image.
  struct Candidate {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Point* pt;       //!< 3D point.
    Vector2d px;     //!< projected 2D pixel location.
    Candidate(Point* pt, Vector2d& px) : pt(pt), px(px) {}
  };
  typedef std::list<Candidate, aligned_allocator<Candidate> > Cell;
  typedef std::vector<Cell*> CandidateGrid;

  /// The grid stores a set of candidate matches. For every grid cell we try to find one match.
  struct Grid
  {
    CandidateGrid cells;
    vector<int> cell_order;
    int cell_size;
    int grid_n_cols;
    int grid_n_rows;
  };

  Grid grid_;
  Matcher matcher_;
  Map& map_;

  static bool pointQualityComparator(Candidate& lhs, Candidate& rhs);
  void initializeGrid(vk::AbstractCamera* cam);
  void resetGrid();
  bool reprojectCell(Cell& cell, FramePtr frame);
  bool reprojectPoint(FramePtr frame, Point* point);
};
```

## reprojectMap

1. resetGrid
2. Identify those Keyframes which share a common field of view.
3. Sort KFs with overlap according to their closeness
4. Reproject all **mappoints** of the closest N kfs with overlap. We only store
in which grid cell the points fall.
    
    ```cpp
     size_t n = 0;
      overlap_kfs.reserve(options_.max_n_kfs);
      for(auto it_frame=close_kfs.begin(), ite_frame=close_kfs.end();
          it_frame!=ite_frame && n<options_.max_n_kfs; ++it_frame, ++n)
      {
        FramePtr ref_frame = it_frame->first;
        overlap_kfs.push_back(pair<FramePtr,size_t>(ref_frame,0));
    
        // Try to reproject each mappoint that the other KF observes
        for(auto it_ftr=ref_frame->fts_.begin(), ite_ftr=ref_frame->fts_.end();
            it_ftr!=ite_ftr; ++it_ftr)
        {
          // check if the feature has a mappoint assigned
          if((*it_ftr)->point == NULL)
            continue;
    
          // make sure we project a point only once
          if((*it_ftr)->point->last_projected_kf_id_ == frame->id_)
            continue;
          (*it_ftr)->point->last_projected_kf_id_ = frame->id_;
          if(reprojectPoint(frame, (*it_ftr)->point))
            overlap_kfs.back().second++;
        }
      }
    ```
    
5. Now project all **point candidates**
    
    ```cpp
        boost::unique_lock<boost::mutex> lock(map_.point_candidates_.mut_);
        auto it=map_.point_candidates_.candidates_.begin();
        while(it!=map_.point_candidates_.candidates_.end())
        {
          if(!reprojectPoint(frame, it->first))
          {
            it->first->n_failed_reproj_ += 3;
            if(it->first->n_failed_reproj_ > 30)
            {
              map_.point_candidates_.deleteCandidate(*it);
              it = map_.point_candidates_.candidates_.erase(it);
              continue;
            }
          }
          ++it;
        }
    ```
    
6.  Now we go through each grid cell and select one point to match. At the end, we should have at **maximum one reprojected point per cell**.
    
    ```
      for(size_t i=0; i<grid_.cells.size(); ++i)
      {
        // we prefer good quality points over unkown quality (more likely to match)
        // and unknown quality over candidates (position not optimized)
        if(reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame))
          ++n_matches_;
        if(n_matches_ > (size_t) Config::maxFts())
          break;
      }
    ```
    
    ```cpp
      /// A candidate is a point that projects into the image plane and for which we
      /// will search a maching feature in the image.
      struct Candidate {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Point* pt;       //!< 3D point.
        Vector2d px;     //!< projected 2D pixel location.
        Candidate(Point* pt, Vector2d& px) : pt(pt), px(px) {}
      };
      typedef std::list<Candidate, aligned_allocator<Candidate> > Cell;
    ```
    
    ### reprojectCell
    
    ```cpp
        // we prefer good quality points over unkown quality (more likely to match)
        // and unknown quality over candidates (position not optimized)
    ```
    
    ```cpp
    bool Reprojector::reprojectCell(Cell& cell, FramePtr frame)
    {
      cell.sort(boost::bind(&Reprojector::pointQualityComparator, _1, _2));
      Cell::iterator it=cell.begin();
      while(it!=cell.end())
      {
        ++n_trials_;
    
        if(it->pt->type_ == Point::TYPE_DELETED)
        {
          it = cell.erase(it);
          continue;
        }
    
        bool found_match = true;
        if(options_.find_match_direct)
          found_match = matcher_.findMatchDirect(*it->pt, *frame, it->px);
        if(!found_match)
        {
          it->pt->n_failed_reproj_++;
          if(it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_failed_reproj_ > 15)
            map_.safeDeletePoint(it->pt);
          if(it->pt->type_ == Point::TYPE_CANDIDATE  && it->pt->n_failed_reproj_ > 30)
            map_.point_candidates_.deleteCandidatePoint(it->pt);
          it = cell.erase(it);
          continue;
        }
        it->pt->n_succeeded_reproj_++;
        if(it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_succeeded_reproj_ > 10)
          it->pt->type_ = Point::TYPE_GOOD;
    
        Feature* new_feature = new Feature(frame.get(), it->px, matcher_.search_level_);
        frame->addFeature(new_feature);
    
        // Here we add a reference in the feature to the 3D point, the other way
        // round is only done if this frame is selected as keyframe.
        new_feature->point = it->pt;
    
        if(matcher_.ref_ftr_->type == Feature::EDGELET)
        {
          new_feature->type = Feature::EDGELET;
          new_feature->grad = matcher_.A_cur_ref_*matcher_.ref_ftr_->grad;
          new_feature->grad.normalize();
        }
    
        // If the keyframe is selected and we reproject the rest, we don't have to
        // check this point anymore.
        it = cell.erase(it);
    
        // Maximum one point per cell.
        return true;
      }
      return false;
    
    ```
    
    **findMatchDirect**
    
    1. success: add feature points and its 3d point
    2. else: add 1 time failure, which accumulate to delete not useless points 
    
    ```cpp
    bool Matcher::findMatchDirect(
        const Point& pt,
        const Frame& cur_frame,
        Vector2d& px_cur)
    {
      if(!pt.getCloseViewObs(cur_frame.pos(), ref_ftr_))
        return false;
    
      if(!ref_ftr_->frame->cam_->isInFrame(
          ref_ftr_->px.cast<int>()/(1<<ref_ftr_->level), halfpatch_size_+2, ref_ftr_->level))
        return false;
    
      // warp affine
      warp::getWarpMatrixAffine(
          *ref_ftr_->frame->cam_, *cur_frame.cam_, ref_ftr_->px, ref_ftr_->f,
          (ref_ftr_->frame->pos() - pt.pos_).norm(),
          cur_frame.T_f_w_ * ref_ftr_->frame->T_f_w_.inverse(), ref_ftr_->level, A_cur_ref_);
      search_level_ = warp::getBestSearchLevel(A_cur_ref_, Config::nPyrLevels()-1);
      warp::warpAffine(A_cur_ref_, ref_ftr_->frame->img_pyr_[ref_ftr_->level], ref_ftr_->px,
                       ref_ftr_->level, search_level_, halfpatch_size_+1, patch_with_border_);
      createPatchFromPatchWithBorder();
    
      // px_cur should be set
      Vector2d px_scaled(px_cur/(1<<search_level_));
    
      bool success = false;
      if(ref_ftr_->type == Feature::EDGELET)
      {
        Vector2d dir_cur(A_cur_ref_*ref_ftr_->grad);
        dir_cur.normalize();
        success = feature_alignment::align1D(
              cur_frame.img_pyr_[search_level_], dir_cur.cast<float>(),
              patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_);
      }
      else
      {
        success = feature_alignment::align2D(
          cur_frame.img_pyr_[search_level_], patch_with_border_, patch_,
          options_.align_max_iter, px_scaled);
      }
      px_cur = px_scaled * (1<<search_level_);
      return success;
    }
    ```
    
    ```cpp
    bool align2D(
        const cv::Mat& cur_img,
        uint8_t* ref_patch_with_border,
        uint8_t* ref_patch,
        const int n_iter,
        Vector2d& cur_px_estimate,
        bool no_simd)
    {
    #ifdef __ARM_NEON__
      if(!no_simd)
        return align2D_NEON(cur_img, ref_patch_with_border, ref_patch, n_iter, cur_px_estimate);
    #endif
    
      const int halfpatch_size_ = 4;
      const int patch_size_ = 8;
      const int patch_area_ = 64;
      bool converged=false;
    
      // compute derivative of template and prepare inverse compositional
      float __attribute__((__aligned__(16))) ref_patch_dx[patch_area_];
      float __attribute__((__aligned__(16))) ref_patch_dy[patch_area_];
      Matrix3f H; H.setZero();
    
      // compute gradient and hessian
      const int ref_step = patch_size_+2;
      float* it_dx = ref_patch_dx;
      float* it_dy = ref_patch_dy;
      for(int y=0; y<patch_size_; ++y)
      {
        uint8_t* it = ref_patch_with_border + (y+1)*ref_step + 1;
        for(int x=0; x<patch_size_; ++x, ++it, ++it_dx, ++it_dy)
        {
          Vector3f J;
          J[0] = 0.5 * (it[1] - it[-1]);
          J[1] = 0.5 * (it[ref_step] - it[-ref_step]);
          J[2] = 1;
          *it_dx = J[0];
          *it_dy = J[1];
          H += J*J.transpose();
        }
      }
      Matrix3f Hinv = H.inverse();
      float mean_diff = 0;
    
      // Compute pixel location in new image:
      float u = cur_px_estimate.x();
      float v = cur_px_estimate.y();
    
      // termination condition
      const float min_update_squared = 0.03*0.03;
      const int cur_step = cur_img.step.p[0];
    //  float chi2 = 0;
      Vector3f update; update.setZero();
      for(int iter = 0; iter<n_iter; ++iter)
      {
        int u_r = floor(u);
        int v_r = floor(v);
        if(u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols-halfpatch_size_ || v_r >= cur_img.rows-halfpatch_size_)
          break;
    
        if(isnan(u) || isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
          return false;
    
        // compute interpolation weights
        float subpix_x = u-u_r;
        float subpix_y = v-v_r;
        float wTL = (1.0-subpix_x)*(1.0-subpix_y);
        float wTR = subpix_x * (1.0-subpix_y);
        float wBL = (1.0-subpix_x)*subpix_y;
        float wBR = subpix_x * subpix_y;
    
        // loop through search_patch, interpolate
        uint8_t* it_ref = ref_patch;
        float* it_ref_dx = ref_patch_dx;
        float* it_ref_dy = ref_patch_dy;
    //    float new_chi2 = 0.0;
        Vector3f Jres; Jres.setZero();
        for(int y=0; y<patch_size_; ++y)
        {
          uint8_t* it = (uint8_t*) cur_img.data + (v_r+y-halfpatch_size_)*cur_step + u_r-halfpatch_size_;
          for(int x=0; x<patch_size_; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy)
          {
            float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1];
            float res = search_pixel - *it_ref + mean_diff;
            Jres[0] -= res*(*it_ref_dx);
            Jres[1] -= res*(*it_ref_dy);
            Jres[2] -= res;
    //        new_chi2 += res*res;
          }
        }
        update = Hinv * Jres;
        u += update[0];
        v += update[1];
        mean_diff += update[2];
    
    #if SUBPIX_VERBOSE
        cout << "Iter " << iter << ":"
             << "\t u=" << u << ", v=" << v
             << "\t update = " << update[0] << ", " << update[1]
    //         << "\t new chi2 = " << new_chi2 << endl;
    #endif
    
        if(update[0]*update[0]+update[1]*update[1] < min_update_squared)
        {
    #if SUBPIX_VERBOSE
          cout << "converged." << endl;
    #endif
          converged=true;
          break;
        }
      }
    
      cur_px_estimate << u, v;
      return converged;
    ```
    
    ```cpp
    
    bool Point::getCloseViewObs(const Vector3d& framepos, Feature*& ftr) const
    {
      // TODO: get frame with same point of view AND same pyramid level!
      Vector3d obs_dir(framepos - pos_); obs_dir.normalize();
      auto min_it=obs_.begin();
      double min_cos_angle = 0;
      for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
      {
        Vector3d dir((*it)->frame->pos() - pos_); dir.normalize();
        double cos_angle = obs_dir.dot(dir);
        if(cos_angle > min_cos_angle)
        {
          min_cos_angle = cos_angle;
          min_it = it;
        }
      }
      ftr = *min_it;
      if(min_cos_angle < 0.5) // assume that observations larger than 60° are useless
        return false;
      return true;
    }
    ```
    
    ```cpp
    void getWarpMatrixAffine(
        const vk::AbstractCamera& cam_ref,
        const vk::AbstractCamera& cam_cur,
        const Vector2d& px_ref,
        const Vector3d& f_ref,
        const double depth_ref,
        const SE3& T_cur_ref,
        const int level_ref,
        Matrix2d& A_cur_ref)
    {
      // Compute affine warp matrix A_ref_cur
      const int halfpatch_size = 5;
      const Vector3d xyz_ref(f_ref*depth_ref);
      Vector3d xyz_du_ref(cam_ref.cam2world(px_ref + Vector2d(halfpatch_size,0)*(1<<level_ref)));
      Vector3d xyz_dv_ref(cam_ref.cam2world(px_ref + Vector2d(0,halfpatch_size)*(1<<level_ref)));
      xyz_du_ref *= xyz_ref[2]/xyz_du_ref[2];
      xyz_dv_ref *= xyz_ref[2]/xyz_dv_ref[2];
      const Vector2d px_cur(cam_cur.world2cam(T_cur_ref*(xyz_ref)));
      const Vector2d px_du(cam_cur.world2cam(T_cur_ref*(xyz_du_ref)));
      const Vector2d px_dv(cam_cur.world2cam(T_cur_ref*(xyz_dv_ref)));
      A_cur_ref.col(0) = (px_du - px_cur)/halfpatch_size;
      A_cur_ref.col(1) = (px_dv - px_cur)/halfpatch_size;
    }
    ```