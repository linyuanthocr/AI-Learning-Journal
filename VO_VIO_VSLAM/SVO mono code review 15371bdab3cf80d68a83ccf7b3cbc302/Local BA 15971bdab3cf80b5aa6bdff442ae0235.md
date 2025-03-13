# Local BA

```cpp
void localBA(
    Frame* center_kf,
    set<FramePtr>* core_kfs,
    Map* map,
    size_t& n_incorrect_edges_1,
    size_t& n_incorrect_edges_2,
    double& init_error,
    double& final_error)
{

  // init g2o
  g2o::SparseOptimizer optimizer;
  setupG2o(&optimizer);

  list<EdgeContainerSE3> edges;
  set<Point*> mps;
  list<Frame*> neib_kfs;
  size_t v_id = 0;
  size_t n_mps = 0;
  size_t n_fix_kfs = 0;
  size_t n_var_kfs = 1;
  size_t n_edges = 0;
  n_incorrect_edges_1 = 0;
  n_incorrect_edges_2 = 0;

  // Add all core keyframes
  for(set<FramePtr>::iterator it_kf = core_kfs->begin(); it_kf != core_kfs->end(); ++it_kf)
  {
    g2oFrameSE3* v_kf = createG2oFrameSE3(it_kf->get(), v_id++, false);
    (*it_kf)->v_kf_ = v_kf;
    ++n_var_kfs;
    assert(optimizer.addVertex(v_kf));

    // all points that the core keyframes observe are also optimized:
    for(Features::iterator it_pt=(*it_kf)->fts_.begin(); it_pt!=(*it_kf)->fts_.end(); ++it_pt)
      if((*it_pt)->point != NULL)
        mps.insert((*it_pt)->point);
  }

  // Now go throug all the points and add a measurement. Add a fixed neighbour
  // Keyframe if it is not in the set of core kfs
  double reproj_thresh_2 = Config::lobaThresh() / center_kf->cam_->errorMultiplier2();
  double reproj_thresh_1 = Config::poseOptimThresh() / center_kf->cam_->errorMultiplier2();
  double reproj_thresh_1_squared = reproj_thresh_1*reproj_thresh_1;
  for(set<Point*>::iterator it_pt = mps.begin(); it_pt!=mps.end(); ++it_pt)
  {
    // Create point vertex
    g2oPoint* v_pt = createG2oPoint((*it_pt)->pos_, v_id++, false);
    (*it_pt)->v_pt_ = v_pt;
    assert(optimizer.addVertex(v_pt));
    ++n_mps;

    // Add edges
    list<Feature*>::iterator it_obs=(*it_pt)->obs_.begin();
    while(it_obs!=(*it_pt)->obs_.end())
    {
      Vector2d error = vk::project2d((*it_obs)->f) - vk::project2d((*it_obs)->frame->w2f((*it_pt)->pos_));

      if((*it_obs)->frame->v_kf_ == NULL)
      {
        // frame does not have a vertex yet -> it belongs to the neib kfs and
        // is fixed. create one:
        g2oFrameSE3* v_kf = createG2oFrameSE3((*it_obs)->frame, v_id++, true);
        (*it_obs)->frame->v_kf_ = v_kf;
        ++n_fix_kfs;
        assert(optimizer.addVertex(v_kf));
        neib_kfs.push_back((*it_obs)->frame);
      }

      // create edge
      g2oEdgeSE3* e = createG2oEdgeSE3((*it_obs)->frame->v_kf_, v_pt,
                                       vk::project2d((*it_obs)->f),
                                       true,
                                       reproj_thresh_2*Config::lobaRobustHuberWidth(),
                                       1.0 / (1<<(*it_obs)->level));
      assert(optimizer.addEdge(e));
      edges.push_back(EdgeContainerSE3(e, (*it_obs)->frame, *it_obs));
      ++n_edges;
      ++it_obs;
    }
  }

  // structure only
  g2o::StructureOnlySolver<3> structure_only_ba;
  g2o::OptimizableGraph::VertexContainer points;
  for (g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer.vertices().begin(); it != optimizer.vertices().end(); ++it)
  {
    g2o::OptimizableGraph::Vertex* v = static_cast<g2o::OptimizableGraph::Vertex*>(it->second);
      if (v->dimension() == 3 && v->edges().size() >= 2)
        points.push_back(v);
  }
  structure_only_ba.calc(points, 10);

  // Optimization
  if(Config::lobaNumIter() > 0)
    runSparseBAOptimizer(&optimizer, Config::lobaNumIter(), init_error, final_error);

  // Update Keyframes
  for(set<FramePtr>::iterator it = core_kfs->begin(); it != core_kfs->end(); ++it)
  {
    (*it)->T_f_w_ = SE3( (*it)->v_kf_->estimate().rotation(),
                         (*it)->v_kf_->estimate().translation());
    (*it)->v_kf_ = NULL;
  }

  for(list<Frame*>::iterator it = neib_kfs.begin(); it != neib_kfs.end(); ++it)
    (*it)->v_kf_ = NULL;

  // Update Mappoints
  for(set<Point*>::iterator it = mps.begin(); it != mps.end(); ++it)
  {
    (*it)->pos_ = (*it)->v_pt_->estimate();
    (*it)->v_pt_ = NULL;
  }

  // Remove Measurements with too large reprojection error
  double reproj_thresh_2_squared = reproj_thresh_2*reproj_thresh_2;
  for(list<EdgeContainerSE3>::iterator it = edges.begin(); it != edges.end(); ++it)
  {
    if(it->edge->chi2() > reproj_thresh_2_squared) //*(1<<it->feature_->level))
    {
      map->removePtFrameRef(it->frame, it->feature);
      ++n_incorrect_edges_2;
    }
  }

  // TODO: delete points and edges!
  init_error = sqrt(init_error)*center_kf->cam_->errorMultiplier2();
  final_error = sqrt(final_error)*center_kf->cam_->errorMultiplier2();
}

```