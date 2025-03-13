# DepthFilter

https://www.cnblogs.com/luyb/p/5773691.html

![image.png](DepthFilter%2017b71bdab3cf80b2ace7f0e612d2c1c4/image.png)

![image.png](image%203.png)

steps:

1. [findepipolarmatch](https://www.notion.so/findEpipolarMatchDirect-17e71bdab3cf804c9c9fe24b873605db?pvs=21)
2. calculate tau
3. update seed: a, b, mu, sigma2

![image.png](DepthFilter%2017b71bdab3cf80b2ace7f0e612d2c1c4/image%201.png)

![image.png](DepthFilter%2017b71bdab3cf80b2ace7f0e612d2c1c4/image%202.png)

### Depth uncertainty

![image.png](DepthFilter%2017b71bdab3cf80b2ace7f0e612d2c1c4/image%203.png)

![image.png](DepthFilter%2017b71bdab3cf80b2ace7f0e612d2c1c4/image%204.png)

![image.png](DepthFilter%2017b71bdab3cf80b2ace7f0e612d2c1c4/image%205.png)

![image.png](DepthFilter%2017b71bdab3cf80b2ace7f0e612d2c1c4/image%206.png)

```cpp
double DepthFilter::computeTau(
      const SE3& T_ref_cur,
      const Vector3d& f,
      const double z,
      const double px_error_angle)
{
  Vector3d t(T_ref_cur.translation());
  Vector3d a = f*z-t;
  double t_norm = t.norm();
  double a_norm = a.norm();
  double alpha = acos(f.dot(t)/t_norm); // dot product
  double beta = acos(a.dot(-t)/(t_norm*a_norm)); // dot product
  double beta_plus = beta + px_error_angle;
  double gamma_plus = PI-alpha-beta_plus; // triangle angles sum to PI
  double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
  return (z_plus - z); // tau
}
```

### UpdateSeed

update: a, b,  mu, sigma2 for **inverse depth**.

![image.png](DepthFilter%2017b71bdab3cf80b2ace7f0e612d2c1c4/image%207.png)

更新过程（*Vogiatzis的Supplementary matterial*）如下：

https://george-vogiatzis.org/publications/ivcj2010supp.pdf

![image.png](DepthFilter%2017b71bdab3cf80b2ace7f0e612d2c1c4/image%208.png)

![image.png](DepthFilter%2017b71bdab3cf80b2ace7f0e612d2c1c4/image%209.png)

```cpp
void DepthFilter::updateSeed(const float x, const float tau2, Seed* seed)
{
  float norm_scale = sqrt(seed->sigma2 + tau2);
  if(std::isnan(norm_scale))
    return;
  boost::math::normal_distribution<float> nd(seed->mu, norm_scale);
  float s2 = 1./(1./seed->sigma2 + 1./tau2);
  float m = s2*(seed->mu/seed->sigma2 + x/tau2);
  float C1 = seed->a/(seed->a+seed->b) * boost::math::pdf(nd, x);
  float C2 = seed->b/(seed->a+seed->b) * 1./seed->z_range;
  float normalization_constant = C1 + C2;
  C1 /= normalization_constant;
  C2 /= normalization_constant;
  float f = C1*(seed->a+1.)/(seed->a+seed->b+1.) + C2*seed->a/(seed->a+seed->b+1.);
  float e = C1*(seed->a+1.)*(seed->a+2.)/((seed->a+seed->b+1.)*(seed->a+seed->b+2.))
          + C2*seed->a*(seed->a+1.0f)/((seed->a+seed->b+1.0f)*(seed->a+seed->b+2.0f));

  // update parameters
  float mu_new = C1*m+C2*seed->mu;
  seed->sigma2 = C1*(s2 + m*m) + C2*(seed->sigma2 + seed->mu*seed->mu) - mu_new*mu_new;
  seed->mu = mu_new;
  seed->a = (e-f)/(f-e/f);
  seed->b = seed->a*(1.0f-f)/f;
}
```