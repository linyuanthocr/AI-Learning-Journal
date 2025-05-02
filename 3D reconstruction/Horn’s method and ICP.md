# Horn’s method & ICP

---

## Horn's Method: A Closed-Form Solution for Absolute Orientation

In 1987, Berthold K. P. Horn published a seminal paper titled "Closed-form solution of absolute orientation using unit quaternions," providing a closed-form solution to the absolute orientation problem. This problem involves determining the rotation, translation, and scale that best align two sets of corresponding 3D points. Horn's method is widely used in computer vision, robotics, and photogrammetry.([CVL](https://www.cvl.iis.u-tokyo.ac.jp/~oishi/Papers/Alignment/Horn_Closed-form_JOSA1987.pdf?utm_source=chatgpt.com), [Robotics Knowledgebase](https://roboticsknowledgebase.com/wiki/math/registration-techniques/?utm_source=chatgpt.com))

![image.png](images/Horn%E2%80%99s%20method%20&%20ICP%201e771bdab3cf802c828ce7070a8aa1fa/image.png)

![image.png](images/Horn%E2%80%99s%20method%20&%20ICP%201e771bdab3cf802c828ce7070a8aa1fa/image%201.png)

![image.png](images/Horn%E2%80%99s%20method%20&%20ICP%201e771bdab3cf802c828ce7070a8aa1fa/image%202.png)

### Advantages of Horn's Method

- **Closed-Form Solution**: Provides an exact solution without iterative optimization.
- **Robustness**: Utilizes all available point correspondences, leading to a stable solution.
- **Efficiency**: Computationally efficient, suitable for real-time applications.
- **Versatility**: Applicable to various fields such as robotics, computer vision, and photogrammetry.([CVL](https://www.cvl.iis.u-tokyo.ac.jp/~oishi/Papers/Alignment/Horn_Closed-form_JOSA1987.pdf?utm_source=chatgpt.com), [Cnblogs](https://www.cnblogs.com/matlabworld/p/17983218?utm_source=chatgpt.com), [ResearchGate](https://www.researchgate.net/publication/2620318_A_New_Closed_Form_Approach_to_the_Absolute_Orientation_Problem?utm_source=chatgpt.com))

### Applications

- **Camera Pose Estimation**: Determining the position and orientation of a camera in space.
- **3D Object Alignment**: Aligning 3D models or point clouds.
- **Robotic Navigation**: Calculating the transformation between different coordinate frames.
- **Augmented Reality**: Overlaying virtual objects onto the real world by aligning coordinate systems.

---

# ICP

![image.png](images/Horn%E2%80%99s%20method%20&%20ICP%201e771bdab3cf802c828ce7070a8aa1fa/image%203.png)

![image.png](images/Horn%E2%80%99s%20method%20&%20ICP%201e771bdab3cf802c828ce7070a8aa1fa/image%204.png)

![image.png](images/Horn%E2%80%99s%20method%20&%20ICP%201e771bdab3cf802c828ce7070a8aa1fa/image%205.png)

![image.png](images/Horn%E2%80%99s%20method%20&%20ICP%201e771bdab3cf802c828ce7070a8aa1fa/image%206.png)

![image.png](images/Horn%E2%80%99s%20method%20&%20ICP%201e771bdab3cf802c828ce7070a8aa1fa/image%207.png)

![image.png](images/Horn%E2%80%99s%20method%20&%20ICP%201e771bdab3cf802c828ce7070a8aa1fa/image%208.png)

![image.png](images/Horn%E2%80%99s%20method%20&%20ICP%201e771bdab3cf802c828ce7070a8aa1fa/image%209.png)

![image.png](images/Horn%E2%80%99s%20method%20&%20ICP%201e771bdab3cf802c828ce7070a8aa1fa/image%2010.png)
