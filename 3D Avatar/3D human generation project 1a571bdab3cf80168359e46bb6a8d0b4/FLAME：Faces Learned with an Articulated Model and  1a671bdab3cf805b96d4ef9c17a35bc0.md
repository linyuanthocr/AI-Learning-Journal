# FLAME：Faces Learned with an Articulated Model and Expressions

In the paper *Learning a Model of Facial Shape and Expression from 4D Scans*, the authors introduce **FLAME**, a 3D morphable model of facial shape and expression. 

![image.png](FLAME%EF%BC%9AFaces%20Learned%20with%20an%20Articulated%20Model%20and%20%201a671bdab3cf805b96d4ef9c17a35bc0/image.png)

To build this model from 4D (time-sequence 3D) scans, they employ a **non-rigid registration process** to align different facial scans to a common topology. Here’s how they do it:

---

![image.png](FLAME%EF%BC%9AFaces%20Learned%20with%20an%20Articulated%20Model%20and%20%201a671bdab3cf805b96d4ef9c17a35bc0/image%201.png)

### **1. Alignment of 3D Scans (Non-Rigid Registration)**

To construct FLAME, the authors process thousands of 3D facial scans with different expressions. The key steps in aligning these scans include:

### **a) Initial Rigid Alignment**

- The raw 3D scans are first **rigidly aligned** to a common coordinate system using **ICP (Iterative Closest Point)**. This ensures that each scan is roughly positioned correctly in space.
- The rigid alignment only corrects for **translation and rotation** but does not adjust for shape differences.

### **b) Non-Rigid Registration to a Template Mesh**

- A **template mesh** (a predefined facial topology) is used as a reference.
- Each 3D scan is **non-rigidly deformed** to fit this template using an **as-rigid-as-possible (ARAP) deformation** strategy.
- A **coarse-to-fine approach** is used:
    - First, large-scale deformations match the template to the scan.
    - Then, finer details (like wrinkles and subtle expression changes) are captured.
- **SMPL body model formulation**
    - SMPL is a parameterized blend-skinned body model that combines an identity shape space, articulated pose, and pose-dependent corrective blendshapes.
    
    ![image.png](FLAME%EF%BC%9AFaces%20Learned%20with%20an%20Articulated%20Model%20and%20%201a671bdab3cf805b96d4ef9c17a35bc0/image%202.png)
    

## Model formulation

![image.png](FLAME%EF%BC%9AFaces%20Learned%20with%20an%20Articulated%20Model%20and%20%201a671bdab3cf805b96d4ef9c17a35bc0/image%203.png)

![image.png](FLAME%EF%BC%9AFaces%20Learned%20with%20an%20Articulated%20Model%20and%20%201a671bdab3cf805b96d4ef9c17a35bc0/image%204.png)

![image.png](FLAME%EF%BC%9AFaces%20Learned%20with%20an%20Articulated%20Model%20and%20%201a671bdab3cf805b96d4ef9c17a35bc0/image%205.png)

![image.png](FLAME%EF%BC%9AFaces%20Learned%20with%20an%20Articulated%20Model%20and%20%201a671bdab3cf805b96d4ef9c17a35bc0/image%206.png)

![image.png](FLAME%EF%BC%9AFaces%20Learned%20with%20an%20Articulated%20Model%20and%20%201a671bdab3cf805b96d4ef9c17a35bc0/image%207.png)

### Temporal registration

The registration process computes an aligned template Ti ∈ R3N. The registration pipeline alternates between registering meshes while regularizing to a FLAME model and training a FLAME
model from the registrations as shown in Figure 4.