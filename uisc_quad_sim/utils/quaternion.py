import numpy as np
import numba

@numba.njit()
def q_inv(q):
    return np.array([q[0],-q[1],-q[2],-q[3]])

@numba.njit()
def q_mult(q1,q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

@numba.njit()
def q_rot(q,v):
    qv = np.array([0,v[0],v[1],v[2]])
    q_inv_v = q_inv(q)
    return q_mult(q_mult(q,qv),q_inv_v)[1:]

@numba.njit()
def q_mat(q):
    w,x,y,z = q
    return np.array(
        [
            [1-2*y*y-2*z*z,2*x*y-2*w*z,2*x*z+2*w*y],
            [2*x*y+2*w*z,1-2*x*x-2*z*z,2*y*z-2*w*x],
            [2*x*z-2*w*y,2*y*z+2*w*x,1-2*x*x-2*y*y]
        ]
    )

def mat_q(rot):
    # Stable implementation of quaternion from rotation matrix
    from scipy.spatial.transform import Rotation as R
    r = R.from_matrix(rot)
    return r.as_quat(scalar_first=True)
    # # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    r11, r12, r13 = rot[0]
    r21, r22, r23 = rot[1]
    r31, r32, r33 = rot[2]
    tr = r11 + r22 + r33
    r = np.sqrt(1 + tr)
    s = 1 / (2*r)
    w = 0.5 * r
    x = (r32 - r23) * s
    y = (r13 - r31) * s
    z = (r21 - r12) * s
    return np.array([w, x, y, z])

if __name__ == "__main__":
    import scipy.spatial.transform as st
    for _ in range(100):
        q = np.random.rand(4)
        q /= np.linalg.norm(q)
        v = np.random.rand(3)
        
        m1 = st.Rotation.from_quat(q,scalar_first=True).as_matrix()
        m2 = q_mat(q)
        assert np.allclose(m1,m2)

        v1 = m1 @ v
        v2 = q_rot(q,v)
        assert np.allclose(v1,v2)
    print("Quaternion test passed")