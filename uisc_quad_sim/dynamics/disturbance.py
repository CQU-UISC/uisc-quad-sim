# raw author: https://github.com/KevinHuang8/DATT/blob/main/python_utils/wind.py
import numpy as np
import numba as nb


@nb.njit
def wind_velocity_field(
    pos, w_pos, w_dir, radius, coneslope, vmax, decay_lat, decay_long
):
    """
    Calculates the wind velocity vector at a specific position.

    Args:
        pos (np.ndarray): Target position [x, y, z], shape (3,)
        w_pos (np.ndarray): Wind source position, shape (3,)
        w_dir (np.ndarray): Wind direction (normalized), shape (3,)
        radius (float): Radius of the wind column at the source
        coneslope (float): Slope of the wind cone expansion
        vmax (float): Maximum wind velocity
        decay_lat (float): Lateral decay rate
        decay_long (float): Longitudinal decay rate

    Returns:
        np.ndarray: Wind velocity vector at pos, shape (3,)
    """
    # Vector from source to target
    dist = pos - w_pos

    # Project distance onto the wind direction (longitudinal distance)
    r_para = np.dot(dist, w_dir)

    # If the target is behind the wind source, there is no wind
    if r_para < 0:
        return np.zeros(3)

    # Calculate perpendicular distance to the wind axis
    # Cross product magnitude gives the area of the parallelogram, divided by norm(w_dir)=1 gives height
    r_perp_vec = np.cross(dist, w_dir)
    r_perp = np.linalg.norm(r_perp_vec)

    # Calculate effective radius at this longitudinal distance
    r_eff = radius + coneslope * r_para

    # Calculate distance outside the effective radius (0 if inside)
    r_perp_excess = max(0.0, r_perp - r_eff)

    # Calculate magnitude using exponential decay
    # Decay laterally based on excess distance, longitudinally based on distance from source
    magnitude = vmax * np.exp(-decay_lat * r_perp_excess) * np.exp(-decay_long * r_para)

    return magnitude * w_dir


class Disturbance:
    name = "D_BASE"

    def force(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def moment(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def __str__(self) -> str:
        return "Base Disturbance"

    def field(self, box=(5, 5, 3), n=(12, 12, 6)) -> np.ndarray:
        # return force field
        nx, ny, nz = n
        x_f, y_f, z_f = box
        x, y, z = np.meshgrid(
            np.linspace(-x_f, x_f, nx),
            np.linspace(-y_f, y_f, ny),
            np.linspace(0, z_f, nz),
            indexing="ij",
        )

        output = np.zeros((nx, ny, nz, 3))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    pos = np.array([(x[i, j, k], y[i, j, k], z[i, j, k])])
                    output[i, j, k] = self.force(pos.T, np.zeros((3, 1))).T
        return np.array(
            (x, y, z, output[:, :, :, 0], output[:, :, :, 1], output[:, :, :, 2])
        )


class EmptyField(Disturbance):
    name = "D_EMPTY"

    def force(self, x, u):
        return np.zeros((3))

    def moment(self, x, u):
        return np.zeros((3))

    def __str__(self):
        return "Empty Field"


class AirDrag(Disturbance):
    name = "D_AIR_DRAG"

    def __init__(self, drag_coeff: np.ndarray):
        self.drag_coeff = drag_coeff

    def force(self, x, u):
        v = x[3:6]  # Velocity components
        return -self.drag_coeff * v

    def moment(self, x, u):
        return np.zeros((3))

    def __str__(self):
        return f"Air Drag Field with coefficient {self.drag_coeff}"


class ConstField(Disturbance):
    name = "D_CONST"

    def __init__(self, force, moment):
        # force and torque has shape (3,)
        self.force_val = force
        self.moment_val = moment

    def force(self, x, u):
        return self.force_val

    def moment(self, x, u):
        return self.moment_val

    def __str__(self):
        return f"Const Field: {self.force_val}, {self.moment_val}"


class TimeVarField(Disturbance):
    name = "D_TVAR"

    def __init__(self, force, moment, dt):
        # force and torque has shape (3,)
        self.force_val = force
        self.moment_val = moment
        self.tf = 0
        self.tm = 0
        self.dt = dt

    def force(self, x, u):
        force_val = np.sin(self.tf) * self.force_val
        self.tf += self.dt
        return force_val

    def moment(self, x, u):
        moment_val = np.cos(self.tm) * self.moment_val
        self.tm += self.dt
        return moment_val

    def __str__(self):
        return f"TimeVar Field: {self.force_val}, {self.moment_val}"


class TimeVarConstField(Disturbance):
    name = "D_TVAR_CONST"

    def __init__(self, force, moment, dt, T):
        # force and torque has shape (3,)
        self.force_val = force
        self.moment_val = moment
        self.T = T
        self.tf = 0
        self.tm = 0
        self.dt = dt

    def force(self, x, u):
        scale = np.clip(self.tf / self.T, 0, 1)  # Ensure scale is between 0 and 1
        force_val = scale * self.force_val
        self.tf += self.dt
        return force_val

    def moment(self, x, u):
        scale = np.clip(self.tm / self.T, 0, 1)  # Ensure scale is between 0 and 1
        moment_val = scale * self.moment_val
        self.tm += self.dt
        return moment_val

    def __str__(self):
        return f"TimeVarConst Field: {self.force_val}, {self.moment_val}"


class WindField(Disturbance):
    name = "D_WIND"

    def __init__(
        self,
        pos,
        to,
        vmax=10.0,
        radius=0.2,
        noisevar=0.5,
        decay_lat=4,
        decay_long=0.6,
        dispangle=15,
    ):
        """
        Initialize the Wind Field disturbance.

        Args:
            pos (np.ndarray): Origin of the wind source.
            to (np.ndarray): Target point to define wind direction.
            vmax (float): Maximum velocity of the wind.
            radius (float): Radius of the non-decaying wind core.
            noisevar (float): Variance of the noise added to the wind force.
            decay_lat (float): Coefficient for lateral decay.
            decay_long (float): Coefficient for longitudinal decay.
            dispangle (float): Dispersion angle of the wind cone in degrees.
        """
        self.pos = np.array(pos, dtype=np.float64)
        direction = np.array(to, dtype=np.float64) - self.pos
        self.dir = direction / np.linalg.norm(direction)

        self.vmax = vmax
        self.radius = radius
        self.noisevar = noisevar
        self.decay_lat = decay_lat
        self.decay_long = decay_long
        self.coneslope = np.tan(np.radians(dispangle))

    def force(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Calculate the external force exerted by the wind on the rigid body.

        Args:
            x (np.ndarray): State vector, shape (19,). x[0:3] is position.
            u (np.ndarray): Control input (unused here).

        Returns:
            np.ndarray: Force vector in inertial frame, shape (3,)
        """
        # Extract position from state (assuming x[0:3] is position)
        current_pos = x[0:3]

        # Calculate deterministic wind velocity field using JIT function
        wind_vel = wind_velocity_field(
            current_pos,
            self.pos,
            self.dir,
            self.radius,
            self.coneslope,
            self.vmax,
            self.decay_lat,
            self.decay_long,
        )

        # Calculate noise scale relative to current wind intensity
        # If wind_vel is small, noise should also be small
        wind_speed = np.linalg.norm(wind_vel)
        if wind_speed > 1e-6:
            noise_scale = wind_speed / self.vmax
        else:
            noise_scale = 0.0

        # Generate random noise (Gaussian)
        # Note: np.random is not JIT-compiled here, which is fine for the python class
        noise = self.noisevar * noise_scale * np.random.normal(size=3)

        # Resulting force (Assuming a simplified drag model: F ~ constant * v_wind)
        # Your previous code used 0.5 factor, the reference used 0.2.
        # Kept 0.5 to match your original logic.
        f = 0.5 * wind_vel + noise

        return f

    def moment(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Calculate the external moment (assumed zero for simple wind model).
        """
        return np.zeros(3)

    def __str__(self):
        return f"""Wind Field:
    pos: {self.pos}
    dir: {self.dir}
    vmax: {self.vmax}
    radius: {self.radius}
    noisevar: {self.noisevar}
    decay_lat: {self.decay_lat}
    decay_long: {self.decay_long}
    coneslope: {self.coneslope}"""


class CompositeField(Disturbance):
    name = "D_COMPOSITE"

    def __init__(self, *disturbances: Disturbance):
        self.disturbances: list[Disturbance] = []
        for d in disturbances:
            if not isinstance(d, Disturbance):
                raise TypeError(f"Expected Disturbance, got {type(d)}")
            self.disturbances.append(d)

    def add(self, disturbance: Disturbance):
        if not isinstance(disturbance, Disturbance):
            raise TypeError(f"Expected Disturbance, got {type(disturbance)}")
        self.disturbances.append(disturbance)

    def clear(self):
        self.disturbances = []

    def __len__(self):
        return len(self.disturbances)

    def force(self, x, u):
        f = np.zeros(3)
        for d in self.disturbances:
            f += d.force(x, u)
        return f

    def moment(self, x, u):
        m = np.zeros(3)
        for d in self.disturbances:
            m += d.moment(x, u)
        return m

    def __str__(self):
        return "Composite Disturbance: " + ", ".join(
            [str(d) for d in self.disturbances]
        )
